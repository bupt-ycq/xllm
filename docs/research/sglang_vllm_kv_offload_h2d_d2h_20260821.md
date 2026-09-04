# SGLang / vLLM KV Cache Offload 的 H2D / D2H 优化调研

日期：2026-08-21

范围：分析上游 KV cache offload 的 GPU/Host 数据通路及其调度，并对照本地 xLLM 当前分支。

代码快照：正文链接固定到 vLLM [`36bad1b90cb2958b6de4abdb5a9249b136f39720`](https://github.com/vllm-project/vllm/commit/36bad1b90cb2958b6de4abdb5a9249b136f39720) 和 SGLang [`0db2c53deca12715d209ffbd46bdfafdfd5a5d35`](https://github.com/sgl-project/sglang/commit/0db2c53deca12715d209ffbd46bdfafdfd5a5d35)。结稿时 vLLM `main` 已前进到 [`e85d1b69cf2f1c6101cfc7c799bb0c457cacc4b3`](https://github.com/vllm-project/vllm/commit/e85d1b69cf2f1c6101cfc7c799bb0c457cacc4b3)，但本文涉及的 `gpu_worker.py`、`swap_blocks_triton.py` 和 `cache_kernels.cu` 与前一快照逐字相同。

本地快照：分支 `feat/mlu-tmo-kv-transfer-new`，HEAD `2509b8a3f58d64dfe20241d0c0d0268cd45809ee`；同时审计当前 worktree，明确区分尚未提交的统计改动。
证据口径：上游只使用官方仓库源码、官方文档和官方 issue；本地结论来自静态代码审计。本文没有复现实机带宽或延迟。

## 结论先行

两边的共同基线已经不是简单的逐块同步 `memcpy`：都预分配 pinned Host KV 池，使用独立 H2D/D2H stream 和 event，把多个离散 KV 段合批，并在数据仍被计算使用时做引用保护。不过二者最值得关注的优化方向不同：

1. **SGLang HiCache 的核心优势是 layer-wise H2D 流水。** 一个 H2D stream 按层搬入并逐层记录完成事件，模型计算流只等待当前层，因此可形成“计算 layer N、同时加载 layer N+1”的流水；D2H 则一次提交 all-layer write-back。官方设计文档也明确描述这一重叠策略。证据见 [`l2_transfer.py#L49-L113`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/l2_transfer.py#L49-L113)、[`cache_controller.py#L50-L97`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/managers/cache_controller.py#L50-L97) 和[设计文档 `#L102-L109`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/docs/docs/advanced_features/hicache_design.mdx#L102-L109)。
2. **vLLM 默认 native `OffloadingConnector` 的核心优势是批量 DMA 与自适应 H2D 后端。** D2H 总选 batch DMA；CUDA H2D 在“小于 28 KiB/descriptor、8 字节对齐且至少 16 个 descriptor”等条件下改用 Triton GPU-pull，否则使用 `cuMemcpyBatchAsync`。阈值是 H100 PCIe Gen5 上的经验值。证据见 [`gpu_worker.py#L38-L61`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/cpu/gpu_worker.py#L38-L61) 和 [`swap_blocks_triton.py#L12-L74`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/cpu/swap_blocks_triton.py#L12-L74)。
3. **最新 vLLM 已经对 load 和 store 都等待当前 compute stream。** 这不是只给 D2H 加依赖：H2D 也必须等待目标 GPU block 上已经排队的写入/清零，避免新 KV 被旧 compute work 覆盖。只有 H2D 的 Host source 是稳定只读，才传 `CU_MEMCPY_SRC_ACCESS_ORDER_ANY` 允许驱动乱序预取；D2H 必须保留 `STREAM` 源顺序。证据见 [`gpu_worker.py#L637-L663`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/cpu/gpu_worker.py#L637-L663)。
4. **两边都利用了 UVA/mapped pinned Host memory，但含义要准确。** vLLM 的小块 Triton H2D kernel 和 SGLang 的 GPU-assisted kernel 都由 GPU SM 直接解引用 pinned Host 地址，把 payload 拉到 GPU；它省掉的是大量小 DMA 提交/中间 payload staging，不等于 Host memory 不经过 PCIe，也不是 storage 到 GPU 的 GPUDirect Storage。
5. **两边都支持 CUDA 12.8 batch copy，但成熟度和 fallback 不同。** vLLM native 将整个 logical job 的 layer/group/block descriptors 合成一次 `cuMemcpyBatchAsync`，旧驱动回退多个 `cudaMemcpyAsync`；SGLang `page_first_direct` 将 page×layer×K/V descriptors 合成一次 batch copy，旧环境回退逐 page 的 non-blocking copy。SGLang 的 HIP batch 分支在当前源码中显式关闭，vLLM native 则在 ROCm 7.1+ 调 `hipMemcpyBatchAsync`。
6. **默认 offload 策略并不相同。** vLLM 默认只 offload prompt，支持 chunk 粒度、`store_threshold`、LRU/ARC 和 per-request `max_offload_tokens`，并把 D2H 推迟到下一个 engine step、采样之后；SGLang 支持 `write_through`、`write_through_selective`、`write_back`，以及 L3 prefetch 的 `best_effort`/`wait_complete`/`timeout`。
7. **不要把 L2↔L3 zero-copy/GDS 能力误报成 L1 GPU↔L3 直连。** vLLM 官方文档明确 secondary tier 一律经 CPU primary staging。SGLang 的 `page_first`/`page_first_direct` 能把一页作为连续 Host object 传给 L3，NIXL 也可使用 GDS 类插件，但当前 HiCache 路径仍以预注册 Host region 为 L3 I/O 目标/来源；这些“zero-copy”主要指 L2 与 L3 间省掉额外 Host bounce copy。

## 对当前 xLLM 分支的直接结论

### 已经具备，不建议重复实现

| 能力 | 当前 xLLM 证据 | 判断 |
| --- | --- | --- |
| Pinned Host KV pool | `xllm/core/platform/mlu/mlu_host_memory.cpp:50-83` 使用 `cnHostMemAlloc(..., CN_MEMHOSTALLOC_PORTABLE)`；`kv_cache_utils.cpp:397-405` 接入 | 已有，和上游共同基线一致 |
| Block/page-first Host layout | `kv_cache_utils.cpp:598-617` 形成 `[host_blocks, layers, ...]`；`hierarchy_kv_cache_transfer.cpp:455-463` 保留层维 | 已有，L2 物理布局已经利于同一 block 的跨层聚合 |
| H2D layer-range overlap | `hierarchy_kv_cache_transfer.cpp:124-143,664-685` 每个 range 提交后记录 event；`mlu_layer_synchronizer.cpp:89-127` 由 compute stream 等待 | 已有，但默认约 4 个 range，不是 SGLang 的严格逐层粒度 |
| Direct 合并与 batch copy | `mlu_batch_memcpy.cpp:189-330,1129-1260` 已有 layer-major flatten、连续 extent、固定 stride 2D 和 batch fallback | 已有，是当前主路径的核心优化 |
| TMO 跨层融合 | `mlu_batch_memcpy.cpp:663-700,1389-1476` 跨层提交，并缓存静态 paged-buffer pointer tensor | 已有；动态 metadata 仍可优化 |
| Indexed staging 快路 | `mlu_indexed_copy.cpp:480-653,691-956` 有 pinned Host/device staging pool、callback 回收和最多 4 个 slot | 已实现但默认关闭，尚未成为生产路由 |
| Restore 带宽预算 | `unified_policy.cpp:151-167,460-607` 有 copy-unit budget、partial restore 和 copy-vs-compute latency 选择 | 已有，是当前分支相对简单上游策略的一项优势 |
| D2H block 聚合与引用保护 | `hierarchy_block_manager_pool.cpp:985-1057` 每个 DP 聚合 offload pairs，完成回调前保留 Block | 已有；但后端仍按 layer range 多次同步 |

### 当前没有、最值得补的部分

| 优先级 | 缺口 | 当前实现与影响 | 建议 |
| --- | --- | --- | --- |
| **P0 正确性** | H2D 缺少 compute→copy destination 依赖 | D2H 在 `hierarchy_kv_cache_transfer.cpp:629-635` 先让 copy stream 等 compute stream；H2D `:651-675` 直接提交。新 forward 等 H2D event，只保护后续读取，不能自动证明旧 compute 已停止写复用的 GPU block | 首次 H2D submit 前增加 copy stream 对 current compute stream 的依赖；除非能够形式化并用测试证明“旧 compute 完成前 block 绝不复用” |
| **P0 生命周期** | H2D completion 不是一等 job | `CopyStreamLease` 在函数返回即归还 stream（`:87-122`），而 Direct H2D 是 submit-only；source/destination block ref、descriptor scratch、stream token 的安全依赖外部隐含约束，也没有 pending bytes/jobs 限流 | 建立方向化 transfer job；end event 完成后再释放 job 持有资源，并增加 bytes/jobs credit |
| **P0 性能** | D2H 每个 layer range 都 `cnQueueSync` | `offload_to_host()` 在 range 循环内调用 D2H（`:636-647`）；Direct 在 `mlu_batch_memcpy.cpp:1263-1274`、TMO 在 `:1469-1476` 每次同步。默认可产生约 4 次以上 Host 阻塞 | 聚合所有 range 的 descriptor/metadata，一次或连续异步 submit，只在全部 range 末尾记录一个 completion event 或同步一次 |
| **P0 资源** | 取消请求会遗留 synchronizer/event | 注册在 `hierarchy_kv_cache_transfer.cpp:527-545`，正常 erase 仅在 `:697-709`；未到达 forward 的 batch 没有清理路径 | 增加 cancellation/timeout cleanup，或统一归入 completion registry |
| **P1** | H2D/D2H 共用 stream pool | 两个 H2D worker 与 D2H caller 共用 `hierarchy_kv_cache_transfer.cpp:253-265` 的 6 条 stream；D2H sync 可能顺带等待此前 H2D，形成方向间 HOL | 像两套上游实现一样拆成独立 handler/pool/credit，并独立调并发度 |
| **P1** | Direct copy plan 和 descriptor scratch 每次重建 | `hierarchy_kv_cache_transfer.cpp:373-500` 反复推导 view；`mlu_batch_memcpy.cpp:268-330,1129-1154` 重建 descriptor 及三组 batch arrays | 预计算静态 layer/role/base/stride plan；每条 stream 使用 grow-only pinned descriptor slab，只填动态 block offset |
| **P1** | H2D 未使用方向特化 source order | `mlu_batch_memcpy.cpp:1145-1147` 对两方向统一使用 `CN_MEMCPY_SRC_ACCESS_ORDER_QUEUE` | 在证明 Host block 生命周期覆盖 DMA 后 A/B H2D-only `ANY`；D2H 必须继续保持 stream order |
| **P1** | 缺少生产级 shape/direction router | Direct/TMO 是全局开关；Indexed 虽按 rows/bytes fallback，但为环境变量路径且默认关闭 | 按 direction、rows、bytes、stride、run/descriptor 数选择 extent/2D/batch、Indexed 或 TMO；阈值必须在 MLU 实测，不能照抄 H100 的 28 KiB |
| **P1** | D2H 缺少价值选择 | `hierarchy_block_manager_pool.cpp:369-405` 对所有符合引用条件的 completed blocks 排队；现有 selective 主要控制 H2D restore 量 | 引入 prompt-only、reuse/store threshold、per-request max tokens 或 selective/write-back policy，优先减少传输字节数 |
| **P1** | TMO 缺显式计算资源配额 | 当前 wrapper/call 没有类似 SGLang `block_quota` 的可调限制 | 增加可配置 core/block quota，并测 overlap 下模型计算退化 |
| **P2** | NUMA、显式 chunking | Host allocator 没有 first-class NUMA policy；已有 block-first + extent/2D 已覆盖主要连续性收益 | 先 profile 再做 NUMA；只有 metadata 或 L3 object 管理成为瓶颈时再引入 `blocks_per_chunk` |

### 一个需要先修的依赖链

正确顺序应当是：

```text
旧 compute 对可复用 GPU block 的最后写
                 │
                 ▼
      H2D copy stream 开始恢复 KV
                 │  range/layer completion event
                 ▼
新 compute stream 读取恢复后的对应层 KV
```

当前分支已经实现下半段“新 compute 等 H2D event”，但没有在 H2D 路径显式建立上半段依赖。vLLM 最新 native 实现同时保留两段依赖；官方 [issue #47282](https://github.com/vllm-project/vllm/issues/47282) 记录了 Simple connector 中 H2D 覆盖仍被 compute 使用的复用 block 后产生乱码的同类故障。另一方向也不能放松：[issue #45704](https://github.com/vllm-project/vllm/issues/45704) 记录了 D2H 未等待 compute、同时错误使用 `ANY` 后的静默 KV 损坏。它们不是 xLLM 缺陷的动态复现，但足以说明这两段 barrier 必须由代码或可验证的 block-lifecycle invariant 保证。

### 建议落地顺序

1. 补 H2D destination dependency，并增加“旧 compute 延迟写 destination、随后 restore”的回归测试。
2. 合并 D2H 全部 layer ranges，只保留一次最终完成；同时补多 range 只完成一次的测试。
3. 引入 completion-aware 的 H2D/D2H 独立 handler、stream/event/descriptor pool 和 inflight credits。
4. 缓存 Direct/TMO 静态 plan 与动态 metadata slab；补齐 per-direction queue wait、submit→complete、pending bytes/jobs、HOL 指标。
5. 把 Indexed/TMO 收敛成配置化的 shape-aware router，在 MLU 上建立阈值曲线。
6. 最后做 selective D2H；它通常比继续微调单次 memcpy API 更直接地降低总字节数。

测试缺口也应同步补齐：当前没有覆盖 H2D 与旧 compute 写 destination 的竞态、双向同池混跑、取消 batch 的 event 清理、D2H 多 range 单次完成、bounded inflight/backpressure。`tests/core/platform/mlu/mlu_batch_memcpy_test.cpp:880-920` 的 `SubmitH2DWaitsForCopyStream` 使用 TMO backend，只能证明 TMO 同步行为，不能覆盖 Direct async H2D 的生命周期。

当前 worktree 对 Direct 路径增加了一部分 run/descriptor/bytes 统计（`mlu_batch_memcpy.h:31-47`、`mlu_batch_memcpy.cpp:626-660,1119-1127`），但这些尚未提交，且仍缺 queue wait、完整传输时延和 pending credit 指标；本文没有修改或覆盖这些用户改动。

## 实现定位与默认性

| 项目 | 路径 | 状态 | 说明 |
| --- | --- | --- | --- |
| vLLM | `OffloadingConnector` + `CPUOffloadingSpec` | **主线默认 native 路径，但 KV offload 本身需显式配置** | `--kv-offloading-backend native` 最终选该 connector；pinned CPU 是 primary tier |
| vLLM | `TieringOffloadingSpec` | 主线可选 | CPU primary + filesystem/NIXL P2P 等 secondary tier；GPU 不直接访问 secondary |
| vLLM | `SimpleCPUOffloadConnector` | **主线内的环境变量门控路径** | 需 `VLLM_USE_SIMPLE_KV_OFFLOAD=1`；另含可选 disk backend |
| SGLang | HiCache | **主线内置、整体 opt-in** | 需 `--enable-hierarchical-cache`；启用后默认 Host layout 是 `page_first`，I/O backend 是 `kernel` |
| SGLang | HiCache `direct` / `page_first_direct` | 主线可选 | 走 async DMA、连续 range 合并和 CUDA batch copy，而非 GPU-assisted kernel |
| SGLang | Mooncake/NIXL/3FS/AIBrix 等 L3 | 插件/connector | L2↔L3 后端；不应与 L1↔L2 H2D/D2H 混为一谈 |
| SGLang | LMCache | 外部替代方案 | 官方文档称为 HiCache 的 alternative，不是 HiCache 内部默认路径 |

vLLM 的 connector 选择逻辑见 [`vllm/config/vllm.py#L1007-L1038`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/config/vllm.py#L1007-L1038)，Simple 路径开关默认 `0`，见 [`vllm/envs.py#L2125-L2128`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/envs.py#L2125-L2128)。SGLang 的 feature flag、Host mode、默认 write policy/I/O backend/layout 及参数归一化见 [`server_args.py#L2690-L2774`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/server_args.py#L2690-L2774)、[`server_args.py#L7422-L7509`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/server_args.py#L7422-L7509) 和 [`server_args.py#L7552-L7589`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/server_args.py#L7552-L7589)。

## vLLM：默认 native OffloadingConnector

### 1. Pinned Host memory 与整池预分配

- 普通 CPU tier 启动时直接按总 block 数分配完整 `torch.zeros(..., pin_memory=True)`；使用共享 mmap/canonical layout 时，则对整个 mmap 一次 `cudaHostRegister`。注册失败时可以继续工作，但会退化为 unpinned DMA。[`gpu_worker.py#L193-L220`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/cpu/gpu_worker.py#L193-L220)、[`gpu_worker.py#L740-L823`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/cpu/gpu_worker.py#L740-L823)。
- copy descriptor 的 source pointer、destination pointer、size 三个数组也在 pinned Host memory 中预分配，避免 batch API 读取 pageable metadata。[`gpu_worker.py#L223-L233`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/cpu/gpu_worker.py#L223-L233)。
- 共享区域使用 `/dev/shm` mmap，创建 tensor 时通过 `torch.frombuffer` 直接引用 mmap；`MADV_POPULATE_WRITE` 预触页，减少首次传输 page fault。worker view 和 canonical view 都只是对同一映射做视图，不复制 payload。[`shared_offload_region.py#L66-L168`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/cpu/shared_offload_region.py#L66-L168)、[`shared_offload_region.py#L173-L268`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/cpu/shared_offload_region.py#L173-L268)。

### 2. 异步 copy、stream/event pool 与正确性依赖

- H2D 和 D2H 各有独立 `SingleDirectionOffloadingHandler`，即两套方向独立的队列、stream/event/descriptor buffer pool。[`gpu_worker.py#L236-L242`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/cpu/gpu_worker.py#L236-L242)、[`gpu_worker.py#L328-L337`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/cpu/gpu_worker.py#L328-L337)、[`gpu_worker.py#L807-L838`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/cpu/gpu_worker.py#L807-L838)。
- 每个 transfer 使用一条独立 stream；同方向后一个 transfer 等待前一个 end event，维持 FIFO。**H2D、D2H 两者都先 `wait_stream(current_stream)`**；之后才分别使用 H2D `ANY` / D2H `STREAM` source order。[`gpu_worker.py#L623-L677`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/cpu/gpu_worker.py#L623-L677)。
- 完成路径只 `event.query()` 非阻塞轮询；完成后 stream、两个 event、三件 descriptor buffers 全部归还池中复用。[`gpu_worker.py#L682-L703`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/cpu/gpu_worker.py#L682-L703)。

这允许不同方向的 DMA 与模型计算异步共存，但 native worker 本身没有像 SGLang 那样发出“第 N 层已经 H2D 完成”的逐层事件：它对一次 request/job 聚合全部层的 descriptors，scheduler 在整个 load job 完成后才恢复请求。因此两者都能异步 overlap，只有 SGLang 是明确的 layer pipeline。

### 3. Batch/Chunk transfer 与 descriptor 合并

- 一个 transfer 会先遍历所有 KV group 和 canonical refs，把所有 block/sub-block 的源地址、目标地址、字节数写进同一组 descriptor arrays；CPU chunk 可以是多个 GPU blocks，首尾还可只搬 chunk 的局部范围。[`gpu_worker.py#L512-L621`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/cpu/gpu_worker.py#L512-L621)。
- CUDA 12.8+ 一次调用 `cuMemcpyBatchAsync` 提交整个 descriptor list，ROCm 7.1+ 使用 `hipMemcpyBatchAsync`；旧 runtime/driver 回退每 descriptor 一个 `cudaMemcpyAsync`。[`cache_kernels.cu#L79-L183`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/csrc/libtorch_stable/cache_kernels.cu#L79-L183)。
- H2D batch copy 的 `CU_MEMCPY_SRC_ACCESS_ORDER_ANY` 允许 DMA engine 对稳定 pinned source 提前/乱序取数，是 copy-engine 内部流水优化；D2H 的 live GPU source 不允许这样做。[`cache_kernels.cu#L130-L153`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/csrc/libtorch_stable/cache_kernels.cu#L130-L153)。
- 用户可用 `block_size` 或 `blocks_per_chunk` 调 Host chunk；大 chunk 降低 metadata/bookkeeping 开销，但 lookup 粒度变粗。[官方配置表 `#L65-L80`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/docs/features/kv_offloading_usage.md#L65-L80)、[调优说明 `#L297-L303`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/docs/features/kv_offloading_usage.md#L297-L303)。

### 4. 小块 H2D 的 GPU-pull/UVA 路径

vLLM 根据方向、平台、每 descriptor 大小和对齐，在初始化 handler 时固定选择 copy function：

- D2H 始终走 batch DMA，源码注释给出的原因是该方向 bandwidth-bound、dedicated copy engine 胜过 Triton。
- CUDA H2D 若最大 payload `< 28 KiB` 且全部 8-byte aligned，可选 Triton；实际 batch 少于 16 个 descriptor 又回退 DMA。kernel 固定最多使用 12 个 SM，避免占满计算资源。
- 三个 pinned Host descriptor arrays 先用 `non_blocking=True` 异步拷到 GPU，Triton kernel 再读取其中的 Host payload 地址，并直接 `tl.load(cpu_ptr)` / `tl.store(gpu_ptr)`。
- ROCm/XPU 不走此分支；vLLM 对 XPU 的注释明确说其缺少 CUDA UVA，GPU kernel 不能解引用 CPU pointer。

证据见 [`gpu_worker.py#L38-L61`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/cpu/gpu_worker.py#L38-L61) 和 [`swap_blocks_triton.py#L12-L74`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/cpu/swap_blocks_triton.py#L12-L74)。这是 GPU-assisted UVA pull，不是 payload 先复制到一个 GPU staging 再 scatter。

### 5. Layout / contiguity

- connector 从真实 KV tensor storage 派生 byte view；对 DSv4 等 packed layout 使用真实 block stride，对普通 layout 使用 page bytes；最终建立 `CanonicalKVCaches` 与 per-layer `CanonicalKVCacheRef`，支持共享/重复布局而不假定每层独立连续分配。[`offloading/worker.py#L69-L175`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py#L69-L175)、[`offloading/worker.py#L177-L243`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py#L177-L243)。
- canonical layout 的静态 copy plan 在初始化时计算，base-pointer scratch arrays 复用，不在每个 job 重新推导全部映射。[`gpu_worker.py#L300-L327`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/cpu/gpu_worker.py#L300-L327)。
- shared canonical view 可去除 replicated KV 的重复物理副本，并向 Host secondary tier 暴露 zero-copy `memoryview`。这优化的是内存布局和 CPU↔secondary 接口，不代表 GPU 直接访问 secondary tier。[`shared_offload_region.py#L212-L268`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/cpu/shared_offload_region.py#L212-L268)。

### 6. 换入/换出调度

- `offload_prompt_only` 默认 `true`；`store_threshold` 可让 cold block 先观察若干次再存；primary cache policy 内置 LRU/ARC。per-request `max_offload_tokens` 还能只保存 prompt 的前一段，当前官方文档标为 experimental。[配置表 `#L65-L80`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/docs/features/kv_offloading_usage.md#L65-L80)、[`max_offload_tokens` 文档 `#L305-L314`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/docs/features/kv_offloading_usage.md#L305-L314)。
- manager 区分 write-pending 和 evictable blocks；load 前增加 refcount、禁止被驱逐；store 只驱逐 idle block，内存不足时返回失败而不阻塞；完成后才把新 block 标为 ready/evictable。[`manager.py#L30-L73`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/cpu/manager.py#L30-L73)、[`manager.py#L113-L236`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/cpu/manager.py#L113-L236)。
- load job 立即提交；store job 刻意推迟到下一个 engine step 开头，使 D2H 排在 sampling 相关 transfer 之后，避免推迟 token generation。[`offloading/worker.py#L292-L344`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py#L292-L344)。
- 调度器按所有 group 聚合 load/store，request 有 load 在途时不发 store；preemption 或 GPU block 被重新分配时强制 flush 对应 D2H job，避免源 block 复用竞态。[`offloading/scheduler.py#L1007-L1101`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py#L1007-L1101)、[`offloading/scheduler.py#L1326-L1441`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py#L1326-L1441)、[`offloading/scheduler.py#L1443-L1474`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py#L1443-L1474)。

## vLLM：SimpleCPUOffloadConnector（环境变量门控）

这个实现值得参考，但不能当作 vLLM 默认实现报告：

- 它从原始 GPU storage 构造每 block 的连续 `int8` view，固定创建低优先级 load/store streams。[`simple_kv_offload/worker.py#L81-L169`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/simple_kv_offload/worker.py#L81-L169)。
- CPU mode 先做普通整池分配，再用 `cudaHostRegister` pin 整块 tensor，特意绕过 PyTorch pinned allocator 的 next-power-of-two rounding（源码举例 100 GB 会变 128 GB）。[`worker.py#L202-L239`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/simple_kv_offload/worker.py#L202-L239)、[`cuda_mem_ops.py#L23-L32`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/simple_kv_offload/cuda_mem_ops.py#L23-L32)。
- layer base pointers、bytes/block、raw stream handle 都在初始化时预计算；每批只用 NumPy broadcasting 构造 layer×block descriptors，然后一次 `cuMemcpyBatchAsync`/`hipMemcpyBatchAsync`。[`cuda_mem_ops.py#L125-L202`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/simple_kv_offload/cuda_mem_ops.py#L125-L202)。
- 一个 background thread 接收任务，使用方向专属 stream；H2D source 用 `ANY`，D2H source 用 `STREAM` 并等待 compute-done event。[`copy_backend.py#L25-L69`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/simple_kv_offload/copy_backend.py#L25-L69)、[`copy_backend.py#L71-L130`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/simple_kv_offload/copy_backend.py#L71-L130)。注意两条 stream 独立，但 CPU submission 仍经过同一 FIFO thread。
- CPU descriptor/build overhead 被延后到 model execution 后的 `get_finished()` 中，注释称约 5 ms 的 CPU block-copy overhead 可被 GPU compute 隐藏；H2D 可直接 launch，D2H 要等 compute event。[`worker.py#L251-L301`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/simple_kv_offload/worker.py#L251-L301)。

其可选 disk mode 更像一个实验性、可复用的两级流水模板：4 KiB aligned pinned staging，H2D/D2H 各自 buffer pool 和 coordinator thread，默认 `O_DIRECT`，用环形 slots 把 GPU DMA 与上一 slot 的 `preadv`/`pwritev` 交错。[`disk_backend.py#L1-L8`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/simple_kv_offload/disk_backend.py#L1-L8)、[`disk_backend.py#L36-L59`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/simple_kv_offload/disk_backend.py#L36-L59)、[`disk_backend.py#L81-L184`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/simple_kv_offload/disk_backend.py#L81-L184)、[`disk_backend.py#L285-L336`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/simple_kv_offload/disk_backend.py#L285-L336)。它明确是 disk↔pinned Host buffer↔GPU，不是 GDS。

## SGLang：HiCache L1↔L2

### 1. Pinned Host pool 与复用 buffer

- 默认 Host allocator 使用 mmap 分配，再对整块 KV buffer 调一次 `cudaHostRegister`；CUDA/HIP 走该路径，NPU/MUSA 改用 `torch.empty(..., pin_memory=True)`。注册失败直接抛错，避免异步传输读到 stale data。[`pool_host/common.py#L15-L27`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/pool_host/common.py#L15-L27)、[`pool_host/common.py#L120-L182`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/pool_host/common.py#L120-L182)。
- Host pool 按固定 GB 或 device-pool ratio 一次性预分配，按每机 rank 数平分可用 RAM，并预留 10 GiB；销毁时主动 unregister 大 pinned regions，避免退出回收长时间卡住。[`pool_host/base.py#L26-L59`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/pool_host/base.py#L26-L59)、[`pool_host/base.py#L113-L205`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/pool_host/base.py#L113-L205)、[`pool_host/base.py#L212-L230`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/pool_host/base.py#L212-L230)。
- `page_first` D2H staged path 预分配最多 64 pages 的 GPU staging K/V buffers，之后循环复用，不按 request 临时申请 payload buffer。[`pool_host/mha.py#L196-L227`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/pool_host/mha.py#L196-L227)。

### 2. H2D per-layer 流水、D2H all-layer 批处理

- `L2TransferEngine` 常驻一条 H2D stream 和一条 D2H stream。D2H 在方向专属 stream 上调用 `backup_from_device_all_layer()`；H2D 则在方向专属 stream 上逐层调用 `load_to_device_per_layer()`，每层完成后执行 callback 记录该层事件。[`l2_transfer.py#L49-L113`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/l2_transfer.py#L49-L113)。
- `LayerDoneCounter` 预建三组 per-layer event，producer/consumer 环形复用；模型 current stream 对当前 layer event 做 `wait_event`，无需等待整个请求 H2D 完成。[`cache_controller.py#L50-L97`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/managers/cache_controller.py#L50-L97)、[`cache_controller.py#L871-L900`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/managers/cache_controller.py#L871-L900)。
- controller 把同一轮队列中的多个 `CacheOperation` 用 `torch.cat` 合并 host/device indices 和 pool transfers；再作为一个 batch 提交 H2D 或 D2H，减少 Python/launch 开销。[`cache_controller.py#L99-L170`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/managers/cache_controller.py#L99-L170)、[`cache_controller.py#L727-L767`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/managers/cache_controller.py#L727-L767)。
- kernel backend 的 indices 通过 `.to(device, non_blocking=True)` 异步搬到 GPU；direct + `layer_first` 则在 CPU 排序 Host indices，并同步重排 device indices，使 source/destination 更容易形成连续 run。[`cache_controller.py#L780-L819`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/managers/cache_controller.py#L780-L819)。

### 3. 默认 GPU-assisted kernel：UVA pull/push 与 cache hint

- HiCache 的 JIT kernel 接受 CPU/GPUHost tensor；GPU kernel 通过 raw Host/device pointers 直接搬 payload。CUDA 用 4/8/16-byte vectorized PTX `ld.global.L1::no_allocate` / `st.global.L1::no_allocate`，ROCm 使用 vectorized non-temporal load/store，尽量不污染 L1 cache。[`hicache.cuh#L35-L121`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/kernels/jit/csrc/kvcacheio/hicache.cuh#L35-L121)、[`hicache.cuh#L126-L180`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/kernels/jit/csrc/kvcacheio/hicache.cuh#L126-L180)。
- 通用 JIT kernel 要求 element size 128-byte aligned；page-first staged D2H 只要求 16-byte aligned。默认 block quota 是 2，用很小的 SM 配额限制 I/O kernel 对模型计算的干扰。[`hicache.py#L13-L24`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/kernels/ops/kvcache/hicache.py#L13-L24)、[`hicache.py#L68-L113`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/kernels/ops/kvcache/hicache.py#L68-L113)。
- MHA Host pool 按 layout 选择 JIT/AOT kernel 或 direct backend；H2D 保持 per-layer dispatch，D2H 进入 all-layer dispatch。[`pool_host/mha.py#L237-L366`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/pool_host/mha.py#L237-L366)、[`pool_host/mha.py#L383-L430`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/pool_host/mha.py#L383-L430)。

官方文档声称这些 GPU-assisted kernels 相对 `cudaMemcpyAsync` baseline 最多可达到 3 倍传输速度；这是上游文档中的声明，并非本调研复测结果。[`hicache_design.mdx#L104-L107`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/docs/docs/advanced_features/hicache_design.mdx#L104-L107)。

### 4. direct backend：连续 range 合并与 batch DMA

- `layer_first` direct 先扫描 source/destination indices，把两边都连续的 token range 合成一个更大的 `tensor.copy_(non_blocking=True)`；只有 run 断开才提交新 copy。[`transfer.cu#L728-L784`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/kernels/aot/csrc/kvcacheio/transfer.cu#L728-L784)。
- `page_first_direct` 在 CUDA 12.8+ 构造所有 page×layer×K/V copy descriptors，以一次 `cudaMemcpyBatchAsync` 提交；对 CUDA 12/13 ABI、driver/runtime/symbol 都做 capability gate，失败回退逐 page non-blocking copy。[`transfer.cu#L937-L995`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/kernels/aot/csrc/kvcacheio/transfer.cu#L937-L995)、[`transfer.cu#L1073-L1120`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/kernels/aot/csrc/kvcacheio/transfer.cu#L1073-L1120)、[`transfer.cu#L1121-L1244`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/kernels/aot/csrc/kvcacheio/transfer.cu#L1121-L1244)。
- 当前 HIP batch path 虽有实现，但 `kEnableHipBatch = false`，所以 ROCm 会回退逐 page copy。[`transfer.cu#L997-L1071`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/kernels/aot/csrc/kvcacheio/transfer.cu#L997-L1071)。
- batch attributes 对两方向都设 `cudaMemcpySrcAccessOrderStream`，同时提供 Host/Device location hint；与 vLLM H2D 的 `ANY` 不同。[`transfer.cu#L1121-L1176`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/kernels/aot/csrc/kvcacheio/transfer.cu#L1121-L1176)。

### 5. Layout、contiguity 与 page-first staged D2H

SGLang 的 Host layout 是显式调优维度：

- `layer_first = [K/V, layer, token, ...]`，最接近 GPU 的逐层访问；`page_first = [K/V, token, layer, ...]`，同一 page 的全层 KV 连续，适合把整页作为单个 L3 object；`page_first_direct = [K/V, page, layer, page-token, ...]`，兼顾 L3 page object 和 L2↔L1 的 page-layer 聚合。[`pool_host/mha.py#L158-L194`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/pool_host/mha.py#L158-L194)、[官方设计 `#L92-L103`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/docs/docs/advanced_features/hicache_design.mdx#L92-L103)。
- `page_first` 的 per-layer refs 由 tensor transpose 得到，只换 stride、不复制 payload；data pointer arrays 预先放在 GPU。[`pool_host/mha.py#L109-L128`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/pool_host/mha.py#L109-L128)。
- `page_first` D2H 不能直接产生 Host 上的 page-contiguous layout，于是分至多 64 pages/chunk：先由 GPU kernel 把 layer-first KV relayout 到连续 GPU staging，再一次 batch DMA 到目标 Host pages，slot/chunk 循环复用。[`hicache.py#L263-L309`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/kernels/ops/kvcache/hicache.py#L263-L309)、[`staged_write_back.cuh#L186-L290`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/kernels/jit/csrc/kvcacheio/staged_write_back.cuh#L186-L290)。这是一条明确的 **device relayout + DMA** 混合路径。

### 6. 换入/换出和热度调度

- 三种 write policy：`write_through` 每次访问都向下写；`write_through_selective` 只在访问次数达到阈值后写 hot data；`write_back` 只在上层 eviction 时写。当前源码给 `write_through` 的阈值是 1、selective 是 2，且 chunked request 不增加 hit count；官方当前默认 policy 是 `write_through`，`write_back` 源码还注明未来会弃用。[设计文档 `#L74-L84`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/docs/docs/advanced_features/hicache_design.mdx#L74-L84)、[`hiradix_cache.py#L195-L210`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/hiradix_cache.py#L195-L210)、[`hiradix_cache.py#L983-L992`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/hiradix_cache.py#L983-L992)、[`hiradix_cache.py#L1189-L1263`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/hiradix_cache.py#L1189-L1263)。
- `load_back` 是 all-or-none；不足 10 tokens（当前 `load_back_threshold`）或超过 memory quota 就跳过。发起 async H2D 前保护 Host nodes；GPU 分配失败先驱逐 GPU cache 后重试，仍失败才放弃。[`hiradix_cache.py#L1374-L1445`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/hiradix_cache.py#L1374-L1445)。
- L3 miss 后只有 hit 长度达到阈值才 prefetch；停止策略有 `best_effort`、`wait_complete`、`timeout`，timeout 随 token 数线性增加并有上限。[设计文档 `#L45-L72`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/docs/docs/advanced_features/hicache_design.mdx#L45-L72)、[`hiradix_cache.py#L1577-L1623`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/hiradix_cache.py#L1577-L1623)。
- MLA 多 TP rank 的 KV 内容相同，因此只由一个 rank write-back，避免重复 D2H/L3 存储。[设计文档 `#L109`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/docs/docs/advanced_features/hicache_design.mdx#L109)。

## L3、zero-copy、copy engine 与 direct storage 的边界

### vLLM

官方定义很明确：只有 CPU primary tier 能直接和 GPU 交换数据，secondary tiers 必须经 CPU primary。因此默认 native tiering 没有 L3/storage↔GPU direct path。[`kv_offloading_usage.md#L3-L23`](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/docs/features/kv_offloading_usage.md#L3-L23)。

- “copy engine”指 H2D/D2H 使用 CUDA/HIP DMA engine；vLLM D2H 总选它，H2D 大块也选它。
- small-H2D Triton 是 SM/UVA pull，会占少量 SM，不是 copy engine。
- shared region 的 `memoryview` 是 secondary tier 对 Host bytes 的 zero-copy view。
- Simple disk backend 是 O_DIRECT filesystem I/O + pinned Host staging，不是 GPUDirect Storage。

### SGLang

HiCacheStorage 的接口把 device↔host 与 host↔storage 明确分开；storage 的 get/set 操作接收 Host indices/keys。[`hicache_storage.py#L94-L107`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/hicache_storage.py#L94-L107)。`page_first`/`page_first_direct` 让同一 page 全层数据成为一个连续 Host object，L3 backend 可直接拿 address+size，免去再 pack 一份 Host payload。[`hicache_design.mdx#L92-L103`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/docs/docs/advanced_features/hicache_design.mdx#L92-L103)。

NIXL 后端会预注册 Host KV pool；page-first 可直接以这块 Host buffer 做 NIXL transfer，layer-first 则需要独立的 load/store bounce buffers。它也要求 O_DIRECT file backend 的 buffer 对齐。[`hicache_nixl.py#L141-L154`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py#L141-L154)、[`hicache_nixl.py#L273-L299`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py#L273-L299)、[`hicache_nixl.py#L335-L408`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py#L335-L408)。官方文档列出 NIXL 可连接 GDS plugin，但当前上述 HiCache 路径的注册和传输对象仍是 Host region；不能据此声称 KV 已经从存储直达 L1 GPU。[`hicache_design.mdx#L115-L129`](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/docs/docs/advanced_features/hicache_design.mdx#L115-L129)。

## 横向对照

| 优化维度 | vLLM native | SGLang HiCache |
| --- | --- | --- |
| Pinned Host memory | 整池 `pin_memory` 或整 mmap `cudaHostRegister`；descriptor arrays 也 pinned | mmap 整池 `cudaHostRegister`；失败为 fatal；NPU/MUSA 用 pinned allocator |
| H2D/D2H async | 两个方向独立 handler，per-transfer stream/event | 常驻 H2D/D2H streams，per-layer events |
| 与 compute 同步 | **load/store 都** wait current compute stream；D2H 延后到 sampling 后 | H2D 逐层 event 等待，形成 layer pipeline；D2H all-layer |
| copy engine | D2H 必选 batch DMA；大 H2D 选 batch DMA | `direct` 走 DMA；`page_first` D2H 走 GPU relayout + batch DMA |
| GPU-assisted/UVA | 小 H2D 走 Triton，12 SM、28 KiB/descriptor、16 descriptor 门槛 | 默认 kernel backend，2-block quota；H2D/D2H 均有 GPU kernel，page-first D2H 特化为 staged |
| Batch/合并 | 整 job 的 group×layer×block descriptors 一次 batch API | queue operations 合并；direct 连续 run 合并；page-first-direct page×layer×K/V 一次 batch API |
| Chunk | Host chunk 可包含多个 GPU blocks，支持首尾 partial | page 为 L3 粒度；staged D2H 固定最多 64 pages/chunk |
| Layout | 从真实 storage/stride 派生 canonical mappings，支持 packed/replicated layout | `layer_first`、`page_first`、`page_first_direct` 显式选择，面向 L1 或 L3 contiguity |
| buffer/pool | KV 整池；stream/event/pinned descriptor/scratch pools | KV 整池；三组 per-layer events；固定 GPU staging；插件可建双向 bounce pool |
| H2D source order | batch DMA 用 `ANY`；稳定 Host source 可由 driver pipeline | direct batch 当前用 `STREAM`；kernel 自己调度 GPU load |
| zero-copy 含义 | GPU SM 可 UVA pull Host；secondary 可对 shared Host mmap 用 memoryview | GPU kernel 可访问 pinned Host；page-first 允许 L2↔L3 无额外 Host pack |
| eviction/store | LRU/ARC，store threshold，prompt-only 默认，per-request cap | write-through/selective/write-back，Host/GPU node protection |
| L3 direct GPU | 无，官方明确 CPU staging | 当前 HiCache data path 仍以 Host pool 为中介；NIXL/GDS 是 backend 能力，不等同端到端直达 GPU |

## 可用于本地分支差距核对的上游能力清单

本节只给出检查项，不判断 xLLM 是否已有：

1. 是否对整个 Host KV pool 一次 pin/register，并对 descriptor metadata 也 pin；是否避免很多小 pinned allocations 及 allocator rounding。
2. 是否把 H2D/D2H 做成独立 stream/event/queue，并复用 stream、event、descriptor、staging buffer，而不是 request 热路径反复分配。
3. H2D 是否不仅保证 source ready，也等待 destination block 上已有的 write/zeroing；D2H 是否等待 compute 完成；GPU block 复用/preemption 是否强制 flush 未完成 store。
4. 是否能把 layer×block 的离散小 copy 合成一个 batch DMA submission；旧 driver/runtime 是否有正确 fallback。
5. 是否针对 H2D 与 D2H 分别选实现：小 H2D 可考虑低-SM GPU-pull/UVA，大 H2D 与 D2H 用 copy engine；阈值必须在目标硬件实测，不应照抄 H100 的 28 KiB。
6. 是否支持真正的 per-layer H2D completion event，使 load layer N+1 与 compute layer N 重叠；仅“异步提交整份 KV”不等价于 layer pipeline。
7. Host layout 是否能兼顾 L1 per-layer 访问与 L3 per-page object；能否预计算 storage/stride mapping，合并连续 run，减少 transpose/gather 和 descriptor 数。
8. 对不匹配 layout 的 D2H，是否值得采用“device relayout 到有限 staging + 大块/batch DMA”，并以固定 chunk 循环复用 staging。
9. 是否有访问热度门槛、prompt-only/per-request cap、write-through-selective 等策略，避免把传输带宽浪费在一次性 KV 上。
10. 如果接 NVMe/RDMA/L3，需区分：Host zero-copy view、O_DIRECT、RDMA registered memory、GDS、真正 storage↔GPU direct path；不能用一个“zero-copy”概括所有层级。

## 证据强度与限制

- 关于接口、默认值、调用顺序、stream/event 依赖、layout 和 fallback 的结论来自固定 commit 的可执行源码，证据强。
- “最多 3x”只来自 SGLang 官方设计文档，本报告未复测；vLLM 的 28 KiB/12 SM/16 descriptors 是源码注明的 H100 PCIe Gen5 经验参数，迁移到其他 GPU/NPU/MLU 必须重新 benchmark。
- `main` 在 2026-08-21 之后仍可能改变；实施前应重新确认目标 release/tag，而不是把本报告的 `main` 快照直接等同于所有已发布版本。
