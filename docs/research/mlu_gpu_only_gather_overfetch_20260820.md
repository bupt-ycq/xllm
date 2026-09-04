# xLLM MLU：取消 CPU payload gather、改为设备侧整理的可行性调研

日期：2026-08-20  
目标分支：`feat/mlu-tmo-kv-transfer-new`  
代码快照：HEAD `8afa4320a`（`perf: add indexed fast path for mlu kv transfers`）  
硬件/软件快照：MLU590-M9DK，driver 6.5.51，CNRT 7.6.3，CNDrv 3.6.6

## 结论先行

**可行，而且本分支已有两条“不做 CPU payload gather”的 MLU 路径。** 用户所说的
“CPU gather -> 一次大 DMA -> MLU scatter”只对应新加、默认关闭的
`mlu_indexed_copy` 实验路径；不是整个分支唯一的 H2D 实现。

1. **现在即可用的首选是 TMO backend。** 审计对应的
   `torch_mlu_ops` 实现后确认：每个 Host object 直接连续
   `cnrtMemcpyAsync` 到 MLU staging，随后 `invokeTransferCacheBlocks` 在 MLU 上按
   block id scatter；CPU 只准备地址和 block-id metadata，不搬 payload。当前分支
   TMO 相对同 binary Direct 的 B4 实测中，32K/64K TTFT mean 分别降低
   **26.34%/20.65%**，正确性检查全部通过。
2. **如果要继续做一个 Direct 类新后端，第一优先不是 overfetch，而是“单边规则
   2D DMA gather”。** 将 Host block id 排序、按等步长 run 分组，用
   `cnMemcpy2DAsync(src_pitch=Host stride, dst_pitch=row_bytes)` 精确收集到连续 MLU
   staging，再做一次 MLU scatter。它不搬无用字节，并且不要求最终 device block id
   也规则；当前 Direct 只有 Host 和 Device 两边同时等步长才会进入 2D。
3. **任意离散段可用 `cnMemcpyBatchAsync`，这本身就是公开的用户态
   scatter-gather/descriptor-list 接口。** 当前 Direct 已在用它，并可直接 DMA 到
   最终随机 device block，完全不需要 CPU payload gather 或 device scatter。只有当
   小段 batch DMA 的有效带宽明显低于连续 DMA 时，才值得增加 staging 和 device
   scatter。
4. **“多传再丢弃”可做，但必须按 gap 自适应合并，不能把一组 block 直接做全局
   min/max bounding span。** 当前 Host 是 `[block, 78 layers, ...]`。若一次只恢复
   `Lr` 层，覆盖 `B` 个连续 Host block 的 bounding-span 放大率是
   `((B-1)*78+Lr)/(B*Lr)`；B4 下 `Lr=19` 已是 3.33 倍，尾部 `Lr=2` 是 29.5 倍。
5. **CUDA 上有公开的 GPU 直接读 mapped pinned Host memory 方案；本机 MLU590 不支持
   mapped-host BANG/IPU kernel 直读这条公开路径。**
   本机 CNDrv 头文件将 mapped Host device pointer 限定为 TNC 使用；实机查询又显示
   Host-register `DEVICEMAP` 不支持，`cnHostMemAlloc(...DEVICEMAP)` 返回
   `CN_OPS_ERROR_NOT_SUPPORTED`。没有一手证据证明普通 BANG/IPU kernel 可以安全解引用
   该 Host pointer。

因此建议顺序是：**保留 TMO 为生产基线 -> 做 source-only 2D staging 原型 -> 对真正
随机的小段再 A/B batch-to-final、batch-to-staging、gap-aware overfetch -> 最后才考虑
Host allocator/layout 改造。**

## 1. 当前分支实际在做什么

### 1.1 调用方和物理布局

`HierarchyKVCacheTransfer` 的 Host cache 是按 block-first 分配的，逻辑形状为
`[num_host_blocks, num_layers, ...per_block_dims]`；每次计划保存 Host block view、
对应的 device block id，以及一段连续 layer range。H2D 逐 layer range 异步提交并用
layer event 放行 forward，D2H 同步完成。参见
[`hierarchy_kv_cache_transfer.h`](../../xllm/core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.h#L48) 和
[`build_copy_plan()`](../../xllm/core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.cpp#L369)。

这意味着 Host 端已经具备适合“整 block object 搬运”的 page/block-first 性质；问题
不是缺少 page layout，而是：

- layer-wise restore 只读取一个窄 layer range，而相邻 Host block 之间的物理 pitch
  是完整 78-layer object；
- 最终 device block id 可任意，device 侧每层 cache 也是独立 tensor；
- 因而“Host 源规则”和“最终 Device 目标规则”是两个不同条件。

### 1.2 Direct backend：已经没有 CPU payload gather

Direct 先把 tensor pair 变成地址 descriptor，然后：双边连续时合并 extent；双边都为
相同固定步长且 run 足够长时用 `cnMemcpy2DAsync`；其余用
`cnMemcpyBatchAsync`，每批最多 4096 个 descriptor。该路径由 DMA 直接读 Host cache
原地址、直接写最终 device block；CPU 只构造地址/长度数组。参见
[`copy_direct()`](../../xllm/core/platform/mlu/mlu_batch_memcpy.cpp#L955) 和
[`find_direct_copy_run()`](../../xllm/core/platform/mlu/mlu_batch_memcpy.cpp#L171)。

所以如果目标只是“不要 CPU gather”，最小动作是保持
`mlu_kv_transfer_backend=direct` 且不要启用 `XLLM_MLU_INDEXED_COPY`。随机映射测试也明确
验证 Direct 会提交 4 个 batch descriptors，而非先 pack payload；连续和固定 stride
则分别压成 extent/2D。参见
[`mlu_batch_memcpy_test.cpp`](../../tests/core/platform/mlu/mlu_batch_memcpy_test.cpp#L415)。

### 1.3 新 `mlu_indexed_copy`：确实仍在 CPU gather

该实验默认 `enabled=false`，只有环境变量 `XLLM_MLU_INDEXED_COPY` 或测试显式打开后
才启用；默认门槛是 512 rows、1 MiB。参见
[`mlu_indexed_copy.h`](../../xllm/core/platform/mlu/mlu_indexed_copy.h#L40) 和
[`indexed_copy_options_from_environment()`](../../xllm/core/platform/mlu/mlu_indexed_copy.cpp#L978)。

H2D 数据路径是：

1. CPU `torch::index_select_out` 把离散 Host rows pack 到 pinned staging；
2. 一次 `cnrtMemcpyAsync_V3` 搬 payload 和两组 index 到 MLU staging；
3. MLU 上 `index_copy_` scatter 到最终 rows。

对应代码在
[`IndexedCopyBackend::Impl::submit_h2d()`](../../xllm/core/platform/mlu/mlu_indexed_copy.cpp#L818)。
因此它减少了 DMA submission，却没有满足“CPU 不碰 payload”。另外当前实现按
Host/Device storage pair 拆 group，每个 group 都单独 staging/copy/device op；H2D
统计的 `host_pack_ns` 正是后续阈值决策可直接利用的观测量。

### 1.4 TMO backend：已经是连续 H2D staging + MLU scatter

本次额外审计了内部 `torch_mlu_ops` commit
`43ba8a3011a874b2b779f1756d3c17786d5765e9` 的
`csrc/torch_api/multi_layer_block_kv_transfer.cpp`（文件 SHA256
`193a75bc0d1e94ac58efdb255449b87cf2645dc6b644d476ea23ce444fbba804`）：

- 创建形如 `[num_objects, num_layers, ...]` 的 MLU staging；
- H2D 对每个 Host object 提交一段连续 `cnrtMemcpyAsync`；
- queue 完成后调用 `invokeTransferCacheBlocks(..., CONTEXT2CACHE)`，按 MLU 上的
  `block_ids` 写最终 paged cache；
- 单次 op 接受 1--4 objects，xLLM 按 4 objects 分 chunk。

可点击的内部源地址：
[`multi_layer_block_kv_transfer.cpp`](http://gitlab.software.cambricon.com/neuware/torch_mlu_ops/-/blob/43ba8a3011a874b2b779f1756d3c17786d5765e9/csrc/torch_api/multi_layer_block_kv_transfer.cpp#L135)，
xLLM 包装和 chunk 调用见
[`mlu_tmo_transfer.cpp`](../../xllm/core/platform/mlu/mlu_tmo_transfer.cpp#L18) 与
[`copy_layered_tmo()`](../../xllm/core/platform/mlu/mlu_batch_memcpy.cpp#L1246)。

当前分支最近一次同 binary、每 arm 5 样本的 B4 报告显示：32K TTFT mean
`1.410 -> 1.038 s`，64K `2.492 -> 1.977 s`；输出/token ids 一致且测量窗口无
copy/queue/fatal/OOM 错误。详见
[`FINAL_REPORT.md`](../../agent-workspace/experiments/mlu_tmo_kv_transfer_new_ab/tmo_opt2_36c471ed_20260820/FINAL_REPORT.md#L1)。

## 2. 五种候选架构

| 路线 | CPU 是否搬 payload | 多传字节 | MLU590 可用性 | 判断 |
|---|---:|---:|---|---|
| 多段异步 H2D 到最终 block | 否 | 0 | **已可用**：`cnMemcpyBatchAsync` / `cnMemcpy2DAsync` / `cnMemcpyAsync_V3` | Direct 已实现；大 row 或 descriptor 不多时通常最省 |
| 连续/等步长 Host run -> staging -> device scatter | 否 | 0 | **可实现**；TMO 已是 1--4 object 版本 | 最值得新增的 Direct 原型；优先用 source-only 2D |
| gap-aware overfetch -> staging -> device gather/scatter | 否 | `G>0` | **可实现**；MLU staging 后可用 `index_copy_`、自定义 kernel，合适目标还可用 BANG `__gather` | 仅当省下的 descriptor 时间大于额外 PCIe 字节和 device op |
| mapped/zero-copy：device kernel 直接读 Host | 否 | 0 | **不作为 MLU590 方案**；公开接口限定 TNC，实机 `DEVICEMAP` 不支持 | CUDA 可类比，MLU 无可移植证据 |
| 改 allocator/layout，使一次请求的 Host blocks 成 run | 否 | 0 或很小 | **可实现但侵入大** | 当前已 block-first；应改“分配局部性/transfer slab”，不是再改一个 page-first 名称 |

### 2.1 多段异步 H2D / DMA descriptor list

本机 CNDrv 3.6.6 的
[`cn_api.h`](/usr/local/neuware/include/cn_api.h#L3708) 公开
`cnMemcpyBatchAsync(dsts, srcs, bytes, count, attrs, attrIdxs, numAttrs, queue)`，动态库
也导出该符号。它是应用层能使用的 scatter-gather descriptor list：CPU 生成元数据，
payload 由 DMA 从多个 Host 地址搬到多个 MLU 地址。公开 CNDrv/CNRT/BANG headers 未发现
更底层的 copy-engine descriptor ring API；`DMA_BUF_FD` 是 buffer sharing，不是任意
Host payload 的 SG list。

batch attributes 还提供 `QUEUE`、`DURING_API_CALL`、`ANY` 三种 source access order，
以及 try/hint 性质的 `CN_MEMCPY_FLAG_OVERLAP_WITH_COMPUTE`。本分支当前固定用
`QUEUE`；只读且生命周期受 event 保证的 H2D Host source 可以 A/B `ANY`，但 D2H
source 仍受 compute stream 写入，不能照搬。定义见
[`cn_api.h`](/usr/local/neuware/include/cn_api.h#L1599)。CUDA 官方
[`cuMemcpyBatchAsync`](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb04ea721b05487b3f79129fa849e2f6b)
也规定 batch 整体服从 stream 顺序、batch 内 copy 不保证相互顺序，且
`CU_MEMCPY_SRC_ACCESS_ORDER_ANY` 允许 Host source 在 API 返回后被乱序读取。

部署上必须做版本/符号 gate：寒武纪公开的旧
[`CNDrv 2.7.1 手册`](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cndrv_2.7.1/index.html)
还没有 batch API；当前是 CNDrv 3.6.6 实机能力，不能假设所有 MLU 环境都有。

### 2.2 首选新方案：单边规则 2D DMA gather

当前 Direct 判断 2D run 时同时要求 source 和 destination 地址步长稳定。因此 device
block id 随机时，即使 Host ids 为连续 block，也会回落到 batch。

可以把 destination 换成连续 MLU staging，从而把条件降为“只要求 Host 源规则”：

```text
Host [block, 78 layers, row...]          MLU staging             final paged KV
  block 10, layers [a,b)  ----\       [compact row 0] --\
  block 11, layers [a,b)  ----- 2D --> [compact row 1] ---- device scatter
  block 12, layers [a,b)  ----/       [compact row 2] --/    dst block ids arbitrary
```

实现步骤：

1. 按 Host storage 和 row width 分组，将 `(host_block_id, device_block_id)` 按 Host id
   排序；CPU 只重排小 metadata。
2. 把连续或固定 gap 的 Host ids 切成 arithmetic runs。每个 run 用
   `cnMemcpy2DAsync(dst_pitch=row_bytes, src_pitch=id_gap*host_object_bytes,
   width=row_bytes, height=run_len)` 精确写连续 staging。
3. 上传按排序后的 device block ids；用一次 `index_copy_`，或一个融合的 BANG
   gather/scatter kernel，写最终 paged cache。
4. 对短而随机的 remainder，比较 batch-to-final 与 batch-to-staging，保留更快者。

CNDrv 的 `cnMemcpy2DAsync` 明确支持 MLU300+，并要求 pitch 不小于 width；本机接口见
[`cn_api.h`](/usr/local/neuware/include/cn_api.h#L3517)。寒武纪官方 BANG C 指南也确认
异步 Memcpy 完全异步、buffer 必须活到同步点，参见
[`BANG-C OPS Develop Guide`](https://github.com/Cambricon/mlu-ops/blob/master/docs/BANG-C-OPS-Develop-Guide.md#3353-host-%E5%92%8C-device-%E7%9A%84%E5%90%8C%E6%AD%A5)。

这条路线相对 TMO 的潜在增量是：不受单 op 4-object 限制，可把几十/几百个规则 Host
rows 压成少量 2D submissions；相对 overfetch 的优势是 payload bytes 不增加。

### 2.3 覆盖区间/页/块 overfetch 后 device 筛选

对 source intervals 排序后，只在局部 gap 满足成本阈值时合并：

```text
requested:  [AAAA] gap [BBBB] huge-gap [CCCC]
copy:       [AAAA gap BBBB]          + [CCCC]
device:     保留 A/B，跳过 gap；         保留 C
```

设备侧可直接使用现有 `index_copy_`；如果 staging 内还要按 byte offsets 做真正 gather，
本机 BANG headers 和官方文档提供 `__gather` / `__gather_async` 的 GDRAM2GDRAM 形态。
官方能力门槛是 `__BANG_ARCH__ >= 592`、CNCC >= 4.0、compute_50；因此要按编译目标
验证，不能把它宣称为 MLU370 通用能力。签名、offset 位宽、对齐和不重叠约束见
[`BANG C Discrete Move / gather`](https://www.cambricon.com/docs/sdk_1.15.0/cntoolkit_3.7.2/cambricon_bang_c_4.7.2/2Builtin-Functions/Discrete%20Move%20Functions.html#gather)。

`cnnlBatchGatherV2_v2` 也是 MLU-memory 内 gather，但更适合等宽 tensor rows；任意 byte
run 仍应使用自定义 kernel/BANG builtin。无论哪种都只能整理已在 GDRAM 的 staging，
不能据此推导 BANG kernel 能直接读普通 Host pinned memory。

### 2.4 mapped Host / zero-copy

CUDA 允许 kernel 直接访问 mapped pinned Host memory，但 NVIDIA 明确说明离散 GPU 上
该访问走 PCIe/NVLink-C2C，延迟更高、带宽更低，只有单次、合并访问等特定场景才合适。
参见官方
[`Mapped Memory`](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#mapped-memory)
和
[`Zero Copy Best Practices`](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#zero-copy)。

网上确有 LLM 工程实例：vLLM 在 H100 PCIe Gen5 上对 H2D 小段做 Triton kernel direct
Host read，实测路由条件为 descriptor 数至少 16、每项小于 28 KiB；更大的段仍走
`cuMemcpyBatchAsync`。这是 CUDA/H100 经验阈值，不可复制到 MLU。参见
[`swap_blocks_triton.py`](https://github.com/vllm-project/vllm/blob/14617c2b6c1257ac0d6c7b5e05b195ca30013827/vllm/v1/kv_offload/cpu/swap_blocks_triton.py#L14-L74)
和
[`gpu_worker.py`](https://github.com/vllm-project/vllm/blob/14617c2b6c1257ac0d6c7b5e05b195ca30013827/vllm/v1/kv_offload/cpu/gpu_worker.py#L62-L104)。

MLU590 本机证据如下：

| CNDrv 查询/调用 | 结果 |
|---|---:|
| `CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM` | 1 |
| `CAN_MAP_HOST_MEMORY` | 1 |
| `PORTABLE_HOST_REGISTER_SUPPORTED` | 1 |
| `DEVICE_MAP_HOST_REGISTER_SUPPORTED` | **0** |
| `IOMEMORY_HOST_REGISTER_SUPPORTED` | **0** |
| `cnHostMemAlloc(4096, CN_MEMHOSTALLOC_DEVICEMAP)` | **100050 / NOT_SUPPORTED** |
| `cnMallocHost` 后 `cnHostMemGetDevicePointer` | 成功，但头文件明确写地址只能用于 **TNC** |

属性和 TNC 限制见本机官方
[`cn_api.h`](/usr/local/neuware/include/cn_api.h#L936) 与
[`cnHostMemGetDevicePointer`](/usr/local/neuware/include/cn_api.h#L5612)。因此本调研不建议
写 BANG kernel 解引用这个地址；除非寒武纪给出针对 MLU590 IPU/BANG 的正式支持说明和
sample，并在目标容器完成正确性/带宽测试。

### 2.5 GPUDirect 与 raw DMA SG

GPUDirect RDMA 是 GPU 与 NIC/采集卡/存储适配器等第三方 PCIe peer 的直连；GDS 是
storage 到 GPU memory 的直连。它们减少 Host bounce buffer，但并不是“GPU kernel
读取任意 CPU DRAM”的 API。官方范围见
[`GPUDirect RDMA Overview`](https://docs.nvidia.com/cuda/gpudirect-rdma/#overview) 和
[`GPUDirect Storage Overview`](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html)。

对 xLLM 的现实含义是：若未来 Mooncake/NIC 能直接把远端 KV 写入 MLU/GPU staging，
可以再做 device scatter；它不能优化当前已经位于本机 Host cache 的 payload。
CNDrv 的 `CN_MEMHOSTREGISTER_IOMEMORY` 也是第三方 I/O memory 注册能力，不等于普通
Host KV 的 gather，而且本机对应 device attribute 为 0。

## 3. 性能模型与可执行阈值

定义：

- `L`：真正需要的 payload bytes；
- `M`：当前精确 batch/direct descriptors 数；
- `J`：run 合并/overfetch 后的 DMA regions 数；
- `G`：overfetch 的无用 bytes，实际传输 `U=L+G`；
- `Bpcie`：实测连续 H2D 带宽；`Bdev`：实测 staging-to-final device op 带宽；
- `tdesc`：每减少一个小 DMA descriptor 带来的平均时间收益；
- `Kdev`：device gather/scatter 的固定 launch 时间；
- `Bcpu`：当前 CPU pack 的有效带宽，由 `L/host_pack_ns` 得到。

可先用以下一阶模型，不应预填厂商峰值：

```text
T_direct    ~= M * tdesc + L / Bpcie
T_cpu_pack  ~= L / Bcpu + L / Bpcie + Kdev + L / Bdev
T_2d_stage  ~= J * tdesc + L / Bpcie + Kdev + L / Bdev
T_overfetch ~= J * tdesc + (L+G) / Bpcie + Kdev + L / Bdev
T_zero_copy ~= Kdev + L_effective / Bpcie_read     # 仅 CUDA 类比
```

overfetch 优于精确 batch 的必要判定可写成：

```text
G < Bpcie * ((M-J)*tdesc - Kdev - L/Bdev)

等价地，rho=(L+G)/L <
1 + Bpcie*((M-J)*tdesc-Kdev)/L - Bpcie/Bdev
```

局部实现时无需猜全局 `rho`：对相邻两个 source intervals 的 gap `g`，只有在
`g/Bpcie < t_saved_descriptor`，且 staging/inflight credit 允许时才合并。再把
`Kdev+L/Bdev` 作为整个 job 的一次性成本加入 route 决策。

CPU pack 与 no-pack 的直接阈值为：

```text
G/Bpcie + (J-1)*tdesc + Kdev + L/Bdev < L/Bcpu
```

右边可由当前 `host_pack_ns` 实测，左边用 MLU notifier/CNPAPI 分离计时。只有这个不等式
稳定成立，overfetch 才比现有 indexed path 值得。

### 当前 78-layer block-first 布局的硬约束

设每层每 block payload 是 `P`，恢复连续 `Lr` 层、`B` 个连续 Host blocks：

```text
useful bytes = B * Lr * P
bounding span = ((B-1)*78 + Lr) * P
rho_block_span = ((B-1)*78 + Lr) / (B*Lr)
lim(B->infinity) rho = 78/Lr
```

| B | Lr | bounding-span 放大率 |
|---:|---:|---:|
| 4 | 19 | 3.33x |
| 4 | 2 | 29.50x |
| 任意大 | 19 | -> 4.11x |
| 任意大 | 2 | -> 39.00x |

所以 B4 layer-wise restore 不能直接“从首 block 的 range 起点 DMA 到末 block 的 range
终点”。source-only 2D 可在这些空洞上跳跃且只传 useful bytes，应始终先测它。另一种
可行选择是一次预取该请求的全 78 层（此时 `rho=1`）并让 staging 跨 layer range
存活，但会增加 staging 峰值、提前占 PCIe、削弱逐层放行；需要单独做 TTFT/HBM credit
实验，不能只看 copy 带宽。

### 本仓库已有测量给出的量级参考

旧 64K trace 中，413,595 个约 15.3 KiB 的小 descriptor 搬 6.332 GB 时只有
3.75 GB/s；压成 101 个 2D regions 后达到 54.9 GB/s。按这组特定 trace 粗略反推，
每消掉一个碎 descriptor 的可回收时间约 3.8 us；在 55 GB/s 连续 H2D 下，相当于约
209 KiB 的额外字节预算。这个数只适合设计 microbenchmark sweep，不是生产常量。
原始结论和限制见
[`xllm_mlu_dma_transfer_optimization_20260820.md`](./xllm_mlu_dma_transfer_optimization_20260820.md#已被-xllm-实验验证的结果)。

建议动态 route 至少记录：`M/J/L/G`、run length、Host id gap、Direct/2D/batch bytes、
CPU pack ns、H2D notifier time、device scatter time、staging peak/inflight、end-to-end TTFT。
当前 `min_rows=512`、`min_bytes=1MiB` 只是静态门槛，不能代表上述成本不等式。

## 4. 网上已有实现怎样做

### vLLM

vLLM 的生产 CPU offload 主要使用 pinned Host cache + `cuMemcpyBatchAsync`，H2D source
只读时设置 `SRC_ACCESS_ORDER_ANY`；对 H100 上大量小于 28 KiB 的 H2D descriptors，
才路由到直接读 Host pointer 的 Triton kernel。代码明确把 XPU/ROCm 排除在该 kernel
路径之外，说明 mapped-host kernel 是平台能力，不是 PyTorch 通用行为。参见
[`gpu_worker.py`](https://github.com/vllm-project/vllm/blob/14617c2b6c1257ac0d6c7b5e05b195ca30013827/vllm/v1/kv_offload/cpu/gpu_worker.py#L62-L104)、
[`cache_kernels.cu`](https://github.com/vllm-project/vllm/blob/14617c2b6c1257ac0d6c7b5e05b195ca30013827/csrc/libtorch_stable/cache_kernels.cu#L2360-L2506)。

### SGLang HiCache

SGLang 同时提供 direct DMA 和 GPU-assisted kernel backend，并将 Host layout 区分为
`layer_first`、`page_first`、`page_first_direct`。官方设计说明 page-first 可把同一页
作为连续对象交给 L3，但 layer-wise GPU restore 仍需按 layer/token 组织；
`page_first_direct` 正是为 direct I/O 把页内同层 tokens 聚合。参见
[`HiCache design`](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/hicache_design.mdx#cpu-to-gpu-transfer-optimizations)
和
[`memory_pool_host.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/memory_pool_host.py)。

这支持两个判断：GPU-assisted Host I/O 确有先例；layout 必须服务实际 transfer
granularity。xLLM 已是 block-first，下一步应优化 run 和 allocator 局部性，而不是照抄
另一个框架的 layout 名称。

### LMDeploy

LMDeploy 当前 main 快照 `0552f22841f7d42ca85949ef58ae076aa829e604` 的
`CacheEngine._swap()` 与本分支 indexed 思路接近：CPU advanced indexing
`scache[:, sidx]` 先选最多 2 blocks，`.to(device)` 后再 `index_copy_`。它证明该方案
常见，但不是“GPU-only gather”的例子。参见
[`cache_engine.py`](https://github.com/InternLM/lmdeploy/blob/0552f22841f7d42ca85949ef58ae076aa829e604/lmdeploy/pytorch/engine/cache_engine.py#L711-L751)。

### 论文给出的架构原则

FlexGen 把 CPU/GPU/disk placement 和 I/O schedule 放进统一成本模型，而不是固定选择
某种 copy primitive；PagedAttention 将 KV 管理单位固定为 block/page；vAttention
进一步说明保持虚拟连续性可以减少非连续布局的软件/执行开销，但标准 CUDA VMM 的物理
分配粒度也会带来浪费。这些工作支持“用实测成本 route + block/page 作为管理单位”，
不能直接证明某个 MLU API 一定更快。原文：
[FlexGen](https://arxiv.org/abs/2303.06865)、
[PagedAttention](https://arxiv.org/abs/2309.06180)、
[vAttention](https://arxiv.org/abs/2405.04437)。

## 5. 推荐落地顺序

### 基线闸口：先把 TMO 当作 no-CPU-gather 对照

- 生产 A/B 继续使用已经验证的 TMO；不要把 indexed path 的 CPU pack 成本误归因到
  TMO。
- 在相同 B4、32K/64K case 增加 TMO 的 payload H2D、device scatter、每 op objects
  统计，明确剩余时间是 4-object chunk 固定成本还是带宽。
- 保留 Direct 作为 fallback 和对照。

### P0：做 source-only 2D staging 原型

- 复用现有 `MLUIndexedCopyRegion`/slot pool，但 H2D 不再
  `torch::index_select_out`；改为 Host-id sort + 2D run H2D 到 device staging。
- 第一版只支持相同 storage、相同 row width、唯一 destination ids；其余明确 fallback。
- device scatter 可先复用已验证的 `index_copy_`，验证收益后再写融合 BANG kernel。
- route 比较三臂：Direct batch-to-final、2D-to-staging+scatter、TMO。

### P1：只对 2D remainder 做 bounded、gap-aware overfetch

- 以局部 gap 不等式合并，设置 `max_overfetch_ratio`、`max_staging_bytes`、
  `max_inflight_bytes` 三重保护。
- 不跨 storage，不跨不同 row width，不覆盖全 Host pool min/max。
- 若目标编译架构满足官方条件，再 A/B BANG `__gather`; 否则先用 PyTorch MLU
  `index_copy_`/自定义 kernel。

### P2：用 allocator/layout 根治地址局部性

当前 Host `[block, layer, ...]` 不需要推倒。更有效的改造是：

- Host block allocator 为同一 prefix/job 优先返回连续 run；
- offload object 以“若干 blocks × 全层/角色”的 transfer slab 分配和复用；
- restore device allocator 可偏好连续 block run，但 paged block table 仍保留任意映射；
- 若一次全层预取的 staging 成本可控，则一个 slab 做少量 H2D，后续按 layer 从 staging
  scatter 并逐层 event 放行。

这会让 source-only 2D/extent 命中率提升，同时保持 PagedAttention 的 block sharing 和
fragmentation 优势。只有 trace 显示 Host block ids 长期随机、run-aware allocation 仍无效，
才值得维护第二套 canonical packed layout。

建议的 Direct-family fallback routing 是：

```text
双边连续/规则                 -> 现有 extent / 双边 2D，直接写 final
仅 Host source 形成收益 run   -> P0 source-only 2D -> staging -> device scatter
任意离散且 batch 模型更优     -> 现有 cnMemcpyBatchAsync，直接写 final
batch 小段效率差且 gap 不等式成立 -> P1 bounded overfetch -> staging -> device gather/scatter
其余/任何能力或资源检查失败    -> 现有 Direct batch/per-copy；不回退到 CPU payload pack
```

TMO 是这棵 Direct-family 路由树之外、已经端到端验证的生产 backend；上线新路线时应始终
保留 TMO/Direct 开关，任一 staging allocation、index 唯一性、架构版本、queue submission
检查失败都走原 backend，不做部分提交后静默换路。

## 6. 最终判断

对当前分支，答案不是简单的“可以多传一些然后 GPU 丢掉”：

- **已经可以完全取消 CPU payload gather**：TMO 已经这样做；Direct batch-to-final
  也已经这样做。
- **最有希望的新实现**是 source-only 2D DMA 精确 gather 到 MLU staging，再 device
  scatter；它直接利用当前 block-first/78-layer stride，不增加 PCIe bytes。
- **overfetch 是随机 remainder 的二级策略**，必须用 gap 成本模型；在 B4 窄 layer
  range 下盲目 bounding span 会放大 3.33--29.5 倍，风险很高。
- **mapped Host kernel 是 CUDA 上存在、MLU590 上缺乏公开支持的类比**，不应进入近期
  实现计划。
- **没有比 `cnMemcpyBatchAsync` 更底层的公开 raw SG descriptor API**；如果 batch
  性能不足，应从 2D run、source order hint、staging/device scatter 和 allocator
  局部性着手，而不是尝试驱动私有 descriptor ring。
