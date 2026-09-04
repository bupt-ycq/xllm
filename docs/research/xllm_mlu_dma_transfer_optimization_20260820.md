# xLLM MLU Host KV Direct DMA 优化调研

日期：2026-08-20  
目标分支：`feat/mlu-tmo-kv-transfer-new`  
范围：xLLM MLU Host KV 的 Direct DMA 路径，并对照 SGLang、vLLM、vLLM-Ascend 当前主线实现。

## 结论

xLLM 当前 Direct 路径已经完成了最重要的一轮优化：Host KV 是
`[host_block, layer, ...]` 的 block/page-first 物理布局，Direct backend 会把
layered plan 展开为 layer-major descriptor，再将规则映射压成连续 extent 或
`cnMemcpy2DAsync`，其余离散映射才进入 `cnMemcpyBatchAsync`。因此，下一步不应
再把“切到 page-first”或“再接一个 batch memcpy”当成新方案。

建议按以下顺序推进：

1. **P0：H2D/D2H 流与完成队列分离，并加入有界 inflight。** 当前 2 个 H2D
   worker 和最多 4 个 D2H caller 共用 6 条 stream，H2D submit-only 后 stream
   会在 DMA 完成前立刻回池。单独的 H2D/D2H stream pool、end-event 回收和
   byte/region credit，可以避免方向间队头阻塞和无界 queue backlog。
2. **P0：对 residual batch H2D A/B `CN_MEMCPY_SRC_ACCESS_ORDER_ANY`。** vLLM
   已在相同的“pinned Host 源在传输期间不被写”条件下使用该提示；xLLM 当前
   H2D/D2H 都硬编码 `QUEUE`。D2H 必须继续使用 `QUEUE`。该提示只影响
   `cnMemcpyBatchAsync`，不会改善已经走 2D 的字节，所以必须先统计 batch
   fallback 占比。
3. **P0：D2H 所有 layer range 一次提交、最后只同步一次。** 当前每个 range
   都进入 `copy_layered_d2h()` 并 `cnQueueSync()`。D2H 没有逐层放行消费者的
   需求，可以合并 plans 后一次 copy/sync，减少固定同步和 RPC copy-thread
   占用。
4. **P0/P1：TP-rank 错峰和 byte-based admission。** vLLM-Ascend 已暴露
   `h2d_stagger_us`、`max_transfer_blocks` 和 `max_transfer_bytes`。xLLM TP8 会在
   多 rank 同时恢复相同请求，结合 Direct H2D submit-only，容易把阻塞从设备
   时间转移到 `cnMemcpy*Async` 的 Host 提交阶段。
5. **P1：让离散 block pair 形成更多规则 run。** 先记录 extent/2D/batch
   的 bytes、run length 和地址步长，再决定是重排相互独立的 copy pair，还是
   给 Host restore 增加 run-aware device block allocation。单纯增大 CPU
   `blocks_per_chunk` 不会自动减少 descriptor。
6. **P1：descriptor scratch/pool 与静态 offset 缓存。** vLLM 和
   vLLM-Ascend 都复用 descriptor buffer、stream 和 event。xLLM Direct 每次调用
   重新构造多组 `std::vector`。但本地 one-pass plan 实验已经证明“只省 plan
   CPU 时间”可能让端到端回退，因此这项必须由 submit wall time 和 TTFT
   联合验收，不能仅看 plan microbenchmark。

## 调研快照

| 项目 | 固定源码快照 |
|---|---|
| SGLang | [`21c88f8625f2e699543a1c34f41d6894ef342903`](https://github.com/sgl-project/sglang/tree/21c88f8625f2e699543a1c34f41d6894ef342903) |
| vLLM | [`14617c2b6c1257ac0d6c7b5e05b195ca30013827`](https://github.com/vllm-project/vllm/tree/14617c2b6c1257ac0d6c7b5e05b195ca30013827) |
| vLLM-Ascend | [`2c8e27aba65c49c41e9ce43795d5db1643032097`](https://github.com/vllm-project/vllm-ascend/tree/2c8e27aba65c49c41e9ce43795d5db1643032097) |
| xLLM | branch HEAD `e7af8f199`，另有未提交的 TMO metadata/cache 优化；本文不修改这些代码 |

## xLLM 当前 Direct DMA 基线

### 已经具备的能力

- Host KV 的 block tensor shape 是 `[num_host_blocks, num_layers,
  ...per_block_dims]`。这与 Direct I/O 所需的 block/page-first 组织一致；不是
  layer-first Host pool。
- `flatten_layered_plans_for_direct()` 按 layer-outer、object-inner 展开 copy。
  对同一层的一批 block，Host 地址通常以“完整 Host object 大小”为 pitch，
  device 地址以“单层 device block 大小”为 pitch。
- `find_direct_copy_run()` 将双边都连续的地址合成一个 extent；三个及以上固定
  双边 stride 的 descriptor 合成一个 `cnMemcpy2DAsync`；其他情况进入
  `cnMemcpyBatchAsync`，单批最多 4096 项。
- Direct H2D 是 submit-only；每个 layer range 在 copy stream 上记录 event，
  compute stream 到对应 range 时再等待。Direct D2H 是同步完成。
- Host memory 使用 `cnHostMemAlloc(..., CN_MEMHOSTALLOC_PORTABLE)`，不是 pageable
  memory。

关键源码：

- `xllm/core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.cpp:233-272`
- `xllm/core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.cpp:369-497`
- `xllm/core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.cpp:617-687`
- `xllm/core/platform/mlu/mlu_batch_memcpy.cpp:103-230`
- `xllm/core/platform/mlu/mlu_batch_memcpy.cpp:877-1024`
- `xllm/core/platform/mlu/mlu_host_memory.cpp:50-82`

### 当前仍存在的固定开销和排队风险

1. `HierarchyKVCacheTransfer` 固定创建 2 个 H2D load worker，并将
   `2 + kOffloadStreamCount(4)` 条 stream 放进同一个 pool。
2. `CopyStreamLease` 析构时直接把 stream 放回 pool；Direct H2D 此时可能仍在
   执行。后续 H2D 会继续追加队列，D2H 若拿到该 stream 并最终同步，还会被迫
   排空与自身无关的旧 H2D。
3. Direct `copy_direct()` 每次重建 descriptor、batch src/dst/size 数组。
4. `CNmemcpyBatchAsyncAttributes` 对 H2D/D2H 都设置
   `CN_MEMCPY_SRC_ACCESS_ORDER_QUEUE` 和 `CN_MEMCPY_FLAG_DEFAULT`。
5. D2H 外层逐 range 调 `copy_layered_d2h()`，而每次调用内部都会
   `cnQueueSync()`。
6. 每个 range 都重新过滤 `BlockTransferInfo`、构造 tensor views 和 layered
   plans。该 CPU 开销存在，但已有实验表明它不是当前首要瓶颈。

## 三个上游实现带来的有效启发

| 能力 | SGLang | vLLM | vLLM-Ascend | 对 xLLM 的含义 |
|---|---|---|---|---|
| 方向隔离 | 独立 D2H/H2D stream | 两个 `SingleDirectionOffloadingHandler` | 两个 `SingleDirectionNPUOffloadingHandler`；Simple path 也用独立 load/store stream | xLLM 的共享 stream pool 值得优先拆分 |
| 异步完成 | event + layer callback，indices `record_stream` | event query，完成后回收 stream/event/buffer | event query/poll，完成后回收 stream/event/descriptor | xLLM 应显式跟踪 pending job，而非 DMA 未完成就无条件回池 |
| descriptor 复用 | 合并 queue operation；indices 一次准备 | pinned descriptor buffer pool、静态 canonical copy plan、scratch base | pinned descriptor buffer pool，预构建 tensor base/stride params | Direct scratch/pool 有价值，但需端到端验证 |
| H2D source ordering | CUDA kernel/direct 各自维护生命周期 | H2D 用 `SRC_ACCESS_ORDER_ANY`，D2H 用 STREAM | Ascend batch API没有完全相同的公开参数 | MLU CNDrv 恰好提供同名 `ANY`，适合做 H2D-only A/B |
| Host chunk | `page_first_direct` 把一页的 layer/KV 放在一起 | `blocks_per_chunk` | 原生 NPU path同样支持 `blocks_per_chunk` | xLLM 已是 block×layer；更大 chunk 只有在改善映射连续性时才有收益 |
| 流量整形 | kernel `block_quota` 限制计算资源干扰 | 单方向有序 job queue | `h2d_stagger_us`、transfer block/byte limit | xLLM 应优先做 DMA queue credit 和 TP-rank 错峰 |

### SGLang

SGLang 的 `L2TransferEngine` 明确创建独立 D2H/H2D stream；H2D 按层提交并在
每层完成后回调，传输 indices 用 `record_stream` 保活。`CacheOperation.merge_ops`
会合并同一轮操作；`LayerDoneCounter` 使用可查询完成状态的 event ring。

SGLang 官方建议 Direct I/O 使用 `page_first_direct`，其 MHA Host shape 为
`[2, page, layer, page_size, head, dim]`。这有利于一个 page 的完整导出和 Direct
I/O，但它不意味着 per-layer restore 自动变成一个连续 DMA；实际 Direct H2D
仍需按 page、layer、K/V 解析地址。xLLM 的 `[block, layer, ...]` 已经具备相同的
核心物理性质。

SGLang 的 HiCache CUDA/ROCm kernel 能用 device indices 直接访问可映射 Host
memory，并以 `block_quota` 限制资源干扰。MLU590 的公开能力不支持把普通 pinned
Host KV 映射成 BANG kernel 可直接解引用的 GDRAM，因此该 kernel 不能直接移植。

源码：

- [SGLang `l2_transfer.py`](https://github.com/sgl-project/sglang/blob/21c88f8625f2e699543a1c34f41d6894ef342903/python/sglang/srt/mem_cache/l2_transfer.py)
- [SGLang `cache_controller.py`](https://github.com/sgl-project/sglang/blob/21c88f8625f2e699543a1c34f41d6894ef342903/python/sglang/srt/managers/cache_controller.py)
- [SGLang MHA Host pool](https://github.com/sgl-project/sglang/blob/21c88f8625f2e699543a1c34f41d6894ef342903/python/sglang/srt/mem_cache/pool_host/mha.py)
- [SGLang HiCache best practices](https://github.com/sgl-project/sglang/blob/21c88f8625f2e699543a1c34f41d6894ef342903/docs/docs/advanced_features/hicache_best_practices.mdx)
- [SGLang HiCache kernel wrapper](https://github.com/sgl-project/sglang/blob/21c88f8625f2e699543a1c34f41d6894ef342903/python/sglang/kernels/ops/kvcache/hicache.py)

### vLLM

vLLM 当前 CPU offload 为每个方向各建一个 handler。每个 job 获取独立 stream、
start/end event 和 pinned descriptor buffers；job 完成后统一回池。方向内通过前一
job 的 end event 保序，两个方向彼此独立。

它还做了三件与 xLLM Direct 相关的事：

- canonical layout 的 fragment offsets 在初始化时预计算，运行时复用 scratch
  base arrays；
- CPU chunk 可包含多个 GPU block，由 `blocks_per_chunk` 做 sub-block pointer
  展开；
- H2D pinned Host 源在传输期间不被并发写，因此 batch DMA 使用
  `CU_MEMCPY_SRC_ACCESS_ORDER_ANY`；D2H 源仍受 compute stream 写入，必须保留
  STREAM ordering。

vLLM 还有一个 H100 专用 Triton 小包路径：descriptor 至少 16 个、每项小于
28 KiB 才选 kernel，否则走 DMA。这个阈值是 H100 PCIe Gen5 实测值，不能直接
用于 MLU；它只证明 backend routing 应由 shape 决定，而不是一个 backend 全包。

源码：

- [vLLM CPU offload worker](https://github.com/vllm-project/vllm/blob/14617c2b6c1257ac0d6c7b5e05b195ca30013827/vllm/v1/kv_offload/cpu/gpu_worker.py)
- [vLLM batch DMA binding](https://github.com/vllm-project/vllm/blob/14617c2b6c1257ac0d6c7b5e05b195ca30013827/csrc/libtorch_stable/cache_kernels.cu)
- [vLLM shape-aware Triton route](https://github.com/vllm-project/vllm/blob/14617c2b6c1257ac0d6c7b5e05b195ca30013827/vllm/v1/kv_offload/cpu/swap_blocks_triton.py)

### vLLM-Ascend

vLLM-Ascend 的原生 NPU offload 基本复刻 vLLM 的生命周期设计：H2D/D2H 各自
持有 handler，复用 stream、timing event 和 pinned descriptor tensors，CPU
allocation 支持 `blocks_per_chunk`，最终通过 `swap_blocks_batch` 调
`aclrtMemcpyBatchAsync`。Simple offload path 也使用独立 load/store stream，完成
由 NPU event 轮询，不在 copy worker 内做全局同步。

它的 layerwise store path 额外暴露：

- `h2d_stagger_us`：按 TP rank/layer 错开 H2D 提交；
- `layerwise_max_transfer_blocks`：限制一次 transfer 的 block 数；
- `layerwise_max_transfer_bytes`：拆分过大的单个 transfer。

这些选项不能证明同一参数在 MLU 上一定有收益，但它们直接支持“多 rank 同时提交
会产生资源争用，需要显式整形”这一优化方向。

源码：

- [vLLM-Ascend native NPU offload](https://github.com/vllm-project/vllm-ascend/blob/2c8e27aba65c49c41e9ce43795d5db1643032097/vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/native/cpu_npu.py)
- [vLLM-Ascend Simple copy backend](https://github.com/vllm-project/vllm-ascend/blob/2c8e27aba65c49c41e9ce43795d5db1643032097/vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/simple/copy_backend.py)
- [vLLM-Ascend batch pointer construction](https://github.com/vllm-project/vllm-ascend/blob/2c8e27aba65c49c41e9ce43795d5db1643032097/vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/simple/npu_mem_ops.py)
- [vLLM-Ascend `aclrtMemcpyBatchAsync` binding](https://github.com/vllm-project/vllm-ascend/blob/2c8e27aba65c49c41e9ce43795d5db1643032097/csrc/torch_binding.cpp)
- [vLLM-Ascend CPU offload guide](https://github.com/vllm-project/vllm-ascend/blob/2c8e27aba65c49c41e9ce43795d5db1643032097/docs/source/user_guide/feature_guide/kv_cache_cpu_offload.md)
- [vLLM-Ascend layerwise transfer shaping](https://github.com/vllm-project/vllm-ascend/blob/2c8e27aba65c49c41e9ce43795d5db1643032097/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py)

## 已被 xLLM 实验验证的结果

### 已证明有效

1. **连续/2D region 合并是 Direct 路径的主收益。** 旧 64K trace 搬运
   6.332 GB，却产生 413,597 个 H2D event，平均 15.3 KiB。把 413,595 个小
   descriptor 合成 101 个 2D region 后，微基准从 3.75 GB/s 提升到
   54.9 GB/s，达到连续大块的 96.7%。
2. **新 Direct 相对旧 Direct 的端到端收益明确。** 32K TTFT mean 提升
   1.589x，64K 提升 1.922x。
3. **device event 的异步 H2D 依赖有增量收益，但并不稳定。** 在已有 2D
   优化上，5 个 cell 中 3 个改善，中位 cell 约 +11.4%；高并发/碎片映射下会
   因 runtime queue backpressure 回退。
4. **当前工作树的 TMO metadata 一次构造/上传与 paged pointer cache 有明确
   收益。** 最新 TMO 相对同 binary Direct 的 32K/64K TTFT mean 分别为
   1.358x/1.260x。这个结论属于整体 Host KV backend 对比，不代表 Direct DMA
   本身已经没有优化空间。

本地报告：

- `agent-workspace/experiments/host_kv_optimization_ab-second/final_report.md`
- `agent-workspace/experiments/mlu_tmo_kv_transfer_new_ab/new_direct_5c261dfa_20260819/REPORT.md`
- `agent-workspace/experiments/mlu_tmo_kv_transfer_new_ab/tmo_opt2_36c471ed_20260820/FINAL_REPORT.md`

### 已证明无效或不应直接重做

1. **只做 one-pass copy plan。** plan mean 从 1.670 ms 降到 0.357 ms，但 H2D
   span mean 增加 5.58%，Host TTFT mean/p50 回退 4.80%/5.68%。
2. **只在 launcher 外套 `numactl`。** Host page placement 几乎没变化，serving
   TTFT 没有稳定收益。
3. **`mmap + mbind + cnHostMemRegister` 替换厂商 pinned allocator。** 1 GiB
   register 超过 3 分钟，D2H 出现数量级退化；该方案已被实验否决。
4. **直接移植 SGLang CUDA host-mapped gather kernel。** MLU590 当前公开能力
   不支持普通 pinned Host memory 被 BANG kernel 直接解引用；CNNL gather/scatter
   只覆盖 MLU memory 内部数据。
5. **把 page-first 当作减少 DMA descriptor 的充分条件。** xLLM 已是
   block-first Host layout；descriptor 数最终取决于 host/device block pair 的
   地址规律，而不是 layout 名称。

对应审计：

- `agent-workspace/experiments/sglang_one_layer_h2d_gap_20260731/sglang_gap_audit.md`
- `agent-workspace/research/mlu_indexed_host_device_copy_support.md`

## 推荐实现与验收顺序

### P0-0：先补 Direct DMA 可观测性

每个 request/range 至少记录：

- coalesce 前 logical descriptor 数和 bytes；
- EXTENT、2D、BATCH 各自的 calls、descriptors、bytes；
- run length 的 mean/p50/p95/max；
- `build_plan_us`、`coalesce_us`、各 CN API submit wall time、stream lease wait；
- 每条 stream 的 pending jobs/bytes，H2D/D2H 分开；
- TP rank、NUMA node 和 request concurrency。

没有这些数据，`ANY`、descriptor pool、run-aware allocation 的优先级无法可靠判断。

### P0-1：方向化 stream + 有界 pending queue

建议把当前共享 `copy_stream_` 拆为：

```text
H2D submit workers -> h2d stream(s) -> per-range ready events -> compute waits
D2H callers       -> d2h stream(s) -> final completion event/sync
```

第一轮不应一次改太多：

1. 先只拆 H2D/D2H pool，保持 H2D=2、D2H=4；
2. 再扫 H2D stream 数 1/2/4、D2H stream 数 1/2；
3. 最后加入 pending byte/region credit，stream 仅在 end event 完成或 credit
   允许时复用。

预期主要改善 C2/C4 的 wave-tail、D2H offload turnaround 和 Host API submit
阻塞；C1 吞吐可能接近持平。

### P0-2：H2D-only source access order

候选改动仅作用于 batch fallback：

```cpp
attr.srcAccessOrder = h2d ? CN_MEMCPY_SRC_ACCESS_ORDER_ANY
                          : CN_MEMCPY_SRC_ACCESS_ORDER_QUEUE;
```

启用前必须确认：H2D 引用的 Host block 在 end event 前不会被 D2H、远端 store
或淘汰复用并发写入。若该生命周期不变量不能由 block refcount/event 明确证明，
就不能开启 `ANY`。

`CN_MEMCPY_FLAG_OVERLAP_WITH_COMPUTE` 应作为第二个独立开关 A/B，不能和
`ANY` 一次合入，否则无法归因；它只是 vendor performance hint，也可能因总线或
memory-controller 竞争使 TTFT 退化。

### P0-3：D2H 一次 sync

`offload_to_host()` 可把所有 range 的 `layered_plans` append 到一个 vector，
只调用一次 `copy_layered_d2h()`。保持：

- 调用前一次 `wait_stream(compute_stream)`；
- 所有 plans 在第一次 submit 前完成校验；
- 失败路径 drain；
- 最终一次同步成功后才 publish Host block。

这项不会直接缩短 Host-hit H2D 的单请求 critical path，但会更快释放 RemoteWorker
copy thread、device block 和 Host publish gate，降低持续 churn 下的间接排队。

### P0-4：TP-rank 错峰与 inflight credit

优先做可配置实验，不硬编码策略：

- `stagger_us = (tp_rank + range_idx) % tp_size * base_us`；
- `base_us` 扫 0/10/25/50/100；
- 同时扫每 rank 最大 pending bytes，而不是只限制 job 数；
- 观察最慢 rank，不只看 rank0 或 aggregate GB/s。

如果 stagger 只降低 aggregate 带宽而没有改善 TTFT/p99，应保留为关闭状态。

### P1：提高规则 run 比例

在 telemetry 证明 BATCH bytes 占比高之后，再比较两条路线：

1. **descriptor 重排：** 对 source/destination 都唯一、彼此无覆盖的 copy pair，
   在同 storage/dtype/byte-count 内重新排序，优先抽取双边等差 run；
2. **run-aware allocation：** Host restore 分配 device blocks 时优先取连续 run，
   并按 token/Host block 顺序配对。

第一条改动局部但要证明 copy 独立性；第二条可能改变 block allocator 碎片与 prefix
cache 行为，风险更高。两者都必须同时观察长期 churn 后的 free-list fragmentation。

### P1/P2：descriptor pool、静态 offset 与 `blocks_per_chunk`

- Direct layered 路径可以直接从 Host base、layer stride、device layer base 和
  block id 生成 raw descriptor，避免先创建 `block × layer` 个 Tensor view，
  随后又逐个提取 `data_ptr/storage/nbytes`。这与“提前一次构建所有 range plan”
  不是同一个优化，但仍要保持原有逐 range ready-event 提交节奏。
- 为每个 H2D worker/stream 保存可增长但不缩容的 descriptor scratch，可避免
  `DirectCopyDescriptor` 和三个 batch vector 的重复 allocation。
- 静态部分只缓存 storage base、layer offset、block stride；每次请求仍根据 block
  ids 填最终地址。
- `blocks_per_chunk` 只有在 block manager 也以 chunk 粒度分配/回收、从而提高
  双边规则 run 比例时才可能减少 Direct API work。仅把多个已连续的 Host blocks
  包成更大逻辑 object，不会改变 xLLM 当前 2D DMA 已能表达的内容。

这些优化应设一个保留门槛：submit CPU wall time显著下降，并且 32K/64K Host
TTFT mean 至少改善 5%、p99 不退化超过 5%。

### P2：真正减少传输字节数

vLLM-Ascend 的 sparse decode 只恢复 hot-buffer 中缺失的 top-k rows，而不是把
完整历史 KV 搬回 NPU。这是本轮上游调研中唯一能从根本上减少 PCIe bytes 的
方向，但它要求 attention、residency table、eviction 和 scheduler 共同改造，
不属于 Direct DMA backend 的局部优化。只有在前述 P0/P1 已经把 DMA 提交与排队
压到合理水平、总传输字节仍是主瓶颈时，才值得立项。

## 建议 A/B 矩阵

固定 GLM-5.2-W4A8、TP8、相同 Decode、相同 prompt namespace 和相同 binary，
至少覆盖：

| 维度 | 建议值 |
|---|---|
| Prompt | 32K、64K；容量允许时加 128K |
| 并发 | C1、C2、C4 |
| Host 状态 | Cold / HBM / Host 三态配对 |
| mapping | 初始连续、稳定 churn、强碎片三档 |
| backend | Direct baseline；每个候选只改一个变量 |
| repetitions | 每 cell 至少 5，tail 判断建议 10+ |

主指标：Host TTFT mean/p50/p99、paired `Host-HBM` gap、effective prompt tok/s。
机制指标：各类 DMA bytes/calls、submit wall time、active DMA、queue pending
bytes、最慢 TP rank、D2H publish latency。任何优化都必须同时通过 cached-token
完整性和输出 token 一致性验证。

## 最终优先级

```text
P0  telemetry
 -> H2D/D2H stream 分离
 -> pending byte credit / TP-rank stagger
 -> H2D batch srcAccessOrder=ANY A/B
 -> D2H all-range submit + single sync

P1  基于 telemetry 的 run 重排或 run-aware allocation
 -> descriptor scratch/pool + 静态 offset cache
 -> shape-aware Direct/TMO/backend route

P2  allocator/NUMA vendor 能力；不要重做 mmap+register
```

如果只选一个最适合马上实现的改动，建议先做 **方向化 stream + pending
bytes/events telemetry**。它同时解决当前代码可见的共享队列问题，也为后续判断
`ANY`、stagger、descriptor pool 和 run-aware allocation 提供可信数据。
