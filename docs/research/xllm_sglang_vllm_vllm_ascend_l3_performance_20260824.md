# xLLM MLU + Mooncake 本地 SSD L3 性能分析及框架对比

日期：2026-08-24（Asia/Shanghai）  
范围：xLLM `feat/mlu-tmo-kv-transfer-new` 的正式 attempt6 TTFT 数据；SGLang、vLLM、vLLM-Ascend 官方源码/文档。  
结论标签：**[源码证实]** 表示代码或官方文档直接给出；**[实测证实]** 表示本次正式产物直接给出；**[推断]** 表示由源码和指标共同支持、但尚未通过单变量 A/B 实验隔离。

## 1. 结论先行

当前 L3 比 cold 慢，主因不是 NVMe 物理带宽，而是 **KV 对象和调用被切得过碎之后，元数据、RPC、TransferEngine TCP 建连、串行等待和 Host staging 的总开销占据了关键路径**。

1. **[实测证实] 盘没有被打满。** 1K/4K 的 L3 p50 端到端有效吞吐分别约 **1.39/1.45 GiB/s**，而单盘 `O_DIRECT` 长读实测约 **6.5 GiB/s**。正式 Mooncake 指标中 6,400 次 SSD `BatchLoad` 平均只有 **1.895 ms**，但框架日志里 `key size: 8` 的端到端 offload call p50 是 **32.5 ms**，约为纯 SSD batch 时间的 **17.1 倍**。[正式 TTFT JSON](../../agent-workspace/experiments/l3_pd/20260824-121713/post_fix_eval/ttft-attempt6-cold-l3.json)、[client metrics](../../agent-workspace/experiments/l3_pd/20260824-185300/metrics/client-after-attempt6-ttft-l3.prom)、[SSD 基准](../../benchmark_results/ssd_to_host_20260824/experiment_report.md)
2. **[源码 + 实测证实] 4K 的关键路径恰好由小批次串行时间解释。** `prefetch_batch_size=8`；每 rank 有 256 个 16-token block，即 32 个 BRPC chunk；MTP target/draft 在每个 chunk 内顺序读取，所以每 rank 是 64 个 Mooncake batch。`64 × 32.5 ms ≈ 2.08 s`，和 4K L3 p50 **2.337 s** 同量级；剩余约 0.26 s 才包括 admission、Host→MLU、调度和首 token。反过来，`64 × 1.895 ms ≈ 0.121 s`，表明盘内 `BatchLoad` 只占小部分。[MTP 顺序调用](../../xllm/core/runtime/mtp_worker_impl.cpp#L1047)、[BRPC chunk 串行循环](../../xllm/core/distributed_runtime/comm_channel.cpp#L552)
3. **[源码 + 实测证实] 一次正式 TTFT replay 产生 51,200 个物理 Store 对象、6,400 次 SSD batch RPC、12,800 次 master metadata query request、102,400 个 query item，以及约 102,400 条短 TCP 数据连接。** 这些数字彼此严格闭合，不是采样推测。
4. **[实测证实] 当前 SSD 配置把 Mooncake 推荐的 bucket 聚合退化了。** `MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT=1`，因此每对象一个 `.bucket` 和一个 `.meta`；同时 `MOONCAKE_OFFLOAD_USE_URING=0`，走 buffered POSIX 文件 I/O；未 drop page cache。故当前 L3 TTFT 更接近 **buffered/page-cache warm 的路径上限**，不能当成“物理冷 NVMe”结果。Mooncake 官方默认 bucket 上限是 500 keys/256 MB，并明确 bucket 是一个 `.bucket` 数据文件加一个 `.meta` 文件。[Mooncake SSD 部署文档](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/deployment/ssd/ssd-offload.md)
5. **[源码证实] 当前 xLLM admission 要等所有 rank 的 Storage→Host 全部完成，才发布 Host prefix、入队调度并发起 Host→MLU。** 已有 Host→MLU layer-wise event，但 L3 read 与 H2D/compute 之间没有流水，因此不能隐藏 SSD/网络等待。[admission barrier](../../xllm/core/scheduler/continuous_scheduler.cpp#L179)、[只有结果全完成才 publish](../../xllm/core/framework/block/hierarchy_block_manager_pool.cpp#L850)、[随后才调度 H2D](../../xllm/core/framework/block/hierarchy_block_manager_pool.cpp#L955)
6. **对比边界：** SGLang 和 vLLM-Ascend 的持久 SSD 路径也可以使用 Mooncake，并没有一个天然更快的 SSD 介质实现；它们主要在框架层提供更大的 page/block batch、registered multi-buffer、后台预取/写回和更清晰的异步状态。vLLM 当前 main 的持久 FS tier 则采用另一套本地文件路径；其设计可参考，但官方当前把 OffloadingConnector 平台支持限定在 CUDA/ROCm/XPU，不能据此声称 MLU/NPU 现成可用。[vLLM KV offloading 官方文档](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/)

因此，当前性能差首先是 **配置与调用粒度问题**，然后才是需要代码改造的流水问题。最值得先做的不是换盘，而是：恢复 bucket 聚合、开启可验证的 direct I/O、复用 TCP 连接、消除重复 metadata query，并把 batch 从 8 提高到 64/128 做 A/B。

## 2. 正式 attempt6 数据闭环

### 2.1 TTFT、对象和有效吞吐

正式测试每个长度 10 个样本，L3 20/20 全命中完整 prompt。1K 的前几个样本受连接冷启动/端口压力影响出现 4.93 s、1.59 s、1.56 s 离群值，因此这里用 p50 描述稳态路径，同时保留 mean 作为尾部风险提醒。

| Prompt | 16-token 逻辑 block | 物理 Store objects/请求 | KV 字节/请求 | cold p50 | L3 p50 | L3/cold | L3 p50 有效吞吐 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K | 64 | 1,024 | 0.8486 GiB | 191.516 ms | 608.816 ms | 3.18× | 1.394 GiB/s |
| 4K | 256 | 4,096 | 3.3945 GiB | 730.479 ms | 2,336.768 ms | 3.20× | 1.453 GiB/s |

1K L3 mean 为 1,234.137 ms，4K L3 mean 为 2,342.630 ms。所有原始记录见[正式 TTFT JSON](../../agent-workspace/experiments/l3_pd/20260824-121713/post_fix_eval/ttft-attempt6-cold-l3.json)。

对象数的推导如下：

- **[源码证实]** key 中编码 `tp_rank` 和 `block_type`；一个 `BlockTransferInfo` 对应一个 Store key。MLU MLA value cache 被移除，本模型每对象含 KEY 和 INDEX 两个 slice。[key/object 构造](../../xllm/core/framework/kv_cache_transfer/kv_cache_store.cpp#L125)、[MLU MLA layout](../../xllm/core/framework/kv_cache/kv_cache_shape.cpp#L404)、[INDEX slice](../../xllm/core/framework/kv_cache/indexed_kv_cache_impl.cpp#L155)
- **[实测证实]** TP=8、MTP target+draft=2 份。因此 `64 × 8 × 2 = 1,024`，`256 × 8 × 2 = 4,096` 个对象/请求。正式日志给出的 target slot 为 1,757,184 B、draft slot 为 22,528 B，平均每对象 `(1,757,184 + 22,528)/2 = 889,856 B`。[node_0 formal log](../../testspace/scripts/logs/l3-mlu-pd-postfix5-20260824-185300-attempt6-replay/prefill_0/node_0.log)
- 故 1K 为 `1,024 × 889,856 = 911,212,544 B = 0.8486 GiB`；4K 为 `4,096 × 889,856 = 3,644,850,176 B = 3.3945 GiB`。
- 20 次请求合计 `10 × 1,024 + 10 × 4,096 = 51,200` 个对象、`45,560,627,200 B`；这和 `mooncake_ssd_read_ops_total=51,200`、`mooncake_ssd_read_bytes_total=45,560,627,200` 完全相等。[client metrics](../../agent-workspace/experiments/l3_pd/20260824-185300/metrics/client-after-attempt6-ttft-l3.prom)

### 2.2 metadata、RPC、SSD batch 与 TCP 次数

| 层次 | attempt6 TTFT 增量 | 为什么是这个数 |
|---|---:|---|
| Mooncake SSD object reads | 51,200 | 上述 20 请求的物理对象总数 |
| `BatchLoad` / offload batch RPC | 6,400 | 1K：`10×8 ranks×8 chunks×2` = 1,280；4K：`10×8×32×2` = 5,120 |
| Master `BatchGetReplicaList` requests | 12,800 | 每个 offload batch 查询两次 metadata |
| Master query items / file hits | 102,400 | `6,400 batches × 8 keys × 2 queries` |
| TCP data connections（估算值，机制和对象数均已证实） | 约 102,400 | 51,200 对象 × KEY/INDEX 两个 Slice；pool 关闭时每 Slice 新建并关闭连接 |

**重复 metadata query 是源码和指标双重证据：**

1. xLLM `MooncakeStoreClient::batch_get` 先 `batch_query(keys)`，用它判断 `LOCAL_DISK`。[xLLM wrapper](../../xllm/core/framework/kv_cache_transfer/mooncake_store_client.cpp#L129)
2. 对 local disk 随后调用 `batch_get_into_multi_buffers`。[xLLM local-disk branch](../../xllm/core/framework/kv_cache_transfer/mooncake_store_client.cpp#L216)
3. Mooncake `RealClient::batch_get_into_multi_buffers_internal` 内部又执行一次 `BatchQuery(keys)`。[patched Mooncake source](../../third_party/Mooncake/mooncake-store/src/real_client.cpp#L4973)
4. Master before/after 正式指标正好增加 12,800 requests、102,400 items 和 102,400 file hits；而实际 SSD reads 是 51,200，恰为一半。[master before](../../agent-workspace/experiments/l3_pd/20260824-185300/metrics/master-before-attempt6-ttft-l3.prom)、[master after](../../agent-workspace/experiments/l3_pd/20260824-185300/metrics/master-after-attempt6-ttft-l3.prom)

**短 TCP 连接同样是源码和观测闭合：**

- Mooncake TCP connection pool 默认 `false`，只有设置 `MC_TCP_ENABLE_CONNECTION_POOL=1` 才开启。[patched TCP header](../../third_party/Mooncake/mooncake-transfer-engine/include/transport/tcp_transport/tcp_transport.h#L111)
- pool 关闭时 `getConnection` 每次 resolve/connect；每个 Slice 的 `startTransfer` 取一条连接，并在完成后立即 close。[patched TCP source](../../third_party/Mooncake/mooncake-transfer-engine/src/transport/tcp_transport/tcp_transport.cpp#L1085)
- 本模型每 Store object 有 KEY/INDEX 两个 Slice，所以一次 1K request 约新建 `1,024×2=2,048` 条数据连接。原始 JSON 中 TIME_WAIT 在连续 1K 样本后大致以这个量级增长，峰值约 21k，并曾出现临时端口耗尽。Mooncake 官方也把“每 transfer 新建短连接→TIME_WAIT→`Cannot assign requested address`”列为已知故障模式，并建议 connection pool，但同时提示当前 pool 的 idle fd 回收与锁存在限制。[Mooncake TCP troubleshooting](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/troubleshooting/troubleshooting.md)

日志统计边界：已有汇总曾记录全量 7,504 次 `key size: 8` call、p50 32.5 ms；本报告在当前保留的 8 个 attempt6-replay node 日志中可重新提取 **6,512 次（每 rank 814）**，p50 同为 **32.5 ms**。其中正式 TTFT 的 **6,400 次**由 SSD metric count、对象数和 batch 公式三路严格锁定；额外 112 次属于正式窗口外 probe/startup。后续因果分析只使用可闭合的 6,400 次，不依赖 7,504/6,512 的差异。

### 2.3 当前实际数据路径

```text
请求进入 admission
  └─ 8 个 TP worker 并行
      └─ 每 worker 按 8 blocks 切 BRPC chunk；chunk 之间串行
          ├─ target: query metadata
          │    └─ local-disk multi-buffer get
          │         ├─ 再 query metadata
          │         ├─ RPC 到 SSD owner，BatchLoad 到 owner staging buffer
          │         └─ TransferEngine 按 KEY/INDEX slice 传到 xLLM Host block
          └─ draft: 完整重复上述流程（与 target 串行）
  └─ 等所有 worker/chunk 全完成，合并命中并发布 Host prefix
      └─ 请求进入调度，才启动 layer-wise Host→MLU
          └─ forward/first token
```

Mooncake owner 把 SSD `BatchLoad` 放进专用线程池，之后通过 TransferEngine 将 staging buffer 传到调用方并释放 lease；这说明“SSD I/O 有后台线程”并不等价于“xLLM request 已与 H2D/compute 流水化”。[owner BatchLoad/RPC](../../third_party/Mooncake/mooncake-store/src/real_client.cpp#L5613)、[owner→caller transfer](../../third_party/Mooncake/mooncake-store/src/real_client.cpp#L5653)

## 3. 当前瓶颈逐项分析

### 3.1 KV 对象粒度与每请求对象数

**[源码证实]** xLLM 的持久对象粒度是“一个 TP rank、一个 block type、一个 16-token hash block”；对象 value 是该 Host block 对应的全部缓存 tensor slices。这个粒度本身保持了 prefix 的细粒度复用，但 TP8 和 MTP target/draft 把逻辑 block 数放大 16 倍。

**[推断]** 889,856 B/object 并非传统意义的 4 KiB 小 I/O，真正的问题是 **每对象都付一次 metadata、bucket/file、RPC/lease 和两次 TCP connection 的固定成本**。应优先批量化这些固定成本；是否再把 8/16 个逻辑 block 聚成一个持久 super-chunk，要在 prefix 命中精度、partial load、eviction 粒度和吞吐之间权衡。

### 3.2 SSD 文件布局

**[源码证实]** Mooncake bucket backend 会按 bucket 聚合 key，每 bucket 一个 `.bucket` 数据文件和一个 `.meta` 文件；读取时先按 bucket 分组，再逐 bucket open 并读取其中对象。[patched bucket load](../../third_party/Mooncake/mooncake-store/src/storage_backend.cpp#L1768)、[Mooncake SSD design](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/store/ssd-offload.md)

**[实测证实]** 当前 `bucket_keys_limit=1`，所以等价于每对象两文件。此前目录统计 75,793 objects 对应 151,586 files，恰为 2×。这不是 Mooncake 推荐生产默认；官方默认 500 keys/256 MB，并把 bucket backend 定位为通用推荐、file-per-key 仅适合调试/小规模。[Mooncake SSD deployment](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/deployment/ssd/ssd-offload.md)

### 3.3 batch I/O、io_uring 与 direct I/O

**[源码证实]** 当前 patched Mooncake 只有编译了 `USE_URING` 且 runtime `use_uring_` 为真时，读 fd 才带 `O_DIRECT` 并使用 `UringFile`；否则使用 `PosixFile` buffered I/O。[file factory](../../third_party/Mooncake/mooncake-store/src/storage_backend.cpp#L836)

**[实测证实]** 正式服务 `MOONCAKE_OFFLOAD_USE_URING=0`，并且没有 drop page cache。因此 1.39/1.45 GiB/s 既不是 direct I/O，也不能证明产生了同等物理盘读流量。独立控制实验显示同一文件 page-cache warm 时可达 6.313 GB/s 且物理盘几乎不读，这正说明必须用 `O_DIRECT` 或盘计数器验证。[SSD 基准页缓存控制](../../benchmark_results/ssd_to_host_20260824/experiment_report.md#页缓存控制)

**[推断]** 当前 `BatchLoad` 平均仅 1.895 ms，很可能有相当比例命中 page cache；即使如此，end-to-end 仍需 32.5 ms/batch，反而更有力地说明控制面/传输面是主要差距。启用 io_uring/O_DIRECT 后第一次 TTFT 可能变慢，但测量才代表真实冷 SSD；优化目标应同时看物理盘吞吐与端到端 TTFT。

### 3.4 CPU staging、D2H/H2D 与流水

**[源码证实]** L1→L3 store 先同步 D2H 到 xLLM Host block，再 `batch_put`；L3→L1 先完整读回 Host block，admission publish 后才启动异步、layer-wise H2D。[store D2H→put](../../xllm/core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.cpp#L593)、[storage get](../../xllm/core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.cpp#L574)、[H2D schedule](../../xllm/core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.cpp#L521)

Host KV buffers已注册给 Mooncake TransferEngine，避免 owner staging 数据再经过一个普通临时 CPU memcpy；但当前仍存在 owner SSD staging buffer→xLLM Host block→MLU 的两段搬运。[Host tensor registration](../../xllm/core/framework/kv_cache_transfer/kv_cache_store.cpp#L61) **[推断]** 现阶段优化重点不是简单删掉 Host 层，因为 Host prefix cache 还有 L2 语义，而是做双缓冲/分批 readiness，让 batch N 的 H2D 和 batch N+1 的 SSD/RPC 重叠。

### 3.5 promotion 与 eviction

Mooncake 官方支持 `promotion_on_hit`，可在 SSD hit 后按 admission threshold 提升回 DRAM，也支持 bucket FIFO/LRU eviction；当前正式测试的 master memory hit 未增长，说明 replay 仍反复走 SSD，而非热 prefix 晋升到 L2。[Mooncake SSD deployment](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/deployment/ssd/ssd-offload.md)

**边界：** promotion 改善的是第二次及后续复用；第一次 SSD hit 的 TTFT 仍要靠 batch、连接复用、direct I/O 和流水优化。打开 promotion 也会增加 DRAM 占用和 SSD→DRAM 后台流量，应单独测命中次数阈值和 eviction 策略。

## 4. 与 SGLang、vLLM、vLLM-Ascend 的源码级对比

版本快照：SGLang `0db2c53deca12715d209ffbd46bdfafdfd5a5d35`；vLLM `36bad1b90cb2958b6de4abdb5a9249b136f39720`；vLLM-Ascend `b6ec3943ff5e73359a5c29c1b29fe17672ca45e3`。三者均来自官方 GitHub 仓库。

| 维度 | 当前 xLLM MLU + Mooncake | SGLang HiCache | vLLM current main | vLLM-Ascend AscendStore |
|---|---|---|---|---|
| KV 对象粒度 | 16-token block × TP rank × target/draft；一个对象含该 block 的多 tensor slices | page 级；Mooncake MLA 一个 page key 可映射为 multi-buffer，MHA 通常 K/V component keys；draft/side pool单独命名。[源码](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L780-L930) | 持久 FS tier 的 CPU primary block 可由 `blocks_per_chunk` 聚合多个 GPU blocks；一个 content hash/group 对应一个 `.bin`。[源码](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/file_mapper.py#L21-L126) | non-layerwise 是 block/chunk × cache group × rank key，value 是覆盖多层的 multi-buffer；key 显式含 TP/PP/PCP/DCP/group。[源码](https://github.com/vllm-project/vllm-ascend/blob/b6ec3943ff5e73359a5c29c1b29fe17672ca45e3/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py#L72-L170) |
| 当前同模型每请求对象数 | 已精确测得：1K=1,024；4K=4,096 | 未在本机跑同模型，不能给伪精确数。MLA cache 被视为 TP rank-replicated，只有 TP rank0 负责 backup，减少持久副本；load 是否仍由各 rank 发起取决于运行配置。[源码](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/managers/cache_controller.py#L500-L529) | 取决于 `blocks_per_chunk`、KV group 和并行布局；不能直接套 xLLM 的 1K/4K 数 | 取决于 group/block/parallel；不可直接套 xLLM 数。支持 TP rank circular shift 来均衡一个 batch 内 key 次序。[源码](https://github.com/vllm-project/vllm-ascend/blob/b6ec3943ff5e73359a5c29c1b29fe17672ca45e3/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py#L895-L1035) |
| SSD 文件布局 | bucket backend，但 keys_limit=1 导致每对象 `.bucket`+`.meta` | Mooncake backend时仍由 Mooncake bucket 配置决定；NIXL/HiCacheFile 是另一套 backend，不应混淆 | 持久 FS tier：hash 前缀两级目录分片、每 chunk/group 一个 `.bin`；另有 Simple disk 单个预分配 slot 文件，但它会在进程生命周期内创建/删除，属于临时 swap。[persistent](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/tiering/fs/manager.py#L1-L16)、[ephemeral](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/simple_kv_offload/disk_backend.py#L148-L218) | Mooncake SSD 文件格式仍由 Mooncake backend 决定；源码为每个 global rank 建独立 SSD 子目录以避免 bucket 冲突。[源码](https://github.com/vllm-project/vllm-ascend/blob/b6ec3943ff5e73359a5c29c1b29fe17672ca45e3/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py#L93-L161) |
| metadata/RPC | 每 batch 两次 metadata query；SSD owner RPC；每 slice 新短 TCP | Mooncake backend同样受 Mooncake client/backend行为约束，但框架直接 batch multi-buffer；embedded mode可避免 xLLM这种额外 wrapper query层。[源码](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L1294-L1333) | 本地 FS tier没有 master/RPC/TCP；existence 走 batched `faccessat` C extension或 `exists`。远端 connector是另一语义。[源码](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/tiering/fs/manager.py#L65-L105) | embedded RealClient，直接 `batch_is_exist`、`batch_put/get_*multi_buffers`；仍受 Mooncake master与传输协议开销影响。[源码](https://github.com/vllm-project/vllm-ascend/blob/b6ec3943ff5e73359a5c29c1b29fe17672ca45e3/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py#L173-L266) |
| batch | xLLM BRPC batch=8；target/draft顺序；chunk顺序 | storage batch固定 128 pages；后台 thread逐 batch处理。[源码](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/hicache_storage.py#L20-L24) | submit load/store 为非阻塞 job；FS manager默认16 read-priority+16 write-priority线程，可跨 job并发并相互 stealing。单个 batch内部当前 C/Python路径仍按文件顺序读写，不应误称 io_uring 聚合。[manager](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/tiering/fs/manager.py#L86-L202)、[C I/O](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/csrc/fs_io.cpp#L264-L337) | 一次 backend call 接受一组 keys 和每 key 的多地址/多 size；独立 sending/receiving thread；`load_async` 可选。[backend](https://github.com/vllm-project/vllm-ascend/blob/b6ec3943ff5e73359a5c29c1b29fe17672ca45e3/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py#L189-L266) |
| io_uring / direct I/O | uring=0，buffered；未验证物理冷读 | Mooncake backend由 Mooncake开关决定；SGLang 的 `hicache-io-backend=kernel/direct` 指 device↔host，不等于 SSD io_uring。NIXL POSIX backend另有 direct-I/O路径。[NIXL源码](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py#L339-L408) | persistent FS先 probe O_DIRECT，支持则优先 direct，失败回退 buffered；使用 `readv` 直接写 shared CPU memoryview，不依赖 io_uring。[manager](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/tiering/fs/manager.py#L163-L202)、[I/O](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/tiering/fs/io.py#L41-L66) | Mooncake backend由同一 Mooncake SSD配置决定；框架本身没有另造 SSD io_uring格式 |
| CPU staging / pinned | xLLM Host KV注册给 TE；owner仍先 SSD→client buffer，再 TE→xLLM Host | Mooncake Host allocator/registered pool + multi-buffer；page-first/NIXL可走 zero-copy，其他布局使用预注册 bounce buffer。[Mooncake](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L35-L67)、[NIXL](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py#L339-L408) | persistent tier以 CPU primary shared memory为一级；secondary必须经 CPU。Simple disk额外提供 aligned pinned double buffer，交错 NVMe和GPU DMA。[tier contract](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/tiering/base.py#L117-L209)、[simple disk](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/simple_kv_offload/disk_backend.py#L32-L180) | 注册 NPU KV地址给 TE；Mooncake multi-buffer可直接覆盖多个层地址。是否 pin/统一地址依赖 Ascend硬件和 fabric-memory配置。[源码](https://github.com/vllm-project/vllm-ascend/blob/b6ec3943ff5e73359a5c29c1b29fe17672ca45e3/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py#L118-L177) |
| D2H/H2D、预取、并发 | H2D layer-wise async，但在完整 L3 barrier后才开始；每 worker chunk串行；TCP不复用 | 独立 prefetch/backup线程和队列；按128-page批次逐步更新完成 token；H2D提供 per-layer event，可让计算按层等待。[controller](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/managers/cache_controller.py#L408-L430)、[batch progress](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/managers/cache_controller.py#L997-L1079)、[layer events](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/managers/cache_controller.py#L50-L93) | secondary load/store非阻塞，完成由scheduler poll；promotion先 secondary→CPU，再CPU→GPU；Simple disk明确双缓冲重叠磁盘与DMA。[tier manager](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/tiering/manager.py#L253-L329)、[simple disk](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/simple_kv_offload/disk_backend.py#L289-L340) | non-layerwise有独立收发thread；`load_async` 默认 false，必须显式开启。官方文档当前把 layerwise限定在 Prefill+Memcache backend，不能把源码里的layerwise类当作 Mooncake SSD正式支持。[官方 KV Pool 文档](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/kv_pool.html) |
| promotion / eviction | Mooncake能力存在；当前 replay未晋升到memory | HiRadix/Host pool维护可逐出状态，storage prefetch有 threshold（默认256 tokens）和 best-effort/wait/timeout政策；write-through/write-back可选。[源码](https://github.com/sgl-project/sglang/blob/0db2c53deca12715d209ffbd46bdfafdfd5a5d35/python/sglang/srt/mem_cache/hiradix_cache.py#L1189-L1263) | secondary命中异步 promotion到CPU primary，再GPU；CPU primary支持LRU/ARC策略，secondary支持cascade。[tier manager](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/kv_offload/tiering/manager.py#L349-L486) | reusable prefix cache；持久层 promotion/eviction最终仍取决于Mooncake配置和AscendStore调度 |

### 4.1 不能混淆的三条路径

1. **SGLang/vLLM-Ascend + Mooncake 持久 SSD**与 xLLM 使用的是同一类 Store/SSD backend。它们的框架编排值得参考，但若同样设置 `bucket_keys_limit=1`、关闭 direct I/O、关闭 TCP pool，也会继承相同底层代价。
2. **vLLM persistent FS tier**是持久、内容寻址、可跨实例共享的 secondary tier，最值得参考的是 chunk、目录分片、O_DIRECT probe、CPU promotion和非阻塞 job；它不是 Mooncake RPC 路径，而且当前平台支持不能直接外推到 MLU。
3. **vLLM Simple disk backend**使用每 rank 单个预分配 slot 文件、O_DIRECT、`preadv/pwritev`、pinned aligned double buffer，适合作为本地 NVMe swap 性能上限设计参考；但文件随进程生命周期清理，不能替代可重启恢复/跨实例复用的持久 L3。[官方源码](https://github.com/vllm-project/vllm/blob/36bad1b90cb2958b6de4abdb5a9249b136f39720/vllm/v1/simple_kv_offload/disk_backend.py)

## 5. 优化建议与优先级

### P0：先做配置/无侵入 A/B，确认剩余上限

1. **恢复 bucket 聚合。** 先测 `MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT=64/128/500`，size limit保持256 MB或明确设置；核对 offload flush/可见性延迟、文件数、恢复时间和 eviction 粒度。`1` 只适合故障定位，不适合性能结论。
2. **打开真实 direct I/O。** 确认 Mooncake 编译含 `USE_URING` 后设置 `MOONCAKE_OFFLOAD_USE_URING=1`；检查 `RLIMIT_MEMLOCK`、4096对齐和实际 fd flag。用块设备 read counters 或 `iostat` 验证物理读，不能只看应用字节数。Mooncake 官方设计是 thread-local ring、queue depth 32、fixed registered buffer和可选 O_DIRECT。[官方 SSD design](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/store/ssd-offload.md)
3. **开启并审计 TCP connection pool。** 设置 `MC_TCP_ENABLE_CONNECTION_POOL=1`，warm-up后再量；预期首先消除约102,400次握手与TIME_WAIT。必须同时监控 `ESTAB`、fd数、重连失败和pool锁等待，因为Mooncake官方已记录当前实现的idle fd回收/锁风险。[官方 troubleshooting](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/troubleshooting/troubleshooting.md)
4. **增大 xLLM prefetch batch。** 先从8→64，再测128。4K每rank BRPC chunks将从32降到4或2；target/draft仍会翻倍，但固定RPC/stream次数先降8–16倍。监控单batch owner buffer、GC lease、BRPC idle timeout和尾延迟。
5. **分离 cold/warm SSD口径。** 每组都报告：(a) direct physical cold；(b) page-cache warm；(c) connection cold/warm。当前结果不可继续只写“SSD TTFT”。

### P1：代码改造

1. **消除第二次 `BatchQuery`。** 给 Mooncake multi-buffer get 增加接收已选择 `QueryResult`/replica的内部API，或把local/regular选择下沉到一次调用；目标是正式 master query requests从12,800降到6,400，items从102,400降到51,200。
2. **真正的 endpoint batch。** xLLM一次把64/128 blocks交给worker；Mooncake应对同一endpoint用一个owner RPC、一个lease、一个TransferEngine task list处理，而不是框架逐8 blocks调用。target/draft若内存独立，可并发或合并发起，最终按两者共同ready发布。
3. **L3→Host→MLU双缓冲流水。** 每个batch独立ready bitmap/event：batch N一旦SSD→Host完成便发H2D；同时batch N+1做metadata/RPC/SSD。进一步按layer readiness让forward只等待当前层。不能继续用“所有TP/所有blocks完成”作为唯一publish barrier。
4. **审慎评估 super-chunk。** 将8/16个连续hash block组成持久对象，可再减少metadata、文件和lease数量；但需设计partial prefix lookup、尾块、eviction和版本兼容。优先做API batch，确认固定成本下降后再决定是否改变持久对象语义。
5. **评估 MLA TP副本去重。** SGLang把MLA视为rank-replicated并仅rank0 backup。xLLM若能证明MLU MLA各TP rank的Host内容字节等价，可改为单持久副本+rank内broadcast/共享读取；在此之前不能直接去掉key中的`tp_rank`，否则存在正确性风险。

### P2：策略与长期结构

1. 对重复访问prefix启用/调优 L3→L2 promotion，配合 LRU/FIFO和admission threshold，避免每轮都读SSD。
2. 为metadata query结果增加短TTL cache/lease复用，减少master压力，但必须处理eviction、replica失效和tenant隔离。
3. 两块NVMe当前LVM为linear，不会自动得到两盘条带带宽；若单盘真正饱和，再做bucket/file按盘分片或RAID0。现在只有约1.4 GiB/s，尚未到此优先级。
4. Mooncake `offset_allocator_storage_backend` 可作为“单预分配大文件、高并发小对象”的实验对照，但它启动会truncate且不恢复metadata，不能替代要求重启可恢复的bucket backend。[官方 SSD deployment](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/deployment/ssd/ssd-offload.md)

## 6. 建议验证矩阵

每次只改一个变量，使用相同20个TTFT prompt和同一GSM8K精度集；先warm连接，再分别测physical cold与page-cache warm。

| 实验 | 单变量 | 必采指标 | 成功判据 |
|---|---|---|---|
| A | TCP pool off→on | TTFT、connect/close数、TIME_WAIT/ESTAB、fd、offload call p50/p99 | 端口耗尽消失；offload p50显著下降；无fd持续增长 |
| B | batch 8→64→128 | BRPC streams、master request/items、owner RPC、buffer peak、TTFT | 4K每rank批次数按预期下降；无lease超时 |
| C | bucket keys 1→64/128/500 | objects、`.bucket/.meta`文件数、open/stat、恢复时间、store/load TTFT | 文件数由2×objects降到约2×buckets；精度不变 |
| D | uring/direct off→on | fd flags、盘read sectors、iostat、SSD BatchLoad、TTFT | 物理读字节≈请求字节；结论不依赖page cache |
| E | 去掉重复query | master request/items增量、TTFT、命中正确性 | request/items各减半；replica选择语义不变 |
| F | L3/H2D pipeline | batch/layer ready timeline、MLU copy/compute stream、TTFT | SSD、H2D、compute出现可量化重叠；无未ready层被消费 |
| G | promotion on-hit | SSD reads、memory hits、L2占用、eviction、重复轮TTFT | 第2次起SSD read下降，且无热点污染/容量失控 |

正确性门槛：GSM8K结果与cold基线一致；20/20 L3 hit；target/draft必须同时命中才发布；取消/timeout后不得泄漏Host block、lease或连接。性能报告同时给p50/p90/p99，不能只给均值。

## 7. 最终判断

**[源码 + 指标支持的强结论]** 当前 L3 慢于 cold 的第一责任不是“本地SSD太慢”，而是 51,200 个对象被组织成 6,400 个小而串行的端到端batch，每batch又承担两次metadata query、owner RPC、两个slice的短TCP建连，并且全部完成后才开始H2D。32.5 ms × 每rank batch数已经能解释绝大多数TTFT；1.895 ms的SSD `BatchLoad`和6.5 GiB/s单盘能力都表明物理介质尚有明显余量。

**[待A/B验证的因果排序]** 预计收益顺序为：TCP复用与batch放大 > 重复query删除 > bucket聚合/direct-I/O正确化 > L3/H2D流水 > 对象super-chunk/TP副本去重 > 多盘分片。前四项应先完成，之后再和SGLang/vLLM/vLLM-Ascend做同模型、同缓存字节、同physical-cold口径的公平benchmark。
