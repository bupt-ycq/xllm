# xLLM、vLLM、SGLang 的 KV Cache Offload / Store 对比

日期：2026-08-24

## 对比范围

- xLLM：当前 `/workspace/xllm` 工作树，基线 commit `2509b8a3f58d64dfe20241d0c0d0268cd45809ee`。相关实现直接按本地工作树核对。
- vLLM：官方 `main` commit `a4d70bef3724edb068c8206804154065acaa4cd4`。
- SGLang：官方 `main` commit `e28b9cf7b038be8a39ecd42a3b0b0229e17e6438`。

这里把“offload”理解为 Device/GPU 到 Host/CPU 的 KV 下沉，把“Store”理解为 Host 之后的文件、对象存储、Mooncake 等更后级介质。vLLM 的 P/D 直传 connector 与持久/共享 Store 路径分开讨论。

## 核心结论

三者的默认或主路径都不是“L2 满后才写 L3”。

- **xLLM**：固定 write-through。完整且可缓存的 HBM block 完成 D2H 后，若 Store 开启，立刻查重并 `BatchPut` 到 Mooncake Store。
- **vLLM 原生 Tiering**：完整 KV chunk 产出后异步写 CPU primary；CPU 写完成后级联到所有 secondary tiers。不是等 CPU tier 满。可选的旧/简化 `SimpleCPUOffloadConnector` 才提供接近 GPU 淘汰时再搬运的 `lazy_offload`。
- **SGLang HiCache**：默认 `write_through`，L1 节点进入备份流程后先 D2H，D2H 完成即排队写 L3；另可选择 `write_through_selective` 或 `write_back`。`write_back` 的触发点是 L1/Device 淘汰，不是 L2 已满。

## 三方差异

| 维度 | xLLM | vLLM | SGLang |
|---|---|---|---|
| 层级模型 | Device HBM → pinned Host → Mooncake Store | GPU → CPU primary；Tiering 可再接多个 secondary tier；另有独立 Store/传输 connectors | 明确的 HiCache L1 Device → L2 Host → L3 Storage |
| L1→L2 触发 | 调度过程中收集已完成的完整 block；请求释放时也补收集 | 默认 OffloadingConnector 对已产出的完整 chunk 建 store job；默认仅 prompt/prefill | 由写策略决定：through/selective 命中阈值触发；back 在 L1 淘汰触发 |
| L2→L3 时机 | D2H 成功后立即 `batch_put` | Tiering 在 CPU store 完成后向全部 secondary tiers 级联 | 默认 D2H ack 后立即排队写 L3；selective 达阈值；back 随 L1 淘汰 |
| 是否等 L2 满 | 否 | 否 | 默认否；write-back 也看 L1 淘汰，不是 L2 满水位 |
| 策略可配置性 | 当前固定 write-through | 原生 single tier 有 `store_threshold`；Tiering 强制向各 secondary tier write-through；Simple alternate 有 lazy 模式 | 最完整：`write_through`、`write_through_selective`、`write_back`；还支持 Host `buffer_only` |
| 淘汰策略 | Device/Host 各自 prefix cache/LRU；Host 压力不决定首次写 Store | CPU primary 默认 LRU，也支持 ARC/custom policy | L1/L2 与 radix tree/cache controller 联动；write-back 下 Host 资源不足可能无法保留待下沉数据 |
| Store 后端 | 当前内建 Mooncake Store | 原生 secondary tier 有 FS、S3-compatible OBJ、NIXL P2P；另有 MooncakeStore、LMCache、HF3FS 等 connectors | HiCache Storage 后端较多，包括 file、Mooncake、HF3FS、NIXL、AIBrix、EIC、MORI、SHM 等 |
| Store 查重/共享 | 内容 hash key；Mooncake put 前批量查存在；支持跨 xLLM 进程/重启共享 | FS/OBJ 使用内容 hash 和运行配置命名，可跨进程共享；MooncakeStore connector 也先查重再 put | L3 查询和传输缺失页；共享能力随 Storage backend 实现 |
| 数据粒度 | 完整 block | 完整 block 或多个 GPU block 组成的 chunk | page/block；L3 以 page 为主要粒度 |
| 恢复路径 | Store → 预分配 Host → 分层 H2D；多 TP 达成一致后发布命中 | secondary → CPU primary → GPU；异步传输未完成时调度等待 | L3 → L2 prefetch → 分层 H2D；支持 best-effort/wait/timeout 策略 |
| 计算/传输重叠 | D2H 异步；H2D 按层配合同步事件与 forward 重叠 | GPU/CPU DMA 异步；store job 延至下一 engine step 启动 | L1/L2 分层搬运与计算重叠；L3 后台批处理 |
| 失败语义 | 所有 TP 的 D2H/RPC 成功才发布 Host prefix；Mooncake 部分 put 失败仅告警，不撤销 Host 成功 | manager 聚合 worker 完成状态；各 secondary/store connector 有各自完成和失败处理 | controller/ack 驱动状态更新；多 rank 对命中长度做一致性归约 |

## 写入时序细节

### xLLM

`HierarchyBlockManagerPool::collect_offload_pairs` 只收集完整、可用于 prefix cache 的 block，并在正常分配推进以及序列释放时调用。调度循环中的 `transfer_blocks` 消费这些任务。`HierarchyKVCacheTransfer::offload` 先执行 D2H；成功后，只要 `enable_kvcache_store` 为真，就直接调用 `KVCacheStorage::batch_put`。`KVCacheStore::batch_put` 先 `batch_is_exist`，仅写缺失对象。

因此，xLLM 当前没有 L2 高低水位、L2 满、Host LRU 淘汰时才触发 L3 的逻辑。Host LRU 只负责容量管理。

值得注意的是，Mooncake `BatchPut` 是 best-effort：部分 Store 写失败会记录 warning，但 `offload` 仍把已成功的 Host copy 视为成功。

### vLLM

默认 `OffloadingConnector` 扩展 prefix cache，在完整 KV chunk 产出时创建异步 store job。worker 将 store 启动推迟到下一个 engine step，使 GPU→CPU copy 在前一轮采样相关传输之后开始，降低对生成关键路径的影响。

使用 `TieringOffloadingSpec` 时，CPU 是唯一能直接与 GPU 交互的 primary tier。primary 写完成后，`TieringOffloadingManager.complete_store` 向所有 secondary tiers 发起 cascade。因此它属于写穿/fan-out，不是“CPU L2 满了再 spill”。secondary 命中时也先提升到 CPU，再从 CPU 恢复 GPU。

需要区分两条旁路：

1. `SimpleCPUOffloadConnector` 可启用 `lazy_offload`，在 GPU block 靠近 free/eviction 队列时下沉，但它不是当前默认 native backend。
2. `MooncakeStoreConnector`、LMCache 等独立 connectors 有自己的直接共享存储语义，不等同于 native Tiering 的 secondary tier。

### SGLang

HiCache 把策略直接暴露为服务参数，默认 `hicache_write_policy=write_through`：新叶节点插入时命中计数达到默认阈值，触发 D2H；D2H ack 后记录 L2，并立即将 L2→L3 写入任务交给后台 storage controller 批处理。

- `write_through`：默认阈值为 1，首次形成可缓存节点即可备份。
- `write_through_selective`：默认阈值为 2，二次访问后再备份，减少一次性前缀污染。
- `write_back`：仅在 L1 节点淘汰时发起备份，先进入 Host，再在 D2H 完成后写 L3。
- `buffer_only`：Host 只作为 L1↔L3 staging buffer，必须配置 L3，且不支持 write-back。

## 对 xLLM 的直接启示

1. 如果目标是降低网络/Store 写放大，可在 block manager 的“是否产生 offload job”处增加 `write_through/selective/write_back` 策略，而不是把策略塞进 DMA/Store 实现。
2. 当前 Host 成功与 Store 成功是两个不同 durability level；应补齐 Store 写成功/失败指标，必要时让 metadata 明确区分 `HOST_READY` 与 `STORE_READY`。
3. xLLM 已有 `KVCacheStorage` 抽象，可以继续把 Mooncake 配置与后端注册解耦，并增加 prefetch budget/timeout 等准入策略。

## 官方与本地一手来源

### xLLM

- `xllm/core/framework/block/hierarchy_block_manager_pool.cpp`
- `xllm/core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.cpp`
- `xllm/core/framework/kv_cache_transfer/kv_cache_store.cpp`
- `docs/src/content/docs/zh/features/global_kvcache.md`

### vLLM

- [KV Offloading Usage Guide](https://github.com/vllm-project/vllm/blob/a4d70bef3724edb068c8206804154065acaa4cd4/docs/features/kv_offloading_usage.md)
- [TieringOffloadingManager](https://github.com/vllm-project/vllm/blob/a4d70bef3724edb068c8206804154065acaa4cd4/vllm/v1/kv_offload/tiering/manager.py)
- [Offloading scheduler](https://github.com/vllm-project/vllm/blob/a4d70bef3724edb068c8206804154065acaa4cd4/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py)
- [Simple offload manager](https://github.com/vllm-project/vllm/blob/a4d70bef3724edb068c8206804154065acaa4cd4/vllm/v1/simple_kv_offload/manager.py)

### SGLang

- [HiCache design](https://github.com/sgl-project/sglang/blob/e28b9cf7b038be8a39ecd42a3b0b0229e17e6438/docs/docs/advanced_features/hicache_design.mdx)
- [HiCache server arguments](https://github.com/sgl-project/sglang/blob/e28b9cf7b038be8a39ecd42a3b0b0229e17e6438/python/sglang/srt/server_args.py)
- [HiRadixCache write path](https://github.com/sgl-project/sglang/blob/e28b9cf7b038be8a39ecd42a3b0b0229e17e6438/python/sglang/srt/mem_cache/hiradix_cache.py)
- [HiCache cache controller](https://github.com/sgl-project/sglang/blob/e28b9cf7b038be8a39ecd42a3b0b0229e17e6438/python/sglang/srt/mem_cache/cache_controller.py)
