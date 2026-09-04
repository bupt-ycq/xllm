# xLLM KV Cache Offload / Store 对照 SGLang 缺陷审计

审计日期：2026-08-25

- xLLM：当前 `/workspace/xllm` 工作树，HEAD `e33b6b52004040d54cdbe3f33f3ae1fe06dbad2f`；相关文件存在未提交修改，因此以下本地行号按审计时工作树定位。
- SGLang：官方 `sgl-project/sglang` `main`，固定 commit [`6a810383176a4e2a5b88f4a28ccf42cce2c1c538`](https://github.com/sgl-project/sglang/commit/6a810383176a4e2a5b88f4a28ccf42cce2c1c538)。
- 证据：仅使用当前 xLLM 源码/测试和 SGLang 官方源码/文档/测试。本文是静态源码审计，不把未运行的场景写成已复现故障。

## 结论

当前 xLLM 已经具备 typed Host pool、TP 结果合并、block-granular Store hit、分层 H2D、取消后延迟释放等重要能力；与 SGLang 的主要差距不再是“有没有 Host cache”，而是以下四类契约还没有闭环：

1. cache invalidation 和超时后的 buffer ownership 仍存在可导致脏 KV 被复用/覆盖的 P0 风险；
2. L3 prefetch 仍是“先占满 Host、再读 Store”，而不是 query-first、按真实命中精确 staging；
3. D2H、L3 write、L3 read 共享较粗的同步/无界执行路径，缺少水位、credit 和 worker 健康管理；
4. backend、metrics、运行时生命周期和策略可配置性明显落后于 SGLang。

优先级总表：

| 优先级 | Finding | 直接后果 |
|---|---|---|
| P0 | 1. 权重更新/缓存重置没有覆盖 Host 与 L3 namespace | 更新权重后可能复用旧权重生成的 KV |
| P0 | 2. prefetch 超时会释放仍可能被 worker 写入的 Host slot | 迟到的 Store get 可能覆盖已复用槽位 |
| P1 | 3. Store prefetch 在 query 前预分配 Host | 长 prompt 制造假 Host 压力、降低有效命中 |
| P1 | 4. load/write 没有独立水位和 byte/job credit | 写入或预取可挤占 Host，队列和尾延迟不可控 |
| P1 | 5. L3 write 串在 D2H RPC 完成路径上 | 慢 Store 延迟 Host 发布和 HBM block 释放 |
| P1 | 6. 只有固定的 request-end write-through | 写放大；不能选择 selective/write-back/buffer-only |
| P1 | 7. backend exception/retry/health 契约缺失 | 异常可越过 worker 边界；无稳态自愈 |
| P1 | 8. 非 MLU prefetch 仍是整 worker 原子完成 | NPU 等平台不能渐进发布连续成功前缀 |
| P2 | 9. reset、attach/detach 等生命周期不完整 | 运维只能重启；reset 语义不一致 |
| P2 | 10. 可观测性只新增 Host used/total | 无法解释 I/O 堵塞、失败、丢弃和重试 |
| P2 | 11. backend/config/capacity 模型仍硬编码 | 难扩后端，且容易配置出无效的大 Host pool |
| P2 | 12. cancelled H2D 与 shutdown 缺少有界回收 | synchronizer 累积或进程退出永久等待 |

## P0

### 1. 权重更新和 prefix reset 没有形成 Host/L3 的版本失效协议

**xLLM 证据**

- `LLMMaster::update_weights()` 直接转发到 engine：`xllm/core/distributed_runtime/llm_master.cpp:679-681`；engine 等所有 worker 更新成功后直接返回，没有调用 scheduler cache reset：`xllm/core/distributed_runtime/llm_engine.cpp:1532-1555`。
- worker 原地载入新权重后直接返回：`xllm/core/runtime/worker_impl.cpp:1686-1700`。
- 即使显式调用 `reset_prefix_cache()`，基类只遍历 device `block_managers_`：`xllm/core/framework/block/block_manager_pool.cpp:333-340`；`HierarchyBlockManagerPool` 没有 override，独立的 `host_block_managers_` 不在该循环内：`xllm/core/framework/block/hierarchy_block_manager_pool.h:45-78,129-140`。
- L3 key 包含 model id、TP、block type/rank、schema 和 token hash，但没有 weight version/generation：`xllm/core/framework/kv_cache_transfer/kv_cache_store.cpp:125-140`。

同一服务进程原地更新权重后，device、Host 和 Mooncake 中旧权重生成的 KV 都可能继续由相同 token hash 命中。仅清 device 也不够，因为下一请求仍可从 Host/L3 恢复旧 KV。

**SGLang 对照**

SGLang 的 weight updater 有显式 `flush_cache_after_weight_update` hook，并在各 update path 成功后调用；同时记录 weight version，见 [`SchedulerWeightUpdaterManager`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/managers/scheduler_components/weight_updater.py#L75-L185)。其 Radix key 还支持 `cache_salt` 隔离，见 [`RadixKey`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/mem_cache/radix_cache.py#L62-L80)。不过 SGLang 的跨模型/跨版本外部 Store key 隔离本身仍有[官方已知问题](https://github.com/sgl-project/sglang/issues/24758)，所以应借鉴其本地 flush/version hook，但不能原样照抄现有 L3 namespace。

**修复要求**

- 更新权重前 quiesce，成功后原子推进 cache generation；
- `HierarchyBlockManagerPool::reset_prefix_cache()` 同时 reset device 与 Host，并取消/等待在途 D2H/H2D/prefetch；
- Store key 加 weight version/cache generation，避免依赖远端物理删除；
- 测试“同 token、换权重后绝不命中旧 Host/L3”。

### 2. prefetch 超时没有保留 in-flight Host slot 的所有权，存在迟到写污染

**xLLM 证据**

- 默认 `prefetch_timeout=0`，并被转换成 BRPC `idle_timeout_ms=-1`，所以一个永不返回的 Store get 可以永久占住 admission slot：`xllm/core/framework/config/kv_cache_store_config.cpp:21-28`，`xllm/core/distributed_runtime/llm_engine.cpp:847-870`，`xllm/core/scheduler/continuous_scheduler.cpp:179-215,234-270`。
- 把 timeout 设成非零也没有真正取消 worker 上的 Mooncake `batch_get`：worker 已经在线程池里执行 `prefetch_kv_blocks()`，客户端 idle timeout 只关闭 stream：`xllm/core/distributed_runtime/worker_service.cpp:592-646`，`xllm/core/distributed_runtime/comm_channel.cpp:557-630`。
- stream 关闭后客户端立即把该 worker 标成 failed/completed；所有 peer 收敛后，master 会执行 `release_prefetch_plan()`，把 Host block 交还 allocator：`xllm/core/framework/kv_cache_transfer/prefetch_result.h:188-214`，`xllm/core/framework/block/hierarchy_block_manager_pool.cpp:986-1024,1292-1303`。
- 但超时 worker 上的 backend 调用仍可能稍后把数据写进原 dst block id；此时该 Host slot 可能已经分给另一个请求或 prefix。现有 `PrefetchH2DReservation` 只保护 MLU 后续 H2D window，不持有 worker 侧 G2H Store get 的 Host owner。

因此当前配置形成两难：timeout 为 0 有永久阻塞风险；timeout 非 0 则在慢/卡住的 backend 上存在 use-after-recycle 式迟到写风险。这是根据异步控制流和所有权生命周期得到的静态结论，需要用可阻塞 fake Store 做确定性回归测试。

**SGLang 对照**

SGLang 默认使用 token-scaled bounded timeout；请求可以按策略继续，但 ongoing prefetch、Host ref 和 storage op 会继续由 controller/ack 路径持有并在真实完成后回收，而不是因 scheduler 放弃等待就立即复用目标页，见 [`HiRadixCache` timeout/progress handling](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/mem_cache/hiradix_cache.py#L1581-L1692) 和 [`PrefetchTimeoutConfig`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/mem_cache/hicache_storage.py#L49-L55)。

**修复要求**

- Store get job 必须持有明确的 Host reservation/epoch，直到 backend 真正 ack；timeout 只能停止等待/发布，不能释放 in-flight dst；
- backend 若不可取消，目标页进入 quarantine，真实完成后检查 generation 再丢弃或发布；
- 默认采用有上限的 token-scaled timeout，同时区分 scheduler deadline 与 backend operation deadline；
- 测试“backend 超时后 slot 被请求复用，迟到完成不得改写/发布新 owner”。

## P1

### 3. L3 prefetch 顺序相反：xLLM 先占 Host，再调用 Store get

**xLLM 证据**

`prefetch_from_storage()` 对 prompt 的每个缺失 full block，先按 `free + prefix-cache-evictable` 估算容量，再实际分配 Host block；只有成功分配的 block 才组成 Store get 请求：`xllm/core/framework/block/hierarchy_block_manager_pool.cpp:831-887,913-935`。

这会带来两个问题：

1. 长 prompt 在还不知道 L3 是否命中前就占用大量 pinned Host page；
2. Host 紧张时，后部 key 根本不会被 query，无法先知道真实 longest-prefix hit 再决定 staging；多个并发请求也会互相制造假压力。

此外，xLLM 会为全部 missing full blocks 建 query，并等待整个 operation 完成；即使平坦 KV 前缀在靠前 block 已 miss，worker 仍会继续拉取后续 batch：`xllm/core/framework/block/hierarchy_block_manager_pool.cpp:890-935,1306-1345`，`xllm/core/distributed_runtime/comm_channel.cpp:565-630`。对当前请求而言，首个 miss 之后的离散 hit 不能增加可复用的连续前缀，却仍增加 Store I/O 和 admission 等待。

**SGLang 对照**

SGLang 是 query -> 跨 rank 合并 hit -> scheduler 精确分配 Host -> get。Host 不足时先 evict，再缩短为可容纳的 page-aligned prefix，最后撤销 prefetch 并重算；见 [`_drain_and_alloc_storage_hit`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/mem_cache/hiradix_cache.py#L593-L665) 与 [`prefetch_from_storage`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/mem_cache/hiradix_cache.py#L1775-L1817)。

**修复要求**

把 `batch_exists` 提升为正式接口，先获得各 pool 的共同连续前缀，再在 scheduler 线程为真实 hit 精确分配 Host；分配不足时缩短而非按整 prompt 预占。平坦 KV 在第一个 miss batch 处停止后续 query/get；SWA 等非平坦 pool 则按各自 coverage policy 处理，不能一刀切。

### 4. Host load/write 没有独立水位，D2H queue 也没有 byte/job credit

**xLLM 证据**

- active sequence 的 Host growth 会 best-effort 一直追到 `num_tokens`：`xllm/core/framework/block/hierarchy_block_manager_pool.cpp:548-587`。
- Store prefetch 直接使用 `free + prefix cache` 可用量：同文件 `:851-867`；没有为 D2H write 留固定 reserve。
- D2H 用无显式容量上限的 `BlockingConcurrentQueue`，每轮会将当前队列全部 drain 成一次 transfer：`xllm/core/framework/block/hierarchy_block_manager_pool.h:47-48,134-149`，`xllm/core/framework/block/hierarchy_block_manager_pool.cpp:1402-1437`。
- scheduler 的 `prefetch_admission_limit_` 只等于 request queue 大小，不是 Host bytes、in-flight pages 或 Store latency credit：`xllm/core/scheduler/continuous_scheduler.cpp:100-105,179-214`。

因此 load 可以吃掉 write staging，write backlog 也没有按 bytes/jobs/age 的界限；压力下只能靠 allocation miss 被动退化。

**SGLang 对照**

普通 cache 模式限制 speculative prefetch 最多占 Host pool 约 50%；buffer-only 模式 load 优先，并以 prefetch 占用、10% reserve、write live cap 和 backlog cap 控制写入，见 [`cache_controller.py`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/managers/cache_controller.py#L478-L541) 与 [`BufferModePipeline.flush_pending_writes`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/mem_cache/buffer_mode/pipeline.py#L406-L515)。

**修复要求**

至少引入 `prefetch_reserved_pages`、`write_reserved_pages`、queued/in-flight bytes/jobs、最大 oldest age；load 优先，但不得让 write 永久饥饿。

### 5. L3 write 同步串在 D2H RPC 上，阻塞 Host 发布和 HBM 释放

**xLLM 证据**

worker 先同步 D2H，随后在同一个 `offload()` 调用中同步 `batch_put()`，最后才返回完整 block 数：`xllm/core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.cpp:604-630,633-660`。master 侧只有 worker future 全部返回后才释放 device block并发布 Host prefix：`xllm/core/framework/block/hierarchy_block_manager_pool.cpp:1434-1468`。

所以 Mooncake 慢写会延长两侧 block 同时 pinned 的时间，直接推高 HBM/Host 瞬时压力。Store put 部分失败虽然只影响 L3 durability，但当前控制流仍等待它完成。

**SGLang 对照**

SGLang 把 D2H ack、Host publish、L3 backup 分成不同阶段；D2H ack 后 Host copy 可先成为有效 L2，L3 write 由后台 queue 批处理，见 [`write_backup`/D2H ack](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/mem_cache/hiradix_cache.py#L845-L946) 和 [`_storage_backup_thread_func`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/managers/cache_controller.py#L1165-L1302)。

**修复要求**

将状态拆成 `D2H_IN_FLIGHT -> HOST_READY -> STORE_IN_FLIGHT -> STORE_READY/FAILED`。D2H 成功即可释放 device transfer pin、发布 Host；L3 durability 用独立 job/ack/metric 表示。

### 6. 固定 request-end write-through，缺少策略层和 active eviction 触发

**xLLM 证据**

D2H pair 只在 `deallocate()` 时由 `collect_offload_pairs()` 收集完整 block：`xllm/core/framework/block/hierarchy_block_manager_pool.cpp:416-479`。Store 开启后，每次成功 D2H 都直接 `batch_put`：`xllm/core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.cpp:619-630`。配置中没有 write policy：`xllm/core/framework/config/kv_cache_store_config.h:41-78`。

这意味着：

- 一次性 prompt 也会写 Host/L3，容易产生写放大；
- 不能按二次命中再写、不能在 device eviction 时 write-back；
- 它不是 active-sequence swap，不能让单个活跃长上下文突破 device KV 工作集上限。

**SGLang 对照**

SGLang 暴露 `write_through`、`write_through_selective`、`write_back`，并有 Host 仅作 bounce buffer 的 `buffer_only` 模式，见 [官方 HiCache write policies](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/docs/docs/advanced_features/hicache_design.mdx#L74-L84) 和 [`buffer_mode/pipeline.py`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/mem_cache/buffer_mode/pipeline.py#L1-L39)。

**修复要求**

策略应位于 block publication/eviction 层，而不是 DMA/Store 内部。先实现 `write_through` 与 `write_through_selective`；是否需要 active swap 应单独设计，不能把 prefix cache offload 宣称为长上下文扩容。

### 7. backend exception、retry 和 worker health 没有稳态契约

**xLLM 证据**

- Mooncake client 的 `batchIsExist`、`batch_put_from_multi_buffers`、`batch_get_into_multi_buffers` 直接调用 backend，没有 exception translation、retry/backoff 或 circuit breaker：`xllm/core/framework/kv_cache_transfer/mooncake_store_client.cpp:100-143`。
- Store `batch_put()` 已返回逐对象成功数，但 `offload()` 只记 warning，仍返回请求的完整 block 数；master 因而无法区分 `HOST_READY` 与 `STORE_READY`，也无法对失败对象 retry：`xllm/core/framework/kv_cache_transfer/kv_cache_store.cpp:143-183`，`xllm/core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.cpp:619-630`。
- D2H RPC handler 直接调用 `worker_->transfer_kv_blocks()`，没有异常边界：`xllm/core/distributed_runtime/worker_service.cpp:578-589`。
- prefetch worker lambda 同样没有 try/catch；异常越过 `ThreadPool::internal_loop()` 的 `runnable()`：`xllm/core/distributed_runtime/worker_service.cpp:615-646`，`xllm/core/util/threadpool.cpp:154-166`。
- `prefetch_timeout` 是 stream idle/停止发新 batch，并不取消已经阻塞在 backend 的 C++ 调用：`xllm/core/distributed_runtime/comm_channel.cpp:565-630`。

这是源码控制流推断：若 backend 向外抛异常，普通 `std::thread` 中未捕获异常可能终止进程；即使 backend 只返回失败码，也没有统一 retry、熔断和 worker unhealthy 指标。

**SGLang 对照**

SGLang 也没有完整的稳态 retry/circuit breaker，且 daemon storage worker 对未捕获 backend exception 同样存在退出风险；Mooncake 仅 warmup 有有限重试，见 [`mooncake_store.py`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L654-L678)。因此这一项应**超过** SGLang，而不是照抄其不足。

**修复要求**

所有 backend 入口统一返回 typed status；worker 最外层捕获异常并把 cache op 降级为 miss/recompute；D2H ack 与 L3 per-object status 分开；增加 deadline、可重试分类、有限退避、熔断和 health state。

### 8. 渐进式 durable prefetch/H2D 目前只在 MLU 编译启用

**xLLM 证据**

`LLMEngine::prefetch_from_storage()` 只有 `USE_MLU` 使用 `DURABLE_BATCH`，其他平台使用 `LEGACY_WORKER_ATOMIC`：`xllm/core/distributed_runtime/llm_engine.cpp:847-870`。相应的 ready-prefix H2D reservation/window 也被 `#if defined(USE_MLU)` 包围：`xllm/core/framework/block/hierarchy_block_manager_pool.h:96-99,111-125`，`xllm/core/framework/block/hierarchy_block_manager_pool.cpp:976-1013,1027-1290`。

非 MLU 平台必须等每个 worker 的整个 prefetch operation 完成后，才能 merge hits 并发布 Host；长请求不能按已完成的连续 batch 渐进恢复，慢 rank 会扩大 tail latency 和 Host 持有时间。

即使在 MLU 路径，默认 `prefetch_batch_size=2`，每个 worker 对 batch 逐个创建 BRPC stream，并同步等待关闭后才进入下一批：`xllm/core/framework/config/kv_cache_store_config.cpp:26-28`，`xllm/core/distributed_runtime/comm_channel.cpp:565-623`。若 block size 为 128，43,008-token prompt 约有 336 个 block，默认会形成约 168 个串行 stream 往返/worker；这会显著放大 TTFT 和 head-of-line blocking。SGLang 的 storage batch 常量是 128 pages，见 [`hicache_storage.py`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/mem_cache/hicache_storage.py#L22-L23)。这里的数量是按 block size=128 的示例，不是所有模型的固定值。

**SGLang 对照**

SGLang storage pipeline 逐 page/batch推进 `completed_tokens`，scheduler 只消费所有 ranks 共同 ready 的连续前缀；见 [`load_storage_page`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/managers/cache_controller.py#L970-L1081) 与 [rank-min ack/prefetch synchronization](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/mem_cache/hiradix_cache.py#L998-L1031)。

**修复要求**

把 `PrefetchResult::DURABLE_BATCH` 和 reservation 状态机做成 backend-neutral 控制面；平台只实现实际 H2D completion event，不应决定是否具备渐进 commit 语义。把 Store page batch 与 H2D layer window 分成两个参数，并用基准确定默认值，避免每两个 page 建一个同步 RPC stream。

## P2

### 9. reset 与 runtime attach/detach 生命周期不完整

除 P0 finding 2 的 Host reset 缺失外，当前 `KVCacheStorage` 只有 init/get/put，没有 `close/health/attach/detach/clear generation`：`xllm/core/framework/kv_cache_transfer/kv_cache_store.h:45-68`。`HierarchyKVCacheTransfer` 仅在构造/析构时创建和销毁 Store：`xllm/core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.cpp:276-323`。切换后端或配置只能重启服务，也没有“必须 fully idle”的显式控制面。

SGLang attach/detach 会检查完全 idle、启动/停止 worker、失败回滚，并拒绝在线程仍存活时释放 backend；见 [`cache_controller.py`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/managers/cache_controller.py#L408-L635) 和 [官方 runtime 文档](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/docs/docs/advanced_features/hicache_storage_runtime_attach_detach.mdx#L37-L68)。不过 SGLang DP partial success 也没有自动回滚，xLLM 实现时应补齐。

### 10. Metrics 无法回答“为什么慢/为什么没命中/为什么丢”

当前新增的标准指标只有 Host used/total token gauges：`xllm/core/common/metrics.cpp:230-233`，由 scheduler 设置：`xllm/core/scheduler/scheduler_policy.cpp:802-808`。D2H/H2D/Store 关键路径主要是 VLOG/WARNING：`xllm/core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.cpp:598-600,623-628`。

缺少至少以下标准指标：

- D2H/H2D tokens、bytes、duration、失败；
- Store query/get/put pages、bytes、bandwidth、partial failure；
- queue depth、queued/in-flight bytes、oldest age；
- prefetch cancel/timeout、write drop reason、Host allocation failure；
- worker alive、backend retry/circuit state。

SGLang 已覆盖 L2 load/backup/evict/drop 与 L3 prefetch/backup/bandwidth，见 [`StorageMetricsCollector`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/observability/metrics_collector.py#L1848-L1978) 和 [`RadixCacheMetricsCollector`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/observability/metrics_collector.py#L1995-L2192)。SGLang 自身仍缺统一 queue age、worker health 和 retry 指标，xLLM 应一次补齐。

### 11. backend/config/capacity 仍硬编码，缺少有效组合校验

**xLLM 证据**

- `KVCacheStorage` 虽是抽象类，但 production 构造逻辑硬编码 `KVCacheStore`/Mooncake：`xllm/core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.cpp:276-299`；配置字段也全是 Mooncake 专属：`xllm/core/framework/config/kv_cache_store_config.h:41-78`。
- Host capacity 只用 `host_blocks_factor` 表示，不提供 per-rank bytes 上限、系统内存 reserve 或按同 host ranks 均分：同文件 `:62-68`。
- 通用校验已经会拒绝 `enable_prefix_cache=false`、无 DMA provider、量化/不支持 layout 等组合：`xllm/core/framework/kv_cache/kv_cache_utils.cpp:498-587`。但 worker 构造点只局部检查 factor，且 `offload()` 仍保留 `batch_memcpy_ == nullptr => full success` 的危险分支：`xllm/core/runtime/worker_impl.cpp:2291-2339`，`xllm/core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.cpp:604-612`。正常入口目前应不可达，仍应删除并改成 fail-fast，避免新入口绕过中央校验后静默发布未复制数据。
- Store key 只有 `tp_rank/tp_size`，没有显式 CP rank/size：`xllm/core/framework/kv_cache_transfer/kv_cache_store.cpp:125-140`。MLU model-CP 特判会把 CP worker 纳入唯一 rank，但 NPU 的 `tp_size` 会除以 `cp_size`，不同 CP rank 可得到相同 `tp_rank`：`xllm/core/runtime/worker_impl.cpp:2304-2319`。若这些 CP shard 的 KV 内容不同，就存在 key collision 风险；SGLang 的 [`HiCacheStorageConfig`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/mem_cache/hicache_storage.py#L27-L40) 显式携带 `attn_cp_rank/size`。这一项应先用 NPU CP=2 的跨 rank key/内容测试确认，再定为 correctness bug。

SGLang 用 `StorageBackendFactory` 注册/校验 file、Mooncake、HF3FS、NIXL 等后端，见 [`backend_factory.py`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/mem_cache/storage/backend_factory.py#L16-L251)；Host pool 会预留 10 GiB、按同 host rank 分摊预算，并在 PP ranks 同步固定容量，见 [`pool_host/base.py`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/mem_cache/pool_host/base.py#L26-L102)。

建议把 production backend 选择移到 registry/factory；容量同时支持 bytes 与 ratio，并在启动时打印每 rank 实际 bytes/tokens、OS reserve 和 pinned-memory 注册结果。删除 `nullptr => success`；Store namespace 显式编码全部会改变 KV shard 语义的并行维度。

### 12. cancelled H2D synchronizer 和 shutdown wait 缺少有界回收

- H2D 为 batch 注册的 `layer_wise_load_synchronizer_` 只在对应 forward 调用 `set_layer_synchronizer()` 时删除；源码自己也承认“batch cancelled”可留下未消费 entry，但只在相同 `batch_id` 冲突时覆盖：`xllm/core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.cpp:533-563,741-753`。batch id 近似单调递增，所以取消/forward 前失败的不同 batch 可长期积累 synchronizer 和 event。
- `KVTransferTracker` 析构使用无 timeout 的 condition-variable wait；只要 Store/RPC callback 永远不返回，shutdown 就可能永久挂住：`xllm/core/framework/kv_cache_transfer/kv_transfer_completion.cpp:66-116`。

SGLang 的 storage detach 会先唤醒线程、释放 pending op、再做有界 join；线程 10 秒仍不退出就明确失败，见 [`HiCacheController` lifecycle](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/managers/cache_controller.py#L408-L635)。xLLM 应让 batch cancellation 显式 erase/abort synchronizer，并为 shutdown 引入 deadline、诊断 pending job 和 fail-safe 策略；补“transfer 已注册但 forward 未发生”和“backend 永不返回时 shutdown 有界”的测试。

## 已有优势，不应重复重写

以下能力当前 xLLM 已有，应在修复中保留：

- Host allocator/Block 引用能在 D2H callback 前同时 pin source 与 destination；成功后才发布 Host，失败不发布：`xllm/core/framework/block/hierarchy_block_manager_pool.cpp:1402-1474`。
- Store get 返回逐 block bitmap，TP worker 结果取交集：`xllm/core/framework/kv_cache_transfer/kv_cache_store.cpp:194-221`，`xllm/core/framework/kv_cache_transfer/prefetch_result.h:217-240`。
- typed cache 已覆盖 KV/SWA/C4/C128 和 tensor role/schema 隔离：`xllm/core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.cpp:149-228`，`xllm/core/framework/kv_cache_transfer/kv_cache_store.cpp:61-108`。
- MLU 的 `PrefetchH2DReservation` 已建立 reserve、in-flight、ready/failed/cancelled、ack 后发布的正确方向，不应退回“RPC 接受即完成”：`xllm/core/framework/block/prefetch_h2d_reservation.h:32-176`。
- 当前测试已经覆盖 Host allocation failure、TP bitmap 合并、prefetch timeout/cancel、batchless H2D real completion 和 event failure drain；主要入口为 `tests/core/framework/block/hierarchy_block_manager_pool_test.cpp:692-1449`、`tests/core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer_test.cpp:359-730`。

## 建议落地顺序

1. 先修 P0：weight/cache generation 覆盖 device/Host/L3；为 Store get 增加 Host reservation/quarantine，确保 timeout 后迟到写不能落到已复用 slot。
2. 再把 L3 read 改成 query-first、真实 hit 精确 staging，并把 MLU durable-batch 状态机平台无关化；同时修正默认 batch=2 的串行 RPC 放大。
3. 拆开 D2H ack 与 L3 write ack，加入 load/write page/byte credits 和水位。
4. 增加 backend exception boundary、deadline、有限 retry、health state 与完整 metrics。
5. 最后引入 write policy、backend factory 和 runtime attach/detach。

这些改动与“43K prompt 的 admission 误判”是互补关系：前者修当前 scheduler 的瞬时资源分类，本文这些 finding 则保证 offload/store 自身不会继续制造假容量、脏 owner 或无法解释的 I/O 堵塞。

## 跨 rank L3 写入语义

结论：SGLang **没有全局 rank 0 聚合/代理写 L3**。每个 scheduler rank 都持有自己的 radix cache、Host pool、cache controller 与 storage worker；D2H 完成后，本 rank 的 radix cache 把本地 Host indices 交给本地 controller，后者的 backup thread 直接调用本地 backend。对应控制流见 [`Scheduler` 为 tensor-parallel GPU worker 构建本地 cache](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/managers/scheduler.py#L400-L414)、[`Scheduler` 构建本地 `tree_cache`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/managers/scheduler.py#L537-L572)、[`HiRadixCache.write_backup_storage`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/mem_cache/hiradix_cache.py#L909-L946) 和 [`HiCacheController.backup_thread_func`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/managers/cache_controller.py#L1260-L1300)。这条 write path 中没有 gather；Mooncake 最终从该 rank 的 Host pool 取得本地 page pointers/sizes 并写对象，见 [`batch_set_v1`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L1076-L1132)。

| 并行/池语义 | 哪些 rank 真正调用 L3 set | 每个调用写什么 | Mooncake namespace |
|---|---|---|---|
| MHA 主 KV | 每个 PP stage 内的**每个 attention-TP rank** | 本 TP rank 的 `1/tp_size` head shard，且只含该 PP stage 的本地 layer slice | PP>1 时为 `{page_hash}_{tp_rank}_{pp_rank}_{k/v}`；PP=1 时省略 PP rank |
| MLA/DeepSeek-V4 compressed 主 KV | 每个 PP stage 内仅 **attention-TP rank 0**；非零 TP rank 的 backup worker跳过主 KV set | 该 TP group 内完整、相同的复制型 KV，但仍只含本 PP stage 的本地 layer slice | 不带 TP rank；PP>1 时为 `{page_hash}_{pp_rank}_k` |
| hybrid sidecar | 由 sidecar 是否 rank-sharded 决定 | Mamba/KDA state 与 Mooncake MHA draft/draft-SWA 即使主池是 MLA，也由各 TP rank 写各自 shard；复制型 sidecar 仍只由 TP0 写 | MHA 型 sidecar 带 TP/PP suffix；复制型 sidecar 只带 PP suffix |

MHA/MLA 的数据分布不是推断：SGLang 官方设计明确写明 MHA 的每个 TP rank 只有 `1/tp_size` KV，而 MLA 各 TP rank 持有完整相同 KV，因此 MLA 只需一 rank write-back，见 [`hicache_design.mdx`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/docs/docs/advanced_features/hicache_design.mdx#L104-L109)。代码中的优化条件也只检查 `is_mla_model && tp_rank != 0`，**不检查 `pp_rank`**，所以这里的“rank 0”是每个 PP stage（以及 DP-attention replica）内部的 TP0，不是整个 job 的 global rank0，见 [`backup_skip`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/managers/cache_controller.py#L508-L516)。PP stage 写的是本地 layer slice，因为 Host pool 继承本地 device pool 的 `start_layer/end_layer`，见 [`pool_host/base.py`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/mem_cache/pool_host/base.py#L149-L161)；Mooncake 再把 PP rank 编入 MHA/MLA suffix，见 [`MooncakeStore` rank suffix](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L549-L593)。因此 PP 没有代理写：MHA 总体是各 stage 的所有 TP shard 分别写，MLA 总体是各 stage 的 TP0 分别写。

DP 要单独理解。开启 DP attention 时，controller 把 `attn_tp_rank/size` 作为 storage TP 维度，同时记录 `dp_rank`，但 `HiCacheStorageConfig` 根本不携带 DP rank；见 [`_generate_storage_config`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/managers/cache_controller.py#L635-L695) 与 [`HiCacheStorageConfig`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/mem_cache/hicache_storage.py#L26-L40)。所以各 DP replica 独立备份自己实际处理并生成的 page，并共享同一 model/TP/PP 内容寻址 namespace；不是每个 DP rank 都重写每一个逻辑 page，也不存在 DP0 代理。若两个 DP replica 产生同一个 page，Mooncake 的 set 路径会先 `batch_exist`，仅 put 缺失对象，见 [`batch_set_v1`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L1086-L1119)。对 MLA 来说，这意味着**每个 DP replica 内的 attention-TP0**都有资格写同一个共享 key，而非全局只写一次。

最后需纠正 API 名称：普通 Mooncake 主 KV 在该 commit 下不是走 `batch_set` 或 `batch_set_v2`，而是 controller 选择 zero-copy `batch_set_v1`，见 [`page_set_func` 选择](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/managers/cache_controller.py#L547-L560) 和 [`_page_set_zero_copy`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/managers/cache_controller.py#L1181-L1192)；generic backend 才走 `batch_set`。`batch_set_v2` 主要用于 hybrid sidecar `PoolTransfer`，而且其 `should_backup` 会保留上述 rank-sharded 例外，见 [`HybridCacheController._page_backup/should_backup`](https://github.com/sgl-project/sglang/blob/6a810383176a4e2a5b88f4a28ccf42cce2c1c538/python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py#L664-L721)。
