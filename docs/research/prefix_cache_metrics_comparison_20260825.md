# xLLM、vLLM-Ascend 与 SGLang Prefix/KV Cache 指标对照

> 调研日期：2026-08-25（Asia/Shanghai）  
> 一手源码快照：vLLM `d3e2888c7588fe3a7be93606c7c626dbfd304d2e`（2026-08-25）、vLLM-Ascend `fa668c6495f21734dfd3b7aab464ac73beb4d28b`（2026-08-25）、SGLang `833be86c15044b312c09b5e3b0ceb73248049ce3`（2026-08-24）。  
> xLLM 口径来自 `/workspace/xllm` 当前工作树；其中本次新增命中率代码尚未提交。本文只调研和比较指标，没有修改生产代码或测试。

## 结论先行

`prefix_cache_requests_total` 与 `prefix_cache_hit_requests_total` **不参与 Token 命中率**。它们是一对请求级 Counter，用来计算：

```promql
请求命中率 = rate(prefix_cache_hit_requests_total[5m])
           / rate(prefix_cache_requests_total[5m])
```

这里“请求命中”指该请求至少有 1 个 Prompt Token 被缓存复用。它回答“多少请求得到了任何缓存收益”，而 Token 命中率回答“总 Prompt Token 中有多少免于计算”。两者不能互相替代：100 个请求各命中 1/1000 Token 时，请求命中率是 100%，Token 命中率只有 0.1%；一个超长请求大量命中、其余短请求全未命中时则可能相反。

对照结论如下：

1. vLLM/vLLM-Ascend 的标准做法是导出 Token 查询数和 Token 命中数 Counter，由 PromQL 按任意时间窗口计算命中率；官方设计明确反对把命中率只做成 Gauge。
2. SGLang 同时有最新 Prefill 报告周期的 `sglang:cache_hit_rate` Gauge，以及更适合聚合、可按 device/host/storage 分层的 `sglang:prefill_effective_tokens_total{mode=...}` Counter。
3. vLLM 和 SGLang 都没有“至少命中一个 Token 的请求数”这一专用 Counter。因此，xLLM 的两个请求 Counter 是**额外能力，不是行业对齐所必需**。若需求只要求 Token 命中率，可以删除；若确实需要请求覆盖面/SLO，则应保留这一对 Counter。
4. xLLM 新增的两个百分比 Gauge 是冗余的，而且不适合多实例聚合；Counter 应作为主接口。建议删除 Gauge，或者至少把它们明确标成“单进程启动以来的 lifetime 比率”，不可做集群平均。
5. xLLM 当前最明显的缺项不是请求 Counter，而是：实际 cache lookup 的 query Token Counter、local/host/external 分层命中数、Host/外部缓存容量，以及 PD 角色/实例维度。

## 1. xLLM 当前工作树

### 1.1 已有及新增指标

| 指标 | 类型 | 口径 | 备注 |
|---|---|---|---|
| `prefix_cache_match_length_total` | `bvar::Adder<double>` Counter | 每次底层 Prefix Cache 匹配返回的 Block 数乘 Block Size 后累加 | 低层 lookup 口径，重试/重新调度可能再次累加；不是去重后的请求口径。位置：`xllm/core/framework/block/block_manager_impl.cpp:160-176`。 |
| `prefix_cache_block_matched_num` | `bvar::LatencyRecorder` | 每次 lookup 命中的 Block 数分布 | 位置：`xllm/core/framework/prefix_cache/prefix_cache.cpp:96-101`；Linear State 路径也在 `linear_state_prefix_cache.cpp:99-110` 更新。 |
| `prefix_cache_block_matched_rate` | `bvar::LatencyRecorder` | 每次 lookup 的整数百分比 `matched_blocks / queried_full_blocks * 100` | 请求级非加权分布，不是全局 Token 比率。 |
| `num_blocks_in_prefix_cache` | `bvar::Status<double>` Gauge | 当前 Prefix Cache 中的 Block 数 | 通用 Scheduler 更新，见 `xllm/core/scheduler/scheduler_policy.cpp:774-788`。 |
| `kv_cache_utilization_perc` | Gauge | `num_used_blocks / num_total_blocks` | 名字含 `perc`，但实现返回 0..1；见 `block_manager_impl.h:84-87`。 |
| `num_free_blocks` / `num_used_blocks` | Gauge | 分配器空闲 Block / 有效在用 Block | 容量诊断。 |
| `prefix_cache_prompt_tokens_total` | Counter（新增） | 已记录 Prefill 请求的 Prompt Token 总数；零 Prompt 请求排除 | 它是工作负载分母，不等于“实际执行了 Prefix Cache 查询的 Token 数”。 |
| `prefix_cache_hit_tokens_total` | Counter（新增） | 请求级去重后，Prefix/Host Cache 复用的 Prompt Token 总数，且 clamp 到 Prompt 长度 | `Sequence::num_prefix_cache_tokens()` 取 device/host shared token 数的最大值，见 `request/sequence.cpp:838-842`。 |
| `prefix_cache_requests_total` | Counter（新增） | 被记录且 Prompt 非空的 Prefill 请求数 | 请求命中率分母。 |
| `prefix_cache_hit_requests_total` | Counter（新增） | 上述请求中 `cached_tokens > 0` 的请求数 | 请求命中率分子。 |
| `prefix_cache_token_hit_rate_perc` | Gauge（新增） | 进程启动以来 `hit_tokens_total / prompt_tokens_total * 100` | Counter 可完全推导；不适合跨实例直接平均。 |
| `prefix_cache_request_hit_rate_perc` | Gauge（新增） | 进程启动以来 `hit_requests_total / requests_total * 100` | 同上。 |

新增 Counter/Gauge 的定义与更新集中在 `xllm/core/common/metrics.cpp:58-115`，请求级“一次且仅一次”门控在 `xllm/core/framework/request/request.cpp:205-220`。

### 1.2 NPU 路径

xLLM **没有 NPU 专属 Prefix Cache 命中指标**。对 `xllm/core/layers/npu/`、`xllm/core/layers/npu_torch/`、`xllm/core/kernels/npu/`、`xllm/core/platform/npu/` 的扫描未发现命中指标注册或写入；NPU 与其他设备共用 `core/common/metrics.*`、Request、Block Manager 和 Scheduler 的统计链路。因此 NPU 正常本地 Prefill 会写同一组指标，Linear State/Hybrid Prefix Cache 则通过 `linear_state_prefix_cache.cpp` 写同一组 Block 级 recorder。

需要注意两个统计边界：

- 分离式 PD 的 D 侧在收到 P 侧返回值后调用带参数重载 `record_num_prefix_cache_tokens(size_t)`，只更新响应里的 `cached_tokens`，不会触发新增全局 Counter；见 `xllm/core/scheduler/disagg_pd_scheduler.cpp:916-926`。合理部署下 P 侧会在实际 Prefill admission 时计数，因此这能避免 P/D 双计，但意味着各实例 `/vars` 不是端到端服务汇总，必须明确按 P 角色采集。
- `FixedStepsScheduler` 对 `num_blocks_in_prefix_cache` 使用了 `kv_cache_manager_->num_blocks_in_prefix_cache().size()`，得到的是 DP 向量长度而非缓存 Block 数；见 `xllm/core/scheduler/fixed_steps_scheduler.cpp:303-308`。这与本次新增命中率无关，但会使该路径的缓存使用量 Gauge 失真。

## 2. vLLM Core

Prometheus Python Client 会给 Counter 的 exposition 名加 `_total`。下表采用 `/metrics` 实际看到的名字。

| 指标 | 类型 | 精确口径 | 官方源码 |
|---|---|---|---|
| `vllm:prefix_cache_queries_total` | Counter | 本地 Prefix Cache 被查询的 Token 数；只统计启用且未跳过 lookup、最终成功 admission 的新请求，preempted 重查单独记在内部字段而不导出到这个 Counter | [声明](https://github.com/vllm-project/vllm/blob/d3e2888c7588fe3a7be93606c7c626dbfd304d2e/vllm/v1/metrics/loggers.py#L584-L602)、[lookup 记录条件](https://github.com/vllm-project/vllm/blob/d3e2888c7588fe3a7be93606c7c626dbfd304d2e/vllm/v1/core/kv_cache_manager.py#L217-L230)、[preempted/new 分流](https://github.com/vllm-project/vllm/blob/d3e2888c7588fe3a7be93606c7c626dbfd304d2e/vllm/v1/metrics/stats.py#L114-L142) |
| `vllm:prefix_cache_hits_total` | Counter | 上述本地查询中命中的缓存 Token 数 | 同上；[写 Prometheus](https://github.com/vllm-project/vllm/blob/d3e2888c7588fe3a7be93606c7c626dbfd304d2e/vllm/v1/metrics/loggers.py#L1123-L1130) |
| `vllm:external_prefix_cache_queries_total` | Counter | KV Connector 跨实例缓存的候选查询 Token 数，即扣除本地命中后仍需查询外部缓存的 Token | [声明](https://github.com/vllm-project/vllm/blob/d3e2888c7588fe3a7be93606c7c626dbfd304d2e/vllm/v1/metrics/loggers.py#L604-L629)、[写入](https://github.com/vllm-project/vllm/blob/d3e2888c7588fe3a7be93606c7c626dbfd304d2e/vllm/v1/metrics/loggers.py#L1132-L1138) |
| `vllm:external_prefix_cache_hits_total` | Counter | KV Connector 返回的外部缓存命中 Token 数 | 同上。 |
| `vllm:prompt_tokens_by_source_total{source="local_compute"|"local_cache_hit"|"external_kv_transfer"}` | Counter | 首次 Prefill 的 Prompt Token 来源分解；三者之和为总 Prompt Token | [来源定义与不变量](https://github.com/vllm-project/vllm/blob/d3e2888c7588fe3a7be93606c7c626dbfd304d2e/vllm/v1/metrics/stats.py#L376-L419)、[指标声明](https://github.com/vllm-project/vllm/blob/d3e2888c7588fe3a7be93606c7c626dbfd304d2e/vllm/v1/metrics/loggers.py#L670-L702) |
| `vllm:prompt_tokens_cached_total` | Counter | 本地 Prefix Cache + 外部 KV Transfer 跳过计算的 Prompt Token 总数 | [声明](https://github.com/vllm-project/vllm/blob/d3e2888c7588fe3a7be93606c7c626dbfd304d2e/vllm/v1/metrics/loggers.py#L694-L702)、[写入](https://github.com/vllm-project/vllm/blob/d3e2888c7588fe3a7be93606c7c626dbfd304d2e/vllm/v1/metrics/loggers.py#L1194-L1202) |
| `vllm:prompt_tokens_total` | Counter | 首次 Prefill Prompt Token 总数，可作为总体复用率分母 | 同上。 |
| `vllm:kv_cache_usage_perc` | Gauge，0..1 | 非空闲 GPU KV Block 比例，1 表示 100%；不是 Prefix Cache 独占量 | [声明](https://github.com/vllm-project/vllm/blob/d3e2888c7588fe3a7be93606c7c626dbfd304d2e/vllm/v1/metrics/loggers.py#L561-L569)、[Block Pool 算法](https://github.com/vllm-project/vllm/blob/d3e2888c7588fe3a7be93606c7c626dbfd304d2e/vllm/v1/core/block_pool.py#L800-L819) |

vLLM 没有 Prometheus 命中率 Gauge，也没有 `prefix_cache_requests_total` 或 `prefix_cache_hit_requests_total`。日志会对最近约 1000 个新请求维护 `hits / queries` 的 Token 命中率；[窗口实现](https://github.com/vllm-project/vllm/blob/d3e2888c7588fe3a7be93606c7c626dbfd304d2e/vllm/v1/metrics/stats.py#L35-L111)。官方设计说明要求 Prometheus 导出 Counter，让用户按时间窗口计算比率，而非固定 Gauge；见 [Prefix Cache metrics 设计](https://github.com/vllm-project/vllm/blob/d3e2888c7588fe3a7be93606c7c626dbfd304d2e/docs/design/metrics.md#L405-L432)。

常用计算：

```promql
# 本地 lookup 效率
sum(rate(vllm:prefix_cache_hits_total[5m]))
/
sum(rate(vllm:prefix_cache_queries_total[5m]))

# 本地 + 外部对总体 Prompt 的计算节省比例
sum(rate(vllm:prompt_tokens_cached_total[5m]))
/
sum(rate(vllm:prompt_tokens_total[5m]))
```

## 3. vLLM-Ascend

vLLM-Ascend **没有另起一套 Prefix Cache hit/query Prometheus 指标**，继承上述 vLLM Core 指标。其自定义 Recompute/Balance/Dyntra 等 Scheduler 仍把本地命中写入 Core `prefix_cache_stats`，把 Connector 命中写入 `connector_prefix_cache_stats`；例如 [RecomputeScheduler](https://github.com/vllm-project/vllm-ascend/blob/fa668c6495f21734dfd3b7aab464ac73beb4d28b/vllm_ascend/core/recompute_scheduler.py#L827-L845)。因此 NPU 服务对外看到的仍是：

- `vllm:prefix_cache_queries_total`
- `vllm:prefix_cache_hits_total`
- `vllm:external_prefix_cache_queries_total`
- `vllm:external_prefix_cache_hits_total`
- `vllm:prompt_tokens_cached_total`
- `vllm:prompt_tokens_by_source_total{source=...}`
- `vllm:kv_cache_usage_perc`

插件自己的可选 `ms_service_metric` 配置只有 `engine:memory:kvcache_gb` 这一项与 KV Cache 容量内存相关，它是 Gauge，不是命中率或查询/命中数；见 [base_metrics.yaml](https://github.com/vllm-project/vllm-ascend/blob/fa668c6495f21734dfd3b7aab464ac73beb4d28b/vllm_ascend/observability/config/base_metrics.yaml#L26-L45)。UCM Connector 可动态转发底层连接器自定义指标，但 vLLM-Ascend 仓库本身不固定这些名字，不能视为稳定的插件指标。

## 4. SGLang

### 4.1 Radix/Prefix Cache 主指标

| 指标 | 类型 | 精确口径 | 官方源码 |
|---|---|---|---|
| `sglang:cache_hit_rate` | Gauge，0..1 | 最新一次 Prefill 统计报告中 `effective_hit_tokens / (effective_input_tokens + effective_hit_tokens)`；扣除 retracted 请求的重处理 Token。Decode 报告会把它写为 0，因此不是进程 lifetime 或稳定窗口比率 | [声明](https://github.com/sgl-project/sglang/blob/833be86c15044b312c09b5e3b0ceb73248049ce3/python/sglang/srt/observability/metrics_collector.py#L298-L303)、[Prefill 计算](https://github.com/sgl-project/sglang/blob/833be86c15044b312c09b5e3b0ceb73248049ce3/python/sglang/srt/managers/scheduler_components/metrics_reporter.py#L676-L698)、[Decode 置零](https://github.com/sgl-project/sglang/blob/833be86c15044b312c09b5e3b0ceb73248049ce3/python/sglang/srt/managers/scheduler_components/metrics_reporter.py#L874-L934) |
| `sglang:prefill_effective_tokens_total{mode="input"|"device_hit"|"host_hit"|"storage_hit"}` | Counter | 首次有效 Prefill 的计算 Token 与各层命中 Token；排除 retracted 重处理，支持按任意窗口算总体/分层命中率 | [声明和官方计算说明](https://github.com/sgl-project/sglang/blob/833be86c15044b312c09b5e3b0ceb73248049ce3/python/sglang/srt/observability/metrics_collector.py#L898-L911)、[累加实现](https://github.com/sgl-project/sglang/blob/833be86c15044b312c09b5e3b0ceb73248049ce3/python/sglang/srt/observability/metrics_collector.py#L1272-L1288)、[分层来源](https://github.com/sgl-project/sglang/blob/833be86c15044b312c09b5e3b0ceb73248049ce3/python/sglang/srt/managers/schedule_policy.py#L905-L920) |
| `sglang:realtime_tokens_total{mode="prefill_compute"|"prefill_cache"|"decode"}` | Counter | 每个报告周期实际处理的 Token，包含 retract 后的重处理；适合吞吐/计算量，不适合唯一请求的全局命中率 | [声明](https://github.com/sgl-project/sglang/blob/833be86c15044b312c09b5e3b0ceb73248049ce3/python/sglang/srt/observability/metrics_collector.py#L890-L897)、[累加](https://github.com/sgl-project/sglang/blob/833be86c15044b312c09b5e3b0ceb73248049ce3/python/sglang/srt/observability/metrics_collector.py#L1250-L1270) |
| `sglang:cached_tokens_total{cache_source=...}` | Counter | 已完成请求的缓存 Prompt Token 数；有详细信息时来源为 `device`、`host`、`storage_<backend>`，旧路径回退为 `total` | [声明](https://github.com/sgl-project/sglang/blob/833be86c15044b312c09b5e3b0ceb73248049ce3/python/sglang/srt/observability/metrics_collector.py#L1599-L1609)、[请求完成时写入](https://github.com/sgl-project/sglang/blob/833be86c15044b312c09b5e3b0ceb73248049ce3/python/sglang/srt/observability/metrics_collector.py#L1747-L1805) |
| `sglang:prompt_tokens_total{is_streaming=...}` | Counter | 已完成请求的 Prompt Token 总数 | 同上。 |
| `sglang:num_requests_total{is_streaming=...}` | Counter | 已完成请求总数，不区分是否缓存命中 | 同上。SGLang 没有 cached-request numerator，不能直接推导请求命中率。 |
| `sglang:uncached_prompt_tokens_histogram` | Histogram | 每请求 `prompt_tokens - cached_tokens` 分布 | [声明](https://github.com/sgl-project/sglang/blob/833be86c15044b312c09b5e3b0ceb73248049ce3/python/sglang/srt/observability/metrics_collector.py#L1573-L1603)。 |

SGLang 官方文档也展示了 `sglang:cache_hit_rate` 与 `sglang:token_usage` 的 `/metrics` 输出；见 [Production Metrics](https://github.com/sgl-project/sglang/blob/833be86c15044b312c09b5e3b0ceb73248049ce3/docs/docs/references/production_metrics.mdx#L6-L25)。

推荐使用有效 Counter 计算，而不是依赖瞬时 Gauge：

```promql
# device + host + storage 总 Prefix 命中率
sum(rate(sglang:prefill_effective_tokens_total{mode=~".*_hit"}[5m]))
/
sum(rate(sglang:prefill_effective_tokens_total[5m]))

# 仅 L1/device 命中率
sum(rate(sglang:prefill_effective_tokens_total{mode="device_hit"}[5m]))
/
sum(rate(sglang:prefill_effective_tokens_total[5m]))
```

### 4.2 KV Cache 使用量与 HiCache

SGLang 对“主动占用”和“可淘汰 Prefix Cache”分得更细：

- `sglang:token_usage`：Full/SWA/Mamba 各池占用率的最大值；名字是历史遗留。
- `sglang:full_token_usage`、`sglang:swa_token_usage`、`sglang:mamba_usage`：各池主动使用率。
- `sglang:kv_available_tokens`、`sglang:kv_used_tokens`、`sglang:kv_evictable_tokens`：Full Attention 池的空闲、运行请求锁定、可淘汰 Radix Cache Token 数。
- `sglang:swa_*_tokens`、`sglang:mamba_*_tokens`：SWA/Mamba 的对应分层。
- `sglang:max_total_num_tokens`、`sglang:kv_cache_memory_usage_gb`：启动时容量与 KV Cache 内存。

定义与不变量见 [SchedulerStats](https://github.com/sgl-project/sglang/blob/833be86c15044b312c09b5e3b0ceb73248049ce3/python/sglang/srt/observability/metrics_collector.py#L69-L112) 和 [Gauge 声明](https://github.com/sgl-project/sglang/blob/833be86c15044b312c09b5e3b0ceb73248049ce3/python/sglang/srt/observability/metrics_collector.py#L311-L401)。

HiCache 额外提供：

- `sglang:hicache_host_used_tokens` / `sglang:hicache_host_total_tokens` Gauge：L2 Host KV 当前占用/总容量；[声明](https://github.com/sgl-project/sglang/blob/833be86c15044b312c09b5e3b0ceb73248049ce3/python/sglang/srt/observability/metrics_collector.py#L633-L648)。
- `sglang:prefetched_tokens_total`：L3 预取 Prompt Token；`sglang:load_back_tokens_total{pool=...}`：L2→L1 加载 Token；`sglang:hicache_backup_tokens_total{pool=...}`：L1→L2 备份 Token；`sglang:evicted_tokens_total` / `sglang:hicache_dropped_tokens_total{reason,pool}`：淘汰/未备份销毁 Token。它们是传输与容量压力指标，不应当直接当作命中数；[Storage collector](https://github.com/sgl-project/sglang/blob/833be86c15044b312c09b5e3b0ceb73248049ce3/python/sglang/srt/observability/metrics_collector.py#L1856-L1895)、[load/backup/evict](https://github.com/sgl-project/sglang/blob/833be86c15044b312c09b5e3b0ceb73248049ce3/python/sglang/srt/observability/metrics_collector.py#L2084-L2162)。

### 4.3 PD 层

SGLang PD 没有另一套 Prefix hit/query Counter。实际缓存 Token 仍进入通用 `prefill_effective_tokens_total` 和请求完成侧 `cached_tokens_total`；PD 专属 Prometheus 指标主要是 transfer queue、吞吐、延迟和失败数，如 `sglang:kv_transfer_speed_gb_s`、`sglang:kv_transfer_latency_ms`、`sglang:num_transfer_failed_reqs_total`，见 [PD 指标声明](https://github.com/sgl-project/sglang/blob/833be86c15044b312c09b5e3b0ceb73248049ce3/python/sglang/srt/observability/metrics_collector.py#L489-L562)。

开启 KV event publisher 时，内部 `KvMetrics` 消息还携带 `gpu_prefix_cache_hit_rate` 和 `gpu_cache_usage_perc`，但它是发往 KV event/路由消费端的结构字段，不是 `/metrics` 中独立的 Prometheus 时序；见 [kv_events_publisher.py](https://github.com/sgl-project/sglang/blob/833be86c15044b312c09b5e3b0ceb73248049ce3/python/sglang/srt/managers/scheduler_components/kv_events_publisher.py#L30-L98)。

## 5. 横向矩阵：xLLM 多什么、缺什么

| 能力 | xLLM 当前工作树 | vLLM / vLLM-Ascend | SGLang | 判断 |
|---|---|---|---|---|
| 全局 Token 命中分子/分母 Counter | 有：hit/prompt | 有：local query/hit；另有总体 prompt cached/total | 有：effective input + 分层 hit | xLLM 基础已补齐。 |
| 实际 Prefix Cache 查询 Token 数 | **无**；prompt 总数并不等于实际 lookup query | 有，本地与 external 分开 | effective 总量可作为首次 Prefill 分母 | xLLM 缺项。 |
| 直接命中率 Gauge | 有，进程 lifetime 百分数 | 无，官方建议 Counter 推导 | 有，最新 Prefill 报告比率 | xLLM 的 Gauge 可推导且不利于集群聚合，建议不作为主接口。 |
| 请求总数 / 至少命中请求数 | **两者都有** | 均无 cache-specific Counter | 只有通用请求总数，无命中请求数 | xLLM 独有；是否保留取决于是否要请求覆盖面。 |
| local/host/external 分层命中 | **无**，合成一个 hit Token 数 | local 与 external 分开，另有 source label | device/host/storage 分层 | xLLM 主要缺项。 |
| 每次 Block 匹配数/率分布 | 有 | 无对应标准 Prometheus 序列 | 无对应标准序列 | xLLM 额外的底层诊断能力。 |
| Prefix Cache 当前驻留量 | 有 Block 数，但 FixedSteps 路径有错误 | 没有 Prefix-only 驻留量；只有总体活跃 usage | 有 evictable Radix Token，且各池分层 | xLLM 比 vLLM 多、比 SGLang 粗。 |
| Host Cache 容量/占用 | 无 | vLLM Core 无；Ascend 插件仅可选 `kvcache_gb` 总内存 | 有 HiCache host used/total Token | xLLM 缺项。 |
| PD / 外部缓存命中 | 无来源标签；P 侧计数、D 侧只传播 usage | `external_prefix_cache_*` + source label | 通用 tier hit + PD transfer 指标 | xLLM 缺分层与角色语义。 |
| 多实例/模型/engine 标签 | bvar 单值，无标签 | `model_name`,`engine` | model/rank 等 labels | xLLM 集群可观测性缺项。 |

## 6. 建议

### P0：先统一口径

1. 将 Counter 作为公开主接口，Dashboard 用 `rate()`/`increase()` 计算窗口命中率；删除或降级 `prefix_cache_token_hit_rate_perc`、`prefix_cache_request_hit_rate_perc`。
2. 明确区分两个分母：
   - `prompt_tokens_total`：所有已 admission 的首次 Prefill Prompt Token，衡量实际计算节省比例；
   - `query_tokens_total`：实际执行了 cache lookup 的 Token，衡量缓存本身效率。
3. 当前新增的 `prefix_cache_prompt_tokens_total` 应保留现名和描述，不要直接改称 `queries`，因为 Prefix Cache 关闭/跳过时它仍可能计入。

### P1：补齐来源分层

优先对齐 vLLM/SGLang，增加带来源的 Counter（若 bvar 标签能力受限，也可先拆成三个 Counter）：

```text
prefix_cache_hit_tokens_total{cache_source="device|host|external"}
prefix_cache_query_tokens_total{cache_source="device|external"}
```

并补 Host Cache used/total Token 或 Block Gauge。这样才能区分“NPU HBM Radix 命中”“Host offload 恢复”“Mooncake/PD 外部复用”。

### P1：两个请求 Counter 的去留

- 若产品明确需要“有多少请求享受到缓存”的覆盖面：保留 `prefix_cache_requests_total` 和 `prefix_cache_hit_requests_total`，删掉可推导 Gauge；它们应成对存在。
- 若目标只是与 vLLM/vLLM-Ascend 对齐 Token 命中率：这两个 Counter 及请求命中率 Gauge 都可以删除，减少指标面。
- 若保留，应把描述写成“admitted first-prefill requests”和“requests with at least one cached prompt token”，并明确 Prefix Cache disabled/skip lookup 是否进入分母。

### P1：修正路径一致性

1. 修复 FixedSteps 的 `num_blocks_in_prefix_cache().size()` 统计错误。
2. 给 PD 指标规定唯一归属：建议只在实际执行 Prefix lookup 的 P 侧计数，并增加 role/instance 维度或在采集规则中只聚合 P；D 侧仅传播 `cached_tokens` 不重复计数。
3. 保留旧 `prefix_cache_match_length_total` 作为底层 lookup 诊断，但不要与新的请求去重 `prefix_cache_hit_tokens_total` 混用；二者数值不应被期待完全相等。

## 7. 推荐的最终最小指标集

若兼顾行业对齐和 xLLM 的请求级需求，最小集合建议为：

```text
# 首次有效 Prefill 工作负载
prefix_cache_prompt_tokens_total
prefix_cache_hit_tokens_total{cache_source="device|host|external"}

# 实际缓存查询效率
prefix_cache_query_tokens_total{cache_source="device|external"}

# 可选：请求覆盖面
prefix_cache_requests_total
prefix_cache_hit_requests_total

# 容量
num_blocks_in_prefix_cache
num_free_blocks
num_used_blocks
host_cache_used_tokens
host_cache_total_tokens
```

比率全部在查询端推导。这样既能得到全局/窗口 Token 命中率，也能得到请求命中率、各层命中率与缓存压力，并避免无法正确聚合的 lifetime Gauge。
