# vLLM Ascend decode preemption CPU KV offload 与 prefix cache 复核

调研日期：2026-08-12。固定源码快照：vLLM
`8c011da6d0a45f2c9316ea3c658da754f770a178`，vLLM Ascend
`63bd2c418c8914201981f1150fc0e0ae5724896e`，xLLM
`6fbce3eb74ff61f46f3ad8b9d16a586ad406005a`（含当前工作树中的 decode
preemption 变更）。

上游调研仅采用以下官方来源：vllm-project/vllm-ascend PR [#10742](https://github.com/vllm-project/vllm-ascend/pull/10742)、issue [#10820](https://github.com/vllm-project/vllm-ascend/issues/10820)、vllm-ascend `main` 的 [recompute_cpu_offload 指南](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/user_guide/feature_guide/recompute_cpu_offload.md) 与 [scheduler 代码](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/vllm_ascend/distributed/kv_transfer/kv_pool/recompute_cpu_offload/manager.py)，以及 vllm-project/vllm 的 RFC/PR [#16144](https://github.com/vllm-project/vllm/issues/16144)、[#13377](https://github.com/vllm-project/vllm/pull/13377)、[#19854](https://github.com/vllm-project/vllm/issues/19854)、[#19848](https://github.com/vllm-project/vllm/pull/19848)、[#24256](https://github.com/vllm-project/vllm/issues/24256) 和 `main` 的 [SimpleCPUOffloadConnector](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py)、[OffloadingConnector](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py)、[config 选择逻辑](https://github.com/vllm-project/vllm/blob/main/vllm/config/vllm.py)。xLLM 行为与性能结论则来自固定代码快照、当前工作树和本轮实测日志。

阅读顺序：第 0 章把本轮问答串成完整因果链；第 1～5 章保留上游源码、已合入方案和未合入 PR 的详细证据。

## TL;DR

1. 结论：Ascend 的 decode preemption CPU KV offload 与“prefix 共享”是解耦的，但与“prefix-cache 命中/恢复口径”没有完全解耦。
   - 证据：`enable_offload_prefix_caching` 默认 `false`，文档明确说仅在显式测试 prefix sharing 时再打开；`manager.py` 中 `block_hash` 只有在 `enable_offload_prefix_caching` 打开时才参与 CPU block 共享，否则 offload block 是请求私有快照。[guide](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/user_guide/feature_guide/recompute_cpu_offload.md) / [manager.py](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/vllm_ascend/distributed/kv_transfer/kv_pool/recompute_cpu_offload/manager.py)
   - 但：恢复仍复用 `get_num_new_matched_tokens()` / `update_state_after_alloc()` 这套“外部 KV 命中 -> 分配 GPU block -> H2D reload”的状态机，所以它不是完全独立于 prefix-cache 体系，而是“共享命中协议、分离存储策略”。[PR #10742](https://github.com/vllm-project/vllm-ascend/pull/10742) / [manager.py](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/vllm_ascend/distributed/kv_transfer/kv_pool/recompute_cpu_offload/manager.py)

2. 结论：Ascend 方案的 token 口径是“稳定的已计算 token 边界”，不是“任意已分配 block 数”。
   - 证据：preempt 时把 `num_computed_tokens` 存入 `PreemptedRequestState`；resume 时先算 `restorable_tokens = min(state.num_computed_tokens, request.num_tokens)`，再用 `hit_length = restorable_tokens - num_computed_tokens` 得到可恢复 token 数；真正 H2D 时再把 token 区间映射回 block 区间。[issue #10820](https://github.com/vllm-project/vllm-ascend/issues/10820) / [manager.py](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/vllm_ascend/distributed/kv_transfer/kv_pool/recompute_cpu_offload/manager.py)
   - 风险：官方文档已点名 Qwen3.5 + async scheduling 仍有 token/state boundary 问题，建议关闭 async scheduling；RFC 里也把“乐观 token 边界”列为后续问题。[guide](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/user_guide/feature_guide/recompute_cpu_offload.md) / [issue #10820](https://github.com/vllm-project/vllm-ascend/issues/10820)

3. 结论：当前 Ascend 方案优先保证 correctness，不追求异步隐藏传输时延。
   - 证据：PR #10742、issue #10820、主线文档都把当前实现描述为 torch `copy_` 的正确性优先版本；worker 里 preempt store 和 pre-forward load 都是 `sync=True` 提交。[PR #10742](https://github.com/vllm-project/vllm-ascend/pull/10742) / [issue #10820](https://github.com/vllm-project/vllm-ascend/issues/10820) / [worker.py](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/vllm_ascend/distributed/kv_transfer/kv_pool/recompute_cpu_offload/worker.py)

4. 结论：vLLM 主线已经从“改核心 KVCacheManager 的 CPU offload”转向“connector 化、异步化、可插拔后端”的路线。
   - 证据：#16144 提的是改 `KVCacheManager` / `BlockPool`、在每个 schedule step 生成 swap plan；#13377 的 TL;DR 也是 “swap GPU KV cache blocks to CPU upon eviction and swap them back if there's a cache hit”；#19854 明确批评这一路线太像 V0，改为 `OffloadingConnector` + `OffloadingManager` + worker 线程异步传输；`main` 默认 native backend 也已优先选 `OffloadingConnector`，只有环境变量才走 `SimpleCPUOffloadConnector`。[#16144](https://github.com/vllm-project/vllm/issues/16144) / [#13377](https://github.com/vllm-project/vllm/pull/13377) / [#19854](https://github.com/vllm-project/vllm/issues/19854) / [config](https://github.com/vllm-project/vllm/blob/main/vllm/config/vllm.py) / [OffloadingConnector](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py)

5. 结论：把“preemption 失败/低命中流式终止”暴露成新 finish_reason 的路子，在 vLLM 官方 RFC 里提出过，但截至 2026-08-12 没被采纳。
   - 证据：#24256 提议 `cache_hit_threshold` 和新 `finish_reason=cache_threshold`，还提议可把已生成 output tokens 返还给 router 继续做 P/D；该 RFC 状态是 “Closed as not planned”。这说明官方没有把“流式/拒绝返回协议”收敛进当前主线 API。[#24256](https://github.com/vllm-project/vllm/issues/24256)

## 0. 从配置故障到性能结论：完整问题链

### 0.1 为什么会出现 `prefix cache > prompt` 和 HTTP 流不结束

这不是用户主动修改造成的，而是此前配置校验引入的间接耦合：为了启用 Decode
抢占的 Host KV，Decode 同时打开了 prefix cache。两者共用了 `shared_tokens` /
`num_prefix_cache_tokens` 口径，Decode 恢复后再次记录 prefix 数时，已经生成的 token
对应 KV 也可能进入该数值。

这里必须拆开两个概念：

| 字段 | 含义 | 合法范围 | 是否对外暴露 |
|---|---|---:|---|
| `prompt_prefix_hit_tokens` | 首次 Prefill 时由 prefix cache 跳过的 prompt token | `0..prompt_tokens` | 是，对应 OpenAI usage 的 `cached_tokens` |
| `decode_resume_tokens` | 被抢占请求恢复时已有 KV 覆盖到的稳定 token 边界 | 可覆盖 prompt + 已生成 token | 否，仅供调度/恢复 |

xLLM 当前 `Sequence::num_prefix_cache_tokens()` 取 device/host
`shared_tokens_num()` 最大值，`Request::record_num_prefix_cache_tokens()` 又采用只增不减的
max 记录；若 Decode 重新入队时调用记录逻辑，就可能把恢复边界误写成对外 prefix hit。
`Request::generate_output()` 最终把该值输出为 `usage.num_cached_tokens`。运行日志已经连续出现
`prefix cache hit tokens must not exceed prompt tokens`，说明终态 generation 被
xllm-service 拒绝；若拒绝分支没有发送 SSE error/`[DONE]` 并清理请求，客户端就表现为 HTTP
流一直不结束。

vLLM 对外统计的定义支持这个边界：`num_cached_tokens` 是 Prefill 阶段跳过的 prompt
token，并断言它不大于 `num_prompt_tokens`；Decode 内部恢复进度不是这个字段。[vLLM
PrefillStats](https://github.com/vllm-project/vllm/blob/8c011da6d0a45f2c9316ea3c658da754f770a178/vllm/v1/metrics/stats.py)

因此不能用 `min(decode_resume_tokens, prompt_tokens)` 作为主修复。这样虽不再越界，却会把错误
语义伪装成合法统计。正确做法是让 prompt hit 在首次 admission/P-D 传递时冻结，Decode 抢占恢复
永远只更新私有 resume cursor。

### 0.2 `main` 在普通 P/D 下如何处理 Decode 抢占

本文讨论的基线是：`enable_disagg_pd=true`、`enable_pd_ooc=false`。在该配置下，Decode
实例使用 `DisaggPDScheduler`，它继承 `ContinuousScheduler` 的普通调度策略，而不是
`PDOOCScheduler`。[scheduler_factory.cpp](https://github.com/jd-opensource/xllm/blob/cd167a6aca8b4de200a1e7ab76b105cf102e7b72/xllm/core/scheduler/scheduler_factory.cpp#L30-L57)

`main` 的 Decode victim 在 HBM allocation 失败时立即执行：

```text
释放 victim 的全部 HBM KV
-> set_preempted()
-> 放入 Decode 实例自己的 prefill_queue
-> 本轮剩余阶段或后续调度轮重新 Prefill
-> 重新达到原 token 位置后继续 Decode
```

对应代码直接在 `schedule_decode_from_queue()` 中完成 `deallocate -> set_preempted ->
prefill_queue.push`。[scheduler_policy.cpp](https://github.com/jd-opensource/xllm/blob/cd167a6aca8b4de200a1e7ab76b105cf102e7b72/xllm/core/scheduler/scheduler_policy.cpp#L657-L676)
这里的 `prefill_queue` 是 **Decode 节点的本地调度队列**，不是通过网络把请求退回独立 Prefill
实例。普通 P/D 主线没有“只要还有 scheduled work 就禁止重算”的全局门禁。

什么时候开始重算取决于 batch policy：

- 默认 `enable_mix_batch=true` 时走 Decode-first：先给活动 Decode 分配本轮 token，再用剩余
  token、sequence、latency 和 HBM budget 调度 `chunk_queue/prefill_queue`。因此其他请求仍在
  Decode 时，被抢占请求也可以开始本地重算。[decode_first_policy.cpp](https://github.com/jd-opensource/xllm/blob/cd167a6aca8b4de200a1e7ab76b105cf102e7b72/xllm/core/scheduler/decode_first_policy.cpp#L53-L83)
- Prefill-first/exclusive 模式下，如果抢占发生在本轮 Decode 阶段，通常下一轮优先处理重新
  Prefill；仍然不需要等待所有活动请求完成。
- 若实际 HBM 或本轮 budget 不足，请求留在 Prefill/chunk queue，后续每轮继续参与正常公平调度。

这解释了为什么 `main` 即使重复计算 KV，也可能比一个被 park 在调度体系之外的 Host snapshot
更快：前者立即重新参与资源竞争，后者若有全局恢复门禁会发生饥饿。

### 0.3 `main` 每次重算多长

被抢占后 sequence 的 KV cursor 被清零，但 token 历史仍保留。`Sequence::num_tokens()` 等于
prompt token 加已经确认的 generated token，因此总重算长度为：

```text
total_recompute_tokens = prompt_tokens + committed_generated_tokens
```

未确认的 speculative/placeholder token 不应作为稳定重算边界。xLLM 的 Sequence 定义也明确以
`num_tokens - num_prompt_tokens` 计算已生成 token 数，并在 token commit 时增长 `num_tokens`。

本轮压测的 Decode 脚本启用了 chunked prefill，设置 `max_tokens_per_batch=4096`，没有单独设置
`max_tokens_per_chunk_for_prefill`。未设置时后者自动继承 `max_tokens_per_batch`，所以单个 scheduler
step 的 **全局 Prefill 上限** 是 4096 tokens，而不是每个请求各 4096。
[xllm.cpp](https://github.com/jd-opensource/xllm/blob/cd167a6aca8b4de200a1e7ab76b105cf102e7b72/xllm/xllm.cpp#L467-L472)

```text
chunk_tokens = min(
    remaining_recompute_tokens,
    max_tokens_per_chunk_for_prefill,  // 本轮为 4096
    remaining_global_token_budget)
```

例如 8000-token prompt 在抢占前已确认生成 600 tokens，总共约需重算 8600 tokens；没有其他
budget 消耗时大致分为 `4096 + 4096 + 408`。若仍有活动 Decode，它们先消费本轮少量 decode/MTP
budget，重新 Prefill 使用剩余部分。主线计算逻辑见
[compute_prefill_tokens](https://github.com/jd-opensource/xllm/blob/cd167a6aca8b4de200a1e7ab76b105cf102e7b72/xllm/core/scheduler/scheduler_policy.cpp#L333-L358)。

### 0.4 xLLM 当前 Host KV 恢复与 `main` 的关键差异

两条路径最终都必须重新获得覆盖旧 cursor 的 HBM KV，但 admission 方式不同：

| 维度 | `main` 本地重算 | 当前 Host snapshot 恢复 |
|---|---|---|
| 抢占后位置 | 立即进入本地 `prefill_queue` | coordinator 私有 parked list |
| 公平性 | 受普通 queue/policy/budget 管理 | 取决于独立 restore policy |
| HBM admission | chunked Prefill 可逐步申请、逐步计算 | 当前要求完整 snapshot 的目标 HBM 一次分配成功 |
| 进度成本 | 重复模型计算 | D2H/H2D copy，不重复模型计算 |
| 极端压力风险 | 计算量大，但持续参与调度 | 完整 snapshot 长期放不回会饥饿 |

旧恢复策略只要本轮存在任何 scheduled work 就拒绝启动 H2D。这相当于把 offloaded request
移出公平调度体系，直到所有活动 workload 清空。实测 D2H ready 后等待 175/221 秒，而真正 H2D
仅约 0.45/0.38 秒，证明主要瓶颈是 admission starvation，不是 CPU-NPU 带宽。

### 0.5 vLLM Ascend 如何恢复，以及它是否要求完整装回

vLLM Ascend 也使用 request-private CPU snapshot，但恢复请求仍位于普通 waiting queue：每轮先满足
running 请求的增量 blocks，然后遍历 waiting。D2H 未 ready 时 connector 返回 `None` 并跳过；ready
后返回完整 restorable token 数和 `load_kv_async=true`。scheduler 将 `num_new_tokens` 设为 0，尝试
为全部 external/restorable tokens 一次分配目标 HBM；成功后进入 `WAITING_FOR_REMOTE_KVS` 并在
forward 前 H2D。[RecomputeScheduler](https://github.com/vllm-project/vllm-ascend/blob/63bd2c418c8914201981f1150fc0e0ae5724896e/vllm_ascend/core/recompute_scheduler.py#L471-L479)
[restore admission](https://github.com/vllm-project/vllm-ascend/blob/63bd2c418c8914201981f1150fc0e0ae5724896e/vllm_ascend/core/recompute_scheduler.py#L613-L716)

因此对普通 full-attention KV，Ascend 当前同样是 **完整 snapshot 一次 admission、一次 H2D**，没有
通用的 chunked H2D restore：

```text
restorable_tokens = min(stored_num_computed_tokens, request.num_tokens)
external_tokens = restorable_tokens - local_computed_tokens
allocate_slots(all external_tokens)
-> success: H2D all mapped blocks
-> failure: stay waiting, retry later
```

本地 cache hit、sliding-window 无效 blocks、Mamba state 和 MTP 尾部裁剪会减少实际 copy，但不等于
通用分块恢复。[manager.py](https://github.com/vllm-project/vllm-ascend/blob/63bd2c418c8914201981f1150fc0e0ae5724896e/vllm_ascend/distributed/kv_transfer/kv_pool/recompute_cpu_offload/manager.py#L195-L235)

Ascend 相比旧 xLLM 策略的改进是“不等 running 清空、每轮按剩余容量 admission”；它没有改变
完整恢复最终需要全部 HBM 的事实，也没有显式最小驻留期、cooldown 或严格的最大等待时间。因此在
持续 HBM 超卖时，Ascend 同样可能等待；其已合入实现主要证明 correctness-first 的生命周期正确，
不能当作极端压力公平性已经解决的证据。[PR #10742](https://github.com/vllm-project/vllm-ascend/pull/10742)

### 0.6 本轮性能结果说明了什么

测试条件：GLM-5.2-W4A8、普通 P/D、`enable_pd_ooc=false`、prefix cache 关闭、MTP=3、DP=2、
1169 个 HBM KV blocks、8×8000 输入、每条输出 2048、并发 8。

| 指标 | `main`：重算 | 当前分支：CPU KV 恢复 | 变化 |
|---|---:|---:|---:|
| 完成率 | 8/8 | 8/8 | 相同 |
| 总耗时 | 116.45 s | 280.74 s | +141.1% |
| 输出吞吐 | 140.70 tok/s | 58.36 tok/s | -58.5% |
| Mean E2E | 78.31 s | 136.34 s | +74.1% |
| P99 E2E | 115.96 s | 278.03 s | +139.8% |
| Mean TPOT | 29.36 ms | 58.70 ms | +100.0% |
| P99 TPOT | 53.88 ms | 134.34 ms | +149.3% |
| P99.99 ITL | 33.06 s | 188.43 s | 5.70× |
| 最大 ITL | 35.32 s | 221.19 s | 6.26× |
| Mean TTFT | 18.21 s | 16.18 s | -11.2% |

并发 2、不触发抢占的控制组中，`main` 为 30.23 s/135.47 tok/s/13.38 ms TPOT，当前分支为
30.53 s/134.18 tok/s/13.43 ms TPOT，底速差异小于 1%。因此高压差距来自恢复调度，不是基础算子
性能。TTFT 发生在抢占前，不是评价该修复的主要指标；E2E、TPOT、极端 ITL 和吞吐更有代表性。

结果文件：

- `testspace/scripts/logs/pd_decode_preemption_main_recompute_1169blocks_c8_20260812.jsonl`
- `testspace/scripts/logs/pd_decode_preemption_branch_cpu_kv_1169blocks_c8_repeat_20260812.jsonl`
- `testspace/scripts/logs/pd_decode_preemption_prefix_off_1169blocks_c8_fair_restore_20260812.jsonl`
- `testspace/scripts/logs/pd_decode_main_no_preemption_c2_20260812.jsonl`
- `testspace/scripts/logs/pd_decode_branch_no_preemption_c2_20260812.jsonl`

这个场景没有“测错”。每个 8000-token full-attention 请求约需 500 个 block，而每个 DP 只有约
1169 blocks；每个 DP 承担约 4 个请求时，稳定 working set 显著超卖。它有效回答：snapshot ready
但 HBM 长期不足时，系统能否有界、公平地恢复或 fallback。当前答案是否定的，所以“当前实现暂不能
作为性能优化合入”成立。

但该测试不能独自证明 Host copy 机制天然比重算慢。它把 raw copy 收益和极端 admission 竞争放在
同一结果中；实际 H2D 只有约 0.4 秒，说明 offload 仍可能在合适工作区间获益。

### 0.7 Host KV offload 的优势区间与 break-even

Host snapshot 是“用 Host 带宽换模型计算”，只有下面不等式成立才有性能收益：

```text
D2H + offloaded_queue_wait + H2D
<
recompute_queue_wait + Prefill(prompt + committed_output)
```

更适合 offload 的条件：

- 长 prompt、晚期抢占，重算历史很长；
- Decode 节点计算昂贵，重 Prefill 会显著干扰其他请求 TPOT；
- 抢占偶发，snapshot 能在少量调度轮内重新获得 HBM；
- Host copy 带宽充足，恢复后有驻留保护，不会立即再次被抢占；
- P/D 架构希望 Decode 保持 decode-only，避免本地长 Prefill。

更适合直接 recompute 或 fallback 的条件：

- prompt/历史较短，重算本来很便宜；
- HBM 持续超卖，完整 snapshot 在可接受时限内放不回；
- 高频反复抢占，Host transfer 形成抖动；
- Host 带宽或容量紧张；
- scheduler 无法提供公平 admission 和恢复后驻留保护。

因此 production policy 不应是“所有 victim 一律 offload”，而应根据预测 copy 成本、预测重算成本、
snapshot 大小、当前 HBM 压力和 offloaded age 动态选择。

### 0.8 推荐落地方案

第一阶段做最小且安全的修复：

1. 允许 P/D Decode 在 `enable_prefix_cache=false` 时独立启用 request-private Host KV
   preemption；Decode 启动脚本保持 prefix cache 关闭。
2. 不要只根据 `instance_role=DECODE` 和 `host_blocks_factor>1` 隐式判断功能。新增明确的
   `decode_preemption_mode=recompute|host_offload`（或等价布尔开关）；旧配置可暂时兼容推导并告警。
3. 私有 D2H/H2D 路径不得 publish 到 prefix hash/LRU，不得增加 `shared_tokens_num()`，恢复后只
   commit `DecodeResumePosition`。
4. `num_prefix_cache_tokens` 只允许在首次 Prefill admission 或 Prefill→Decode 元数据接收时写入；
   Decode 抢占恢复和重新入队不得再次调用无参 `record_num_prefix_cache_tokens()`。
5. 在 xllm-service 保留 `cached_tokens <= prompt_tokens` 校验，但失败必须形成完整终态：发送结构化
   SSE error，随后发送 `[DONE]`，并清理请求/连接状态。不能仅丢弃 terminal generation。
6. 删除“存在任意 scheduled work 就禁止恢复”的全局门禁。保留 Decode-first：当前 running 先申请
   本轮增量 blocks，然后按 offload-ready FIFO 尝试一个 snapshot；容量不足就保持 `OFFLOADED`，
   下一轮重试。
7. 完整恢复 admission 至少预留 snapshot blocks 加下一 decode quantum/lookahead 的安全余量；同一时刻
   限制 H2D 数量，恢复后设置最小驻留步数或 victim penalty，避免刚恢复就再次被抢占。
8. 为 offloaded wait 设置 deadline/aging。等待达到阈值后提升恢复优先级；仍无法恢复时明确转
   chunked recompute 或 router retry，保证有界等待。
9. 根据 `estimated_copy_cost`、`estimated_recompute_cost`、snapshot 大小、HBM 压力和历史抢占次数动态
   选择 `host_offload` 或 `recompute`，不要把 Host offload 设为所有 victim 的无条件路径。

配置矩阵应明确为：

| Prefix cache | Decode host preemption | Host tier | 结果 |
|---|---|---|---|
| 关 | 关 | 关 | 普通 Decode/recompute |
| 开 | 关 | 开 | 共享 Host Prefix Cache |
| 关 | 开 | 开 | request-private Decode 抢占快照；本次目标 |
| 开 | 开 | 开 | 两种能力共存，但统计与生命周期仍隔离 |

第二阶段再做结构化收口：物理 Host allocator 可以暂时共用，但逻辑所有权和预算要分开。
`HostPreemptSnapshot` 是 request-scoped、传输期间 pinned、恢复/取消后释放；`HostPrefixCache`
是 hash-scoped、可跨请求共享、受 LRU 淘汰。分别增加 `prefix_hit_tokens`、
`preemption_restored_tokens/bytes/latency` 和 fallback 指标。

失败路径必须是显式状态机：

```text
RUNNING -> OFFLOAD_PENDING -> OFFLOADING -> OFFLOADED
        -> RESTORING -> RUNNING
        -> RECOMPUTE/ROUTER_RETRY/TERMINAL_ERROR  (容量或传输失败)
```

D2H 在所有 TP worker 完成前不得释放/复用源 HBM；H2D 在所有 worker 完成前不得提交 cursor
或参加 forward。P/D Decode 无法 offload 时，如果 Decode 节点不允许本地 Prefill，就应显式 abort/
router retry，而不是悄悄把请求放回 Decode 做全量 Prefill。vLLM Ascend 尚未合入的
[#14062](https://github.com/vllm-project/vllm-ascend/pull/14062) 正因这一 fallback 风险被指出：
offload 失败后普通 preempt 会导致 Decode 本地 Prefill。

### 0.9 必须覆盖的验证矩阵

- `prompt=128`，生成若干 token 后触发 D2H/H2D；恢复边界允许大于 128，但对外
  `cached_tokens` 必须保持首次 Prefill 值且不大于 128，输出连续且流正常 `[DONE]`。
- Prefix cache 开/关 × Host preemption 开/关四象限；相同 prompt 真 prefix hit 与 Decode
  抢占同时发生时，两个计数互不污染。
- Host 容量不足、D2H/H2D 失败、恢复时暂时无 HBM、请求取消/超时；无死锁、无 block 泄漏，
  fallback 只有一个明确入口。
- TP 多 rank 完成栅栏，以及 MTP 部分接受/全部拒绝、尾块未满时抢占；恢复后不重 token、不漏 token。
- 构造非法 usage 的 service 契约测试：即使拒绝 generation，也必须发 error + `[DONE]` 并释放请求。
- 单次强制抢占机制测试：D2H 后及时释放 HBM，单独比较 `D2H + H2D` 与 8K/16K 历史重算，确认
  raw offload 收益。
- 中等压力收益测试：扫描 concurrency、HBM blocks 和 prompt 长度，寻找 offload 的实际收益区间，
  同时记录 ready-to-H2D wait、重复抢占次数和 restored tokens。
- 保留 `c8 + 1169 blocks` 极限超卖测试：要求有界等待；超过恢复 deadline 后必须提升优先级或转
  recompute，不能依赖其他请求最终结束才取得进展。
- 对照测试始终包含无抢占底速，避免把版本/算子差异归因于 offload。

## 1. 是否与 prefix cache 解耦

我的结论分两层：

- 与“prefix 共享存储”解耦：是。
  - Ascend recompute offload 即使关闭 `enable_offload_prefix_caching` 也能工作；此时 CPU 侧只保存“这个被 preempt 请求自己的快照”，不做 hash 共享。[guide](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/user_guide/feature_guide/recompute_cpu_offload.md) / [manager.py](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/vllm_ascend/distributed/kv_transfer/kv_pool/recompute_cpu_offload/manager.py)

- 与“prefix 命中状态机”不解耦：不是。
  - 官方 PR 明说 multi-connector 下会优先处理 recompute-offloaded request 的 `get_num_new_matched_tokens`，确保私有/unhashed blocks 先恢复；文档也写了“先查 local prefix cache，再返回 external restorable tokens”。所以它依然挂在“外部命中 token 数 -> alloc -> reload”这个统一协议上。[PR #10742](https://github.com/vllm-project/vllm-ascend/pull/10742) / [issue #10820](https://github.com/vllm-project/vllm-ascend/issues/10820)

对照 vLLM 主线也能看出差异：

- `SimpleCPUOffloadConnector` 明确要求 prefix caching 开启，否则直接禁用 CPU offload；这代表“通用 prefix-cache offload”与 prefix cache 强绑定。[SimpleCPUOffloadConnector](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py)
- Ascend recompute offload 则只把“共享”做成可选增强，而不是前置条件。[guide](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/user_guide/feature_guide/recompute_cpu_offload.md)

## 2. 状态机与 token 口径

按主线代码，状态机可以压成：

1. `RUNNING`
2. preempt 前调用 `update_state_before_preempt(request, block_ids, num_computed_tokens)`，生成 `PreemptedRequestState`
3. 若有新 D2H store，状态是 `ready=False`
4. worker 完成 store，scheduler 聚合所有 worker 完成事件后置 `ready=True`
5. request 再次被调度时，`get_num_new_matched_tokens()` 返回可恢复 token 数
6. alloc 完新 GPU blocks 后，`update_state_after_alloc()` 准备 H2D block map
7. pre-forward H2D reload
8. worker 上报 `finished_recving`
9. scheduler 清理该请求 CPU/GPU 临时状态

证据都在官方设计和代码里：

- `PreemptedRequestState` 持有 `num_computed_tokens`、`ready`、`finished`、`load_start_tokens`、store/load metadata。[issue #10820](https://github.com/vllm-project/vllm-ascend/issues/10820) / [manager.py](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/vllm_ascend/distributed/kv_transfer/kv_pool/recompute_cpu_offload/manager.py)
- `get_num_new_matched_tokens()` 的 token 口径是：
  - 输入：当前已本地命中的 `num_computed_tokens`
  - 上界：`min(state.num_computed_tokens, request.num_tokens)`
  - 输出：还能从 offload 恢复多少 token
  - 这说明它按 token 边界做口径，再映射到 block；不是简单拿“所有 CPU blocks 都回灌”。[manager.py](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/vllm_ascend/distributed/kv_transfer/kv_pool/recompute_cpu_offload/manager.py)
- sliding-window 的处理也佐证这一点：它保留逻辑 block 位置，只搬非零 block，避免 token/block 对齐漂移。[issue #10820](https://github.com/vllm-project/vllm-ascend/issues/10820) / [guide](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/user_guide/feature_guide/recompute_cpu_offload.md)

对 xLLM 的直接含义：

- token 命中口径必须使用“稳定已完成边界”；
- block 只是传输载体；
- 任何 async/placeholder/spec decode 都必须先定义稳定边界，再谈 host reload。

## 3. 失败路径与“流式终止”

### 3.1 offload 失败

Ascend 当前官方行为很直接：offload 准备失败就回退到原 recompute。

- CPU free blocks 不够：`_create_preempt_state()` 直接返回 `False`，日志提示 skip offload。[manager.py](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/vllm_ascend/distributed/kv_transfer/kv_pool/recompute_cpu_offload/manager.py)
- 文档与 PR 都明确写了 fallback 到 original recompute behavior。[guide](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/user_guide/feature_guide/recompute_cpu_offload.md) / [PR #10742](https://github.com/vllm-project/vllm-ascend/pull/10742)

也就是说：当前主线没有把这类失败提升成用户可见 finish_reason / error contract。

### 3.2 请求在 store/load 过程中结束

当前代码已经有“流式结束/提前结束”处理，但仍是内部清理语义，不是外部 API 语义：

- 如果 request 在 store 完成前结束，`request_finished()` 会把 `state.finished=True`；等 store 真完成后再清理 CPU 状态。[manager.py](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/vllm_ascend/distributed/kv_transfer/kv_pool/recompute_cpu_offload/manager.py)
- 如果 request 在 load 完成后被 `finished_recving` 汇报，`update_connector_output()` 会进入 `_cleanup_preempt_load_request()` 清理状态。[manager.py](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/vllm_ascend/distributed/kv_transfer/kv_pool/recompute_cpu_offload/manager.py)

### 3.3 “低命中就终止/返回 router”是否已有官方定案

没有。

- #24256 提议：
  - 在 `get_num_new_matched_tokens()` 之后算 hit rate；
  - 若低于阈值，返回 `finish_reason=cache_threshold`；
  - phase 2 还想把 preempt 前已生成的 output tokens 一并返还给 router，拼回 prompt 再送 prefiller。
- 但该 RFC 已 “Closed as not planned”。[ #24256 ](https://github.com/vllm-project/vllm/issues/24256)

因此，截至 2026-08-12，官方没有收敛出一个“decode preemption / host miss / 低命中”的标准流式终止协议。

## 4. 未合入方案的主要教训

### 教训 1：不要把 offload 深嵌进核心 KVCacheManager

- #16144 的方向是改 `KVCacheManager` / `BlockPool`，在 schedule step 末尾生成 swap plan。[#16144](https://github.com/vllm-project/vllm/issues/16144)
- #13377 也是“eviction 时 swap-out、cache hit 时 swap-in”的核心内嵌式实现。[#13377](https://github.com/vllm-project/vllm/pull/13377)
- #19854 明确把这条路归纳成“像 V0”，然后转向 `OffloadingConnector` + `OffloadingManager` + worker 异步线程，尽量少改 core。[#19854](https://github.com/vllm-project/vllm/issues/19854)

对 xLLM：host KV 不应深埋进 scheduler/KV allocator 的主路径；应该是 connector / tier manager / external cache manager 形式。

### 教训 2：通用 offload 与 preemption-recovery 是两类问题

- vLLM 主线的 native offload 目标更像“扩大 prefix-cache / inactive cache 容量”，默认走 `OffloadingConnector`，`SimpleCPUOffloadConnector` 也要求 prefix caching 打开。[config](https://github.com/vllm-project/vllm/blob/main/vllm/config/vllm.py) / [SimpleCPUOffloadConnector](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py)
- Ascend #10742 处理的是“decode preemption 后不要重算长 prefill”，它把“请求私有快照恢复”放在第一位，prefix sharing 只是实验增强。[PR #10742](https://github.com/vllm-project/vllm-ascend/pull/10742) / [issue #10820](https://github.com/vllm-project/vllm-ascend/issues/10820)

对 xLLM：不要一开始就把“host prefix cache 扩容”和“preempt snapshot 恢复”混成一个特性。

补充看了一个直接相关但未合入的上游尝试：vLLM
[#43839](https://github.com/vllm-project/vllm/pull/43839) 原本准备新增
`offload_preempted_request()`，维护者指出现有 `OffloadingConnector` 已在每一步 write-through
保存完整 blocks，`handle_preemptions()` 也会 flush/wait 尚未完成的尾部传输；作者确认新接口重复
工作后关闭 PR。它给 xLLM 的教训不是“无需专用抢占语义”，而是：若底层已经持续保存 KV，抢占
路径只需要完成屏障和所有权切换，不能再发布一份重复快照；若底层是 xLLM 当前这种按需私有快照，
则仍需明确的 D2H 生命周期，但不能借 prefix-cache publication 来冒充它。

### 教训 3：API 级 rejection 方案没有共识

- #24256 想用 `cache_hit_threshold` + `finish_reason=cache_threshold`，但没被接受。[#24256](https://github.com/vllm-project/vllm/issues/24256)

对 xLLM：如果要做“低命中回退 prefiller/路由器”，最好先做内部控制面信号，不要先改用户面 OpenAI finish_reason 语义。

## 5. 对 xLLM 的建议

### 建议 A：拆成两个层次

1. `HostPreemptSnapshot`
   - 只服务 decode preemption 恢复
   - 默认不做 hash sharing
   - 键是 `request_id + stable_token_boundary + logical_block_layout`

2. `HostPrefixCache`
   - 单独服务 prefix reuse / cross-request sharing
   - 需要独立 hash、一致性、淘汰、引用计数

原因：这正是 Ascend #10742 与 vLLM 主线 native offload 的最重要边界。前者先求恢复正确，后者才追求共享与多层 tier。

### 建议 B：统一“token 命中协议”，不要统一“物理存储”

- 可以复用一个统一接口：
  - `get_num_new_matched_tokens`
  - `update_state_after_alloc`
  - `build_transfer_meta`
- 但内部要允许两种来源：
  - 私有 preempt snapshot
  - 共享 prefix cache

这与 Ascend 当前做法一致：协议统一，是否共享由 `enable_offload_prefix_caching` 控制。[manager.py](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/vllm_ascend/distributed/kv_transfer/kv_pool/recompute_cpu_offload/manager.py)

### 建议 C：先定义稳定 token 边界，再做 async

先把下面三件事钉死：

- `num_computed_tokens` 是否排除 placeholder / speculative optimistic progress
- preempt 时记录哪个边界
- resume 时 local hit 与 host hit 如何相加

在这之前，不建议上激进异步。Ascend 官方文档已经证明边界语义不稳时，Qwen3.5 + async scheduling 会出问题。[guide](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/user_guide/feature_guide/recompute_cpu_offload.md)

### 建议 D：失败先走内部 fallback，不急着暴露 finish_reason

- host 容量不足
- host miss
- load/store timeout
- router 不可达

第一阶段都建议回落到原有 recompute / resend 流程，并打内部 reason code；不要一开始就学 #24256 改外部 completion finish_reason，因为官方这条路没有定案。[#24256](https://github.com/vllm-project/vllm/issues/24256)

### 建议 E：如果未来要做共享 tier，再走 vLLM 主线 connector 思路

- 把 host/dram/ssd/remote tier 做成可插拔 manager
- 通过 worker->scheduler completion metadata 汇总完成事件
- 保持核心 scheduler/KV allocator 尽量少改

这与 #19854 / #19848 / 当前 `OffloadingConnector` 的方向一致。[#19854](https://github.com/vllm-project/vllm/issues/19854) / [#19848](https://github.com/vllm-project/vllm/pull/19848) / [OffloadingConnector](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py)

## 6. 专项复核：完整 snapshot 放不回 HBM 时，上游解决到了哪一步

截至 2026-08-12，没有在 vLLM、vLLM Ascend 或 SGLang 的官方仓库中找到一个已经落地的、面向
普通 full-attention Decode preemption snapshot 的通用 **partial/progressive H2D restore**：也就是
“先恢复部分历史 KV、先让请求运行，剩余历史以后再补”。原因不是只缺调度代码，而是普通 full
attention 的下一次 Decode 必须读取全部有效历史 KV；只装回一部分不能执行正确的 forward。上游实际
采用的是以下几类办法。

最值得参考的上游 PR/RFC 汇总如下：

| 项目 | PR/RFC | 状态 | 实际解决方式 |
|---|---|---|---|
| vLLM | [#44560](https://github.com/vllm-project/vllm/pull/44560) | 已合入 | 为先进入的 async KV load 预留完整剩余容量，阻止后来请求抢走容量 |
| vLLM | [#33526](https://github.com/vllm-project/vllm/issues/33526) | 未实施，Closed as not planned | 提议分批 progressive onload，主要解决传输队头阻塞 |
| vLLM | [#37307](https://github.com/vllm-project/vllm/pull/37307) | 已合入 | 按完整 ISL 做 admission，避免长请求只因首个 chunk 放得下就进入 |
| vLLM | [#44594](https://github.com/vllm-project/vllm/pull/44594) | 已合入 | 通过 KV watermark 预留少量空闲 blocks，减少抢占发生 |
| SGLang | [#7196](https://github.com/sgl-project/sglang/pull/7196) | 已合入 | 完整恢复 CPU snapshot，并对恢复债务做容量记账 |
| SGLang | [#29471](https://github.com/sgl-project/sglang/pull/29471) | 开放中 | 如果 GPU 仍有 radix prefix，只从 CPU snapshot 恢复缺失 suffix |
| SGLang | [#17216](https://github.com/sgl-project/sglang/pull/17216) | 已合入 | NSA/DSA 稀疏注意力在运行期逐页 offload；不适用于普通 full attention |
| vLLM Ascend | [#10742](https://github.com/vllm-project/vllm-ascend/pull/10742) | 已合入 | 在正常 waiting queue 中反复尝试完整恢复；没有 reservation/aging |
| vLLM Ascend | [#12128](https://github.com/vllm-project/vllm-ascend/pull/12128) | 开放中 | 移植 KV watermark，只减少抢占概率，不解决已 offload snapshot 的恢复饥饿 |

### 6.1 SGLang Decode retract：完整恢复，但用硬 reservation 保证最终能放回

SGLang 已合入 [PR #7196](https://github.com/sgl-project/sglang/pull/7196) 的 P/D Decode retract
与本问题最接近。抢占时，`Req.offload_kv_cache()` 把 `seqlen - 1` 范围的请求 KV 复制到 CPU；恢复
时先重新分配完整 token footprint，再由 `load_kv_cache()` 一次装回相同范围，因此它同样不是分块
恢复。[snapshot copy](https://github.com/sgl-project/sglang/blob/197832bcf536543092e621e03d61ae2602a392d0/python/sglang/srt/managers/schedule_batch.py#L1716-L1733)

它真正解决饥饿的方式是 **恢复债务记账和新 admission 限流**：

1. `resume_retracted_reqs()` 每轮先扫描 `retracted_queue`，只有
   `full_required + reserved decode growth` 能完整容纳时才执行 H2D；队首放不下就停止，不做 partial
   load。[restore admission](https://github.com/sgl-project/sglang/blob/197832bcf536543092e621e03d61ae2602a392d0/python/sglang/srt/disaggregation/decode.py#L704-L752)
2. 计算新请求的 allocatable budget 时，会扣除所有 retracted request 的恢复 footprint，防止后来者
   持续吃掉这部分容量。[capacity accounting](https://github.com/sgl-project/sglang/blob/197832bcf536543092e621e03d61ae2602a392d0/python/sglang/srt/disaggregation/decode.py#L1405-L1459)
3. 只要还有 retracted request 未恢复，`process_decode_queue()` 就不再给新的 P/D 请求做
   preallocation/transfer admission。[admission gate](https://github.com/sgl-project/sglang/blob/197832bcf536543092e621e03d61ae2602a392d0/python/sglang/srt/disaggregation/decode.py#L2350-L2362)

这比“有任何 running work 就禁止恢复”强得多：已有 running 请求仍可继续并逐渐释放空间，同时新请求
不能无限抢占恢复容量，所以队首 snapshot 最终能取得进展。代价是强 head-of-line blocking：一个很大
的 snapshot 放不回时，新请求 admission 会整体暂停。因此它是 **有界进展策略**，不是吞吐最优或
partial restore 策略。

SGLang 的通用 HiCache `load_back()` 也在代码中明确写着 `load it all or not at all`：待加载节点的
Host indices 总量超过本轮 `mem_quota` 时，整段 load 被跳过，只退回到已有 device ancestor，未实现
把同一段 Host KV 分多轮逐步装回。[HiRadixCache load_back](https://github.com/sgl-project/sglang/blob/dc5f6c488317645d96dc630b1f410e4dfb6f9667/python/sglang/srt/mem_cache/hiradix_cache.py#L1368-L1459)
开放的 [PR #29471](https://github.com/sgl-project/sglang/pull/29471) 提供了一个有条件的优化：若 Decode
radix tree 中仍有该请求的 device-resident prefix，就只从 CPU snapshot 恢复缺失 suffix；若没有本地
prefix，仍走完整恢复。它是“利用已有 prefix 缩小 snapshot”，不是任意 snapshot 的渐进恢复，且截至
本文日期尚未合入。

### 6.2 SGLang 页级动态 offload：对稀疏注意力有效，但不是 full-attention snapshot 恢复

SGLang 已合入 [PR #17216](https://github.com/sgl-project/sglang/pull/17216)，把 Decode KV 从“请求
结束后整体 offload”改成“运行期间按 page/stride 持续 D2H，并及时释放旧 GPU pages”。在其
DeepSeek-V3.2 NSA 压测中，并发和端到端吞吐约提升一倍。这个方案从源头缩小 active working set，
所以不会等到抢占后再为完整 snapshot 找一大块 HBM。

但 PR 讨论明确把成立条件归因于 DSA/NSA：每步只选择少量相关历史 KV，作者认为很早的历史 page
很少再被选中；同时也承认会增加 H2D/D2H。PR 合入后另有官方讨论参与者报告 GLM5 page offload
持续慢于 baseline。因而它应被理解为 **稀疏/分层注意力的工作集迁移**，不是普通 full-attention
请求的通用渐进恢复。对 GLM-5.2 full-attention 路径，直接释放仍会被下一 Decode 所需的历史 KV
会破坏正确性，除非注意力内核本身支持按需 paging/load。

### 6.3 vLLM：有直接针对 HOL/deadlock 的方案，但没有通用 partial restore

vLLM 对这个问题的识别最直接。官方 [RFC #33526](https://github.com/vllm-project/vllm/issues/33526)
明确指出：当前 CPU onload 按整个 request 发起，会造成 head-of-line blocking；它提议把 load 切成
`BatchTransferSpec`，前 100 个 block 每 10 个一批，后续每 100 个一批，并逐批向 scheduler 上报完成。
但该 RFC 没有关联实现 PR，已于 2026-06-13 因长期无进展关闭为 `not planned`。维护者还指出，逐批完成
需要 connector API 和 vLLM core 都新增“已完成 batch 进入 GPU prefix cache”的能力；提交者的 L4
实验也发现 scheduler↔worker 的每请求延迟明显大于纯 H2D block copy，因此没有证明该复杂度值得投入。

更重要的是，#33526 解决的是“一个长 onload 阻塞共享其部分 prefix 的短请求”，不是“普通
full-attention snapshot 只恢复一半就开始 Decode”。即使传输分批，单个请求 forward 前仍需全部有效
历史 KV 驻留；所以它能降低 transfer HOL，却不消除完整 snapshot 的 HBM footprint 门槛。

vLLM 最接近可复用、且已经合入的办法是容量 reservation：

- 已合入 [PR #44560](https://github.com/vllm-project/vllm/pull/44560) 修复多个 async KV load
  的 deadlock。scheduler 跟踪正在 async load/chunked prefill 的请求还差多少 blocks 才能容纳完整
  sequence；新的 async load 只能使用 `free_blocks - in_flight_reserved_blocks`。这样先进入的请求 A
  可以保住完成所需容量，后来的 B 留在 `WAITING` 且不持有 blocks，不会形成两个半加载请求互相卡死。
  这是 **先到请求的完整恢复 reservation**，不是 partial H2D。
- 已合入 [PR #37307](https://github.com/vllm-project/vllm/pull/37307) 增加 full-ISL admission，避免只因
  第一个 chunk 放得下就接纳超长请求，随后反复 preempt/re-prefill；当前 `main` 的
  `scheduler_reserve_full_isl` 默认开启。[scheduler config](https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py)
- 已合入 [PR #44594](https://github.com/vllm-project/vllm/pull/44594) 增加可配置 KV watermark，在新分配
  时保留一小部分 blocks。其压测报告在 5% watermark 下 preemption 减少 82%、P99 ITL 降低 56%；
  这是减少问题发生概率，不负责恢复已经 offloaded 的大 snapshot。
- 未合入 [PR #40968](https://github.com/vllm-project/vllm/pull/40968) 曾提出 lateral preemption：当
  `WAITING_FOR_REMOTE_KVS` 等非 running block-holder 形成 wedge 时，抢占其中一个以让另一个完成。
  最终主线选择了 #44560 的预留式 admission，避免先制造 wedge 再抢占。

这组方案的工程含义很清楚：vLLM 主线选择的是 **完整 footprint 预留 + 限制后来者 admission**，
而不是 partial restore。它能防死锁和可预测的反复抢占，但没有给任意 offloaded request 提供
aging/deadline 或超时转 recompute；若队首 snapshot 本身大于可用/可恢复容量，仍需上层拒绝、重算或
重新路由。

### 6.4 vLLM Ascend：正常 waiting admission，但缺少 reservation/aging

- vLLM Ascend [PR #10742](https://github.com/vllm-project/vllm-ascend/pull/10742) 每轮把 ready 的
  offloaded request 放在正常 waiting admission 中尝试，不要求 running 清空；但普通 full-attention
  仍需一次分配全部 external/restorable blocks。RFC [#10820](https://github.com/vllm-project/vllm-ascend/issues/10820)
  没有 aging、deadline、恢复容量 reservation 或 partial restore 设计。
- vLLM 的 [RFC #16144](https://github.com/vllm-project/vllm/issues/16144) 和未合入
  [PR #13377](https://github.com/vllm-project/vllm/pull/13377) 提过逐 scheduler step 生成 block-level
  swap plan；这能让传输按 block 组织，但 full-attention forward 前仍须具备全部有效历史 blocks，
  因此没有消除完整 resident footprint。
- vLLM 当前开放的 [issue #41951](https://github.com/vllm-project/vllm/issues/41951) /
  [PR #41952](https://github.com/vllm-project/vllm/pull/41952) 给 priority queue 增加
  `num_preemptions` tie-breaker，让被抢占请求优先于同优先级的新请求。它解决队列饥饿，但不预留 HBM，
  也不是 offload restore；FCFS 原本就用 `appendleft` 实现相近的重入优先级。
- vLLM [RFC #24256](https://github.com/vllm-project/vllm/issues/24256) 提出的低 cache-hit 时显式返回
  router/Prefill，是“放不回或不值得恢复时退出 Decode”的 fallback 方向，但已 `Closed as not
  planned`，不是主线已有能力。Ascend 开放的 [PR #14062](https://github.com/vllm-project/vllm-ascend/pull/14062)
  也仍在讨论 offload 不可用时应 abort/resend 还是 Decode 本地 recompute。

Ascend 仓库另有开放的 [PR #12128](https://github.com/vllm-project/vllm-ascend/pull/12128)，准备移植
vLLM #44594 的 KV watermark，但描述仍标注测试待补；它只能降低抢占概率，不解决已经 ready 的完整
snapshot 如何保证恢复。当前没有找到 Ascend 移植 vLLM #44560 reservation，或实现 restore
aging/deadline、partial H2D 的 PR。

### 6.5 对 xLLM 的收敛建议

第一版不应把通用 chunked H2D 当作解法，因为它只能分摊 copy 时间，不能降低 full-attention 开始
Decode 前最终必须驻留的 HBM footprint。更可行的是组合 Ascend 的“每轮尝试”与 SGLang 的“恢复
债务记账”：

1. ready snapshot 进入正常 FIFO/aging admission；running 本轮增量先分配，但不再以
   `has_scheduled_work` 为恢复门禁。
2. 为队首 snapshot 逐轮累积 `restore_credit_blocks`，并在新请求 admission budget 中扣除；credit
   达到完整 snapshot 加一个 Decode quantum/lookahead 后启动一次 H2D。
3. credit 应有上限和单 snapshot 所有权，避免多个半恢复请求占住 HBM；未完整分配前不启动 partial
   H2D，从而避免“装回一半后永久等待”的新死锁。
4. 配置 `max_offloaded_wait_ms`。超过软期限后停止接纳新请求；超过硬期限且 credit 仍不足时，可定向
   抢占足够数量的低优先级 running victims，只为队首 snapshot 补齐容量，恢复后设置最小驻留期，避免
   立即反向抢占形成 ping-pong。若 snapshot 超过物理可恢复容量、传输失败，或策略不允许再抢占，则
   释放 Host snapshot，明确转 chunked recompute / router retry。
5. 对 NSA/HiSparse 等确实支持按需历史工作集的模型，可另立 capability 后借鉴 SGLang #17216 做
   page-level active offload；不能把这条模型相关路径作为普通 KV 的默认实现。

因此，上游给出的最可靠工程答案不是“已经有通用 partial restore”，而是：**普通 attention 保持完整
恢复；通过 reservation、停止新 admission、aging/deadline 和 fallback 保证有界进展；只有稀疏注意力
才考虑真正的页级工作集迁移。**

## 最终判断

如果 xLLM 现在要落地，我建议：

- 短期：照 Ascend #10742 做“私有 preempt snapshot 恢复”，默认关闭 prefix sharing。
- 中期：把共享 host prefix cache 单独建模成第二层，不和 preempt snapshot 共用生命周期。
- 长期：若要多 tier / 异步隐藏时延 / 统一事件流，再靠近 vLLM `OffloadingConnector` 路线，而不是回到 #16144 / #13377 那种深改 core 的 swap-space 方案。

这也是本次官方来源复核后最稳的结论。
