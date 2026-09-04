# vLLM 与 SGLang 的 KDA cache / prefill cache 实现调研

- 日期：2026-08-28
- 范围：Kimi Delta Attention（KDA）的运行态 cache、chunked prefill、跨请求 prefix cache，以及 KDA/MLA 混合模型的 cache 管理。
- 来源：仅使用 Kimi Linear 原始论文，以及 vLLM、SGLang 官方 GitHub 主干源码。本文把“prefill 结束后供 decode 使用的运行态”与“跨请求复用的 prefix checkpoint”分开讨论。

## 调研快照

- Kimi Linear：[`8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3`](https://github.com/MoonshotAI/Kimi-Linear/commit/8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3)，论文 [arXiv:2510.26692v2](https://arxiv.org/abs/2510.26692v2)。
- vLLM `main`：[`53bf99050213da20fe29bb6dbb49eed5fad813dc`](https://github.com/vllm-project/vllm/commit/53bf99050213da20fe29bb6dbb49eed5fad813dc)。
- SGLang `main`：[`6480cce9bc7bb83e18537be3995a094cf2426e74`](https://github.com/sgl-project/sglang/commit/6480cce9bc7bb83e18537be3995a094cf2426e74)。

## 结论先行

KDA **不保存逐 token 的 K/V**。每个 KDA 层、每个本地 head 只保存两部分：

1. 固定大小的递归矩阵 state，逻辑上每 head 为 `d_k × d_v`；vLLM/SGLang 的 kernel-facing 存储通常按 `[H_local, d_v, d_k]` 理解，Kimi K3 中 `d_k=d_v=128`，所以源码 shape 都表现为 `[H_local, 128, 128]`；
2. ShortConv 的最近 `kernel_size - 1` 个投影输入，即 Q/K/V 合并后的卷积窗口。

因此，单个**运行态**的大小与上下文长度无关：

```text
state_per_layer_per_slot
  = conv_tail[(Kconv - 1), (Q + K + V channels) / TP]
  + recurrent[H / TP, d_v, d_k]
```

按 Kimi 的 `d_k=d_v=128` 计算，FP32 recurrent matrix 单 head 为 `128*128*4 = 64 KiB`；因此 state 内存的主项通常是 recurrent matrix，ShortConv tail 相对较小。完整 checkpoint 的字节数还要乘本 rank 的本地 head 数和 KDA layer 数。

原始论文明确把 prefill 定义为 chunk kernel、decode 定义为逐 token recurrent kernel，并说明每 head 的固定 state 为 `d_k × d_v`。[KDA 递推 Eq. (1)](https://arxiv.org/html/2510.26692v2#S3.E1) [推理策略与固定 state](https://arxiv.org/html/2510.26692v2#S6.SS3)

“prefill cache”有两层含义：

- **Prefill → decode handoff**：prefill chunk kernel 从初始 state 出发，输出最后一个 chunk 的 terminal state；卷积 kernel 同时把最后 `Kconv-1` 个原始 QKV 投影写回 conv tail。这一份 active state 随后被 decode 原地更新。
- **跨请求 prefix cache**：额外在可命中的 token 边界保存一整份 `{conv tail, recurrent state}` checkpoint。命中时必须同时恢复两者，并让 full-attention/MLA 的 KV 命中长度不超过这个 checkpoint 边界。

两套框架的核心差异是 checkpoint 所有权：

| 维度 | vLLM | SGLang |
|---|---|---|
| Active state | 沿用统一 block/page 抽象；一个 Mamba block 的 payload 是完整 state | `MambaPool` 中一个固定 slot；请求映射到 slot |
| 默认 recurrent dtype | KDA 强制 FP32 | 默认 FP32，可由 config/env 改为 BF16/FP16 |
| 默认 conv dtype | `mamba_cache_dtype`，通常模型 dtype | BF16，可由环境变量修改 |
| Prefill kernel | Triton；满足约束时自动 FlashKDA | Triton/Helion/FlashKDA/CuTeDSL/NVIDIA KDA/PTX KDA 等 dispatcher |
| Chunked prefill | `align` 模式由 scheduler 强制在 state boundary 切分；FlashKDA 可在一次调用中额外吐一个内部 checkpoint | active slot 跨 chunk 复用；`extra_buffer` 在 chunk/page 网格上把中间或 final state scatter 到 ping-pong track slot |
| Prefix checkpoint 索引 | block hash + `MambaManager`，`align` 默认仅保留滚动态与必要边界 | `MambaRadixCache` 节点的 `mamba_value` 指向完整 state slot |
| Prefix hit restore | block/page copy，随后 kernel 从选中的 state slot 续算 | 从 radix checkpoint COW 到新 active slot；可选 int8 recurrent checkpoint |
| Hybrid KDA + MLA | Mamba group 和 attention group 分别管理，调度粒度取各 group block size 的 LCM | MLA token pool 与 Mamba state pool 分开；可选 unified-memory 把两侧放进一个 byte buffer |

## 1. KDA 为什么只需要固定 state

KDA 的递推为：

```text
S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_(t-1)
      + beta_t k_t v_t^T
o_t = S_t^T q_t
```

论文把 `S_t` 定义为 `R^(d_k × d_v)`；历史 token 的键值关联已压入 `S_t`，decode 不再读取逐 token K/V。[论文记号](https://arxiv.org/html/2510.26692v2#S2.SS1) [KDA 递推](https://arxiv.org/html/2510.26692v2#S3.E1)

Prefill 为了使用 Tensor Core，不逐 token 跑这条递推，而是把序列切成 chunk：chunk 间串行传递 `S`，chunk 内通过 WY/UT 形式并行计算；块末用 Eq. (8) 得到下一块初态。[chunk state 更新 Eq. (8)](https://arxiv.org/html/2510.26692v2#S3.E8) [chunk 输出 Eq. (9)](https://arxiv.org/html/2510.26692v2#S3.E9)

Kimi K3 的 Q/K/V 在 KDA 前还有 causal ShortConv，所以完整的可恢复状态不能只有 `S`。卷积参数化见论文 §4；Q/K/V 都经过 ShortConv，`d_k=d_v=128`。[模型参数化](https://arxiv.org/html/2510.26692v2#S4)

## 2. vLLM 实现

### 2.1 Cache shape、dtype 与物理布局

`MambaStateShapeCalculator.kda_state_shape()` 直接给出：

```text
conv_dim       = num_heads * head_dim + 2 * num_k_heads * head_k_dim
conv_state     = [Kconv - 1 + num_spec, conv_dim / TP]  # 默认 SD 布局
recurrent      = [num_heads / TP, head_dim, head_dim]
```

默认 conv 布局为 state-major `SD = [state_len, dim]`，也可通过 `VLLM_SSM_CONV_STATE_LAYOUT` 切到 `DS = [dim, state_len]`。[conv layout](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/model_executor/layers/mamba/mamba_utils.py#L23-L48) [KDA shape](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/model_executor/layers/mamba/mamba_utils.py#L281-L304)

KDA 的 dtype 是一个重要特例：`conv_state` 使用 `mamba_cache_dtype` 解析出的 dtype，`recurrent_state` **固定为 FP32**；通用的 `mamba_ssm_cache_dtype` 文案不能用来推断 KDA state dtype。[`kda_state_dtype()`](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/model_executor/layers/mamba/mamba_utils.py#L131-L137)

物理上，vLLM 先分配一个 page-aligned、清零的 `int8` backing buffer，再为各 layer 创建 `[B,H,N,C]` byte view。Mamba layer 收到的逻辑 page 是 `[B,1,1,C]`；`bind_kv_cache()` 按各 state 的 byte size 切片、reinterpret dtype，再 reshape 为 `[num_blocks, *state_shape]`。[统一 backing allocation](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/v1/worker/utils.py#L378-L429) [view 构造](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/v1/kv_cache_interface.py#L307-L371) [Mamba page 解包](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/model_executor/layers/mamba/abstract.py#L29-L43)

`MambaSpec.page_size_bytes` 等于一份完整 state 的总字节数；在 hybrid cache 中 page 还可能 padding 到公共物理 page size，但 block size 的变化不会改变 Mamba state payload。[`MambaSpec`](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/v1/kv_cache_interface.py#L811-L850) [Mamba page padding](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/v1/core/kv_cache_utils.py#L1057-L1100)

### 2.2 Prefill 如何生成 active cache

Kimi K3 的 `KimiK3DeltaAttention` 先从 `kv_cache` 解出 `conv_state` 和 `recurrent_state`。prefill 路径的完整数据流是：

```text
mixed_qkv
  -> split Q/K/V
  -> causal_conv1d_fn(Q/K/V,
       conv_states=对应 slot 的 conv tail,
       has_initial_state=(num_computed_tokens > 0))
  -> gather_initial_states(recurrent_state, state_indices, has_initial_state)
  -> FlashKDA 或 Triton chunk_kda(initial_state)
  -> last_recurrent_state
  -> recurrent_state[state_indices] = last_recurrent_state
```

对应源码在 [`KimiK3DeltaAttention.forward()` prefill 分支](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/models/kimi_k3/nvidia/kda.py#L821-L972)。`has_initial_state` 来自每个请求已有的 `num_computed_tokens > 0`，因此同一请求的后续 prefill chunk 会读取上一 chunk 的 cache，而首 chunk 从清零 state 开始。[metadata 构造](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/models/kimi_k3/nvidia/kda_metadata.py#L565-L580)

卷积 cache 不是在 KDA chunk kernel 中产生，而是 `causal_conv1d_fn` 在计算 Q/K/V 时更新对应 slot 的窗口；KDA kernel 只消费卷积后的 Q/K/V，并从 initial recurrent state 计算 final recurrent state。

Decode 则用同一 slot：`causal_conv1d_update()` 原地推进 conv tail，再由 packed recurrent KDA kernel 原地推进矩阵 state。[普通 decode 分支](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/models/kimi_k3/nvidia/kda.py#L974-L1002)

### 2.3 Chunked prefill 与 prefix checkpoint

vLLM 有三种 Mamba cache mode：

- `none`：prefix caching 关闭，只保留运行态；
- `all`：每个 `i * mamba_block_size` 边界缓存一份完整 state，单请求最坏为 `ceil(max_model_len / block_size)` 份；
- `align`：只保留滚动态、spec slot 和必要 checkpoint；启用 prefix caching 时的默认模式。

配置定义及默认切换见 [`CacheConfig.mamba_cache_mode`](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/config/cache.py#L176-L196) 与 [`MambaModelConfig`](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/model_executor/models/config.py#L609-L657)。`align` 强制要求 chunked prefill。

`align` 的内存上界是每请求 `2 + num_speculative_blocks + num_prefill_checkpoint_blocks` 份完整 state，而不是随序列长度分配 state；block table 仍按位置展开，但旧位置会被置为 null。[`MambaSpec.max_memory_usage_bytes()`](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/v1/kv_cache_interface.py#L841-L864) [`remove_skipped_blocks()`](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/v1/core/single_type_kv_cache_manager.py#L1435-L1463)

普通 Triton KDA 不返回 chunk 内任意边界 state，所以 scheduler 会把 prefill clip 到 Mamba block boundary。代码明确维护不变量：逻辑 slot `p` 保存处理完 `(p+1)*block_size` 个 token 后的 state；若 chunk 从 block 中间开始，就不能越过下一个边界。[`_mamba_block_aligned_split()`](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/v1/core/sched/scheduler.py#L385-L465)

FlashKDA 是优化特例：

- 仅在 CUDA SM90/SM10x/SM12x、BF16、`head_dim=128`、bounded gate 时可用；否则回退 Triton。[backend 选择](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/models/kimi_k3/nvidia/kda.py#L180-L190) [`resolve_kda_prefill_backend()`](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/models/kimi_k3/nvidia/kda.py#L300-L317)
- 它声明一个额外 `num_prefill_checkpoint_blocks=1`，所以从 block boundary 起跑的 prefill 可以跨过一个 cache boundary，而不必在 scheduler 层切成两次调用。[KDA cache spec](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/models/kimi_k3/nvidia/kda.py#L594-L600)
- metadata 只在 checkpoint offset 位于当前 chunk 内、且满足 FlashKDA 内部 16-token chunk 对齐时请求中间 state。[checkpoint metadata](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/models/kimi_k3/nvidia/kda_metadata.py#L582-L636)
- FlashKDA 同时输出 final state 和 checkpoint state；`_store_cache_checkpoints_kernel` 从 pre-conv `mixed_qkv` 取边界前最后 `Kconv-1` 行写 conv tail，并把对应 recurrent checkpoint 写入目标物理 slot。这样 `{conv, recurrent}` 在同一 token 边界一致。[checkpoint store kernel](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/models/kimi_k3/nvidia/kda.py#L237-L297) [prefill 调用与写回](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/models/kimi_k3/nvidia/kda.py#L864-L953)

Prefix hit 时，`MambaManager.find_longest_cache_hit()` 只需找到最深的单个 state block，因为该矩阵已汇总此前全部 KDA 历史；hybrid coordinator 再与 full-attention group 求共同可恢复长度。[Mamba prefix lookup](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/v1/core/single_type_kv_cache_manager.py#L1299-L1375)

### 2.4 Hybrid KDA + MLA

vLLM 为 KDA 和 MLA/full attention 建立不同 cache group。attention group 的 KV 随 token 增长并可按 DCP 分片；Mamba/KDA state 不按 DCP/PCP 切分，且调度对多 group 取有效 block size 的 LCM。TP 仍会把 KDA heads 分给各 rank，所以 state shape 是 `H/TP`。[hybrid block-size 解析](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/v1/core/kv_cache_utils.py#L650-L711) [Mamba manager 的 state 语义](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/v1/core/single_type_kv_cache_manager.py#L1269-L1286)

这意味着 Kimi 的整体 cache 仍会随上下文增长，只是只有周期性的 MLA 层增长；论文的结构是 3 个 KDA 层配 1 个 MLA 层，并称 KV cache 最多减少 75%，而不是完全消除 KV cache。[hybrid 架构](https://arxiv.org/html/2510.26692v2#S4)

## 3. SGLang 实现

### 3.1 Cache shape、dtype 与 pool

`KimiLinearStateShape.create()` 给出与 vLLM 等价的 shape：

```text
conv_state = [Kconv - 1,
              (num_v_heads*d_v + 2*num_k_heads*d_k) / attn_TP]
temporal   = [num_v_heads / attn_TP, d_v, d_k]
```

源码将 KDA conv 固定为 `[K-1, sharded_channels]`，`conv_slice_axis=1`。[KDA state shape](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/configs/mamba_utils.py#L246-L314)

默认 `conv_dtype=BF16`、`temporal_dtype=FP32`。与 vLLM 不同，SGLang 允许通过模型 config 或 `SGLANG_MAMBA_SSM_DTYPE` 把 temporal state 改成 BF16/FP16，也允许 `SGLANG_MAMBA_CONV_DTYPE` 改 conv dtype。[state dtype 解析](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/configs/mamba_utils.py#L47-L107)

默认 `MambaPool` 直接分配：

```text
conv:     [num_kda_layers, num_slots + 1, K-1, sharded_channels]
temporal: [num_kda_layers, num_slots + 1, H/TP, d_v, d_k]
```

`num_slots` 是并发 active/checkpoint state slot 数，不是 token 数；`+1` 是保留 padding slot。[`MambaPool.State`](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/mem_cache/memory_pool.py#L377-L435) [pool allocation](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/mem_cache/memory_pool.py#L505-L619)

请求到 active state 的映射由 `HybridReqToTokenPool` 维护；继续做 chunked prefill 的请求复用已有 `req.mamba_pool_idx`，新请求才分配并清空 slot。[active slot allocation](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/mem_cache/memory_pool.py#L1355-L1405)

可选 `--enable-page-major-kv-layout` 时，所有 KDA layer/slot 可共享一个 `uint8` backing；每 slot 的 envelope 是：

```text
[conv[0] × all KDA layers][conv[1] × all layers]...[temporal × all layers]
```

各 layer tensor 是带 slot stride 的 view，而不是复制。[Mamba page-major builder](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/mem_cache/layout/page_major.py#L179-L212) [view stride](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/mem_cache/layout/page_major.py#L229-L297)

### 3.2 Prefill 如何生成 active cache

SGLang 的 `KDAAttnBackend.forward_extend()` 与 vLLM 的逻辑一致：

```text
cache_indices = req -> physical mamba slot
has_initial_state = extend_prefix_lens > 0
Q/K/V causal_conv1d_fn(... active conv slot ...)
KDAKernelDispatcher.extend(... active temporal slot ...)
kernel 原地提交 final temporal state
```

三个 conv 调用读取/更新同一个请求 slot 的 Q/K/V 窗口；chunk KDA 接收 `ssm_states`、`cache_indices` 和 ragged `query_start_loc`。源码见 [`KDAAttnBackend.forward_extend()`](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/layers/attention/linear/kda_backend.py#L613-L735)。

Decode 同样先 `causal_conv1d_update()`，再对 `ssm_states[cache_indices]` 做 packed/fused recurrent update，全部原地更新。[decode 路径](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/layers/attention/linear/kda_backend.py#L451-L611)

SGLang 的 prefill backend 选择面比 vLLM 更宽：Triton、Helion、FlashKDA、CuTeDSL、NVIDIA KDA、PTX KDA；不同 GPU/backend 不满足能力门槛时回退 Triton。部分 backend 不能返回 prefix tracking 所需的中间 chunk state，因此开启 `extra_buffer` 时也可能回退。[KDA dispatcher](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/layers/attention/linear/kda_backend.py#L40-L193)

一个值得单独指出的实现差异是：vLLM 的 FlashKDA wrapper 能按指定 offset 额外返回一份 intermediate checkpoint；SGLang 当前 FlashKDA wrapper 不能返回 per-chunk `h`，所以 `extra_buffer` 请求中间 checkpoint 时会整 batch 回退 Triton。SGLang FlashKDA 还只覆盖 SM90+、BF16、`d_k=d_v=128`、bounded gate，以及每请求长度 64–2048 的 prefill。[SGLang FlashKDA fallback 与约束](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/layers/attention/linear/kernels/kda_flashkda.py#L71-L193)

### 3.3 Chunked prefill 与 Radix prefix cache

SGLang 将 active state 与 radix checkpoint 的所有权明确分开：

- `MambaPool` 的 active slot 供正在运行的请求原地更新；
- `MambaRadixCache.TreeNode.mamba_value` 只保存一个 state slot id，代表该节点 token prefix 结束后的完整 `{conv, temporal}` checkpoint；
- 命中后为请求分配新的 active slot，把 checkpoint COW 到 active slot，避免后续 decode 改写共享 checkpoint。[radix node 与 cache 类](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/mem_cache/mamba_radix_cache.py#L444-L528) [prefix-hit COW](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/mem_cache/mamba_radix_cache.py#L1138-L1207) [forward-stream restore](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/model_executor/model_runner.py#L1655-L1701)

Radix tree 若在一个已缓存 node 内部发生 split，新中间 node 的 `mamba_value` 会显式设为 `None`，因为更深 prefix 的 recurrent state 无法逆推出任意更短 prefix 的 state。匹配因此必须回退到最深且真实拥有 checkpoint 的节点。[`_split_node()`](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/mem_cache/mamba_radix_cache.py#L1209-L1225)

它支持两类策略：

#### `no_buffer`

仅适用于 `page_size=1`。chunked prefill handoff 时复制或 donate 当前 active slot 给 radix tree，再为请求准备可继续运行的 slot；这样 checkpoint 一定是 chunk 末 terminal state，但没有额外的块内 tracking buffer。[page-size 约束](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/mem_cache/mamba_radix_cache.py#L444-L470) [`cache_unfinished_req()`](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/mem_cache/mamba_radix_cache.py#L675-L804)

#### `extra_buffer` / `extra_buffer_lazy`

每请求维护 ping-pong track slots。prefill metadata 在 checkpoint grid 上选一个边界：

```text
cache_chunk_size = max(model mamba_chunk_size, page_size)
checkpoint_grid  = lcm(cache_chunk_size, actual radix page_size)
```

若一次 extend 至少跨过一个 grid，scheduler 设置 `mamba_track_mask/index/seqlen`。[grid 定义](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/runtime_context.py#L1633-L1652) [extend tracking plan](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/managers/schedule_batch.py#L2688-L2780)

在 KDA forward 中：

1. 边界 conv tail 直接从 pre-conv `mixed_qkv` 的最后 `Kconv-1` 行拷到 track slot；
2. 若边界正好是 KDA kernel 的最后 state，则从 active final state 拷贝；
3. 若边界位于 extend 内部，则 kernel 返回 per-chunk intermediate `h`，再 scatter 到 track slot。

对应的 conv/recurrent source-index 计算见 [`_init_track_conv_indices()` / `_init_track_ssm_indices()`](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#L310-L379)，实际写入见 [`forward_extend()`](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/layers/attention/linear/kda_backend.py#L642-L733) 与 [`_track_mamba_state_extend()`](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#L820-L841)。

Decode 则每到 `lcm(checkpoint_grid, mamba_track_interval)` 边界，把当前 active state 复制到交替的 track slot；默认 `mamba_track_interval=256`。[decode tracking plan](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/managers/schedule_batch.py#L3222-L3252) [all-layer state tracking](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#L776-L818)

`extra_buffer` 的优势是一个大 prefill chunk 可同时产生内部 checkpoint，无需为了 prefix cache 把模型执行拆到每个边界；代价是每请求需要更多完整 state slot。若 `chunked_prefill_size < mamba_cache_chunk_size`，官方代码会警告 unfinished chunk 可能无法生成 checkpoint。[参数与告警](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/server_args.py#L6316-L6351)

请求结束或 unfinished chunk handoff 时，最深的已 tracking 边界被插入 radix tree；节点只需要最深 state，因其已累计所有更早历史。Radix 的 full-KV 路径可复用整条 prefix，但 KDA 只消费 `last_node.mamba_value`。[finished checkpoint commit](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/mem_cache/mamba_radix_cache.py#L546-L673) [只触碰最深 KDA state](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/mem_cache/mamba_radix_cache.py#L1149-L1158)

SGLang 还支持可选 int8 prefix checkpoint：只把 radix-owned recurrent state 量化为 int8，scale 为 `[L,slot,H,1,d_k]`，conv tail 保持原 dtype；命中时一次性反量化到 full-precision active slot，后续 recurrence 不在量化 state 上运行。[`Int8CheckpointStore`](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/mem_cache/mamba_checkpoint_pool.py#L42-L151) [active/checkpoint transfer](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/mem_cache/mamba_checkpoint_pool.py#L200-L259)

### 3.4 Hybrid KDA + MLA

Kimi model config 直接把 layer id 分为 `linear_layer_ids` 与 `full_attention_layer_ids`。[Kimi layer classification](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/configs/kimi_linear.py#L145-L180)

默认内存结构是：

```text
full/MLA layers -> token-indexed KV/latent pool
KDA layers      -> request/checkpoint-indexed MambaPool
```

两侧容量由 `mamba_full_memory_ratio` 协调；因此 KDA state pool 限制的是同时 active/cached 的请求或 checkpoint 数，MLA pool 限制的是 token 数。

可选 unified-memory 路径把 MLA/full 子池从高地址向下增长、Mamba 子池从低地址向上增长，共用一个 raw buffer；但 KDA state 的寻址仍是 per-request slot、`page_size=1`，不是把 KDA state 复制到每个 MLA token page。[unified hybrid pool](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/mem_cache/unified_memory_pool.py#L1100-L1177) [Mamba slot view](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/mem_cache/unified_memory_pool.py#L711-L824)

## 4. 两者关键差异与限制

### 4.1 相同的不变量

无论采用哪套框架，正确实现必须满足：

1. **同边界一致性**：checkpoint 的 conv tail 与 recurrent matrix 必须对应同一个已处理 token 数；只存 `S` 或两者错一 token 都会使恢复后的输出错误。
2. **chunk continuation**：后续 prefill chunk 必须把前一 chunk terminal state 作为 initial state；不能因为 forward mode 仍是 prefill 就清零。
3. **mixed attention 共同命中**：hybrid KDA+MLA 的可复用 prefix 长度是各 cache 类型可恢复边界的交集；MLA KV 命中比 KDA checkpoint 深时必须回退。
4. **共享 checkpoint 不可原地 decode**：跨请求命中的 checkpoint 必须 COW/复制到 private active state，再继续更新。

### 4.2 vLLM 的取舍

- 优点：KDA state 完全进入通用 block/page、hash、hybrid KV manager；`align` 每请求常驻 state 数小，和现有 scheduler 结合紧密。
- 代价：普通 backend 为了产生边界 checkpoint，需要 scheduler 切分 prefill；只有 FlashKDA 可在一次 prefill 中吐一个内部 checkpoint。
- `all` 模式并非逐 token KV，而是每个 Mamba block boundary 一份完整 recurrent snapshot；上下文很长时仍可能显著占内存。
- Kimi K3 的生产 recurrent cache 固定 FP32，不能直接依赖通用 SSM dtype flag 压缩。
- 当前源码中 `MambaManager.find_longest_cache_hit()` 明确断言 DCP/PCP 为 1；state 本身在 context-parallel ranks 上采用复制语义。[限制](https://github.com/vllm-project/vllm/blob/53bf99050213da20fe29bb6dbb49eed5fad813dc/vllm/v1/core/single_type_kv_cache_manager.py#L1311-L1321)

### 4.3 SGLang 的取舍

- 优点：active slot、radix checkpoint ownership 很清楚；`extra_buffer` 能在一个大 prefill 内抓取中间 checkpoint；可选 int8 只压缩 idle prefix checkpoint。
- 代价：ping-pong/track slot 增加每请求固定内存，调度 metadata 和 speculative/overlap 状态机更复杂。
- `no_buffer` 要求 radix page size 为 1，而且不支持 overlap scheduling；page size 大于 1 或需要 overlap 时应走 extra-buffer 路径。[`no_buffer` 校验](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/server_args.py#L6307-L6314)
- 部分高性能 prefill backend 无法返回任意中间 `h`，prefix tracking 时会回退 Triton或受限制。
- `extra_buffer_lazy` 不支持 PD disaggregation；相关校验在 server args 中显式存在。[限制](https://github.com/sgl-project/sglang/blob/6480cce9bc7bb83e18537be3995a094cf2426e74/python/sglang/srt/server_args.py#L6316-L6330)
- 将 SSM state 改成 BF16/FP16 可省内存，但精度风险由部署者承担；vLLM KDA 当前选择固定 FP32。

## 5. 映射到 xLLM

xLLM 当前的 LINEAR cache 基础设施在所有权模型上更接近 SGLang：**正在执行的 live slot** 与 **prefix-cache-owned checkpoint slot** 分离，但两者共用统一 slot id pool。边界 state 算完后，`LinearStateBlockManager` 将旧 live slot donate/登记为 checkpoint，并为请求 rotate 出新 live slot。[slot ownership 与生命周期](../../xllm/core/framework/block/linear_state_block_manager.h#L29-L100) [save rotation](../../xllm/core/framework/block/linear_state_block_manager.cpp#L65-L103)

`FLAT_KV_LINEAR` 已实现 hybrid 命中的关键约束：取最深 LINEAR checkpoint 作为 restore source，并把 KV shared blocks clamp 到 LINEAR 可恢复 token 数。[mixed KV/LINEAR trim](../../xllm/core/framework/block/composite_block_manager.cpp#L517-L573) [restore source 挂载](../../xllm/core/framework/block/composite_block_manager.cpp#L691-L720)

Restore 路径已经跨所有 linear layers 同时 copy conv + SSM，因此 KDA 可以复用，不应另建一套 prefix cache manager。[跨层 state copy](../../xllm/core/framework/kv_cache/linear_state_restore.cpp#L83-L135)

适配 KDA 的最小工程范围建议是：

1. 将 GDN 专用命名推广为 linear-attention/KDA 可共享的 model args、shape 和 kernel dispatch；保留 `BlockType::LINEAR` 与现有 active/checkpoint 生命周期。
2. 为 KDA 定义 conv shape `[Kconv-1, (Q+K+V channels)/TP]` 与 recurrent shape `[H/TP,d_v,d_k]`，并明确 device kernel 所需的最后两维方向；不要因为 K3 的 `d_v=d_k=128` 而忽略语义。
3. Prefill kernel 必须接受 initial state、返回 final state；chunked prefill 继续使用同一 live slot。若要在一个 chunk 内生成 prefix checkpoint，需要像 vLLM FlashKDA/SGLang extra-buffer 一样额外返回中间 state，并从同一边界的原始 mixed QKV 生成 conv tail。
4. 继续维持 `{conv, SSM}` 同边界保存、mixed KV 共同命中、hit 后 checkpoint→private live COW 三个不变量。

特别要注意 `checkpoint_stride`：xLLM 当前 SSM 第 0 维按 `num_linear_state_blocks * checkpoint_stride` 分配，而 conv 仍按 logical slot 分配；restore 用 `ssm_cache.size(0) / conv_cache.size(0)` 推导一个 logical slot 对应的连续 SSM rows。[SSM shape](../../xllm/core/framework/kv_cache/kv_cache_shape.cpp#L381-L401) [stride-aware copy](../../xllm/core/framework/kv_cache/linear_state_restore.cpp#L112-L133)

因此，接 KDA 时不能把 `checkpoint_stride` 误解释为“每个 prefix boundary 多一份 KDA state”。它是**一个 logical LINEAR slot 内的物理 SSM row 展开倍率**，常用于 speculative state；KDA baseline active/checkpoint 仍应以 logical slot 为所有权单位。若 KDA speculative 实现需要多个中间矩阵，应放在该 slot 的 stride 范围内，并确保保存/恢复整段 stride，而 conv tail 仍只对应同一个已提交边界。

## 6. 对实现评审最有用的检查清单

- [ ] Active cache 是否恰好包含 conv tail 和 recurrent matrix，而不是逐 token K/V？
- [ ] 首个 prefill chunk 是否清零，后续 chunk 是否加载上一 chunk terminal state？
- [ ] Prefill kernel 的 final state 是否写回正确 request/physical slot？
- [ ] Prefix checkpoint 的 conv 与 recurrent 是否来自完全相同的 token 边界？
- [ ] Hybrid MLA KV hit 是否 clamp 到最深可恢复 KDA checkpoint？
- [ ] Prefix hit 后是否复制到 private live slot，避免修改共享 checkpoint？
- [ ] TP 后的 head/channel shape、`[d_v,d_k]` 方向和 dtype 是否与 kernel 一致？
- [ ] `checkpoint_stride > 1` 时是否保存/清零/恢复整段 SSM rows？
- [ ] Chunk size、radix/page size、checkpoint boundary 不整除时，是切分、返回 intermediate state，还是明确拒绝？
