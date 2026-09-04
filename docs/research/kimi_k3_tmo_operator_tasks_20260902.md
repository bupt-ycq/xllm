# Kimi K3 TMO 算子任务解读：KDA、FlashKDA、输出门与 AttnRes

> 调研日期：2026-09-02  
> 对应任务：TMO-3595、TMO-3620、TMO-3619、TMO-3613、TMO-3612、TMO-3611  
> 范围：公开一手资料、Kimi K3 官方模型代码、Moonshot FlashKDA、FLA/vLLM 实现，以及当前 `/workspace/xllm` 工作树；未修改产品代码。

## 结论先行

这六张单子共同指向 **Kimi K3 在 TMO/MLU 上的端到端推理适配**，不是六种彼此独立的新注意力算法。

| 工单 | 实际角色 | 主要阶段 | 一句话解释 |
|---|---|---|---|
| TMO-3611 `chunk_kda_with_fused_gate` | KDA 通用 prefill 主路径 | prefill、多 token | 把 raw KDA gate / beta 的激活与 chunkwise KDA 流水组合起来；通常仍是多 kernel pipeline |
| TMO-3620 `flashkda` | KDA 高性能 prefill 后端评估 | inference prefill | Moonshot 的 CUTLASS 特化实现，是 `chunk_kda` 的替代后端，不是 decode kernel |
| TMO-3619 `fused_recurrent_kda` | KDA 通用递推路径 | decode、短序列、spec verify | 按 token 更新固定矩阵状态；能处理 `T>1`，但 serving 中通常用于短序列 |
| TMO-3613 `fused_recurrent_kda_packed_decode` | KDA 单 token decode 特化 | plain decode | 直接从 post-conv packed QKV 读入，每个请求更新其 state slot，减少拆分、搬运和 launch 开销 |
| TMO-3612 out gate / sigmoid / product | 注意力输出门后处理 | prefill + decode | 对 attention 输出做逐元素 `sigmoid(gate)` 乘法；标题更像 K3 的 Gated MLA 后处理 |
| TMO-3595 AttnRes 融合算子 | 跨网络深度的残差路由 | prefill + decode | 对之前 block 的表示沿“层深”做 RMSNorm、打分、softmax 和加权聚合；不是沿 token 序列做 attention |

最重要的任务关系是：

```text
hidden states
  ├─ KDA layers (69/93)
  │    ├─ prefill ── TMO-3611 chunk_kda_with_fused_gate
  │    │                └─ TMO-3620 FlashKDA（更特化的替代后端/优化参考）
  │    ├─ decode/短 verify ── TMO-3619 fused_recurrent_kda
  │    │                         └─ TMO-3613 packed_decode（T=1 更窄特化）
  │    └─ KDA output: gated RMSNorm -> o_proj
  ├─ Gated MLA layers (24/93)
  │    └─ attention output × sigmoid(output gate) ── TMO-3612
  └─ 每层 attention/MLP 前后的跨深度路由 ── TMO-3595 AttnRes
```

因此，`chunk_kda_with_fused_gate` 与 FlashKDA 更像 **prefill 基线和高性能后端**；`fused_recurrent_kda` 与 packed decode 更像 **通用 decode 和单 token 快路**。后两对都不应按名称理解为必须重复计算两遍。

## 1. 先明确 Kimi K3 是什么

### 证据事实

Kimi 官方报告把 K3 描述为 2.8T 参数、104B 激活参数、原生多模态、1M context 的 MoE，并明确说模型建立在 Kimi Delta Attention（KDA）和 Attention Residuals（AttnRes）之上。[Kimi K3 官方报告](https://arxiv.org/abs/2607.24653) 官方模型卡给出的文本骨干是 93 层，其中 **69 层 KDA、24 层 Gated MLA**，hidden size 7168、96 个 attention heads。[Kimi K3 官方模型卡](https://huggingface.co/moonshotai/Kimi-K3)

官方 `config.json` 进一步给出 KDA `head_dim=128`、short convolution kernel size 4、gate lower bound `-5.0`、`use_full_rank_gate=true`，以及 `attn_res_block_size=12`。[Kimi K3 官方 config](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)

这解释了为什么工单集中在 KDA state 更新、prefill/decode 两套 kernel、输出门和 AttnRes：它们正好覆盖 K3 attention/residual 热路径。

## 2. KDA 的数学与张量流

### 2.1 核心递推

KDA 是带逐 key-channel 衰减的 Delta Rule。按单个 value head 写，固定大小状态为

$$
S_t\in\mathbb{R}^{K\times V},
$$

递推为

$$
\boxed{
S_t=(I-\beta_t k_tk_t^\top)\operatorname{Diag}(\alpha_t)S_{t-1}
    +\beta_t k_tv_t^\top,
\qquad o_t=S_t^\top q_t
}
$$

等价的实现顺序是：

$$
\begin{aligned}
\widetilde S_t &= \operatorname{Diag}(\alpha_t)S_{t-1},\\
e_t &= v_t-\widetilde S_t^\top k_t,\\
S_t &= \widetilde S_t+\beta_t k_te_t^\top,\\
o_t &= S_t^\top q_t.
\end{aligned}
$$

官方 FLA naive reference 正是先做 `S *= exp(g)`，再计算 `v - k^T S`、乘 beta、外积写回，最后用 q 读 state。[FLA KDA naive reference](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/naive.py)

其中 `g=log(alpha)` 是 `[B,T,HV,K]` 的逐 key-channel log decay；`beta` 是 `[B,T,HV]`。这与 Gated DeltaNet 的 `[B,T,HV]` 标量 gate 不同，是 KDA kernel 不能仅靠改名复用 GDN kernel 的核心原因。Kimi Linear 官方论文将 KDA定义为 Gated DeltaNet 的 fine-grained gating 扩展，并给出专用 chunkwise DPLR 算法。[Kimi Linear 官方论文](https://arxiv.org/abs/2510.26692)

### 2.2 K3 实际张量流

对 K3 的一个 KDA layer，官方代码中的主干是：

```text
x [B,T,7168]
  ├─ q/k/v projection
  ├─ 各自 short conv (kernel=4, SiLU)
  │      -> q,k,v [B,T,96,128]
  ├─ fine-grained raw gate projection
  │      -> raw_g [B,T,96,128]
  ├─ beta projection
  │      -> raw_beta [B,T,96]
  ├─ q/k L2 norm
  ├─ KDA gate + beta activation
  ├─ chunk KDA 或 recurrent KDA
  │      -> o [B,T,96,128] + state [N,96,128,128]
  ├─ output gate + gated RMSNorm
  └─ o_proj -> [B,T,7168]
```

官方实现对长/多 token 路径调用 `chunk_kda`，对 cached `q_len==1` 路径调用 `fused_recurrent_kda`，并在两者中请求 kernel 内 Q/K L2Norm、gate 激活和 beta sigmoid。[Kimi K3 官方 `KimiDeltaAttention`](https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py#L706-L892)

K3 使用 bounded gate。其 log decay 为

$$
g_t=-5\cdot\sigma\!\left(\exp(A_{\log})\odot(raw\_g_t+dt\_bias)\right)\in[-5,0),
\qquad \alpha_t=\exp(g_t).
$$

FLA 同时保留不带 lower bound 的 `-exp(A_log) * softplus(...)` 形式；K3 config 和官方调用启用了 `lower_bound=-5` / `safe_gate`。[FLA KDA gate reference](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/gate.py) [FLA `chunk_kda` API](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/chunk.py)

### 2.3 为什么 prefill 和 decode 需要两类算法

- **Decode**：每条请求通常只有 1 个新 token，直接执行上述递推最自然；计算量不随历史 context 增长，只读写固定 `[H,K,V]` state。
- **Prefill**：如果仍逐 token 串行，长 prompt 会让设备并行度很差。`chunk_kda` 把一个 chunk 内的递推精确改写为小型下三角/WY/DPLR 问题和 GEMM；chunk 内并行，chunk 间仍传 state。

所以 KDA 虽然在 decode 形态上是 RNN，prefill 仍可使用精确的 chunk-parallel 算法。这两类 kernel 是同一数学的不同执行形态，不是两个模型层。

## 3. 六张工单逐项解释

## 3.1 TMO-3611：`chunk_kda_with_fused_gate`

### 证据事实

vLLM 的同名函数是 KDA 的 Triton prefill 入口，输入 `q/k/v`、`raw_g`、`raw_beta`、`A_log`、`g_bias/dt_bias`、initial state、`cu_seqlens` 等。它先调用 `fused_kda_gate_chunk_cumsum`，生成激活后的逐维 log gate 及 chunk 内 cumsum，同时得到 `sigmoid(raw_beta)`；随后进入 chunk KDA 的 intra-chunk、state propagation 和 output 流水。[vLLM `chunk_kda_with_fused_gate`](https://github.com/vllm-project/vllm/blob/main/vllm/models/kimi_k3/nvidia/ops/third_party/kda/chunk.py#L696-L814)

这里的 “fused gate” 指 **KDA forget gate / beta 的激活与 chunk prefix 处理**，不是模型末尾的 output gate。它也不表示整个 prefill 只有一次物理 launch。FLA 的默认 chunk path 本身由 gate/cumsum、intra/WY、跨 chunk state 和 output finalize 等阶段组成。[FLA `chunk_kda`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/chunk.py)

### 用途与融合价值

- 用途：长 prompt prefill、chunked prefill、packed variable-length prefill。
- 融合价值：避免先单独 materialize raw gate 激活结果和 beta sigmoid，再启动 chunk prefix kernel；减少 `[tokens,H,K]` 大张量的全局内存往返和 launch。
- 仍需保留：initial/final recurrent state、`cu_seqlens` 隔离、tail、不等长 batch、跨 prefill chunk 携带 state 的正确性。

### 基于工单命名的推断

TMO-3611 很可能是 **MLU 上功能完整的 KDA prefill 基线**。它与 TMO-3620 不一定是两个都必须用于生产的前后串联步骤：后者更可能是前者的高性能替代 backend 或优化参考。

## 3.2 TMO-3620：评估支持 FlashKDA

### 证据事实

Moonshot 官方 FlashKDA 是基于 CUDA/CUTLASS 的 inference-only KDA prefill forward backend。它由 FLA 的 `chunk_kda` 自动 dispatch，当前公开 API 要求 BF16、`K=V=128`，接受 raw gate、raw beta、`A_log`、`dt_bias`、lower bound、initial/final state 和 varlen `cu_seqlens`；公开 README 要求 SM90+、CUDA 12.9+。[Moonshot FlashKDA 官方仓库](https://github.com/MoonshotAI/FlashKDA)

FlashKDA 并非把所有工作塞进一个 kernel。官方 deep dive 明确拆为：

- K1，按 token/chunk 并行：gate activation、L2 norm、decay、局部矩阵构造和逆；
- K2，按 sequence/head 并行：chunk recurrence、output、running/final state。

官方称早期单 kernel 原型因两阶段并行度不匹配而浪费 SM，拆成 K1/K2 后端到端至少快 15%。它选择 chunk size 16 以控制 BF16 数值范围并降低小矩阵求逆成本。[FlashKDA 官方 deep dive](https://github.com/MoonshotAI/FlashKDA/blob/master/docs/20260420-flashkda-v1-deep-dive.md)

### 用途与评估重点

- 用途：K3 的 inference prefill；不是 decode，也不负责 AttnRes/output gate。
- 对 TMO/MLU：CUTLASS CUDA binary 不能直接搬到 MLU；可评估其 API 契约、K1/K2 融合边界、chunk=16、片上 state 和 workspace 设计是否值得复刻。
- 关键验收维度：目标 MLU 型号、BF16 state 精度、8K～1M 长度、dense/varlen、initial/final state、chunked prefill 等价性、workspace 峰值、相对通用 TMO chunk pipeline 的端到端收益。

### 基于工单命名的推断

“评估支持”比“实现支持”更合理，因为它是硬件专用 CUDA 后端。评估结果可能是：做 TMO 版 FlashKDA 风格双 kernel、只借鉴部分 schedule，或保留 TMO 多阶段 `chunk_kda_with_fused_gate`。

## 3.3 TMO-3619：`fused_recurrent_kda`

### 证据事实

FLA 的 `fused_recurrent_kda` 接收 `q [B,T,H,K]`、`k [B,T,H,K]`、`v [B,T,HV,V]`、逐维 gate `[B,T,HV,K]`、beta `[B,T,HV]` 和 state `[N,HV,K,V]`，在单个 recurrent 主 kernel 中可融合：

- Q/K L2 normalization 和 q scale；
- raw gate + `A_log`/`dt_bias` 激活；
- beta sigmoid；
- state decay、delta correction、KV write、output 和 final state。

它支持 equal-length、packed varlen 和 initial/final state。[FLA `fused_recurrent_kda`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/fused_recurrent.py)

它在数学上不限制 `T=1`。但 recurrent loop 对 T 串行，所以 serving 通常把它用于单 token decode、较短 speculative verify，长 prefill 用 chunk path。

### 用途与融合价值

- 用途：decode、短 MTP/spec verify、必要时很短的多 token 请求。
- 融合价值：state `[K,V]` 可以在 kernel 内持续驻留，避免每个 token 把 state 写回再读回；同时消除 norm/gate/beta 的中间张量。
- 融合边界：官方 FLA 版本不包含前面的 causal conv，也不包含后面的 output gated RMSNorm / `o_proj`。

### 基于工单命名的推断

TMO-3619 应是通用语义入口/回退路径；TMO-3613 是其单 token packed specialization，而不是另一个不同算法。

## 3.4 TMO-3613：`fused_recurrent_kda_packed_decode`

### 证据事实

vLLM 的同名实现限定为 **one-token KDA decode directly from packed post-conv QKV**。输入 `mixed_qkv [N, 2*H*K + H*V]`，再带 raw gate、raw beta、`A_log`、`dt_bias`、lower bound、state pool 和 `state_indices [N]`。kernel 内直接切分 q/k/v，执行 L2Norm、gate/beta 激活、一次 KDA state update，并把每行写回对应 state slot。[vLLM packed decode 实现](https://github.com/vllm-project/vllm/blob/main/vllm/models/kimi_k3/nvidia/ops/third_party/kda/fused_recurrent.py#L489-L671)

“packed”在这里主要指：

1. Q/K/V 在一个连续 row 中，避免先拆成三个大 tensor；
2. batch 中每行对应一个请求的一个新 token；
3. `state_indices` 把请求映射到 continuous-batching state cache slot。

它不是“把长 prompt 打包后做 prefill”。vLLM K3 layer 在 pure non-spec decode 中先做 `causal_conv1d_update`，再调用 packed decode；spec verify 则走能处理多 token/varlen 的通用 `fused_recurrent_kda`。[vLLM K3 KDA dispatch](https://github.com/vllm-project/vllm/blob/main/vllm/models/kimi_k3/nvidia/kda.py)

### 用途与融合价值

- 用途：continuous batching 下的 plain one-token decode。
- 价值：避免 QKV split/copy，减少小 batch decode 的 launch 和 HBM round trip，并直接原位更新 state pool。
- 限制：应显式验证“一请求一 token”不变量、strided/contiguous state indices、空/无效 slot、不同 batch size，以及状态布局 `[V,K]` / `[K,V]`。

### 基于工单命名的推断

如果 TMO-3619 已达到目标延迟，3613 可能是二阶段性能优化而非首个功能 blocker；但 K3 plain decode 每层都会调用它，69 个 KDA layers 会放大单次小开销，所以专用快路通常有实际价值。

## 3.5 TMO-3612：out gate / sigmoid / product 融合

### 证据事实

K3 有两种相近但不完全相同的 output gating：

1. **Gated MLA**：官方代码执行

   ```python
   g = self.g_proj(hidden_states).sigmoid()
   attn_output = attn_output * g
   attn_output = self.o_proj(attn_output)
   ```

   即 `sigmoid(output_gate_logits)` 与 attention output 的逐元素乘法。[Kimi K3 官方 MLA output gate](https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py#L683-L703)

2. **KDA**：先生成 output gate logits，然后调用 `FusedRMSNormGated(o, gate, activation='sigmoid')`，语义为 `RMSNorm(o) * sigmoid(gate)`，再做 `o_proj`。[Kimi K3 官方 KDA output path](https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py#L760-L770) [FLA fused norm/gate kernel](https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/fused_norm_gate.py)

### 基于工单命名的推断

标题只写 “out gate/sigmoid/product”，没有写 RMSNorm，因此它 **更像 Gated MLA 的 sigmoid+multiply 融合**，也可能是供两类 attention 复用的底层 elementwise primitive。仅凭标题无法确认是否还包含 gate projection、RMSNorm 或 `o_proj`；这些必须以工单接口/shape 说明为准。

### 用途与融合价值

该算子 token-wise 独立，prefill/decode 都会用。若 gate logits 和 attention output 各自 materialize 后再启 sigmoid、mul，会产生多次全量读写；融合至少可把 sigmoid 与乘法放在一次遍历中。是否值得继续和 attention finalize、RMSNorm 或 output projection 融合，要看布局和 producer/consumer 是否兼容。

## 3.6 TMO-3595：AttnRes 融合算子

### 证据事实

AttnRes 不是 sequence attention。它把标准 PreNorm residual 的固定求和，改成对之前层/块表示的 **depth-wise attention**。官方论文给出

$$
h_l=\sum_{i<l}\alpha_{i\rightarrow l}v_i,
$$

并用 Block AttnRes 将候选从所有层压缩到 block-level representations，以降低深层模型的 memory/communication 开销。[Attention Residuals 官方论文](https://arxiv.org/abs/2603.15031) [Moonshot 官方仓库](https://github.com/MoonshotAI/Attention-Residuals)

K3 官方代码的 `_apply_attn_res` 对每个 token 执行：

```text
v = concat(previous block residuals, current prefix_sum)
  shape: [num_tokens, num_sources, hidden_size]
k = RMSNorm(v)                         # 每个 source 沿 hidden dim 归一化
scores = dot(k, learned pseudo-query)  # [num_tokens, num_sources]
probs = softmax(scores, dim=sources)
h = probs @ v                          # [num_tokens, hidden_size]
```

其中 pseudo-query 可由 `norm.weight * proj.weight` 合并得到。这个路径在每层 self-attention 前、MLP 前调用，并维护 block residual；K3 config 的 block size 为 12。[Kimi K3 官方 AttnRes 实现](https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py#L1135-L1317)

### 用途与融合价值

- 用途：prefill 和 decode 都执行；它在每个 token 内沿 network depth 聚合，不跨 token 混合，因此没有 causal mask/KV cache。
- 典型融合范围：concat/读取 source、RMS statistics、score dot、softmax over sources、weighted reduction；避免 materialize `v_float`、normalized keys、scores 和 probabilities。
- 性能特征：source 数较小、hidden dim 很大（7168），因此更偏 memory-bandwidth 与小 softmax/launch overhead；decode 的每步 token 数更少，launch 开销尤其敏感。

### 基于工单命名的推断

工单只写 “AttnRes 融合算子”，最合理的语义是上述 `_apply_attn_res` 整段，而不是融合普通 `residual + hidden`。是否包含 block append/边界管理或 final AttnRes，标题无法确认。

## 4. 当前 `/workspace/xllm` 已有什么、缺什么

以下是当前工作树的直接代码证据，不代表工单最终分支状态。

### 已有的相邻能力

1. **MLU GDN prefill pipeline** 已有 `ChunkGatedDeltaRuleImpl`，按 gate cumsum、KKT、三角求解、W/U、跨 chunk state、output 七阶段组织。[本地 MLU chunk GDN](../../xllm/core/kernels/mlu/chunk_gated_delta_rule.cpp#L48-L111)
2. **MLU GDN recurrent 与 packed decode** 已有 TMO kernel。通用 `fused_recurrent_gated_delta_rule` 明确把 `is_kda` 硬编码为 0；packed decode 的 `a/dt_bias` 也是每 head 标量语义，因此二者当前不能直接等同于 KDA 逐 channel gate。[本地 recurrent GDN](../../xllm/core/kernels/mlu/fused_recurrent_gated_delta_rule.cpp#L32-L179) [本地 packed GDN decode](../../xllm/core/kernels/mlu/fused_recurrent_gated_delta_rule_packed_decode.cpp#L32-L166)
3. **一个低层 recurrent kernel 已有 KDA 分支**：`fused_sigmoid_gating_delta_rule_update(..., is_kda=true)` 接受 `a/dt_bias [tokens or heads, HV*K]`，并有 dense KDA reference test 验证逐 key-channel decay、beta、state update 和 output。[本地 KDA-capable wrapper](../../xllm/core/kernels/mlu/fused_sigmoid_gating_delta_rule_update.cpp#L65-L140) [本地 KDA 测试](../../tests/core/layers/mlu/qwen3_5_gated_delta_net_ops_test.cpp#L1068-L1192)
4. **MLU full attention 的 output gate 语义已经存在但未融合**：Qwen3.5 attention 直接执行 `gate = sigmoid(gate); out = out * gate`，可作为 TMO-3612 的本地需求旁证。[本地 MLU output gate](../../xllm/core/layers/mlu/qwen3_5/qwen3_5_attention.cpp#L204-L245)
5. **gated RMSNorm 抽象已存在**，Qwen3.5 GDN output path 调用 `norm_->forward(core_attn_out, z)`；但这不是 AttnRes，也不能证明 K3 的 sigmoid-only KDA gate 已完整对齐。[本地 GDN output path](../../xllm/core/layers/mlu/qwen3_5/qwen3_5_gated_delta_net.cpp#L446-L457)

### 当前未发现的完整能力

- `xllm/models` 下只有 Kimi K2/K2.5 相关模型文件，未发现 Kimi K3 model/layer integration。
- 未发现命名为 `chunk_kda` / `chunk_kda_with_fused_gate` / `FlashKDA` 的当前 xLLM 产品实现。
- 未发现 K3 Block AttnRes 实现或 `_apply_attn_res` 等价融合算子。
- 未发现 KDA 专用 packed decode；现有 packed kernel 是 GDN 标量 gate 版本。

所以当前仓库并非“从零”：GDN chunk/recurrent、state cache、packed decode 和 KDA recurrent 分支可复用设计与测试框架；但 **逐 channel KDA gate 会改变 gate shape、state decay 和 chunk output 路径**，不能把已有 GDN kernel 直接改名当完成。

## 5. 建议如何理解优先级与依赖

若目标是先跑通 K3 inference，合理依赖顺序是：

1. 固定 KDA/AttnRes/output gate 的接口、layout、dtype、state slot 和 varlen 契约；
2. 用 TMO-3611 建立可验证的 prefill 路径；
3. 用 TMO-3619 建立通用 decode/短 verify 路径；
4. 接入 TMO-3612 与 TMO-3595，保证层输出和跨层 residual 数学完整；
5. 以 TMO-3613 优化 plain one-token decode；
6. 通过 TMO-3620 决定 prefill 是否需要 FlashKDA 风格专用快路。

这不是公开工单的已确认排期，而是基于算子数据依赖的工程推断。实际 blocker 还取决于是否已有其他分支提供 K3 layer、MLA、MoE、量化和 cache metadata。

## 6. 容易混淆的四点

1. **FlashKDA 不是 FlashAttention。**它是 KDA 的专用 chunkwise prefill backend，不计算 softmax attention，也不用于 Gated MLA。
2. **`chunk_kda_with_fused_gate` 的 fused gate 不是 output gate。**前者生成 KDA 的 forget decay；后者在 attention 输出端做 sigmoid 乘法。
3. **packed decode 不是 packed prefill。**它是 continuous batching 中“一请求一 token”的 QKV/state-slot 快路。
4. **AttnRes 不是在 token 维做 attention。**它沿 layer/block 深度选历史表示；KDA/MLA 则沿 sequence 处理信息。

## 主要一手资料

- [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653)
- [Kimi K3 官方 Hugging Face 仓库与模型代码](https://huggingface.co/moonshotai/Kimi-K3)
- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- [MoonshotAI/Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear)
- [Attention Residuals 论文](https://arxiv.org/abs/2603.15031)与[官方仓库](https://github.com/MoonshotAI/Attention-Residuals)
- [MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA)
- [FLA KDA operators](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda)
- [vLLM Kimi K3 KDA implementation](https://github.com/vllm-project/vllm/tree/main/vllm/models/kimi_k3)

