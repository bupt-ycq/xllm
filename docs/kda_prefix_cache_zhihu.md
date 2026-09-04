# KDA 都没有 KV Cache 了，Prefix Cache 到底缓存什么？

做大模型推理优化时，一个很自然的问题是：

> Kimi Delta Attention（KDA）都不保存逐 token 的 KV Cache 了，它怎么判断两个请求命中了同一段历史前缀？命中后又能复用什么？

先给结论：

> **KDA 不是没有缓存，而是把“逐 token 的 KV”换成了“前缀结束时的递归状态快照”。**
>
> 前缀是否相同，仍然通过 token ID 及其哈希索引判断；命中后复用的载荷，不再是 KV block，而是该前缀边界上的 **recurrent state + short-convolution state**。

如果把普通 Attention 的 Prefix Cache 称为 **KV Block Cache**，那么 KDA 的 Prefix Cache 更像 **State Checkpoint Cache**。

这篇文章把这个过程从模型公式一直拆到推理框架实现。

## 一、先区分两个经常混在一起的“缓存”

讨论这个问题前，最好先把两个概念分开：

1. **Decode Cache**：同一个请求生成下一个 token 时，怎样避免重新计算全部历史。
2. **Prefix Cache**：另一个请求拥有相同前缀时，怎样复用上一个请求已经完成的 prefill。

普通全注意力在两个场景里都复用逐 token 的 K/V，所以大家习惯把“历史状态”和“KV Cache”画等号。

KDA 打破了这个习惯：它在 decode 时保存固定大小的递归状态；要跨请求复用前缀，则需要把某些历史位置的递归状态额外保留为 checkpoint。

## 二、普通 Attention 为什么很容易做 Prefix Cache？

对标准因果 Attention 来说，历史可以写成：

\[
\mathcal{H}_t=\{K_1,V_1,K_2,V_2,\ldots,K_t,V_t\}
\]

服务框架通常把 KV Cache 分成固定大小的 block，并为每个完整 token block 计算链式哈希。假设请求 A 是：

```text
[token 0~15] [token 16~31] [token 32~47] [后缀 A]
```

请求 B 的前三个 block 完全相同：

```text
[token 0~15] [token 16~31] [token 32~47] [后缀 B]
```

框架通过 token 哈希找到这三个 block，然后直接复用对应的 K/V，只从后缀 B 开始 prefill。

这里有两个相互独立的部分：

- **索引（key）**：前缀 token 及其哈希；
- **缓存载荷（value）**：这些 token 对应的 K/V 张量。

理解这个区分，是理解 KDA Prefix Cache 的关键。

## 三、KDA 没有逐 token KV，但它有递归状态

KDA 用一个矩阵状态不断吸收历史信息。省略 head 下标后，其递推形式为：

\[
S_t=
(I-\beta_t k_tk_t^\top)
\operatorname{Diag}(\alpha_t)S_{t-1}
+\beta_t k_tv_t^\top
\]

输出为：

\[
o_t=S_t^\top q_t
\]

其中：

\[
S_t\in\mathbb{R}^{d_k\times d_v}
\]

这是单个 head 的语义形状。若一层有 \(H\) 个 KDA heads，整层状态可写作 `[H,d_k,d_v]`；做 \(P\) 路 Tensor Parallel 并按 head 切分后，每张卡约为 `[H/P,d_k,d_v]`。某些 kernel 为了访存会交换最后两个维度，但不改变它表示的内容。

[Kimi Linear 论文](https://arxiv.org/pdf/2510.26692)中，实验采用 \(d_k=d_v=128\)。无论序列是 1K、100K 还是 1M token，单个 head 的当前递归状态尺寸都不随序列长度增长。

因此，普通 Attention 保存的是：

```text
K1,V1  K2,V2  K3,V3  ...  Kt,Vt
```

KDA 保存的则是：

```text
x1,x2,...,xt  ──不断递推──>  S_t
```

过去的某个 \(K_i,V_i\) 已经无法单独取出，但继续计算下一个 token 也不再需要它们。对 KDA 自身的递推来说，正确的 \(S_t\) 就是继续执行所需的全部长期状态：

\[
S_{t+1}=F(S_t,x_{t+1})
\]

这里要区分两个层面的“精确”：

- 从信息表达角度看，固定大小的 \(S_t\) 是对任意长历史的压缩，可能遗忘或相互干扰；
- 从执行角度看，只要恢复的是正确前缀边界上的 \(S_t\)，后续递推在数学上与从头 prefill 到该位置再继续执行等价，并不是一次近似的 cache hit；不同 kernel 或计算顺序仍可能带来细小的浮点误差。

## 四、KDA 到底怎么命中相同前缀？

答案是：**用 token 找，用 state 接着算。**

第一次请求处理前缀 \(P=[x_1,\ldots,x_p]\) 后，框架得到状态 \(S_p\)，并建立类似下面的缓存项：

```text
Key   = Hash(model, adapter, token_ids[0:p], ...)
Value = StateCheckpoint(prefix_length=p, recurrent_state=S_p, ...)
```

这里的 \(S_p\) 只是为了讲解而使用的简写；真实模型需要保存各个 KDA layer/head 在该边界上的状态，并同时保存对应的卷积状态。

第二个请求到来后，框架仍然比较 token ID 或 token block 的链式哈希。若最长公共前缀正好命中了已保存的边界 \(p\)，就恢复 \(S_p\)，直接处理新后缀：

```text
请求 A：A B C D E F G H | X Y Z
                         └─ 保存 S_8

请求 B：A B C D E F G H | P Q R
        └── token 哈希命中 ──┘
                         └─ 加载 S_8，从 P 开始计算
```

所以，“没有 KV Cache，怎么知道前缀相同”其实混淆了缓存索引与缓存内容：

> **判断是否命中并不依赖 KV。KV 或 recurrent state 只是命中以后被取出的计算结果。**

而且这里命中的必须是严格相同的 token 前缀，不是语义相似的文本。后者属于 Semantic Cache，是另一类优化。

## 五、ShortConv 到底保存什么？Q/K/V 都要过吗？

答案是：**Q、K、V 三路都会经过 ShortConv。Q、K 随后还会做 L2Norm，V 不做。**

先定义三个线性投影结果：

\[
z_t^q=W_qx_t,\qquad
z_t^k=W_kx_t,\qquad
z_t^v=W_vx_t
\]

KDA 论文给出的后续路径为：

\[
q_t,k_t=\operatorname{L2Norm}
(\operatorname{Swish}(\operatorname{ShortConv}(z_t^{q/k})))
\]

\[
v_t=\operatorname{Swish}(\operatorname{ShortConv}(z_t^v))
\]

也就是说：

```text
                         hidden state x_t
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
         W_q                  W_k                  W_v
          │                    │                    │
          ▼                    ▼                    ▼
         z_q                  z_k                  z_v
          │                    │                    │
      ShortConv           ShortConv           ShortConv
          │                    │                    │
         SiLU                 SiLU                 SiLU
          │                    │                    │
       L2Norm               L2Norm                 │
          ▼                    ▼                    ▼
          Q                    K                    V
          └────────────────────┼────────────────────┘
                               ▼
                              KDA
```

这里的 ShortConv 是按 channel 独立执行的 depth-wise causal 1D convolution。若卷积宽度为 \(W\)，第 \(c\) 个 channel 可以写成：

\[
\widetilde q_t[c]
=\sum_{j=0}^{W-1}w_j[c]z_{t-j}^q[c]
\]

所以处理新 token \(t+1\) 时，只有新的 \(z_{t+1}^q\) 还不够，还需要之前 \(W-1\) 个位置的投影特征；K、V 同理。

因此 ShortConv cache 保存的不是：

- 原始 token ID；
- hidden state \(x_t\)；
- 已经卷积完成的最终 Q/K/V。

它保存的是：

> **最近 \(W-1\) 个 token 在 Q/K/V 三条路径上的 Linear Projection 输出，也就是 \(z^q,z^k,z^v\)。**

其语义形状为：

\[
C_Q\in\mathbb{R}^{(W-1)\times H_qd_q}
\]

\[
C_K\in\mathbb{R}^{(W-1)\times H_kd_k}
\]

\[
C_V\in\mathbb{R}^{(W-1)\times H_vd_v}
\]

当三路的 head 数和 head dimension 相同时，vLLM 会把它们打包成一块：

\[
C_t\in\mathbb{R}^{(W-1)\times 3Hd}
\]

默认 `SD` 布局是 `[state_len, dim]`，也可以使用转置后的 `DS` 布局 `[dim, state_len]`。需要注意：某些通用 ShortConv 接口为了 rolling update 方便会暴露 \(W\) 个槽位；而当前 vLLM 的 KDA serving state 明确保存 \(W-1\) 个旧位置。若开启多 token 推测解码，还可能为 speculative tokens 追加槽位。

因此，一个真正可恢复的 KDA 前缀缓存应写成：

\[
\boxed{
\text{KDA checkpoint}_p=
(S_p,C_p^Q,C_p^K,C_p^V)
}
\]

其中 \(S_p\) 是长期递归记忆，三份 \(C_p\) 是生成下一个 Q/K/V 所需的局部卷积历史。vLLM 对 KDA Prefix Caching 的描述也是“matrix-like recurrent state + short convolution state”。

## 六、真正困难的地方：不能在每个 token 后都存一份状态

KDA 的“当前状态”相对整段序列很小，但它相对单个 token 的 KV 并不小。

以 \(d_k=d_v=128\) 为例，一个 head 的矩阵有 16,384 个元素：按 BF16 理论载荷计算是 32 KiB；而当前 vLLM 的 KDA recurrent state 使用 FP32，因此实际是 64 KiB/head。这个数字还要乘以 head 数与 KDA 层数，再加上卷积状态、对齐和框架元数据。

如果为了 Prefix Cache 在每个 token 后都保存一份：

```text
token 1  -> S_1
token 2  -> S_2
token 3  -> S_3
...
token T  -> S_T
```

那就会重新产生随序列长度增长的巨大存储，抵消 KDA 在 decode 阶段的常量状态优势。

因此工程实现会有选择地保留 checkpoint，例如：

- prompt 结束位置；
- 每隔一段较长 token 区间；
- 被多次访问、证明值得缓存的热点前缀。

这也带来一个重要边界：假设只保存了 \(S_{32K}\) 和 \(S_{64K}\)，新请求与旧请求共享 50K token，那么 \(S_{64K}\) 已混入 50K～64K 的信息，不能使用，也通常无法“倒推”出 \(S_{50K}\)。如果没有恰好位于 50K 的状态快照，只能从最近的有效 checkpoint 恢复，再重算剩余部分：

```text
0 -------- 32K -------- 50K -------- 64K
           ↑             ↑
        load S_32K    公共前缀结束
           └── 重新 prefill 32K~50K ──┘
```

因此更严谨的复杂度表述是：

> KDA 为一个正在 decode 的序列维护的最新状态，对序列长度是 \(O(1)\)；但服务端为了跨请求复用而保留的历史 checkpoint 总量，并不会天然是 \(O(1)\)。

## 七、Kimi Linear 其实仍然有传统 KV Cache

还需要纠正另一个常见误解：**KDA 层没有逐 token KV，不等于整个 Kimi Linear 模型没有 KV Cache。**

Kimi Linear 采用 3:1 的层级混合结构：

```text
KDA -> KDA -> KDA -> Full MLA
KDA -> KDA -> KDA -> Full MLA
...
```

也就是：

- KDA 层保存 recurrent state + convolution state；
- Full MLA 层仍保存普通的 paged KV blocks。

论文报告“KV Cache 最多减少 75%”，而不是完全消失，正是因为模型仍周期性保留全注意力层。

因此，一个混合模型的 Prefix Cache 实际是双轨制：

| 对比项 | Full Attention / MLA 层 | KDA 层 |
|---|---|---|
| 历史表示 | 每 token 的 K/V | 固定大小递归状态 |
| Decode 时更新 | append KV | 原地更新状态 |
| Prefix Cache 载荷 | KV blocks | state checkpoint |
| 是否需额外局部状态 | 通常不需要 | 需要 ShortConv state |
| 能否保存任意细粒度历史 | 相对容易 | 每个边界都存代价过高 |
| 从命中点继续 | 读取历史 KV | 载入 \(S_p,C_p\) |

调度器还必须让两类 cache 对同一个逻辑前缀达成一致。MLA 即使命中了 50K KV，只要 KDA 没有可恢复到 50K 的状态，整个模型也不能声称已经计算完 50K；两组 cache 最终必须共享一致的 `num_computed_tokens`。

## 八、把维度代进去：Kimi Linear 48B 与 Kimi K3

前面讲的是通用公式。下面把官方配置代进去，看看 state checkpoint 到底有多大。

### Kimi Linear 48B

[Kimi-Linear-48B-A3B-Instruct 官方配置](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/config.json)给出：

\[
H=32,\qquad d=128,\qquad W=4
\]

模型共有 20 个 KDA 层和 7 个 Full MLA 层。单个请求、单个 KDA 层、未做 TP 切分时：

\[
S_t:[32,128,128]
\]

而 vLLM 风格的 packed ShortConv state 为：

\[
C_t:[3,12288]
\]

这里：

\[
3=W-1,\qquad 12288=3_{\mathrm{QKV}}\times32\times128
\]

按 vLLM 当前的默认 dtype 计算，recurrent state 为 FP32，约 2 MiB/层；ShortConv state 对 BF16 模型默认是 BF16，约 72 KiB/层。

### Kimi K3

[Kimi K3 官方配置](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)则给出：

\[
\text{hidden size}=7168,\quad
H=96,\quad d=128,\quad W=4
\]

它共有 93 个 token-mixing 层，其中 69 个 KDA 层、24 个 Gated MLA 层。每条 Q/K/V 投影的宽度为：

\[
Hd=96\times128=12288
\]

所以三条路径可以画成：

```text
hidden [7168]
   │
   ├── Wq ──> [12288] ──> ShortConv ──> Q
   ├── Wk ──> [12288] ──> ShortConv ──> K
   └── Wv ──> [12288] ──> ShortConv ──> V
```

单个请求、单个 KDA 层、未做 TP 切分时，recurrent state 为：

\[
\boxed{S_t:[96,128,128]\ \text{FP32}}
\]

元素数量为：

\[
96\times128\times128=1,572,864
\]

因此大小恰好约为：

\[
1,572,864\times4\ \text{Bytes}=6\ \text{MiB}
\]

ShortConv 最少需要最近三个旧位置。vLLM 将 Q/K/V 三份状态打包后，默认 `SD` 布局为：

\[
\boxed{C_t:[3,36864]\ \text{BF16}}
\]

其中：

\[
36864=3_{\mathrm{QKV}}\times96\times128
\]

元素总数为 \(3\times36864=110,592\)，按 BF16 计算约 216 KiB/层。`DS` 布局只是把它转置成 `[36864,3]`，语义不变。

两种模型可以汇总成下表。这里的数字都是“单个 state slot、单个 KDA 层、TP=1”，不包含 MLA KV Cache、额外 checkpoint、框架元数据和推测解码记录：

| 模型 | KDA 配置 | Recurrent state | ShortConv state（vLLM 默认布局） | 约占用 |
|---|---|---|---|---|
| Kimi Linear 48B | 32 heads，dim 128，W=4 | `[32,128,128]` FP32 | `[3,12288]` BF16 | 2 MiB + 72 KiB |
| Kimi K3 | 96 heads，dim 128，W=4 | `[96,128,128]` FP32 | `[3,36864]` BF16 | 6 MiB + 216 KiB |

K3 的 69 个 KDA 层合计约需要 414 MiB recurrent state 和 14.6 MiB ShortConv state，也就是约 428.6 MiB/request；这仍未包含 24 个 MLA 层随序列长度增长的 KV Cache。

实际部署会按 Tensor Parallel 切分 head 与投影维度。例如 TP=8 时，每张卡每个 KDA 层只保留 12 个 head：recurrent state 约 768 KiB，ShortConv state 约 27 KiB，69 层合计约 53.6 MiB/card/request。该估算按普通 decode state 计算；开启 RecoverSSM、推测解码或额外 Prefix Checkpoint 时还会增加其他状态。

这组数字解释了一个看似矛盾的现象：

> **KDA state 不随 sequence length 增长，但“常数”本身可以很大；decode 只保留最新状态很划算，Prefix Cache 若保留大量历史快照仍然会迅速吃满显存。**

## 九、vLLM 是怎么把这件事工程化的？

需要说明的是：[Kimi Linear 原始论文](https://arxiv.org/pdf/2510.26692)主要定义 KDA 与混合架构，没有详细展开跨请求 Prefix Cache 的状态保留策略。下面的工程细节来自 vLLM 后续对 KDA 混合模型的生产实现，尤其是 Kimi K3 的支持方案。

### 1. 把三个“粒度”解耦

vLLM 不再强迫下面三者完全相同：

- **物理状态块大小**：GPU 上怎样分配较大的 KDA state block；
- **调度对齐点**：执行在哪些位置暂停并注册有效状态；
- **前缀匹配粒度**：token 哈希可以在哪些更细的边界命中。

这样可以用大物理块摊薄状态存储成本，同时允许更细粒度的逻辑前缀命中。具体设计可参考 vLLM 的 [Kimi K3 Prefix Caching 说明](https://vllm.ai/blog/2026-07-22-kimi-k3-preview)。

### 2. Copy-on-write

KDA decode 会原地更新 \(S_t\)。如果两个请求直接共享同一份命中状态，其中一个请求向后计算时就会污染公共缓存。

所以命中后的正确做法是：

```text
                     ┌─ copy -> 请求 A 私有状态 -> 继续更新
cached S_prefix  ----┤
                     └─ copy -> 请求 B 私有状态 -> 继续更新

公共 cached S_prefix 保持不变
```

vLLM 会先把共享状态复制到请求自己的目标块，再继续执行后缀。

### 3. 有选择地保留 checkpoint

vLLM 后续还提供了两类保留思路：

- **按间隔保留**：例如每 32K token 保存一次，同时优先保留 prompt end；
- **按复用价值保留**：一个前缀再次出现、证明它确实是热点时，再为它保留昂贵的 KDA state。

本质上是在权衡：checkpoint 越密，重算越少，但缓存占用越大；checkpoint 越稀，显存更省，但命中公共前缀后可能仍需补算一段。

## 十、四个最容易踩的误区

### 误区 1：KDA 没有 KV Cache，所以它没有 cache

错。KDA 没有的是**随 token 数增长的逐 token KV Cache**，它仍需要当前递归状态和短卷积状态。

### 误区 2：没有 KV，就无法判断前缀是否相同

错。前缀匹配靠 token ID 和哈希索引；KV 或 state 是哈希命中后复用的载荷。

### 误区 3：KDA Prefix Cache 可以像 KV block 一样任意拼接片段

通常不行。\(S_p\) 是严格按顺序处理前缀 \(x_{1:p}\) 的结果，具有路径依赖。它适合恢复某个完整前缀边界，不等价于可任意拼接的独立 token KV block。

### 误区 4：KDA 的 Prefix Cache 一定是 \(O(1)\)

只有每个活跃序列的**最新 decode state**对序列长度是 \(O(1)\)。跨请求缓存多少个历史 checkpoint，是服务系统的容量与淘汰策略问题。

## 十一、一句话收尾

如果一定要用一个比喻：

> Full Attention 像把历史原文逐页保存；KDA 像保存“读到这一页时的大脑状态”。

下次遇到同一本书的相同前缀，系统先用页码与内容哈希确认“确实读过”，然后不再逐页重读，而是恢复当时的大脑状态继续往后看。

所以，KDA Prefix Cache 最准确的总结是：

\[
\boxed{
\text{用 token hash 命中前缀，}
\quad
\text{用 state checkpoint 恢复计算}
}
\]

真正难的从来不是“能不能缓存”，而是：**在状态快照很贵、递归状态会原地更新、模型又混有传统 KV Cache 的前提下，怎样选择值得保存的边界，并让两类缓存始终对齐。**

---

## 参考资料

1. Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
2. MoonshotAI, [Kimi-Linear 官方代码与模型说明](https://github.com/MoonshotAI/Kimi-Linear)
3. vLLM Team, [A Preview of Production-Scale Kimi K3 Support on vLLM](https://vllm.ai/blog/2026-07-22-kimi-k3-preview)
4. vLLM Team and Inferact, [Kimi K3 Is Here: Efficient Day-0 Support on vLLM](https://vllm.ai/blog/2026-07-27-k3)
5. MoonshotAI, [Kimi-Linear-48B-A3B-Instruct config.json](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/config.json)
6. MoonshotAI, [Kimi K3 config.json](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)
7. vLLM, [KDA state dtype 与 shape calculator](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/mamba_utils.py)
8. vLLM, [Kimi K3 KDA NVIDIA 实现](https://github.com/vllm-project/vllm/blob/main/vllm/models/kimi_k3/nvidia/kda.py)
