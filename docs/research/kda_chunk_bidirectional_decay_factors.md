<!--
Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# KDA chunk 内为何同时出现 \(e^{\sum g}\) 与 \(e^{-\sum g}\)

> 日期：2026-09-01
>
> 结论范围：Kimi Delta Attention（KDA）的 chunk/WY/UT 前向算法，以及
> `/workspace/xllm-glm5-next` 中固定 chunk size 64 的 MLU 实现。
>
> 本地基线版本：`04fae8b3ad00202c91fd9628568713159326b37f`。当前工作树另有一个尚未提交的
> optimized candidate，文中会明确区分。

## 结论先行

在 safe gate \(g_t\in[-5,0]\) 下，chunk 算法里出现
\(e^{\sum g}\) 和 \(e^{-\sum g}\)，不是因为模型需要同时做“衰减”和
“反衰减”，也不是 safe gate 额外引入了某个逆向门。

它们来自同一个累计对角衰减矩阵及其逆：

\[
P_r=\operatorname{Diag}(e^{G_r}),\qquad
P_r^{-1}=\operatorname{Diag}(e^{-G_r}),\qquad
G_r=\sum_{j=1}^{r}g_j.
\]

把逐 token 的 KDA 状态递推换到去累计衰减的坐标系后：

- \(P_r k_r=e^{G_r}\odot k_r\) 是当前 token 在该坐标系里用于读取、比较旧状态的 key；
- \(P_r^{-1}k_r=e^{-G_r}\odot k_r\) 是当前 token 写入该坐标系时的 key；
- 两者在任意因果 token 对 \(i<r\) 上相乘，得到的才是模型真正需要的相对衰减：

\[
(e^{G_r}\odot k_r)^\top(e^{-G_i}\odot k_i)
=\sum_d k_{r,d}k_{i,d}e^{G_{r,d}-G_{i,d}}.
\]

因为 \(G_r-G_i=\sum_{j=i+1}^{r}g_j\le 0\)，这里表达的是 token \(i\)
写入之后、到 token \(r\) 之前实际经历的遗忘。正负两个绝对指数只是把这个相对量拆成
两个可做矩阵乘的操作数。

因此还需要区分两件事：

1. **数学公式会暴露两个互逆因子。** Kimi Linear 的 UT 形式明确包含
   \(\Gamma\odot K\) 与 \(K/\Gamma\)。
2. **kernel 不一定、也不应该完整物化二者。** 对 chunk 64，直接物化
   \(e^{-G}\) 会溢出；本地 MLU 实现实际使用相对差值或 anchor 分解，避免构造这个
   危险的完整张量。

Kimi Linear 论文给出了 KDA 递推、WY 展开、UT 变换和输出公式；其中 UT 矩阵正是由
\((\Gamma\odot K)(K/\Gamma)^\top\) 构成。[Kimi Linear，第 3.1 节及式 (1)–(9)](https://arxiv.org/html/2510.26692#S3.SS1)

## 1. 从原始 KDA 递推开始

以下先只看一个 head。KDA 的 per-key-dimension gate 为向量
\(\alpha_t=e^{g_t}\)，对应对角矩阵

\[
D_t=\operatorname{Diag}(\alpha_t).
\]

原始递推是

\[
S_t=(I-\beta_t k_tk_t^\top)D_tS_{t-1}
    +\beta_tk_tv_t^\top,
\qquad
o_t=S_t^\top q_t.
\tag{1}
\]

这也是 Kimi Linear 论文的式 (1)：先让旧状态经过 per-dimension decay，再执行
delta-rule erase，最后写入当前 key-value。[Kimi Linear，式 (1)](https://arxiv.org/html/2510.26692#S3)

式 (1) 的直接实现必须依赖 \(S_{t-1}\) 才能得到 \(S_t\)，所以 token 维天然串行。
chunk 算法要做的，是把 chunk 内连续 \(C\) 次“对角衰减 + rank-1 更新”压缩成
一次小型三角系统和几次大矩阵乘；chunk 之间只保留一次状态递推。论文的式 (2)–(5)
先把这串变换展开成 WY 表示，式 (6)–(9) 再给出适合硬件的 UT 形式。
[Kimi Linear，chunkwise/WY/UT 推导](https://arxiv.org/html/2510.26692#S3.SS1)

## 2. 两个指数的代数来源：一次坐标变换

### 2.1 定义 chunk 内累计衰减

在一个 chunk 内令

\[
G_r=\sum_{j=1}^{r}g_j,
\qquad
P_r=\prod_{j=1}^{r}D_j
   =\operatorname{Diag}(e^{G_r}).
\tag{2}
\]

这里的指数、求和和乘法都按 key dimension 逐元素进行。论文把相同的累计乘积写成
\(\gamma^r=\prod_{j=1}^{r}\alpha_j\)，并用 \(\Gamma\) 堆叠一个 chunk 内所有
\(\gamma^r\)。[Kimi Linear，Decay Formulation](https://arxiv.org/html/2510.26692#S2.SS1)

本地实现接收 log-space gate，并先在每个 64-token chunk 内用 FP32 计算前缀和：
[chunk_kda_fwd.py:23](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/chunk_kda_fwd.py:23)、
[chunk_kda_fwd.py:68](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/chunk_kda_fwd.py:68)。
调用链把这个 `cumulative_gate` 依次交给 KKT/QKT、三角求解后的 W/U 重建，以及
state/output 阶段：[chunk_kda.cpp:572](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/chunk_kda.cpp:572)。

### 2.2 去掉显式累计衰减

定义一个新状态

\[
R_r=P_r^{-1}S_r,
\qquad S_r=P_rR_r.
\tag{3}
\]

因为 \(P_r=D_rP_{r-1}\)，把式 (3) 代回式 (1)：

\[
\begin{aligned}
P_rR_r
&=(I-\beta_rk_rk_r^\top)P_rR_{r-1}
  +\beta_rk_rv_r^\top,\\
R_r
&=R_{r-1}
  -\beta_r(P_r^{-1}k_r)(k_r^\top P_r)R_{r-1}
  +\beta_r(P_r^{-1}k_r)v_r^\top.
\end{aligned}
\]

定义

\[
k_r^+=P_rk_r=e^{G_r}\odot k_r,
\qquad
k_r^-=P_r^{-1}k_r=e^{-G_r}\odot k_r,
\tag{4}
\]

就得到

\[
R_r=R_{r-1}
  +\beta_r k_r^-
   \left(v_r-R_{r-1}^\top k_r^+\right)^\top.
\tag{5}
\]

式 (5) 给出了最直接的直觉：

- \(k_r^+\) 用来从去衰减坐标下的旧状态读取当前 key 已有的 value，即
  \(R_{r-1}^\top k_r^+\)；
- \(k_r^-\) 用来把当前残差写入去衰减坐标下的状态；
- \(P_r\) 和 \(P_r^{-1}\) 是一对坐标变换，不是两种相反的模型动力学。

输出也印证这一点：

\[
o_r=S_r^\top q_r
   =R_r^\top(P_rq_r)
   =R_r^\top(e^{G_r}\odot q_r).
\tag{6}
\]

所以 query 与“读侧”key 使用正累计指数；写入变换状态的 key 使用负累计指数。

### 2.3 为什么两者相乘后正好是相对衰减

当前 token \(r\) 读取先前 token \(i\) 的写入时，出现

\[
\begin{aligned}
(k_r^+)^\top k_i^-
&=(P_rk_r)^\top(P_i^{-1}k_i)\\
&=\sum_d k_{r,d}k_{i,d}e^{G_{r,d}-G_{i,d}}\\
&=\sum_d k_{r,d}k_{i,d}
  \exp\left(\sum_{j=i+1}^{r}g_{j,d}\right).
\end{aligned}
\tag{7}
\]

下标从 \(i+1\) 开始很重要：token \(i\) 的新 value 是在第 \(i\) 步 decay
之后才写入状态的，所以它不应再次承受 \(g_i\)；它只经历后续 \(i+1,\ldots,r\)
步的遗忘。

这也说明所谓 \(e^{-G_i}\) 并没有造成真实的放大。它最终必须与更晚位置的
\(e^{G_r}\) 配对，组合结果是 \(e^{G_r-G_i}\le 1\)。

## 3. 为什么不直接只保留 \(e^{G_r-G_i}\)

从数学值看，只计算相对衰减当然最安全；但从并行算法看，\(G_r-G_i\) 同时依赖
row token \(r\)、column token \(i\) 和 key dimension \(d\)。如果先为所有 token
对显式生成

\[
E_{r,i,d}=e^{G_{r,d}-G_{i,d}},
\]

就会得到一个 \(C\times C\times d_k\) 的三维中间量，再沿 \(d_k\) 归约，难以直接
映射到高吞吐 GEMM。

指数恒等式把它分离为两个 \(C\times d_k\) 操作数：

\[
e^{G_{r,d}-G_{i,d}}
=e^{G_{r,d}}e^{-G_{i,d}}.
\]

于是全部 token 对可以写成一次矩阵乘：

\[
(\Gamma\odot K)(K/\Gamma)^\top.
\tag{8}
\]

Kimi Linear 的 UT 矩阵正是

\[
M=\left(I+\operatorname{StrictTril}\left(
\operatorname{Diag}(\beta)
(\Gamma\odot K)(K/\Gamma)^\top
\right)\right)^{-1}\operatorname{Diag}(\beta),
\tag{9}
\]

随后计算 \(W=M(\Gamma\odot K)\)、\(U=MV\)。输出的 chunk 内项同样使用
\((\Gamma\odot Q)(K/\Gamma)^\top\)。这些分别是论文式 (6)、(7) 和 (9)。
[Kimi Linear，UT 与输出公式](https://arxiv.org/html/2510.26692#S3.SS1)

所以“同时构造两个东西”的硬件原因是：**把 pair-dependent 的相对衰减分解到
GEMM 的左右操作数上**。两个操作数是获得矩阵乘并行度的代价，而不是模型定义里有
两套 gate。

## 4. 两个因子在 chunk 各计算阶段分别做什么

设 chunk 长度为 \(C\)，最后一个有效位置为 \(C\)，并定义

\[
Q^+=\Gamma\odot Q,\quad
K^+=\Gamma\odot K,\quad
K^-=K/\Gamma.
\]

| 阶段 | 数学对象 | 两类因子的用途 | 本地实现 |
| --- | --- | --- | --- |
| gate prefix | \(G_r=\sum_{j\le r}g_j\) | 先保留 log-space 累计量，不急于求指数 | FP32 chunk-local cumsum：[chunk_kda_fwd.py:68](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/chunk_kda_fwd.py:68) |
| KKT / QKT | \(K^+(K^-)^\top\)、\(Q^+(K^-)^\top\) | 构造 delta-rule 的严格下三角系统，以及输出的因果下三角交互 | 跨 16-token block 使用 anchor 分解：[chunk_kda_fwd.py:175](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/chunk_kda_fwd.py:175)；block 内直接算 \(e^{G_r-G_i}\)：[chunk_kda_fwd.py:294](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/chunk_kda_fwd.py:294) |
| triangular solve | 式 (9) 的 \(M\) | 已经进入 \(C\times C\) 系数矩阵，不再需要单独操作 gate 指数 | `solve_tril`：[chunk_kda.cpp:389](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/chunk_kda.cpp:389) |
| W/U 重建 | \(W=M K^+\)、\(U=MV\) | \(K^+\) 编码从 chunk 起点到每个 token 的旧状态读取/擦除强度 | `weighted_keys = k * beta * exp(G)`：[chunk_kda_fwd.py:444](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/chunk_kda_fwd.py:444) |
| inter-chunk output | \(Q^+S_{\mathrm{in}}\) | \(Q^+\) 读取从 chunk 起点衰减到当前位置的输入状态 | `q * exp(G)`：[chunk_kda_fwd.py:587](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/chunk_kda_fwd.py:587) |
| chunk 末状态 | \(\operatorname{Diag}(e^{G_C})S_{\mathrm{in}}\) | 输入状态整体衰减到 chunk 末尾 | [chunk_kda_fwd.py:621](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/chunk_kda_fwd.py:621) |
| chunk 内写入带到末尾 | \(e^{G_C-G_i}\odot k_i\) | token \(i\) 的新写入只承受其后的 gate，避免错误包含 \(g_i\) | `gated_k = k * exp(last_gate - gate)`：[chunk_kda_fwd.py:454](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/chunk_kda_fwd.py:454) |

最后两个量组合成论文式 (8) 的 chunk 状态更新：

\[
S_{\mathrm{out}}
=\operatorname{Diag}(e^{G_C})S_{\mathrm{in}}
+\left(e^{G_C-G_i}\odot K_i\right)^\top
 \left(U-WS_{\mathrm{in}}\right).
\tag{10}
\]

论文把它写成
\(\operatorname{Diag}(\gamma^C)S+(\Gamma^{i\to C}\odot K)^\top(U-WS)\)。
[Kimi Linear，式 (8)](https://arxiv.org/html/2510.26692#S3.SS1)

本地 `state_output` kernel 的结构与式 (10) 一一对应：先算
`new_values = u - w @ state`，再用 `q * exp(G)` 读取旧状态、叠加 intra-chunk
输出，最后让旧状态乘 `exp(last_gate)` 并加上 `gated_k.T @ new_values`。
[chunk_kda_fwd.py:549](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/chunk_kda_fwd.py:549)、
[chunk_kda_fwd.py:591](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/chunk_kda_fwd.py:591)、
[chunk_kda_fwd.py:633](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/chunk_kda_fwd.py:633)。

FlashKDA 的官方 Torch reference 更直接地展示了原始双因子形式：先计算
`g_cumsum`，再生成 `k_decayed = k * exp(g_cumsum)`、
`q_decayed = q * exp(g_cumsum)`、`k_inv = k * exp(-g_cumsum)`，以
`k_decayed @ k_inv.T` 和 `q_decayed @ k_inv.T` 构造 KKT/QKT；随后用
`k_restored = k_inv * exp(g_total)` 得到 \(k_i e^{G_C-G_i}\)。
[FlashKDA Torch reference:208–239](https://github.com/MoonshotAI/FlashKDA/blob/7afb9f454f160a6c4bbc0999beca0a8c40a38934/tests/torch_ref.py#L208-L239)
官方 CUDA K1/K2 也分别构造这些 decay operands 并在递推阶段消费
`k_decayed/q_decayed/k_restored/g_total/INV/Mqk`。
[FlashKDA K1:450–485](https://github.com/MoonshotAI/FlashKDA/blob/7afb9f454f160a6c4bbc0999beca0a8c40a38934/csrc/smxx/fwd_kernel1.cuh#L450-L485)
[FlashKDA K2:533–660](https://github.com/MoonshotAI/FlashKDA/blob/7afb9f454f160a6c4bbc0999beca0a8c40a38934/csrc/smxx/fwd_kernel2.cuh#L533-L660)

## 5. 一个三 token 小例子

为了只看 gate，先忽略 key dimension，取

\[
g_1=-0.2,\qquad g_2=-0.7,\qquad g_3=-0.1.
\]

累计量和两个形式上的坐标因子是：

| token \(r\) | \(G_r\) | \(e^{G_r}\) | \(e^{-G_r}\) |
| ---: | ---: | ---: | ---: |
| 1 | \(-0.2\) | \(0.8187\) | \(1.2214\) |
| 2 | \(-0.9\) | \(0.4066\) | \(2.4596\) |
| 3 | \(-1.0\) | \(0.3679\) | \(2.7183\) |

token 3 读取 token 1 的写入时，两侧因子相乘：

\[
e^{G_3}e^{-G_1}
=e^{-1.0}e^{0.2}
=e^{-0.8}
\approx0.4493.
\]

而

\[
G_3-G_1=g_2+g_3=-0.8.
\]

这说明 token 1 的新写入只经历了后续的 \(g_2\) 与 \(g_3\)，没有把写入发生前的
\(g_1\) 错误地再乘一次。

在 chunk 末状态里也一样：

- chunk 起始状态走到末尾，乘 \(e^{G_3}=e^{g_1+g_2+g_3}\approx0.3679\)；
- token 2 的写入走到末尾，只乘
  \(e^{G_3-G_2}=e^{g_3}\approx0.9048\)；
- token 3 的写入刚发生，不再衰减，乘 \(e^{G_3-G_3}=1\)。

这正是 `gated_k = k * exp(last_gate - gate)` 所编码的三种距离。

## 6. safe gate 为什么没有自动解决数值问题

### 6.1 safe gate 保证的是单步收缩，不是逆累计量有界

本地 GLM5-Next safe gate 的实现是

\[
g=\text{lower\_bound}\cdot
\sigma\!\left(e^{A_{\log}}(g_{\mathrm{raw}}+\text{bias})\right),
\]

其中 `lower_bound < 0`；默认下界为 \(-5\)，所以 gate 位于 \((-5,0)\)。
[glm5_next_kda.cpp:63](/workspace/xllm-glm5-next/xllm/core/layers/mlu/glm5_next/glm5_next_kda.cpp:63)、
[glm5_next_kda.h:68](/workspace/xllm-glm5-next/xllm/core/layers/mlu/glm5_next/glm5_next_kda.h:68)。
prefill 路径先生成这个 log gate，再传入 ChunkKDA。
[glm5_next_kda.cpp:573](/workspace/xllm-glm5-next/xllm/core/layers/mlu/glm5_next/glm5_next_kda.cpp:573)

若 chunk 长度为 \(C\)，最坏情况下

\[
G_r\in[-5C,0],\quad
e^{G_r}\in[e^{-5C},1],\quad
e^{-G_r}\in[1,e^{5C}].
\tag{11}
\]

因此 safe gate 只保证每一步 \(e^{g_t}\le1\)；它并不保证代数变换产生的
\(e^{-G_r}\) 不会越来越大。

### 6.2 chunk 16 与 chunk 64 的范围差异

| chunk | \(G\) 最小值 | \(e^G\) 最小值 | \(e^{-G}\) 最大值 | 直接分解风险 |
| ---: | ---: | ---: | ---: | --- |
| 16 | \(-80\) | \(1.80\times10^{-35}\) | \(5.54\times10^{34}\) | FlashKDA v1 认为仍处于 BF16 可表示范围 |
| 64 | \(-320\) | \(1.06\times10^{-139}\) | \(9.42\times10^{138}\) | BF16/FP32 下分别趋向下溢与上溢 |

MoonshotAI 对 FlashKDA v1 的官方说明明确把内部 chunk 选为 16；首要原因就是在
`lower_bound=-5` 时，让 `exp(cumsum(g))` 保持在 BF16 可表示范围，从而不需要大
chunk 的复杂 intra-chunk rescaling。[FlashKDA v1 deep dive，Chunk Size Selection](https://github.com/MoonshotAI/FlashKDA/blob/7afb9f454f160a6c4bbc0999beca0a8c40a38934/docs/20260420-flashkda-v1-deep-dive.md#L11-L18)
其 C++ 入口也固定 `CHUNK = 16`，并为 `k_decayed/q_decayed/k_restored` 分配 workspace。
[FlashKDA C++ entry point](https://github.com/MoonshotAI/FlashKDA/blob/7afb9f454f160a6c4bbc0999beca0a8c40a38934/csrc/flash_kda.cpp#L9-L24)

本地 MLU ChunkKDA 则固定 chunk size 64，并进一步切成 16-token block：
[chunk_kda.h:25](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/chunk_kda.h:25)、
[chunk_kda.cpp:36](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/chunk_kda.cpp:36)。

### 6.3 真正危险的是“绝对大数 × 绝对小数”

假设两个较晚位置满足

\[
G_i=-290,\qquad G_r=-300.
\]

模型需要的相对量只是

\[
e^{G_r-G_i}=e^{-10}\approx4.54\times10^{-5},
\]

它完全正常。但若先分别计算，则会接近

\[
e^{-300}\cdot e^{290}=0\cdot\infty,
\]

从而产生人工的 `0 * inf`、NaN 或完全丢失精度。这不是模型的真实数值范围，而是
不稳定因式分解造成的中间量范围。

Kimi Linear 论文也明确指出，fine-grained gate 会在除法（例如式 (9) 的
intra-chunk 计算）中带来数值精度问题；已有算法需要 log-domain 和 full-precision
secondary chunking。[Kimi Linear，第 3.2 节](https://arxiv.org/html/2510.26692#S3.SS2)

## 7. 本地 MLU 实现如何避免直接物化 \(e^{-G}\)

### 7.1 同一个 16-token block 内：直接算差值

对 block 内的因果 token 对，本地 kernel 直接计算

\[
e^{G_r-G_i},
\]

然后再与 key/query 相乘，而不是先生成 \(e^{G_r}\) 与 \(e^{-G_i}\)。对应代码是
`tl.exp(gate_rows - gate_column)`，并分别用严格下三角和含对角下三角 mask 生成
KKT 与 QKT。[chunk_kda_fwd.py:294](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/chunk_kda_fwd.py:294)

因果区域中 \(r\ge i\)，而 gate 非正，所以 \(G_r-G_i\le0\)。这条路径不会发生
正指数溢出；若间隔衰减真的极强，可能下溢为 0，但那对应的真实相对贡献本来就已经
极小。

### 7.2 不同 16-token block 之间：用 anchor 拆分

为了继续使用 GEMM，跨 block 路径选择 row block 起点的累计 gate \(a\) 作为 anchor：

\[
e^{G_r-G_i}
=e^{G_r-a}\,e^{a-G_i}.
\tag{12}
\]

对“更晚的 row block × 更早的 column block”：

- row 位于 anchor 之后，所以 \(G_r\le a\)，即 \(e^{G_r-a}\le1\)；
- column 位于 anchor 之前，所以 \(G_i\ge a\)，即 \(e^{a-G_i}\le1\)。

两个 GEMM 操作数都不超过 1，但乘积仍精确等于原来的相对衰减。本地实现加载
`normalizer`，分别形成 `exp(gate_rows - normalizer)` 和
`exp(normalizer - gate_columns)`，再执行 KKT/QKT 矩阵乘。
[chunk_kda_fwd.py:173](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/chunk_kda_fwd.py:173)

当前工作树里的 optimized candidate 对这一意图写得更明确：它保留归一化 Q/K 与
log-space cumulative gate，注释说明 chunk 64 最坏可到 \(-320\)，不能物化
`exp(-cumulative_values)`；KKT kernel 随后按 row-block anchor 生成两个相对因子。
该文件目前是**未跟踪的本地候选实现**，不能当作固定提交中的能力。
[chunk_kda_fwd_optimized.py:118](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/chunk_kda_fwd_optimized.py:118)、
[chunk_kda_fwd_optimized.py:215](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/chunk_kda_fwd_optimized.py:215)。

FLA 的上游 KDA 也采用同类 secondary-block rescaling：对不同子块选择局部 gate
normalizer，再以 `exp2(g_row - anchor)` 和 `exp2(anchor - g_column)` 形成 GEMM
操作数；对单个子块则明确限制指数范围以避免 `exp2` 溢出。
[FLA `chunk_intra.py` 的跨块 rescaling](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/kda/chunk_intra.py#L145-L218)
[FLA `chunk_intra.py` 的子块双因子](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/kda/chunk_intra.py#L742-L760)

### 7.3 W、Q 和 chunk 末状态只需要真实的正向衰减

本地实现仍会计算 `exp(G)`，但这些位置没有与一个全局 `exp(-G)` 张量相乘：

- `W` 的 seed 使用 \(k_i\beta_i e^{G_i}\)，表达输入状态从 chunk 起点走到 token
  \(i\) 的真实衰减；
- inter-chunk output 使用 \(q_i e^{G_i}\)，表达输入状态对 token \(i\) 的真实贡献；
- 旧状态到 chunk 末尾使用 \(e^{G_C}\)；
- chunk 内新写入使用稳定的相对量 \(e^{G_C-G_i}\le1\)。

证据分别见
[chunk_kda_fwd.py:446](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/chunk_kda_fwd.py:446)、
[chunk_kda_fwd.py:591](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/chunk_kda_fwd.py:591)、
[chunk_kda_fwd.py:633](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/chunk_kda_fwd.py:633)、
[chunk_kda_fwd.py:460](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/chunk_kda_fwd.py:460)。

这些 \(e^G\) 在极端长衰减下可能下溢，但它们表示的旧状态贡献本身也确实趋近于 0；
它与“先上溢、再依赖另一个因子抵消”的人工数值问题不同。实现仍应通过高精度参考
逐 case 验证这种下溢是否符合误差目标。

## 8. 实现建议

### 必须保持的数学约束

1. **累计 gate 保持在 FP32 log-space。** 不要把 \(e^{-G}\) 作为 chunk 级持久
   workspace；本地基线已经让 `cumulative_gate` 为 FP32，并仅在使用点求相对指数。
2. **任何 KKT/QKT 因果项最终都必须等于 \(e^{G_r-G_i}\)。** 要特别检查下标：
   token \(i\) 的写入到 token \(r\) 应包含 \(g_{i+1},\ldots,g_r\)，不包含
   \(g_i\)。
3. **chunk 末写入 key 使用 \(e^{G_C-G_i}\)，旧状态使用 \(e^{G_C}\)。** 两者
   不能混用，否则会给新写入多乘一次当前 gate，或让旧状态少衰减。
4. **anchor 必须在代数上成对出现。** 若左操作数减去 \(a\)，右操作数必须加回
   \(a\)，确保乘积仍是 \(e^{G_r-G_i}\)。

### chunk 64 的推荐策略

1. 对 16-token 对角子块直接计算因果 `exp(G_row - G_col)`。
2. 对非对角子块使用位于 row/column 之间的 per-dimension anchor，让两个指数均非正，
   同时保留 GEMM 形式。
3. 对 W/U 和 triangular inverse 使用 secondary-block 或分块合并；不要把
   `K / exp(G)` 先作为全 chunk BF16/FP32 张量落盘。FLA 的官方实现展示了 64-token
   chunk 内四个 16-token 子块的 rescaling 与 inverse merge。
   [FLA KDA secondary chunking](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/kda/chunk_intra.py#L807-L859)
4. 如果不想承担多级 rescaling 与 block inverse 的复杂度，可以像 FlashKDA v1 一样
   直接把计算 chunk 固定为 16；这会改变 chunk 数、状态递推次数和性能权衡，但数值范围
   更容易控制。[FlashKDA v1 deep dive](https://github.com/MoonshotAI/FlashKDA/blob/7afb9f454f160a6c4bbc0999beca0a8c40a38934/docs/20260420-flashkda-v1-deep-dive.md#L11-L31)

### 建议的数值测试

至少覆盖以下 adversarial cases，并与 tokenwise FP64/高精度参考递推比较：

- 整个 chunk 的 gate 全为 \(-5\)，直接触达 \(G_C=-5C\)；
- 很晚的两个 token 只有较小相对差，例如 \(G_i=-290,G_r=-300\)，用于发现
  `0 * inf` 型错误；
- gate 接近 0，用于确认累计衰减接近 1 时没有额外缩放；
- 16/17、63/64/65 token 边界与 tail chunk；
- 非零 initial state，分别检查 inter-chunk output 和 final state；
- packed variable-length sequence，确认 cumsum 与 anchor 不跨序列或 chunk 泄漏。

本地接口固定 BF16 Q/K/V、FP32 recurrent state，并支持 packed variable-length
`cu_seqlens/chunk_indices`；这些 dtype 与布局约束应纳入测试基线。
[chunk_kda.cpp:488](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/chunk_kda.cpp:488)

## 9. 常见误解

### “既然 \(g\le0\)，为什么还会有一个正指数？”

\(-G\ge0\) 来自坐标变换的逆 \(P^{-1}\)，不是一个新的正 gate。模型最终使用的
因果指数仍是 \(G_r-G_i\le0\)。

### “safe gate 是否意味着可以放心直接算 \(e^{-G}\)？”

不能。safe gate 限制每一步 \(g_t\)，但累计 \(-G\) 会随 chunk 长度线性增长。
chunk 64、下界 \(-5\) 时最坏达到 320，直接指数远超 BF16/FP32 范围。

### “实现是不是必须同时保存 \(e^G\) 和 \(e^{-G}\) 两个张量？”

不是。公式为了展示可分离 GEMM 会写出二者；稳定 kernel 可以保存 \(G\)，在 tile
内部用直接差值或 anchor 构造两个有界操作数。本地 MLU baseline 正是这样处理
KKT/QKT，且 state 更新只保存 `gated_k = k * exp(G_C-G_i)`。

### “能否所有地方都只算一次 \(e^{G_r-G_i}\)？”

数学上可以，硬件上未必划算。为所有 \((r,i,d)\) 生成三维相对衰减会失去
`[C,d] @ [d,C]` 的标准 GEMM 结构。更合适的办法是按 tile 选择 anchor，既保持
左右操作数可分离，又控制指数范围。

## 10. 一句话总结

KDA chunk 里的 \(e^{\sum g}\) 与 \(e^{-\sum g}\)，本质上是把状态搬入、搬出
“去累计衰减坐标系”的一对变换；它们让 token-pair 相对衰减可以分解成 GEMM 的
左右操作数。真正有物理意义的是二者组合后的

\[
e^{G_r-G_i}=e^{\sum_{j=i+1}^{r}g_j}\le1.
\]

因此公式需要理解两个因子，kernel 却应优先保存 log-space \(G\)，并用直接差值、
anchor 或 secondary chunking 避免物化危险的 \(e^{-G}\)。

## 一手来源

- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/html/2510.26692)
- [MoonshotAI FlashKDA v1 deep dive（固定提交）](https://github.com/MoonshotAI/FlashKDA/blob/7afb9f454f160a6c4bbc0999beca0a8c40a38934/docs/20260420-flashkda-v1-deep-dive.md)
- [MoonshotAI FlashKDA C++ entry point（固定提交）](https://github.com/MoonshotAI/FlashKDA/blob/7afb9f454f160a6c4bbc0999beca0a8c40a38934/csrc/flash_kda.cpp)
- [MoonshotAI FlashKDA Torch reference（固定提交）](https://github.com/MoonshotAI/FlashKDA/blob/7afb9f454f160a6c4bbc0999beca0a8c40a38934/tests/torch_ref.py#L208-L239)
- [MoonshotAI FlashKDA CUDA K1/K2（固定提交）](https://github.com/MoonshotAI/FlashKDA/blob/7afb9f454f160a6c4bbc0999beca0a8c40a38934/csrc/smxx/fwd_kernel1.cuh#L450-L485)
- [FLA KDA chunk implementation（固定提交）](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/kda/chunk_intra.py)
- [xLLM MLU ChunkKDA header](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/chunk_kda.h:25)
- [xLLM MLU ChunkKDA orchestration](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/chunk_kda.cpp:296)
- [xLLM MLU ChunkKDA Triton baseline](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/chunk_kda_fwd.py:23)
- [xLLM MLU ChunkKDA optimized candidate（当前工作树未跟踪）](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/chunk_kda_fwd_optimized.py:40)
