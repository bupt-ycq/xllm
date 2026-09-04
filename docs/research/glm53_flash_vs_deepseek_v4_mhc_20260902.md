# GLM-5.3-Flash 与 DeepSeek-V4 的 mHC 对比

调研日期：2026-09-02

## 结论

两者的**逐层 mHC 完全相同**：GLM-5.3-Flash 的 HyperConnection 直接继承
DeepSeek-V4 的实现，pre/post/comb 参数化、Sinkhorn 归一化和残差更新公式均一致。

两者的**模型末尾多流汇聚不同**：GLM-5.3-Flash 对 4 路残差流直接取算术平均；
DeepSeek-V4 使用带 RMSNorm 和可学习门控参数的 HyperHead 做加权求和。因此如果把
“mHC”理解为从输入展开、多层更新到最终汇聚的完整机制，则不能说完全相同。

## 逐层 mHC

固定 Transformers 提交中，`Glm5NextTextHyperConnection` 直接继承
`DeepseekV4HyperConnection`，类体为 `pass`。两者在 attention 和 FFN 前后都使用同一
HyperConnection 更新。

令残差流为 `X ∈ R^{H×D}`，子层为 `F`，核心计算可概括为：

1. 从归一化并展平的 `X` 生成 `pre ∈ R^H`、`post ∈ R^H` 和 `comb ∈ R^{H×H}`。
2. `pre` 使用 sigmoid 门控；`post` 使用两倍 sigmoid；`comb` 先 softmax，再经多轮
   Sinkhorn 行列归一化成为近似双随机矩阵。
3. 子层输入为 `z = Σ_i pre_i X_i`。
4. 更新为 `X' = post ⊗ F(z) + comb^T X`。

官方配置也一致采用 `hc_mult=4`、`hc_sinkhorn_iters=20`、`hc_eps=1e-6`。

## 最终汇聚差异

| 项目 | GLM-5.3-Flash | DeepSeek-V4 |
|---|---|---|
| 最终 4→1 汇聚 | `hidden_streams.mean(dim=2)` | 学习得到每路 gate 后加权求和 |
| 归一化 | 无 | flattened 4 路状态上的无权重 RMSNorm |
| 可学习 head 参数 | 无 | `hc_fn`、`hc_base`、`hc_scale` |
| 权重是否 softmax | 不适用 | 否；sigmoid gate，不要求和为 1 |

对 Flash 的 `H=4, D=4096`，DeepSeek 最终 HyperHead 额外包含
`H²D + H + 1 = 65,541` 个参数；GLM 的平均汇聚没有参数。官方权重索引也印证了这一点：
DeepSeek-V4-Flash 有 `hc_head_fn/base/scale`，GLM-5.3-Flash 没有对应参数。

## 不应混淆的地方

- 算子/公式相同不代表权重相同或可直接互换；这些参数由各自模型独立训练。
- GLM-5.3-Flash 是 45 层、混合 KDA 与稀疏注意力；DeepSeek-V4-Flash 是 43 层并有
  自己的 attention/MoE 结构。mHC 包裹的子层不同，整体网络当然不同。
- DeepSeek-V4-Pro 虽然使用同一 mHC 公式，但 hidden size 等配置不同，相应权重形状也不同。

## 一手来源

- [Transformers：GLM HyperConnection 继承及平均 HyperHead](https://github.com/huggingface/transformers/blob/ac3244569528944b9d5773cafea525cd8a8b63de/src/transformers/models/glm5_next/modular_glm5_next.py#L364-L372)
- [Transformers：DeepSeek-V4 HyperConnection 与可学习 HyperHead](https://github.com/huggingface/transformers/blob/ac3244569528944b9d5773cafea525cd8a8b63de/src/transformers/models/deepseek_v4/modular_deepseek_v4.py#L769-L864)
- [Transformers：GLM decoder 中两处 mHC 更新](https://github.com/huggingface/transformers/blob/ac3244569528944b9d5773cafea525cd8a8b63de/src/transformers/models/glm5_next/modular_glm5_next.py#L1142-L1205)
- [Transformers：DeepSeek-V4 decoder 中两处 mHC 更新](https://github.com/huggingface/transformers/blob/ac3244569528944b9d5773cafea525cd8a8b63de/src/transformers/models/deepseek_v4/modular_deepseek_v4.py#L977-L1025)
- [GLM-5.3-Flash 官方配置](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/03eb5366286afd40d2221b1d9c63a6dd1ba4832e/config.json)
- [GLM-5.3-Flash 官方权重索引](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/03eb5366286afd40d2221b1d9c63a6dd1ba4832e/model.safetensors.index.json)
- [DeepSeek-V4-Flash 官方配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json)
- [DeepSeek-V4-Flash 官方权重索引](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/model.safetensors.index.json)
- [mHC 论文](https://arxiv.org/abs/2512.24880)
