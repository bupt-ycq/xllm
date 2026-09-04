# xLLM NPU 对 GLM-5.3-Flash / KDA 的支持调研

> 调研日期：2026-08-31（Asia/Shanghai）  
> 范围：仅使用一手来源，包括 xLLM 官方仓库/提交/PR、Z.ai 官方模型配置以及 KDA NPU 算子上游仓库。  
> xLLM 代码快照：`main@d441cabf6be49ccba344852e3f34ee2500a3f376`；实现分支 `preview/glm-5.3-flash@a6ae154c5f95343d53b563634b66ba79d7b7390b`。

## 结论

xLLM 官方已经宣布对 GLM-5.3-Flash 的 day-0 支持，但截至调研快照，**可运行实现仍在 `preview/glm-5.3-flash` 专用分支，默认 `main` 只有公告和部署文档，最近的发布标签也不含模型实现**。因此准确表述是：

- 支持：有完整的 preview 模型注册、权重加载、KDA/DSA 混合模型、NPU 状态缓存和 8 卡单机启动链路。
- 支持等级：preview / 受限支持，而不是当前 release 上的稳定支持。
- 官方明确限制：DP、EP、prefix cache、PD 分离和 MTP 仍处于开发或验收阶段，文档目前只承诺单机、非 PD 部署。[官方中文部署文档 L7-L16](https://github.com/xLLM-AI/xllm/blob/d441cabf6be49ccba344852e3f34ee2500a3f376/docs/src/content/docs/zh/cookbook/autoregressive_models/glm/glm_5_3_flash.md#L7-L16)

他们并没有在 xLLM 内重新实现 Ascend KDA 核心算子。实际 engine 路径把：

- prefill 交给外部 `fla_npu.ops.ascendc.chunk_kda_fwd`；
- decode / MTP verify 交给外部 `fla_npu.ops.ascendc.recurrent_kda`；
- width-4 depthwise causal conv1d、状态编排、变长 batch 拆分、ACL Graph 兼容以及 MTP commit 协议留在 xLLM。

这也是和仓内自研一体化 KDA/GDN 实现对比时最核心的边界：**xLLM-NPU 的优势主要在模型/调度/状态集成，KDA 算力核心依赖外部 `fla_npu`。**

## 1. 仓库与分支定位

没有单独的官方 `xllm-npu` GitHub 仓库；NPU 后端位于统一仓库 [xLLM-AI/xllm](https://github.com/xLLM-AI/xllm)。xLLM-AI 组织的一手仓库列表中有 `xllm`、`xllm-ops`、`torch-npu-ops` 等，但没有 `xllm-npu`：[GitHub organization repos API](https://api.github.com/orgs/xLLM-AI/repos?per_page=100&type=public)。

| 对象 | 快照 / 状态 | 含义 |
|---|---|---|
| 默认分支 `main` | [`d441cabf`](https://github.com/xLLM-AI/xllm/commit/d441cabf6be49ccba344852e3f34ee2500a3f376) | 合入 PR [#2289](https://github.com/xLLM-AI/xllm/pull/2289)，公告支持并新增部署文档；没有合入模型实现 |
| 实现分支 | [`preview/glm-5.3-flash@a6ae154c`](https://github.com/xLLM-AI/xllm/tree/a6ae154c5f95343d53b563634b66ba79d7b7390b) | GLM-5.3-Flash 模型与 KDA 实现所在分支 |
| merge-base | `c56db5b3c54f6485616b6d3990e4056c08d3a2c3` | preview 相对该基线有 34 个提交；调研时相对 main 已分叉 |
| 最近 release | [`v0.10.1@066781fa`](https://github.com/xLLM-AI/xllm/tree/066781fa223a387269cd1f50decc3328ac2da4dc)（2026-07-14） | 早于 GLM-5.3-Flash preview 开发，不含对应模型源文件 |

官方 `main` README 已明确写“day-0 支持”，但链接指向 preview 分支的启动脚本：[README_zh L27-L31](https://github.com/xLLM-AI/xllm/blob/d441cabf6be49ccba344852e3f34ee2500a3f376/README_zh.md#L27-L31)。正式部署文档同样要求 `git checkout preview/glm-5.3-flash` 后再执行 NPU build：[部署文档 L48-L63](https://github.com/xLLM-AI/xllm/blob/d441cabf6be49ccba344852e3f34ee2500a3f376/docs/src/content/docs/zh/cookbook/autoregressive_models/glm/glm_5_3_flash.md#L48-L63)。

当前支持矩阵仍只列到 GLM-5，没有列 GLM-5.3-Flash：[supported_models.md L11-L21](https://github.com/xLLM-AI/xllm/blob/d441cabf6be49ccba344852e3f34ee2500a3f376/docs/src/content/docs/zh/supported_models.md#L11-L21)。

## 2. 支持链路是否完整

preview 分支已经具备一条端到端代码链路：

1. 官方权重仍使用 `model_type=glm5_next` / `Glm5NextForConditionalGeneration`：[官方 config L1-L5、L281-L289](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/04c4e9e95c5da8862dced7e5056455116f83a7e0/config.json#L281-L289)。xLLM 在 C++ registry 中把旧名称映射到 `glm5_3_flash`，并把 Python model executor 作为实际模型实现：[model_registry.cpp L103-L143](https://github.com/xLLM-AI/xllm/blob/a6ae154c5f95343d53b563634b66ba79d7b7390b/xllm/models/model_registry.cpp#L103-L143)。
2. Python registry 注册 `Glm53FlashVLModel` 和 `Glm53FlashMtpForCausalLM`：[registry.py L102-L108](https://github.com/xLLM-AI/xllm/blob/a6ae154c5f95343d53b563634b66ba79d7b7390b/xllm/python/registry.py#L102-L108)。
3. C++ config loader 解析 KDA 的 `full_attn_layers`、heads、head dim、conv width，并声明 SSM state 使用 FP32：[glm5_3_flash.h L102-L140](https://github.com/xLLM-AI/xllm/blob/a6ae154c5f95343d53b563634b66ba79d7b7390b/xllm/models/llm/glm5_3_flash.h#L102-L140)。
4. 模型层 `Glm53FlashKdaAttention` 把 KDA 执行委托给 NPU attention backend：[glm5_3_flash.py L790-L911](https://github.com/xLLM-AI/xllm/blob/a6ae154c5f95343d53b563634b66ba79d7b7390b/xllm/python/models/glm5_3_flash.py#L790-L911)。
5. backend 为 KDA 层分配并维护 conv/SSM 状态槽，同时允许同一模型中的 DSA index cache 共存：[kv_cache_shape.cpp L75-L94](https://github.com/xLLM-AI/xllm/blob/a6ae154c5f95343d53b563634b66ba79d7b7390b/xllm/core/framework/kv_cache/kv_cache_shape.cpp#L75-L94)。
6. 官方文档给出 A2/A3、W8A8、单机 8 卡、`--model_impl=python --backend=vlm` 的启动方案：[部署文档 L115-L169](https://github.com/xLLM-AI/xllm/blob/d441cabf6be49ccba344852e3f34ee2500a3f376/docs/src/content/docs/zh/cookbook/autoregressive_models/glm/glm_5_3_flash.md#L115-L169)。

因此，“代码是否能识别并运行这个模型”的答案是有；“默认分支和发布版是否已经稳定承载”的答案是否。

## 3. 模型里的 KDA 结构

Z.ai 官方模型配置（[`04c4e9e9`](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/04c4e9e95c5da8862dced7e5056455116f83a7e0/config.json)）显示：

- 45 层中有 34 层 KDA linear attention 和 11 层 DSA sparse/full attention，顺序基本为 `KDA, KDA, KDA, DSA`，最后第 44 层也是 KDA：[官方 config L79-L180](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/04c4e9e95c5da8862dced7e5056455116f83a7e0/config.json#L79-L180)。
- KDA 为 64 heads、head dim 128、short conv kernel size 4、forget gate lower bound `-5.0`：[官方 config L126-L180](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/04c4e9e95c5da8862dced7e5056455116f83a7e0/config.json#L126-L180)。
- 模型最大位置为 1,048,576，启用 mHC：[官方 config L181-L183](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/04c4e9e95c5da8862dced7e5056455116f83a7e0/config.json#L181-L183)。DSA 部分是 NoPE MLA，`qk_rope_head_dim=0`：[官方 config L245-L248](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/04c4e9e95c5da8862dced7e5056455116f83a7e0/config.json#L245-L248)。

xLLM 的 KDA layer 由以下部分组成：

1. q/k/v 三个独立的 column-parallel projection，按 head 做 TP shard。
2. q/k/v 拼成 `[q_local | k_local | v_local]`，经过 full-depthwise causal conv1d；每个通道单独卷积，kernel width 4，随后 SiLU。
3. forget gate：默认安全门控分支为 `g = -5 * sigmoid(exp(A_log) * gate_projection)`；写入状态时使用 `exp(g)` 作为衰减。
4. beta：`sigmoid(b_proj(hidden_states))`。
5. delta-rule 输出再经过 per-head RMSNorm、额外 sigmoid output gate、row-parallel `o_proj`，TP>1 时 all-reduce。

对应实现见 [forget gate L748-L787](https://github.com/xLLM-AI/xllm/blob/a6ae154c5f95343d53b563634b66ba79d7b7390b/xllm/python/models/glm5_3_flash.py#L748-L787) 和 [KDA layer L790-L911](https://github.com/xLLM-AI/xllm/blob/a6ae154c5f95343d53b563634b66ba79d7b7390b/xllm/python/models/glm5_3_flash.py#L790-L911)。

### 3.1 Delta-rule 数学

xLLM 文件中保留了一份 pure-Torch reference。每个 token 的核心递推可概括为：

```text
q, k <- L2Norm(q), L2Norm(k)
q    <- q / sqrt(d)
S    <- exp(g) * S
m    <- k^T S
delta <- beta * (v - m)
S    <- S + k outer delta
o    <- q^T S
```

源码锚点：[fused_recurrent_kda L389-L440](https://github.com/xLLM-AI/xllm/blob/a6ae154c5f95343d53b563634b66ba79d7b7390b/xllm/python/models/glm5_3_flash.py#L389-L440)。多 token prefill 的 chunk reference 用 chunk size 64，把 chunk 内三角递推和跨 chunk recurrent state 合并计算：[chunk_kda L443-L538](https://github.com/xLLM-AI/xllm/blob/a6ae154c5f95343d53b563634b66ba79d7b7390b/xllm/python/models/glm5_3_flash.py#L443-L538)。

这两份 pure-Torch 函数主要是数学参考；实际 engine path 不调用它们完成核心 KDA。

## 4. 实际 NPU 执行路径

### 4.1 依赖边界

`NpuPagedAttentionBackend.execute_linear()` 无条件从 `fla_npu.ops.ascendc` 导入 `chunk_kda_fwd` 和 `recurrent_kda`：[npu_paged_attention.py L1372-L1403](https://github.com/xLLM-AI/xllm/blob/a6ae154c5f95343d53b563634b66ba79d7b7390b/xllm/python/attention/npu_paged_attention.py#L1372-L1403)。`fla_npu` 的一手上游是 [flashserve/flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu)。

xLLM 会在 `aclInit` 之前 best-effort 注册 `fla_npu` 自带 OPP，否则 KDA 调用会报 `aclnnStatus=561103`：[xllm.cpp L275-L291](https://github.com/xLLM-AI/xllm/blob/a6ae154c5f95343d53b563634b66ba79d7b7390b/xllm/xllm.cpp#L275-L291)。但是在该 preview 快照的 `setup.py`、`pyproject.toml` 和 requirements 中未找到明确的 `fla_npu` 版本 pin；兼容版本实际依赖官方开发镜像环境。这是复现和后续合并时需要补齐的契约。

### 4.2 Prefill

prefill 路径：

- causal conv1d 仍由 xLLM 的 PyTorch 实现完成；多序列变长 batch 按 sequence 做 conv loop；
- q/k 先以 FP32 做 L2Norm，再转 BF16；v 为 BF16，gate/beta 为 FP32；
- 调 `chunk_kda_fwd(..., chunk_size=64, layout="BSND", state_v_first=True)`；
- 默认多序列 prefill 逐 sequence 调 kernel，目的是让并发结果和单请求的状态数值路径一致。

源码锚点：[prefill dispatch L2283-L2343](https://github.com/xLLM-AI/xllm/blob/a6ae154c5f95343d53b563634b66ba79d7b7390b/xllm/python/attention/npu_paged_attention.py#L2283-L2343)。

### 4.3 Decode

decode 路径：

- q/k/v 打包为 TND `[T, heads_local, 128]`，用 `cu_seqlens` 表达多请求变长边界；
- q/k L2Norm 在 kernel 内完成；gate、beta 已在模型侧算好，因此 `use_gate_in_kernel=False`、`use_beta_sigmoid_in_kernel=False`；
- 调 `recurrent_kda(..., state_v_first=True)` 并返回/更新最终 SSM state；
- MTP 或多 token decode 仍按 metadata 的 decode/prefill 类型路由，而不是靠 `seq_len == 1` 猜测。

源码锚点：[decode dispatch L2154-L2282](https://github.com/xLLM-AI/xllm/blob/a6ae154c5f95343d53b563634b66ba79d7b7390b/xllm/python/attention/npu_paged_attention.py#L2154-L2282)。引入 fused KDA 的提交 [`6355455d`](https://github.com/xLLM-AI/xllm/commit/6355455dd77ed6dd83ff9cab64b90987f0cd9add) 也明确说明了这些参数，并记录了 GLM-5-Next W8A8 TP8 的简单端到端生成验证。

### 4.4 Conv 与 recurrent state

KDA 每层不是传统 KV cache，而是两类固定大小状态：

- conv state：最近 `kernel_size - 1 = 3` 个 raw q/k/v 投影窗口；
- FP32 SSM/recurrent state：每个本地 head 一份 `128 x 128` 状态矩阵。

状态按 sequence slot 从 framework pool gather，cold-start slot 清零，调用结束后再写回：[state gather/cold-start L1843-L1859](https://github.com/xLLM-AI/xllm/blob/a6ae154c5f95343d53b563634b66ba79d7b7390b/xllm/python/attention/npu_paged_attention.py#L1843-L1859)、[state commit L2345-L2374](https://github.com/xLLM-AI/xllm/blob/a6ae154c5f95343d53b563634b66ba79d7b7390b/xllm/python/attention/npu_paged_attention.py#L2345-L2374)。

conv 没有交给 `fla_npu`：eager 使用 `F.conv1d`，ACL Graph decode 因为 `F.conv1d` 不可捕获，改为展开的逐 tap multiply-add：[conv reference L296-L337](https://github.com/xLLM-AI/xllm/blob/a6ae154c5f95343d53b563634b66ba79d7b7390b/xllm/python/models/glm5_3_flash.py#L296-L337)、[graph conv L352-L386](https://github.com/xLLM-AI/xllm/blob/a6ae154c5f95343d53b563634b66ba79d7b7390b/xllm/python/models/glm5_3_flash.py#L352-L386)。

## 5. MTP / speculative verify 的特殊实现

普通 delta-rule 状态不能像 KV cache 那样简单覆写一个尚未确认的 draft token。xLLM 为此实现了专门的 commit 协议：

- eager 路径默认 lazy commit：当前 verify rows 只读计算输出并暂存 raw qkv/g/beta；下一步根据实际 KV 长度增长，只推进真正被确认的 1 或 2 个 token。
- ACL Graph 路径默认启用 V3：每层持久化 `[base | draft]` 双倍 conv/SSM 状态池。
- `num_accepted_tokens` 在 device 上决定下一步从 base 还是 draft state 开始；`ssm_state_indices=[base, draft]` 让一次 `recurrent_kda` 同时保留 after-base 和 after-draft 两种结果。
- V3 消除了 host 选择和每层多次 recurrent 调用，适配固定地址的 graph replay。

默认开关与动机见 [npu_paged_attention.py L50-L91](https://github.com/xLLM-AI/xllm/blob/a6ae154c5f95343d53b563634b66ba79d7b7390b/xllm/python/attention/npu_paged_attention.py#L50-L91)，V3 的状态协议和 fused call 见 [L1069-L1110](https://github.com/xLLM-AI/xllm/blob/a6ae154c5f95343d53b563634b66ba79d7b7390b/xllm/python/attention/npu_paged_attention.py#L1069-L1110) 与 [L1123-L1287](https://github.com/xLLM-AI/xllm/blob/a6ae154c5f95343d53b563634b66ba79d7b7390b/xllm/python/attention/npu_paged_attention.py#L1123-L1287)。

多序列 prefill/decode 默认还有 seq-wise dispatch。代码注释给出的原因是：merged multi-segment kernel 会选择不同 tiling，1 ULP 状态漂移经过 mHC 多层放大后可能翻转 argmax：[decode L2217-L2247](https://github.com/xLLM-AI/xllm/blob/a6ae154c5f95343d53b563634b66ba79d7b7390b/xllm/python/attention/npu_paged_attention.py#L2217-L2247)、[prefill L2289-L2330](https://github.com/xLLM-AI/xllm/blob/a6ae154c5f95343d53b563634b66ba79d7b7390b/xllm/python/attention/npu_paged_attention.py#L2289-L2330)。

## 6. 关键演进提交

| Commit | 作用 |
|---|---|
| [`0616df9b`](https://github.com/xLLM-AI/xllm/commit/0616df9b104ae5b05fdce15a602efdfc5299160f) | 初始 GLM-5-Next adaptation |
| [`6355455d`](https://github.com/xLLM-AI/xllm/commit/6355455dd77ed6dd83ff9cab64b90987f0cd9add) | KDA 从 pure Torch 切到 `fla_npu` fused prefill/decode |
| [`0e9a2069`](https://github.com/xLLM-AI/xllm/commit/0e9a20697f86bdc1149b0b64d8f73c20d530c52e) | 变长、多 token decode 与 metadata 路由 |
| [`1aef59d3`](https://github.com/xLLM-AI/xllm/commit/1aef59d3cdebdeba56a8dba73a8e52de7fd247dc) | `aclInit` 前注册 `fla_npu` OPP |
| [`b629531a`](https://github.com/xLLM-AI/xllm/commit/b629531a4ae4b0590e653592790cce3d30bc10ed) | recurrent decode in-place state 与 `state_v_first` |
| [`b9d0ae9a`](https://github.com/xLLM-AI/xllm/commit/b9d0ae9a2ecc582d07b11ccf091dfb4ad750da85) | 修复 prefill/decode state layout 一致性 |
| [`c7e4a664`](https://github.com/xLLM-AI/xllm/commit/c7e4a6643e6cc03c7c5f15ef29903a2588a28beb) | ACL Graph KDA spec verify，V3 默认路径及并发精度修复 |
| [`29cecbc8`](https://github.com/xLLM-AI/xllm/commit/29cecbc8c034b2e519b74ac67d5967421e988c45) | 全仓从 GLM-5-Next 重命名为 GLM-5.3-Flash，并保留旧 checkpoint alias |
| [`a6ae154c`](https://github.com/xLLM-AI/xllm/commit/a6ae154c5f95343d53b563634b66ba79d7b7390b) | preview 调研快照；加入 8 卡 W8A8 非 MTP 启动脚本 |

## 7. 证据强度与风险

### 强证据

- 官方 main 已公告支持并合入正式部署文档。[PR #2289](https://github.com/xLLM-AI/xllm/pull/2289)
- preview 分支具备 registry、config、模型、状态 cache、KDA backend、MTP、VLM 和启动脚本的完整代码链。
- [`6355455d`](https://github.com/xLLM-AI/xllm/commit/6355455dd77ed6dd83ff9cab64b90987f0cd9add) 的提交说明记录了 W8A8 TP8 启动和生成正确的端到端验证；[`c7e4a664`](https://github.com/xLLM-AI/xllm/commit/c7e4a6643e6cc03c7c5f15ef29903a2588a28beb) 记录了 MTP on/off、采样和 10 并发 prefill 验证。

### 预览信号 / 风险

- 实现不在 main 或 release tag；main 文档仍要求 checkout preview。
- 官方文档明确把 DP、EP、prefix cache、PD、MTP 标成开发/验收中。
- `tests/` 下唯一 GLM-5.3 专项自动化测试是 W8A8 MoE CPU test，并不覆盖 KDA：[test_glm53flash_w8a8_moe.py L1-L64](https://github.com/xLLM-AI/xllm/blob/a6ae154c5f95343d53b563634b66ba79d7b7390b/tests/python/test_glm53flash_w8a8_moe.py#L1-L64)。仓内 `quality_battery.sh` 只是发请求并保存结果，没有自动正确性断言：[quality_battery.sh L1-L21](https://github.com/xLLM-AI/xllm/blob/a6ae154c5f95343d53b563634b66ba79d7b7390b/quality_battery.sh#L1-L21)。
- `fla_npu` 是实际 KDA kernel 的硬依赖，但 preview 源码没有清晰版本 pin；开发镜像以外的可复现性有风险。
- KDA backend 中有较多针对 1 ULP、kernel tiling、MTP accept/reject 和 graph capture 的特殊分支。这说明团队已遇到并修复真实数值问题，但也意味着直接移植时不能只复制 delta-rule 数学，必须一起复制状态协议和 batch layout 契约。

## 8. 与当前本地实现对照时的检查清单

建议按下列边界逐项 diff，而不是只比较公式：

1. **Kernel ownership**：本地是自研 KDA kernel，还是同样委托 `fla_npu`/通用 GDN kernel？
2. **Gate 粒度**：GLM-5.3 KDA 的 `g` 是 `[T,H,K]` 的 per-channel gate，不是常见 GDN 的 `[T,H]` per-head scalar gate。
3. **State layout**：是否固定 `state_v_first=True`；prefill 和 decode 是否共享完全相同的 `[H,V,K]` 解释。
4. **Conv ownership**：q/k/v 是否先做 width-4 full-depthwise conv；conv state 的布局、cold-start、chunked prefill 更新是否一致。
5. **Batch contract**：prefill BSND、decode TND、`cu_seqlens`、多请求 flatten/restore、TP head shard 是否一致。
6. **MTP commit**：reject 时 draft state 是否会形成 phantom write；是否具备 base/draft 双状态或等价 rollback/commit 机制。
7. **Graph capture**：conv 是否可捕获；capture warmup 是否 snapshot/restore recurrent state；动态 host sync 是否完全移出 replay。
8. **数值策略**：q/k L2Norm、gate/beta FP32、SSM FP32、RMSNorm rowwise 以及多序列 tiling 是否和单请求保持一致。

这八项通常比 `S <- exp(g)S + k delta^T` 本身更能解释两套实现的正确性与性能差异。

## 9. 与本地 MLU 实现的直接对比

本节审查的“本地实现”是相邻 worktree `feat/mlu-glm5-next@49f80dae3`，即提交：

- `bd7567cc7 feat(mlu): add GLM5-Next hybrid model support`
- `49f80dae3 feat(mlu): enable GLM5-Next W4A8 MTP decoding`

### 9.1 一句话结论

两边的 **KDA 数学、门控粒度、FP32 recurrent state、TP head shard 和 V-first 状态语义基本一致**。真正的架构分水岭是：

- NPU 在 prefill 使用 AscendC `chunk_kda_fwd`，decode/MTP 使用 AscendC `recurrent_kda`；
- 本地 MLU 在 decode/MTP 已有融合 Triton recurrent kernel，但 prefill 仍按 token 执行 eager FP32 recurrence。

所以，本地实现当前主要不是“公式和官方不同”，而是 **缺少适配 per-key gate 的 MLU chunk KDA prefill kernel**。这会直接限制长上下文吞吐，并且在 1M context 的 GLM-5.3-Flash 上尤其突出。

### 9.2 差异矩阵

| 维度 | xLLM NPU preview | 本地 MLU | 判断 |
|---|---|---|---|
| 模型接入 | Python `PyCausalLM`，同时注册 VLM 和 MTP，并保留 `glm5_next` alias | C++ LLM backend，注册 `glm5_next`、`glm5_next_text`、`glm5_next_mtp`；从 root 或 `text_config` 读参数 | 官方权重的 `model_type=glm5_next` 可被本地识别；NPU 的多模态链路更完整 |
| q/k/v 与 gate | 独立 q/k/v、beta、低秩 forget/output gate projection | 同样的独立 projection 和 TP shard | 结构一致 |
| Forget gate | `-5 * sigmoid(exp(A_log) * (raw + dt_bias))`，per-key `[T,H,K]` | 完全相同；eager 和 Triton kernel 都保留 per-key gate | 一致，不应退化成 per-head gate |
| Beta / output gate | beta 用 sigmoid；RMSNorm 后乘 `sigmoid(output_gate)` | 完全相同 | 一致 |
| Prefill KDA | 外部 AscendC `chunk_kda_fwd`，chunk size 64，FP32 state accumulation | `eager_prefill()` 对每个 sequence 再逐 token 调 LibTorch recurrence | **本地最大性能差距** |
| Decode KDA | 外部 AscendC `recurrent_kda` | 自有 Triton `fused_sigmoid_gating_delta_rule_update(is_kda=true)` | 都是融合 recurrent 路径；本地不需要照搬 `fla_npu` |
| causal conv1d | eager 用 PyTorch `F.conv1d`；graph decode 手工展开 taps | MLU 原生 `causal_conv1d_fn` / `causal_conv1d_update_decode` | 本地的职责边界更清晰，具备设备原生 kernel |
| 状态布局 | 显式 `state_v_first=True`，语义 `[slot,H,V,K]` | recurrence 也按 `[slot,H,V,K]` 访问；通用 cache shape 在非 MUSA 分支声明为 `[slot,H,K,V]` | GLM 当前 K=V=128，运行时看不出差异；这是应显式固化的潜在契约风险 |
| 变长 batch | `cu_seqlens`，但默认 seq-wise dispatch 以稳定数值 | prefill 把 `q_cu_seq_lens` 拷到 CPU 并逐 sequence `.item()`；decode kernel 直接接 `cu_seqlens` | 本地 prefill 有 device-to-host 同步和 host loop |
| MTP 状态 | eager lazy commit；ACL Graph V3 使用 `[base\|draft]` 双状态池和 device-side accept 选择 | cache 预留 `num_speculative_tokens+1` checkpoint stride；conv/KDA kernel 接受 `num_accepted_tokens` 和 rebased state indices | 两者协议不同但目标等价；本地方案更通用、简单，不必复制 NPU V3 Python 状态机 |
| 量化 | 官方部署 W8A8；KDA projection 在量化配置中被排除 | 整体支持 W4A8 MTP；KDA 内部所有 projection 显式使用 `no_quant_args` | 都让 KDA 保持非量化，差异主要在模型其余部分 |
| 测试 | 未发现 tracked KDA 专项测试；主要依赖提交说明中的 TP8/并发验证 | 有 gate/output gate/分段 recurrence CPU test，也有 MLU KDA fused-kernel 与 accepted-prefix 测试 | 本地 kernel 级回归更强；两边都缺完整模型级 KDA parity 矩阵 |

本地关键源码：

- KDA 公式与 forward dispatch：[glm5_next_kda.cpp L63-L185](/workspace/xllm-glm5-next/xllm/core/layers/mlu/glm5_next/glm5_next_kda.cpp#L63-L185)、[L470-L662](/workspace/xllm-glm5-next/xllm/core/layers/mlu/glm5_next/glm5_next_kda.cpp#L470-L662)
- MLU recurrent KDA gate/kernel：[fused_sigmoid_gating_delta_rule_update.py L193-L211](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/fused_sigmoid_gating_delta_rule_update.py#L193-L211)
- MTP checkpoint cache：[kv_cache_estimation.cpp L731-L737](/workspace/xllm-glm5-next/xllm/core/framework/kv_cache/kv_cache_estimation.cpp#L731-L737)
- 当前 KDA 单测：[glm5_next_kda_test.cpp L27-L114](/workspace/xllm-glm5-next/tests/core/layers/mlu/glm5_next_kda_test.cpp#L27-L114)

### 9.3 对本地实现的审查结论

#### Important：prefill 是逐 token eager recurrence

`eager_prefill()` 明确说明通用 `ChunkGatedDeltaRule` 只接受 `[B,T,H]` 的 headwise gate，而 KDA 要求 `[B,T,H,K]` per-key gate，因此当前对每条 sequence、每个 token 做 FP32 tensor recurrence。同时还把 `q_cu_seq_lens` 拷到 CPU，并逐项 `.item()` 读取边界。

这条路径适合作为 correctness oracle，但不适合作为 GLM-5.3-Flash 的生产 prefill。它不仅是 O(T) 递推本身，还包含大量小算子 launch 和 host/device 同步；上下文越长，和 NPU chunk kernel 的差距越明显。

#### Important：状态维度语义没有在通用 cache shape 中显式表达

本地 eager recurrence 和 fused kernel 按 `[H,V,K]` 使用 state，但通用非 MUSA cache shape 当前按 `[H,K,V]` 分配。GLM-5.3-Flash 的 K/V dimension 都是 128，所以物理 shape 相同，现有测试不会暴露转置错误；一旦 kernel、dump/restore 或后续模型出现 K/V 不等维，契约会变得脆弱。

建议至少用命名、注释和一个非方阵 reference test 固化 `state_v_first`，不要依赖 `128 == 128` 隐式成立。

#### Important：缺少跨执行阶段的模型级 parity 测试

本地已有的三类基础测试是有效优势：safe gate、sigmoid output gate、full-vs-segmented eager recurrence；MLU ops test 也覆盖了 KDA fused kernel 和 accepted-prefix API。但还缺以下能直接拦住真实集成问题的组合：

1. 同一序列一次 prefill 与分段 chunked prefill 的 output/final-state parity；
2. prefill 后接单 token decode，与纯 eager reference parity；
3. 多 sequence 变长 batch、cold/warm state 混排；
4. MTP accept=1、accept=2/更多以及 reject 后下一轮 state parity；
5. TP1/TP8 的输出和 state parity；
6. conv state + SSM state 联合 checkpoint/restore。

### 9.4 推荐实现顺序

1. **P0：实现 MLU per-key chunk KDA prefill。** 保留当前 eager recurrence 作为 oracle。可以借鉴 `fla_npu` 的阶段划分——gate cumulative sum、chunk 内 WY/三角修正、跨 chunk state 推进、finalize output——但不需要复制 NPU backend 的 Python 调度代码。
2. **P0：先补 prefill/decode/state parity 测试矩阵。** chunk kernel 上线前先冻结正确性契约，特别是 `state_v_first`、varlen 与 chunk boundary。
3. **P1：移除 prefill 的 CPU `cu_seqlens` 同步和 per-sequence host loop。** 即使 chunk kernel 分阶段落地，也应让 batch metadata 和 dispatch 保持 device-side。
4. **P1：保留当前 MLU recurrent/MTP 路径。** 它已经在 kernel 内融合 q/k L2Norm、safe gate、beta sigmoid 和 state checkpoint；没有证据表明照搬 NPU 的 `[base|draft]` V3 会更好。
5. **P2：补 GLM-5.3-Flash 命名和 VLM 能力时再扩 registry。** 当前官方 config 仍叫 `glm5_next`，所以这不阻塞文本模型运行；它是产品覆盖面差异，不是 KDA 正确性问题。

## 10. 最终判断

- **xLLM NPU 是否支持 GLM-5.3-Flash：支持，但属于 preview / 受限支持。** 官方给出了单机 8 卡 W8A8 运行链路，核心实现尚未进入 main/release，DP/EP/prefix cache/PD/MTP 仍有成熟度限制。
- **他们的 KDA 和本地 KDA 是否公式不同：基本没有。** safe per-key forget gate、beta、q/k normalization、FP32 state、output sigmoid gate 和 V-first recurrence 的语义是一致的。
- **最值得借鉴什么：AscendC chunk KDA prefill 的算法与性能工程，以及对 MTP/ACL Graph 状态提交协议的显式建模。**
- **本地最该优先补什么：不是重写 decode/MTP，而是把逐 token eager prefill 替换为 MLU native per-key chunk KDA，并用跨阶段 parity 测试锁死状态契约。**
