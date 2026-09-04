# KDA 算子与 GPU/NPU 后端实现调研

> 调研日期：2026-09-01  
> 主要快照：FLA 指定提交 [`9c8e42e`](https://github.com/fla-org/flash-linear-attention/commit/9c8e42e762fce087c27b673af4922795d9edb85e)（2026-07-28）；FLA 当前主干 [`35dceaee`](https://github.com/fla-org/flash-linear-attention/commit/35dceaee5408e69a555fec34cb215c93c375dabe)（2026-08-30）。

## 结论

在 **FLA 对外的 KDA API** 这一层，主要就是两个算子：

1. `chunk_kda`：chunk-parallel 路径，主要服务训练和多 token prefill，支持 backward；
2. `fused_recurrent_kda`：recurrent forward 路径，主要服务 decode、spec verify 和很短的多 token 序列。

指定提交与当前主干的 `fla/ops/kda/__init__.py` 都只导出这两个名字。[指定提交导出表](https://github.com/fla-org/flash-linear-attention/blob/9c8e42e762fce087c27b673af4922795d9edb85e/fla/ops/kda/__init__.py) [当前主干导出表](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/kda/__init__.py)

但如果“算子”指实际 device kernel，答案就不是两个：

- 默认 `chunk_kda` 是 gate/cumsum、intra-chunk、跨 chunk state propagation、output finalize 等多个 Triton kernel 组成的流水；
- `fused_recurrent_kda` 是一个 recurrent 主 kernel，但不包含模型层 causal conv 和输出 gated RMSNorm；
- 另外还有 forward/backward 子算子、naive reference，以及 FlashKDA、TileLang、Triton-Ascend、AscendC、CUDA/ROCm fused decode 等后端。

还有一个需要纠正的常见说法：`fused_recurrent_kda` **不是算子契约上只支持 `q_len == 1`**。它可以处理 `T > 1`、packed varlen、continuous batching 和 speculative state index；只是 recurrent 计算随 T 串行，因此 serving 通常把它用于单 token decode 或很短的 verify。指定提交的 FLA `KDALayer` 在推理态对 `q_len <= 64` 自动选 recurrent，并非检查 `use_cache && q_len == 1`。[KDALayer 路由](https://github.com/fla-org/flash-linear-attention/blob/9c8e42e762fce087c27b673af4922795d9edb85e/fla/layers/kda.py#L207-L216) [recurrent API 与 varlen/state-index 参数](https://github.com/fla-org/flash-linear-attention/blob/9c8e42e762fce087c27b673af4922795d9edb85e/fla/ops/kda/fused_recurrent.py#L234-L440)

## 1. FLA 中究竟有哪些 KDA 相关算子

### 1.1 模型层入口

| 入口 | 主要场景 | backward | 关键说明 |
| --- | --- | --- | --- |
| `chunk_kda` | training、prefill、长序列 | 有 | chunk-parallel；支持 varlen、initial/final state、GVA、CP 和 intermediate state 等功能 |
| `fused_recurrent_kda` | decode、spec verify、短序列 | 无 | forward-only recurrent；可原位更新 state；支持 varlen、continuous batching 和 speculative decoding |

`chunk_kda` 通过 `ChunkKDAFunction` 接入 autograd；`fused_recurrent_kda` 直接调用 forward kernel，没有相应 backward function。[chunk autograd](https://github.com/fla-org/flash-linear-attention/blob/9c8e42e762fce087c27b673af4922795d9edb85e/fla/ops/kda/chunk.py#L24-L176) [recurrent 实现](https://github.com/fla-org/flash-linear-attention/blob/9c8e42e762fce087c27b673af4922795d9edb85e/fla/ops/kda/fused_recurrent.py#L33-L334)

### 1.2 `chunk_kda` 的实际 forward 流水

默认 Triton 路径不是一个单 kernel，而是：

```text
raw gate
  -> kda_gate_chunk_cumsum / chunk_local_cumsum
  -> chunk_kda_fwd_intra
       -> W、U、qg、kg、Aqk、Akk
  -> chunk_gated_delta_rule_fwd_h
       -> 跨 chunk recurrent state、v_new、final_state
  -> chunk_gla_fwd_o_gk
       -> output finalize
```

这个调用链可直接从 [`chunk_kda_fwd`](https://github.com/fla-org/flash-linear-attention/blob/9c8e42e762fce087c27b673af4922795d9edb85e/fla/ops/kda/chunk_fwd.py#L20-L132) 看到。这里还复用了 `fla.ops.common` 和 GLA 的通用 kernel，所以只遍历 `fla/ops/kda/` 会漏掉 state propagation 和 output finalize。

KDA namespace 内的重要子算子包括：

- gate：`fused_kda_gate`、`kda_gate_fwd/bwd`、`kda_gate_chunk_cumsum`；
- intra forward：`chunk_kda_fwd_intra`、token-parallel intra 变体；
- WY：`recompute_w_u_fwd`、`prepare_wy_repr_bwd`；
- backward：`chunk_kda_bwd_intra`、`chunk_kda_bwd_wy_dqkg_fused`、`chunk_kda_bwd_dAv`；
- reference：`naive_recurrent_kda`、`naive_chunk_kda`，只用于正确性参考，不是生产 kernel。

源码索引：[gate](https://github.com/fla-org/flash-linear-attention/blob/9c8e42e762fce087c27b673af4922795d9edb85e/fla/ops/kda/gate.py) [intra](https://github.com/fla-org/flash-linear-attention/blob/9c8e42e762fce087c27b673af4922795d9edb85e/fla/ops/kda/chunk_intra.py) [WY](https://github.com/fla-org/flash-linear-attention/blob/9c8e42e762fce087c27b673af4922795d9edb85e/fla/ops/kda/wy_fast.py) [backward](https://github.com/fla-org/flash-linear-attention/blob/9c8e42e762fce087c27b673af4922795d9edb85e/fla/ops/kda/chunk_bwd.py) [naive reference](https://github.com/fla-org/flash-linear-attention/blob/9c8e42e762fce087c27b673af4922795d9edb85e/fla/ops/kda/naive.py)

### 1.3 `fused_recurrent_kda` 的融合边界

FLA recurrent kernel 可以融合：

- Q/K L2Norm 与 q scale；
- raw gate + `A_log`/`dt_bias` 的激活；
- beta sigmoid；
- state decay、delta update、output 和 optional final state。

它不融合：

- KDA 前面的 causal conv state update；
- KDA 后面的 output gate、RMSNorm 和 `o_proj`。

因此 vLLM 的“fused KDA decode”虽然数学核心对应 `fused_recurrent_kda`，但它的融合边界更大，二者不能按名字视为完全相同的底层算子。

## 2. GPU 侧实现

| 实现 | 阶段 | 技术路线 | 融合度与限制 |
| --- | --- | --- | --- |
| FLA 默认 KDA | prefill + backward；decode | Triton | 功能覆盖最完整；prefill 是多阶段 pipeline，recurrent 是单主 kernel；可运行于相应 Triton CUDA/ROCm 后端 |
| FLA FlashKDA backend / MoonshotAI FlashKDA | prefill forward | CUDA + CUTLASS | K1 prepare + K2 recurrence/output 两个大型主 kernel；varlen 另有小 metadata kernel；BF16、K=V=128、同头数、safe bounded gate、inference-only |
| FLA TileLang backend | backward 子阶段 | TileLang | 指定提交只替换 `chunk_kda_bwd_wy_dqkg_fused`，不是完整 forward backend |
| NVIDIA cudnn-frontend KDA | prefill/training fwd+bwd | FROST + cuTile OSS engine | 有 `kda`/`kda_bwd` graph 和 PyTorch op；不是 native cuDNN backend primitive；已发布版本未见 decode engine |
| vLLM NVIDIA | prefill + decode | FlashKDA/FLA Triton + native CUDA | prefill 自动选 FlashKDA 或 Triton；decode 可用单 launch `conv + KDA + gated RMSNorm` 特化 kernel |
| vLLM AMD | prefill + decode | Triton + native ROCm C++/HIP | prefill 当前走 vendored Triton KDA；decode 同样有 `conv + KDA + norm` fused kernel |
| SGLang | prefill/decode/verify | Triton、FlashKDA、CuTeDSL、PTX、Helion、FlashInfer | dispatcher 按架构、shape、长度与 checkpoint 能力选择；多数 prefill 路线是多阶段 pipeline |
| FlashInfer | prefill/decode/spec | CuTeDSL/Cake/CUDA | 提供架构和 shape 专用的 KDA prefill/decode facade；不同 route 的 launch 数和约束不同 |

### 2.1 默认 FLA Triton

这是最通用的正确性与功能基线。它覆盖训练/反向、varlen、GVA、context parallel、initial/final state 和 intermediate state，但多阶段调度会产生更多 launch 和中间张量。[FLA chunk orchestration](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/kda/chunk_fwd.py#L20-L132)

### 2.2 Moonshot FlashKDA

FlashKDA 没有追求整个 prefill 单 launch。官方说明把计算拆成：

- K1：gate、Q/K norm、decay、局部矩阵与逆；
- K2：跨 chunk recurrence、output 和 final state。

其早期单 kernel 原型反而更慢，拆分后端到端至少提升 15%。所以它应描述为“两个大型高度融合 kernel”，而不是“一个 prefill kernel”。[FlashKDA 设计说明](https://github.com/MoonshotAI/FlashKDA/blob/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b/docs/20260420-flashkda-v1-deep-dive.md#L28-L43) [实际 launch](https://github.com/MoonshotAI/FlashKDA/blob/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b/csrc/smxx/fwd_launch.cu#L145-L215)

FLA 的 FlashKDA verifier 还明确限制 BF16、K=V=128、无 GVA、kernel 内 Q/K norm/gate/beta、V-first state、无 CP、无 intermediate state、safe gate 和 inference mode。[FLA FlashKDA backend](https://github.com/fla-org/flash-linear-attention/blob/9c8e42e762fce087c27b673af4922795d9edb85e/fla/ops/kda/backends/flash_kda.py#L22-L103)

### 2.3 vLLM fused decode

vLLM 的 NVIDIA 专用 decode kernel 把 causal-conv state update、KDA recurrent update 和 output gated RMSNorm 合在一个 CUDA launch 中，但只覆盖严格的 Kimi K3 decode shape，不能替代 prefill `chunk_kda`。[启用条件与调用](https://github.com/vllm-project/vllm/blob/45aed9b0cdbd281af6ba84ac124775f26b030879/vllm/models/kimi_k3/nvidia/kda.py#L460-L493) [CUDA kernel](https://github.com/vllm-project/vllm/blob/45aed9b0cdbd281af6ba84ac124775f26b030879/csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel.cu#L977-L1077)

AMD 路径的 prefill backend 当前只接受 `triton`，而 decode 也提供对应的 ROCm fused kernel。[AMD KDA layer](https://github.com/vllm-project/vllm/blob/main/vllm/models/kimi_k3/amd/kda.py#L143-L220) [ROCm decode wrapper](https://github.com/vllm-project/vllm/blob/main/vllm/models/kimi_k3/amd/ops/kda_decode.py)

### 2.4 其他 GPU 路线

NVIDIA cudnn-frontend v1.27.0 提供 KDA graph、PyTorch op、FROST forward 和 cuTile forward/backward engine，但官方文档把它归为开源 execution engine，并说明这类 linear-attention graph node 没有 native cuDNN backend lowering；固定 release 中也没有命名的 recurrent/decode engine。[cudnn-frontend API/engine 表](https://github.com/NVIDIA/cudnn-frontend/blob/f77fbc3d21be3f24cd0286b9b368105f7c518b8a/docs/python_graph_and_execution_backends.md#L64-L105) [KDA PyTorch op](https://github.com/NVIDIA/cudnn-frontend/blob/f77fbc3d21be3f24cd0286b9b368105f7c518b8a/python/cudnn/linear_attention/ops/kda.py)

SGLang 的 dispatcher 还包含 Triton、FlashKDA、NVIDIA-vendored Blackwell pipeline、SM103 PTX、CuTeDSL、Helion 和 FlashInfer 等 route。它们并非全部等价：有的只做 prefill，有的只做 decode/verify，有的不能返回 intermediate checkpoint。[SGLang KDA dispatcher](https://github.com/sgl-project/sglang/blob/71cee04ebe8061af901e4880169f6a5e86f7c8c1/python/sglang/srt/layers/attention/linear/kda_backend.py) [FlashInfer KDA API](https://github.com/flashinfer-ai/flashinfer/blob/5a8e62ac5d359a70307c1ac163a1974dfcdcd660/flashinfer/kda.py)

## 3. Ascend NPU 侧实现

### 3.1 FLA 指定提交：Triton-Ascend 子算子后端

在 `9c8e42e` 中，`TritonAscendKDABackend` 不是整体替换 `chunk_kda`，而是替换 KDA pipeline 中的 device-specific 子算子：

- `chunk_kda_fwd_intra` 与 token-parallel intra；
- `recompute_w_u_fwd`；
- `chunk_kda_bwd_intra`、`chunk_kda_bwd_wy_dqkg_fused`、`chunk_kda_bwd_dAv`；
- `kda_gate_fwd/bwd`、`kda_gate_chunk_cumsum`、`fused_kda_gate`。

其余 state propagation、output 等继续经 FLA 的共享 dispatch 路径组合。[指定提交 Triton-Ascend backend](https://github.com/fla-org/flash-linear-attention/blob/9c8e42e762fce087c27b673af4922795d9edb85e/fla/ops/kda/backends/triton_ascend/__init__.py)

指定提交中没有 NPU 专用的 `fused_recurrent_kda_fwd` override。FLA 在 2026-08-01 的提交 [`5d226052`](https://github.com/fla-org/flash-linear-attention/commit/5d2260527010e2281de4bab1d5377a400800433c) 才加入 Triton-Ascend recurrent forward；当前主干已通过同一个 KDA dispatcher 路由到该 NPU kernel。[当前 Ascend recurrent kernel](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/kda/backends/triton_ascend/fused_recurrent.py) [当前 backend 注册](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/kda/backends/triton_ascend/__init__.py#L122-L130)

因此版本结论是：

- 对你给的 commit：NPU 有 chunk 路径的 Triton-Ascend 子 kernel，recurrent 还是通用实现；
- 对当前 FLA main：chunk 与 fused recurrent 都已有显式 Triton-Ascend backend。

### 3.2 `flashserve/flash-linear-attention-npu`：AscendC 完整 forward 算子

`fla_npu` 提供两个与 FLA 顶层语义最直接对应的 AscendC 算子：

| FLA 语义 | AscendC 算子 | 用途 |
| --- | --- | --- |
| `chunk_kda` / `chunk_kda_fwd` | `fla_npu.ops.ascendc.chunk_kda_fwd` | prefill、多 token forward |
| `fused_recurrent_kda` | `fla_npu.ops.ascendc.recurrent_kda` | decode、MTP/spec 短序列 |

它还暴露 `kda_gate_cumsum` 和 `chunk_kda_bwd_intra` 等局部算子；但当前仓库没有一个与 FLA 完整 `chunk_kda` autograd backward 等价的单体 AscendC API，所以应区分“完整 inference forward”与“完整训练 backward”。[fla_npu 算子清单](https://github.com/flashserve/flash-linear-attention-npu/blob/66ed5f0ee879ec7e1d6e2a057551cff2d0f6df5b/README.md#L180-L216) [KDA 源码目录](https://github.com/flashserve/flash-linear-attention-npu/tree/66ed5f0ee879ec7e1d6e2a057551cff2d0f6df5b/fla/ops/ascendc/kda)

`chunk_kda_fwd` 顶层数学阶段仍是 gate cumsum → Prepare/Post-WU → FwdH → Finalize。A5 的 BF16、chunk=64、K=V=128 dense 对齐快路可保持一次物理 L0；A5 的 tail/varlen/泛化 shape 会按四阶段提交。A2/A3 和单 chunk 场景有自己的单 L0 路径。因此 AscendC “一个公开 op”也不等于所有 shape 都只有一次物理 launch。[AscendC chunk 设计](https://github.com/flashserve/flash-linear-attention-npu/blob/66ed5f0ee879ec7e1d6e2a057551cff2d0f6df5b/fla/ops/ascendc/kda/chunk_kda_fwd/docs/design.md)

`recurrent_kda` 在一个 AICore kernel 内完成 Q/K norm、raw gate、beta、state decay/delta update 和 output/final state；同样不包含模型外部 causal conv 与 output norm。当前接口面向 decode/MTP，每段 packed sequence 长度不超过 8。[AscendC recurrent README](https://github.com/flashserve/flash-linear-attention-npu/blob/66ed5f0ee879ec7e1d6e2a057551cff2d0f6df5b/fla/ops/ascendc/kda/recurrent_kda/README.md)

### 3.3 serving 框架中的实际接法

当前 vLLM-Ascend 的 Kimi K3 路径在 prefill 调 `torch.ops._C_ascend.chunk_kda_fwd`，在 decode/spec 调 `torch.ops._C_ascend.recurrent_kda`；仓内同时保留一套 Triton NPU `chunk_kda`/`fused_recurrent_kda` 实现和精度测试。[vLLM-Ascend Kimi K3 KDA](https://github.com/vllm-project/vllm-ascend/blob/8b72fdb97b2b0d8eccaf6ac7ac414a151e09810e/vllm_ascend/ops/kimi_kda.py#L228-L320) [Triton KDA](https://github.com/vllm-project/vllm-ascend/blob/8b72fdb97b2b0d8eccaf6ac7ac414a151e09810e/vllm_ascend/ops/triton/kda/kda.py)

xLLM 的公开 NPU GLM-5.3-Flash 预览路径也没有在 xLLM 内重写 KDA 核心：prefill 委托 `fla_npu.ops.ascendc.chunk_kda_fwd`，decode/MTP 委托 `fla_npu.ops.ascendc.recurrent_kda`。[xLLM NPU dispatch](https://github.com/xLLM-AI/xllm/blob/a6ae154c5f95343d53b563634b66ba79d7b7390b/xllm/python/attention/npu_paged_attention.py#L2154-L2343)

在本次核对的公开 `torch-npu` 源码中没有检出同名的内置 `chunk_kda`/`recurrent_kda` 产品算子；这里的 `_C_ascend` KDA 能力来自 vLLM-Ascend/`fla_npu` 的自定义 AscendC 扩展，不能表述成 torch-npu 原生 API。

## 4. 语义对应关系

| KDA 阶段 | GPU 可用实现 | Ascend NPU 可用实现 |
| --- | --- | --- |
| gate 激活 / chunk cumsum | FLA Triton `fused_kda_gate`、`kda_gate_chunk_cumsum` | FLA Triton-Ascend gate backend；`fla_npu` AscendC `kda_gate_cumsum` |
| intra-chunk / WY | FLA Triton intra + WY；FlashKDA K1；cuDNN FE/SGLang/FlashInfer 专用 pipeline | FLA Triton-Ascend intra/WY；`fla_npu` AscendC ChunkKdaFwd Prepare/Post-WU |
| 跨 chunk state | FLA Triton shared delta-state kernel；FlashKDA K2 | FLA shared Triton-Ascend state path；`fla_npu` AscendC FwdH |
| output finalize | FLA Triton GLA output；FlashKDA K2 | FLA Triton-Ascend shared output；`fla_npu` AscendC Finalize |
| 完整 prefill forward | FLA Triton `chunk_kda`；FlashKDA；cudnn-frontend；SGLang/FlashInfer | FLA Triton-Ascend 组合；`fla_npu.chunk_kda_fwd` AscendC |
| recurrent decode | FLA Triton `fused_recurrent_kda`；vLLM CUDA/ROCm fused decode；FlashInfer/SGLang | FLA 当前 Triton-Ascend recurrent；`fla_npu.recurrent_kda` AscendC |
| training backward | FLA Triton full backward；cudnn-frontend cuTile；部分 TileLang | FLA Triton-Ascend KDA-specific backward 子算子 + shared backward；`fla_npu` 目前只有部分 AscendC backward 子算子 |

## 5. 对实现与选型的直接判断

1. **接口数量看两个，kernel 数量看一组。**模型适配层通常只需要 prefill `chunk_kda` 与 decode `fused/recurrent_kda` 两个入口；性能调优和后端移植必须按内部阶段拆解。
2. **不要把 `q_len == 1` 写进算子定义。**它是常见调度策略，不是 recurrent KDA 的数学限制；MTP/spec verify 本来就会让 recurrent 一次处理多个短 token。
3. **不要只按函数名比较融合度。**FLA/fla_npu recurrent 的边界是 KDA core；vLLM fused decode 还包含 conv 与 output norm。
4. **GPU prefill 的直接高性能参考是 FlashKDA，通用正确性参考是 FLA Triton。**前者融合更深但限制更严，后者功能面完整。
5. **Ascend 上有两套成熟度不同的路线。**FLA Triton-Ascend 便于保持上游算法结构和训练能力；`fla_npu` AscendC 提供 serving-oriented 的整段 prefill/recurrent forward 和更深硬件特化。

## 6. xLLM MLU 侧补充

当前独立工作树 `/workspace/xllm-glm5-next` 已有与上述语义对应的 MLU Triton 实现：

- prefill：`ChunkKDA`，基础或优化路径均按 gate/KKT/inverse/WU/state-output 多阶段组织；
- decode：`fused_sigmoid_gating_delta_rule_update(is_kda=true)`；causal conv 与输出 norm 仍在 layer 外部分开执行。

对应源码：[GLM5 Next KDA layer](/workspace/xllm-glm5-next/xllm/core/layers/mlu/glm5_next/glm5_next_kda.cpp:518) [ChunkKDA C++ 调度](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/chunk_kda.cpp:488) [基础 Triton prefill](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/chunk_kda_fwd.py:23) [优化 Triton prefill](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/chunk_kda_fwd_optimized.py:38) [MLU recurrent update](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/fused_sigmoid_gating_delta_rule_update.py:31)
