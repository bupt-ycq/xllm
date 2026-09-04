# CUDA/GPU 侧 KDA 融合算子调研

> 调研快照：2026-09-01  
> 问题：CUDA/GPU 侧是否有可与 `fla_npu` Ascend C `chunk_kda_fwd` 对照的专用 KDA 融合算子？

## 结论先行

有，而且已经不止一种。但“专用 KDA 融合算子”和“整个 prefill 只 launch 一次”不是同一件事：

1. **NVIDIA 官方已经公开并发布 KDA prefill/training 实现。**`NVIDIA/cudnn-frontend` v1.27.0（2026-08-06）提供单节点 `kda` / `kda_bwd` graph、PyTorch op `cudnn.linear_attention.ops.kimi_delta_attention`、Blackwell FROST forward engine 与 cuTile forward/backward engine。但它是 cudnn-frontend 仓库里的开源 Python engine/kernel，官方文档明确说这组线性注意力节点没有 native cuDNN backend lowering；因此不能称为 CUDA Toolkit 原语或 native cuDNN 产品 primitive。在该 release 固定树中也未发现 KDA decode engine。[v1.27.0 release commit](https://github.com/NVIDIA/cudnn-frontend/commit/f77fbc3d21be3f24cd0286b9b368105f7c518b8a) [engine/API 说明](https://github.com/NVIDIA/cudnn-frontend/blob/f77fbc3d21be3f24cd0286b9b368105f7c518b8a/docs/python_graph_and_execution_backends.md#L64-L105)
2. **最直接的 serving CUDA prefill 对标仍是 MoonshotAI/FlashKDA。**它是专用 KDA prefill forward 后端，用两个高度融合的 CUDA/CUTLASS kernel 替代普通 FLA Triton 的多阶段流水。官方明确说明，早期单 kernel 方案反而慢，拆成 K1 prepare 与 K2 recurrence/output 后端到端至少快 15%；packed varlen 还会有一个很小的 prefix 元数据 kernel。因此应称为“专用双 kernel 融合实现”，不能称为“完整 prefill 单 launch”。[FlashKDA deep dive](https://github.com/MoonshotAI/FlashKDA/blob/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b/docs/20260420-flashkda-v1-deep-dive.md#L28-L43) [实际 launch](https://github.com/MoonshotAI/FlashKDA/blob/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b/csrc/smxx/fwd_launch.cu#L145-L215)
3. **vLLM 另有真正单 launch 的专用 decode kernel。**它融合了 causal-conv state update、KDA recurrent update 和 output gated RMSNorm，但只覆盖严格的 Kimi K3 decode shape，不是 prefill `chunk_kda_fwd` 的替代物。[vLLM 启用点](https://github.com/vllm-project/vllm/blob/45aed9b0cdbd281af6ba84ac124775f26b030879/vllm/models/kimi_k3/nvidia/kda.py#L460-L493) [CUDA kernel](https://github.com/vllm-project/vllm/blob/45aed9b0cdbd281af6ba84ac124775f26b030879/csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel.cu#L977-L1077)
4. **SGLang 还集成了 Blackwell 专用的 NVIDIA/CuTeDSL/PTX/Helion 路线。**其中 `ptx_kda` 在 GB300/SM103 的长单序列 route 可以用一个 grid 完成 prefill 主体，最接近“单网格 prefill 融合”；其余多为 2～4 阶段 pipeline。代价是可移植性非常低或依赖运行时编译器。[PTX wrapper](https://github.com/sgl-project/sglang/blob/71cee04ebe8061af901e4880169f6a5e86f7c8c1/python/sglang/srt/layers/attention/linear/kernels/kda_ptx.py#L1-L82) [PTX/CUDA 源码](https://github.com/sgl-project/sglang/blob/71cee04ebe8061af901e4880169f6a5e86f7c8c1/python/sglang/kernels/jit/csrc/attention/kda_prefill.cu)
5. **普通 Triton `chunk_kda` 仍是功能最完整、覆盖面最广的 fallback。**它不是整个 KDA 的单个融合 kernel，而是 gate/cumsum、intra-chunk、跨 chunk state recurrence、output finalize 等多阶段组合，优点是训练、GVA、context parallel、intermediate state 等功能覆盖更全。[FLA Triton orchestration](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/kda/chunk_fwd.py#L20-L132)
6. **`fla_npu` 自身也不是所有 shape 都强制单 launch。**A5 的 BF16、chunk=64、K=V=128 dense 对齐快路是一次物理 `ChunkKdaFwd`；A5 tail/varlen、泛化 shape 会把同一私有入口按 Gate/Prepare、Post-WU、FwdH、Finalize 四阶段提交。它和 CUDA 世界的共同经验是：应优化融合边界，而不是把“launch 数必须为 1”当成目标。[Ascend C 设计](https://github.com/flashserve/flash-linear-attention-npu/blob/d4d2dc4b7a07316871088d5207a2aeb6d819a3db/fla/ops/ascendc/kda/chunk_kda_fwd/docs/design.md)

## 1. “融合”分级，避免概念混淆

本文按以下口径分类：

| 等级 | 含义 | 代表实现 |
| --- | --- | --- |
| 单 kernel / 单 launch | 一个 GPU kernel 完成目标调用的全部主计算 | vLLM fused decode；SGLang SM103 PTX 的部分 prefill route |
| 专用少 kernel 融合 | 为 KDA 定制，把原先很多阶段压成少数大 kernel | cudnn-frontend FROST/cuTile；Moonshot FlashKDA；SGLang NVIDIA KDA pipeline |
| 通用多阶段实现 | 多个相对通用的 Triton/CuTe/Helion kernel 显式串联 | FLA Triton、SGLang CuTeDSL/Helion prefill |

单 launch 不天然更快。KDA 的 chunk-local 逆矩阵与跨 chunk recurrence 并行结构不同，FlashKDA 官方实测正是把它们拆开后更快。[设计取舍](https://github.com/MoonshotAI/FlashKDA/blob/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b/docs/20260420-flashkda-v1-deep-dive.md#L28-L43)

## 2. 能力总表

| 实现 | 阶段 | 实现/融合度 | 主要设备 | 关键限制 | varlen / state / checkpoint |
| --- | --- | --- | --- | --- | --- |
| FLA Triton `chunk_kda` | prefill + backward | 多阶段 Triton | Triton 可用 GPU | K≤256；chunk 32/64；FlashKDA 不满足时的默认实现 | varlen；initial/final；可返回 chunk intermediate；支持训练/CP/GVA |
| NVIDIA cudnn-frontend v1.27 | prefill forward + backward | 官方 OSS Python graph engine；FROST forward + cuTile fwd/bwd | FROST SM100/103；cuTile CUDA 13.3+ GPU | 非 CUDA Toolkit/native-cuDNN primitive；FROST FP16/BF16、K=V=128、BT16；cuTile K≤256；无 official decode | THD packed varlen、initial/final；两个 engine 均拒绝 per-chunk H/checkpoint |
| Moonshot FlashKDA v1 | prefill forward | K1+K2 两个大型 CUTLASS kernel；varlen 加 prefix kernel | SM90a/100a/103a/120a | BF16，K=V=128，等 Q/V heads，safe bounded gate，inference only | fixed/varlen；BF16/FP32 外部 initial/final；上游无中间 checkpoint |
| vLLM FlashKDA fork | prefill forward | 同 FlashKDA 两阶段 | SM90/10x/12x，按 CUDA 版本构建 | BF16，D=128，bounded gate；Kimi K3 集成 | varlen、initial/final；每序列一个、16-token 对齐 checkpoint |
| vLLM native fused decode | decode | 真正单 CUDA kernel，融合 conv+KDA+norm | SM90/10x/12x | H∈{12,24,48,96}，D=128，conv width=4，非 speculative | FP32 recurrent state；无 prefill/checkpoint |
| SGLang `nvidia_kda` | prefill | K1 CuTe + K2 CuTe + K3 Triton + K4 cuTile | datacenter Blackwell major 10，排除 SM120 | BF16，K=V=128；多序列；B≤8；bucket 到 2K/4K/8K/16K | varlen、initial/final；无中间 checkpoint |
| SGLang `ptx_kda` | prefill | SM103 手写 CUDA/PTX；按 shape 单网格或两 launch | 仅 GB300/SM103 | BF16，K=V=128，chunk=64 | varlen、initial/final；支持 64-stride chunk checkpoint |
| SGLang CuTeDSL | prefill/decode | prefill 多 kernel；decode recurrent 融合 | prefill major≥10；decode SM90+ | D=128；prefill 无 safe gate；服务包装不导出 intermediate | varlen、state pool initial/final；无服务层 intermediate |
| SGLang Helion | prefill/decode | Helion 生成式多阶段 prefill；packed recurrent decode | CUDA + Helion 1.4.0 | 依赖 JIT/运行时编译；不是固定手写 kernel | prefill varlen/initial/intermediate；decode 无 conv/norm 融合 |
| FlashInfer 当前 KDA | prefill/decode/spec | CuTeDSL 与冻结 Cake schedule 组合 | SM100/103/120 各自专用 prefill；CuTe decode | BF16，主要优化面 D=128；不同 arch contract 不同 | varlen、state pool、final；新 prefill 支持周期 checkpoint |
| `fla_npu` Ascend C | prefill + decode | dense 快路单 L0；泛化 prefill 四阶段；decode 单 AIV kernel | A2/A3/A5；op 注册含 910B/910_93/950 | prefill FP16/BF16，K=128，V=128/256，chunk=64/128；decode 每段≤8 | varlen、GQA、initial/final、intermediate；prefill state FP32 |

上表是各 serving wrapper 的“实际可选面”，并不表示所有底层库自身只支持表中的范围。例如 SGLang 对 FlashKDA 设置的 2048 token 上限是性能选择策略，不是 Moonshot kernel 的硬上限。

## 3. 普通 FLA Triton `chunk_kda`：功能基线

FLA 的默认 forward 不是单一 kernel。`chunk_kda_fwd` 依次组织：

1. raw gate 激活与 chunk-local cumsum；
2. intra-chunk `W/U/Aqk/Akk`；
3. 跨 chunk state propagation；
4. output finalize。

这条调用链可在 [FLA `chunk_fwd.py`](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/kda/chunk_fwd.py#L20-L132) 直接看到。它的完整接口还支持训练/反向、varlen、GVA、context parallel、initial/final state，以及推理时的 chunk intermediate state；相应约束在 [`chunk.py`](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/kda/chunk.py) 中定义。

decode/小步长不是走这个 prefill 流水，而是使用 [Triton fused recurrent KDA](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/kda/fused_recurrent.py)，它在一个 recurrent kernel 内处理 gate/QK norm/beta/state update，并支持 continuous batching、varlen 和 speculative state index。这仍不包含模型外部的 conv update 或 output norm。

FLA 的 dispatcher 会按 priority 调用已安装 backend 的 verifier，只有 available、enabled 且全部条件通过才替换默认函数，否则回落 Triton。[backend 选择器](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/backends/__init__.py#L160-L219)

## 4. NVIDIA 官方 cudnn-frontend v1.27.0：有 prefill/training OSS engine，无 native backend primitive

### 4.1 发布状态与公开 API

这不是尚未合并的实验分支：`NVIDIA/cudnn-frontend` 的 `main` 与 tag `v1.27.0` 都固定在 commit `f77fbc3d...`，release commit 日期为 2026-08-06。该版本公开了：

- graph 单节点 `kda` 与 `kda_bwd`；
- PyTorch custom op `cudnn.linear_attention.ops.kimi_delta_attention`；
- `KdaFrostEngine` forward，以及作为 fallback/反向实现的 `KdaCuTileEngine`。

官方设计文档明确说明 KDA 是 per-key-channel decay，FROST 是 SM100/SM103 forward 默认 engine；FROST 的 backward 是 stub，梯度转给 cuTile engine。[v1.27.0 commit](https://github.com/NVIDIA/cudnn-frontend/commit/f77fbc3d21be3f24cd0286b9b368105f7c518b8a) [engine 与 op 清单](https://github.com/NVIDIA/cudnn-frontend/blob/f77fbc3d21be3f24cd0286b9b368105f7c518b8a/docs/python_graph_and_execution_backends.md#L64-L105) [PyTorch op](https://github.com/NVIDIA/cudnn-frontend/blob/f77fbc3d21be3f24cd0286b9b368105f7c518b8a/python/cudnn/linear_attention/ops/kda.py#L1-L34)

### 4.2 关键分类：官方 OSS，不等于 CUDA/cuDNN 产品 primitive

`cudnn-frontend` 同时承载 cuDNN Graph API frontend 和可独立 JIT 的 Python DSL engines。官方文档把这组线性注意力节点定义为 **python-engine-only ops**，没有 cuDNN backend lowering；强行路由到 backend 会得到 `cudnnGraphNotSupportedError`。[无 backend lowering 的说明](https://github.com/NVIDIA/cudnn-frontend/blob/f77fbc3d21be3f24cd0286b9b368105f7c518b8a/docs/python_graph_and_execution_backends.md#L64-L92)

所以三种说法的准确性是：

| 说法 | 判断 |
| --- | --- |
| “NVIDIA 官方开源仓库已经提供 KDA prefill/training kernel 与 Python API” | 正确 |
| “CUDA Toolkit 已提供 `cuda::kda` 原语” | 错误；所核验 Toolkit/CUDA samples 固定主干没有该产品 API |
| “native cuDNN backend 已有 KDA primitive” | 错误；官方明确写明无 backend lowering |

在同日核验的 `NVIDIA/cutlass` commit [`dc45f979...`](https://github.com/NVIDIA/cutlass/tree/dc45f979ae336a235da1676b311f35efeb30149a) 与 `NVIDIA/TransformerEngine` commit [`1ff9c372...`](https://github.com/NVIDIA/TransformerEngine/tree/1ff9c3725a66dfbe230efa8e6f9fd8c101523adc) 中也未找到命名的 KDA operator/API。CUTLASS/CuTe 是 FROST 和 FlashKDA 使用的实现工具，不等于 CUTLASS 对外提供了 `cutlass::kda` 产品算子。

### 4.3 forward/backward 支持边界

FROST forward 是 Blackwell chunked prefill kernel：

- SM100～SM103，内部 BT=16；
- Q/K/V 为 FP16 或 BF16，THD packed layout，K=V=128；
- `cu_seqlens` 为 INT32；initial/final state 可 FP32 或 BF16且 dtype 匹配；
- 支持 in-kernel Q/K L2 norm、safe bounded gate 与 beta sigmoid；
- 拒绝 `checkpoint_every_n_tokens` 和 per-chunk `H` intermediate output。

条件直接编码在 [FROST KDA engine](https://github.com/NVIDIA/cudnn-frontend/blob/f77fbc3d21be3f24cd0286b9b368105f7c518b8a/python/cudnn/linear_attention/frost/kda_engine.py#L1-L116) 中。这里的“融合”是专用 chunked prefill engine，不应在没有逐 launch 证明时写成“完整 KDA 单 launch”。

cuTile engine 同时覆盖 forward 与 backward，要求 CUDA Runtime 13.3+、Q/K/V FP16 或 BF16、THD packed、INT32 `cu_seqlens`、K≤256；它同样拒绝 per-chunk `H`/checkpoint。其 backward 会在内部重算 gate cumsum 与 intra-chunk WY 等 forward intermediates。[cuTile engine 条件](https://github.com/NVIDIA/cudnn-frontend/blob/f77fbc3d21be3f24cd0286b9b368105f7c518b8a/python/cudnn/linear_attention/cutile/kda_engine.py#L125-L200) [重计算说明](https://github.com/NVIDIA/cudnn-frontend/blob/f77fbc3d21be3f24cd0286b9b368105f7c518b8a/python/cudnn/linear_attention/ops/kda.py#L20-L34)

release v1.27.0 固定树内没有命名的 KDA decode/recurrent engine；因此 NVIDIA 官方这一项当前只能回答 **prefill/training forward+bwd**，不能把 vLLM/FlashInfer 的 decode kernel归入 CUDA Toolkit 或 native cuDNN 产品 API。

### 4.4 release 与 develop 的区别

本报告的官方发布判断以 v1.27.0 为准。`develop` commit `1dc536ec...` 又增加了 FLA `chunk_kda` drop-in adapter：Blackwell、BF16，safe gate/beta sigmoid/QK norm 可下沉，普通 `-exp*softplus` gate 仍在 Torch 预处理；CP、intermediate state 等变体回退 FLA。[develop-only FLA adapter](https://github.com/NVIDIA/cudnn-frontend/blob/1dc536ec56992b79a0055f4b8745f17b27bb6ee7/python/cudnn/fla/kda.py#L1-L20) [adapter dispatcher](https://github.com/NVIDIA/cudnn-frontend/blob/1dc536ec56992b79a0055f4b8745f17b27bb6ee7/python/cudnn/fla/kda.py#L129-L212)

这个 FLA adapter 截至快照不在 v1.27.0 release tree，不能用它来描述已发布接口；已发布的 graph/PyTorch KDA op 与 FROST/cuTile engines 本身则是确定存在的。

## 5. MoonshotAI/FlashKDA：CUDA prefill 的直接答案

### 5.1 融合边界

FlashKDA v1 的内部 chunk 为 16，而普通 FLA Triton 常用 64。K1 负责 gate activation、Q/K L2 norm、decay、`L/Mqk` 构造与逆矩阵；K2 负责 chunk 间 recurrence、output 和 final state。[算法拆分与 chunk=16 原因](https://github.com/MoonshotAI/FlashKDA/blob/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b/docs/20260420-flashkda-v1-deep-dive.md#L11-L43)

这比普通 Triton 的多个细阶段更“融合”，但严格说仍是两次主 kernel launch。packed varlen 在 K1 前还需要 prefix metadata launch。[launch 源码](https://github.com/MoonshotAI/FlashKDA/blob/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b/csrc/smxx/fwd_launch.cu#L145-L215)

### 5.2 支持契约

- 架构与工具链：官方 README 要求 SM90+、CUDA 12.9+、PyTorch 2.4+；构建脚本列出 `90a/100a/103a/120a`。[README](https://github.com/MoonshotAI/FlashKDA/blob/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b/README.md#L9-L12) [setup.py](https://github.com/MoonshotAI/FlashKDA/blob/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b/setup.py#L19-L52)
- dtype/shape：Q/K/V/G/Beta/Output 为 BF16，`A_log/dt_bias` 为 FP32，K=V=128。[API contract](https://github.com/MoonshotAI/FlashKDA/blob/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b/README.md#L79-L109)
- 融合语义：输入 raw gate 与 raw beta logits，kernel 内执行 bounded safe gate、Q/K L2 norm 和 beta sigmoid。[API contract](https://github.com/MoonshotAI/FlashKDA/blob/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b/README.md#L92-L103)
- layout/head：FLA verifier 要求 `HV==H`，即该路不支持 GVA；要求 `state_v_first=True`。[FLA verifier](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/kda/backends/flash_kda.py#L64-L88)
- fixed/varlen：支持普通 batch 与 packed varlen；packed 时 B=1，`cu_seqlens` 是 INT64。[C++ validation](https://github.com/MoonshotAI/FlashKDA/blob/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b/csrc/flash_kda.cpp#L144-L180)
- state：支持 `[N,H,V,K]` 的 optional initial/final state，外部 state 可 BF16 或 FP32且输入输出 dtype 一致；内部片上 recurrence state 以 BF16 保存，更新用 FP32 FMA，因此不能把它描述为“全程 FP32 state”。[README](https://github.com/MoonshotAI/FlashKDA/blob/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b/README.md#L102-L109) [deep dive](https://github.com/MoonshotAI/FlashKDA/blob/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b/docs/20260420-flashkda-v1-deep-dive.md)
- training/intermediate：仅 forward/inference；有 grad、context parallel 或 `return_intermediate_states=True` 时 FLA verifier 拒绝。[FLA backend](https://github.com/fla-org/flash-linear-attention/blob/35dceaee5408e69a555fec34cb215c93c375dabe/fla/ops/kda/backends/flash_kda.py#L24-L88)
- 许可证：MIT。[LICENSE](https://github.com/MoonshotAI/FlashKDA/blob/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b/LICENSE)

### 5.3 超长上下文边界

Moonshot API 没有写 2K 或 128K 的硬上限。官方测试包含 fixed `T=131072`、`T=1048576`，以及两条各 524288 token 的 packed varlen；但这些极长 case 只使用 H=1 和 BF16 state，所以它们证明的是“长度可运行”，不等于 Kimi H96 下已经给出相同规模性能结论。[长序列测试](https://github.com/MoonshotAI/FlashKDA/blob/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b/tests/test_fwd_full.py#L208-L274)

公开性能表主要是 H20、T=8192、H=64/96，较 FLA Triton 约 1.85×～2.31×；对于 16K/32K/64K/128K 仍应单独实测，而不是外推 8K 表格。[H20 benchmark](https://github.com/MoonshotAI/FlashKDA/blob/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b/BENCHMARK_H20.md#L1-L26)

## 6. vLLM：FlashKDA prefill + 原生 fused decode

### 6.1 prefill dispatcher

vLLM 的 Kimi K3 NVIDIA 路径支持 `auto/triton/flashkda`。`auto` 只有在 CUDA、compute capability major 为 9/10/12、BF16、head_dim=128 且 `lower_bound` 非空时选 FlashKDA；不满足则回 Triton。这里没有 sequence-length 阈值。[支持判断和选择](https://github.com/vllm-project/vllm/blob/45aed9b0cdbd281af6ba84ac124775f26b030879/vllm/models/kimi_k3/nvidia/kda.py#L176-L190) [dispatcher](https://github.com/vllm-project/vllm/blob/45aed9b0cdbd281af6ba84ac124775f26b030879/vllm/models/kimi_k3/nvidia/kda.py#L300-L317)

构建不是浮动拉 Moonshot HEAD，而是固定 vLLM 自己的 FlashKDA fork commit `ee0be888...`。CMake 根据 CUDA 版本构建 SM90、SM10x、SM12x。[CMake integration](https://github.com/vllm-project/vllm/blob/45aed9b0cdbd281af6ba84ac124775f26b030879/cmake/external_projects/flashkda.cmake#L7-L78)

### 6.2 checkpoint 不是通用 intermediate state

vLLM fork 给 FlashKDA 加了 `checkpoint_state [N,H,128,128]` FP32 与 `checkpoint_offsets [N]`。每条 sequence 每次调用只能指定一个 checkpoint；kernel 仅在 `checkpoint_offset == (t+1)*16` 时写出，所以必须对齐 FlashKDA 内部 16-token chunk。它不能替代 FLA 的“返回所有 chunk state”。[fork API](https://github.com/vllm-project/FlashKDA/blob/ee0be888cd0e972f9409bf53756f8c38c6652173/csrc/flash_kda.cpp#L58-L99) [写 checkpoint 条件](https://github.com/vllm-project/FlashKDA/blob/ee0be888cd0e972f9409bf53756f8c38c6652173/csrc/smxx/fwd_kernel2.cuh#L450-L490)

### 6.3 真正单 launch 的 decode 融合

vLLM 自己的 native CUDA kernel 把 conv1d cache update、KDA recurrent state update、output gated RMSNorm 合在一次 launch 中。启用条件包括：

- H ∈ {12, 24, 48, 96}；D=128；conv width=4；
- activation 与 conv state 为 BF16，recurrent state 为 FP32，权重/`A_log/dt_bias` 为 FP32；
- SM90、SM10x 或 SM12x；
- 当前 batch 无 prefill、无 speculative decode，并已准备融合权重和 norm。

条件检查见 [`is_fused_kda_decode_supported`](https://github.com/vllm-project/vllm/blob/45aed9b0cdbd281af6ba84ac124775f26b030879/vllm/models/kimi_k3/nvidia/kda.py#L148-L173)，实际调用分支见 [KDA forward](https://github.com/vllm-project/vllm/blob/45aed9b0cdbd281af6ba84ac124775f26b030879/vllm/models/kimi_k3/nvidia/kda.py#L460-L493)，CUDA shape/dtype 校验见 [kernel API](https://github.com/vllm-project/vllm/blob/45aed9b0cdbd281af6ba84ac124775f26b030879/csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel.cu#L977-L1077)。

vLLM 为 Apache-2.0，固定拉取的 FlashKDA fork 为 MIT。[vLLM LICENSE](https://github.com/vllm-project/vllm/blob/45aed9b0cdbd281af6ba84ac124775f26b030879/LICENSE) [fork LICENSE](https://github.com/vllm-project/FlashKDA/blob/ee0be888cd0e972f9409bf53756f8c38c6652173/LICENSE)

## 7. SGLang：dispatcher 实际会选什么

SGLang 把 phase 分开选择：decode、prefill、verify 各自有 backend；safe bounded gate 对不支持的 backend 会回 Triton。prefill 可选 Triton、Helion、FlashKDA、CuTeDSL、NVIDIA KDA、PTX KDA；decode/verify 则主要是 Triton、Helion、CuTeDSL、FlashInfer。[总 dispatcher](https://github.com/sgl-project/sglang/blob/71cee04ebe8061af901e4880169f6a5e86f7c8c1/python/sglang/srt/layers/attention/linear/kda_backend.py)

### 7.1 SGLang 的 FlashKDA 长度策略

SGLang 的 FlashKDA wrapper 要求 prefill-only、BF16、K=V=128、无 GVA、bounded safe gate，并在下列情况回 Triton：需要 intermediate state、无 `lower_bound`、speculative extend，或者任一 sequence 长度不在 `[64, 2048]`。[功能条件](https://github.com/sgl-project/sglang/blob/71cee04ebe8061af901e4880169f6a5e86f7c8c1/python/sglang/srt/layers/attention/linear/kernels/kda_flashkda.py#L71-L176) [长度条件](https://github.com/sgl-project/sglang/blob/71cee04ebe8061af901e4880169f6a5e86f7c8c1/python/sglang/srt/layers/attention/linear/kernels/kda_flashkda.py#L177-L193)

因此在当前 SGLang dispatcher 下，8K/16K/32K/64K/128K **都不会走 FlashKDA**，会整批回 Triton。这个 2048 上限是 SGLang 根据 crossover 做的策略，不是 Moonshot API 的硬限制。

### 7.2 `nvidia_kda`

这是 SGLang dispatcher 的 backend 名称。vendored 文件带 NVIDIA 2025 copyright，并自述来自 “NVIDIA KDA_prefill package (benchmark/Blackwell path)”；但没有找到可公开核验的独立 NVIDIA `KDA_prefill` 产品仓库/API。因此最稳妥的归类是 **SGLang 内的 NVIDIA-copyrighted、vendored benchmark/research implementation**，不能只凭 `nvidia_kda` 名字认定它是 CUDA Toolkit 或 native cuDNN 算子。[vendored 来源与版权头](https://github.com/sgl-project/sglang/blob/71cee04ebe8061af901e4880169f6a5e86f7c8c1/python/sglang/kernels/ops/attention/linear/kda_nvidia_prefill/chunk_fwd.py#L1-L27)

它是 split KDA prefill pipeline，不是单 kernel：K1 gate/cumsum/scale 用 CuTe，K2 intra-subchunk 用 CuTe，K3 inter-chunk solve 用 Triton，K4 state/output 用 cuTile persistent kernel。

SGLang 只在 datacenter Blackwell capability major 10 且非 SM120 上启用；服务 wrapper 只接多序列 ordinary prefill，单序列回 Triton，B≤8，序列 bucket 为 2K/4K/8K/16K，最大 16K；不支持 speculative 或中间 checkpoint。[设备和 wrapper 边界](https://github.com/sgl-project/sglang/blob/71cee04ebe8061af901e4880169f6a5e86f7c8c1/python/sglang/srt/layers/attention/linear/kernels/kda_nvidia.py#L1-L29) [调度实现](https://github.com/sgl-project/sglang/blob/71cee04ebe8061af901e4880169f6a5e86f7c8c1/python/sglang/srt/layers/attention/linear/kernels/kda_nvidia.py#L260-L325)

### 7.3 `ptx_kda`

这是 hand-written CUDA/PTX/tcgen05 prefill kernel，只支持 GB300/SM103，固定 K=V=128、chunk=64。源码按 shape 选择：长单序列可走一个 fused grid，短 sequence/chunk 少时走 two-launch route，varlen 和多 sequence/high-H 也有各自路线。[wrapper](https://github.com/sgl-project/sglang/blob/71cee04ebe8061af901e4880169f6a5e86f7c8c1/python/sglang/srt/layers/attention/linear/kernels/kda_ptx.py#L1-L82) [CUDA/PTX 源码](https://github.com/sgl-project/sglang/blob/71cee04ebe8061af901e4880169f6a5e86f7c8c1/python/sglang/kernels/jit/csrc/attention/kda_prefill.cu)

它支持 packed varlen、initial/final state，并可输出 per-chunk state；SGLang 只有在 checkpoint stride 为 64 时才接这条 intermediate 路，否则回 Triton。单序列 scratch bucket 覆盖到 32K，更长长度没有当前 wrapper 契约。因此它展示了 GPU prefill 单网格融合的可行性，但不是跨 GPU 可移植的参考实现。

### 7.4 CuTeDSL 与 Helion

- SGLang CuTeDSL prefill 是 Triton prologue + CuTe KKT inverse/UW + CuTe state + CuTe output 的多 kernel pipeline，chunk=64、K=V=128、BF16；支持 varlen 与 state pool initial/final，但服务 wrapper 不导出 intermediate，且不支持 bounded safe gate。[wrapper](https://github.com/sgl-project/sglang/blob/71cee04ebe8061af901e4880169f6a5e86f7c8c1/python/sglang/srt/layers/attention/linear/kernels/kda_cutedsl.py) [Blackwell pipeline](https://github.com/sgl-project/sglang/tree/71cee04ebe8061af901e4880169f6a5e86f7c8c1/python/sglang/kernels/ops/attention/linear/kda_blackwell)
- Helion prefill 也是 L2 norm、gate/cumsum、intra solve、state、output 多阶段生成式 pipeline；支持 varlen、initial/intermediate 和两种 gate，T=1 交给 Triton。packed decode 会融合 recurrent KDA 的 gate/beta/QK norm/state/output，但不融合 conv update 和 output norm。[SGLang wrapper](https://github.com/sgl-project/sglang/blob/71cee04ebe8061af901e4880169f6a5e86f7c8c1/python/sglang/srt/layers/attention/linear/kernels/kda_helion.py) [Helion prefill](https://github.com/sgl-project/sglang/blob/71cee04ebe8061af901e4880169f6a5e86f7c8c1/python/sglang/kernels/ops/attention/helion/kda_prefill.py) [Helion decode](https://github.com/sgl-project/sglang/blob/71cee04ebe8061af901e4880169f6a5e86f7c8c1/python/sglang/kernels/ops/attention/helion/kda_decode.py)

SGLang 为 Apache-2.0。[LICENSE](https://github.com/sgl-project/sglang/blob/71cee04ebe8061af901e4880169f6a5e86f7c8c1/LICENSE)

## 8. FlashInfer 当前实现：框架集成可能滞后于底层库

当前 FlashInfer 已有独立的 KDA prefill/decode facade。普通多 token prefill 在 SM120 可走架构专用 CuTeDSL；SM100/SM103 可在 CuTeDSL 和冻结 Cake schedules 间选择。Cake 包含 direct、persistent、small-BH 与 two-stage BT16 prepare/chain 等 shape/device-specific 路线，不应统称成单 kernel。[公开 API 与 backend 说明](https://github.com/flashinfer-ai/flashinfer/blob/5a8e62ac5d359a70307c1ac163a1974dfcdcd660/flashinfer/kda.py#L85-L105) [prefill backend](https://github.com/flashinfer-ai/flashinfer/blob/5a8e62ac5d359a70307c1ac163a1974dfcdcd660/flashinfer/kda_prefill.py)

其公开契约包括 BF16 Q/K/V/G/Beta、D128 优化面、fixed/packed varlen、state-pool index、initial/final state；还支持周期 checkpoint，通常间隔需为 32 的倍数，SM100-family exact-N16 route 也接受 16 的倍数。[KDA API](https://github.com/flashinfer-ai/flashinfer/blob/5a8e62ac5d359a70307c1ac163a1974dfcdcd660/flashinfer/kda.py#L107-L258)

decode 侧支持 recurrent、packed 与 speculative fused 路线；例如 serving-native packed decode 融合 QKV extraction、Q/K norm、safe gate、beta sigmoid 与 state update，严格特化为 SM100/103、T=1、H=12、K=V=128，但仍不等同于 vLLM 的 conv+KDA+norm 全融合。[FlashInfer decode API](https://github.com/flashinfer-ai/flashinfer/blob/5a8e62ac5d359a70307c1ac163a1974dfcdcd660/flashinfer/kda_decode.py)

重要的是：当前 SGLang commit 的 dispatcher 只把 FlashInfer 接到 decode/verify，并没有使用这个更新的 FlashInfer prefill。因此判断“服务实际走哪条路”必须看 serving 框架 dispatcher，不能只看底层库是否已经有 kernel。[SGLang dispatcher](https://github.com/sgl-project/sglang/blob/71cee04ebe8061af901e4880169f6a5e86f7c8c1/python/sglang/srt/layers/attention/linear/kda_backend.py)

FlashInfer 为 Apache-2.0。[LICENSE](https://github.com/flashinfer-ai/flashinfer/blob/5a8e62ac5d359a70307c1ac163a1974dfcdcd660/LICENSE)

## 9. 与 `fla_npu` Ascend C `chunk_kda_fwd` 对照

### 9.1 NPU prefill 不是一个固定融合形态

`fla_npu` 顶层数学流水为：raw gate → gate cumsum → Prepare/Post-WU → FwdH → Finalize。A5 的 BF16、chunk=64、K=V=128 dense 对齐快路保持一次物理 `ChunkKdaFwd` L0；A5 的多 chunk tail/varlen 及泛化 shape 用私有 `stage` 属性把同一个 L0 入口分四次提交；A2/A3 与单 chunk 仍可用单次物理 L0。[设计文档](https://github.com/flashserve/flash-linear-attention-npu/blob/d4d2dc4b7a07316871088d5207a2aeb6d819a3db/fla/ops/ascendc/kda/chunk_kda_fwd/docs/design.md)

这和 FlashKDA 的架构判断相似：不同阶段并行结构与资源需求不同，融合到合理粒度优先于追求形式上的一次 launch。

### 9.2 NPU 支持面

当前验证范围为 FP16/BF16，K=128，V=128/256，chunk=64/128，BSND/BNSD/TND/NTD，dense/varlen/tail/GQA，safe/unbounded gate，initial/final state 和 optional intermediate；state 输入/输出固定 FP32。[shape/layout 契约](https://github.com/flashserve/flash-linear-attention-npu/blob/d4d2dc4b7a07316871088d5207a2aeb6d819a3db/fla/ops/ascendc/kda/README.md) [验证报告](https://github.com/flashserve/flash-linear-attention-npu/blob/d4d2dc4b7a07316871088d5207a2aeb6d819a3db/fla/ops/ascendc/kda/KDA_FORWARD_VALIDATION_REPORT.md)

报告中还有 `T=131072`、8 条非对齐 varlen、K=V=128 的 NTD 模型形状回归。这是 NPU 长上下文 correctness 的直接对照点。[长序列验证](https://github.com/flashserve/flash-linear-attention-npu/blob/d4d2dc4b7a07316871088d5207a2aeb6d819a3db/fla/ops/ascendc/kda/KDA_FORWARD_VALIDATION_REPORT.md#L69-L73)

decode 另有独立 `recurrent_kda`，单个 AIV kernel 内完成 Q/K norm、raw gate、state decay/delta update、output/final state；支持 BF16 QKV、FP16/BF16/FP32 gate/beta、BF16/FP32 state、varlen/state-pool/spec slot，但当前每条 recurrent sequence 长度限制为 ≤8。[recurrent design](https://github.com/flashserve/flash-linear-attention-npu/blob/d4d2dc4b7a07316871088d5207a2aeb6d819a3db/fla/ops/ascendc/kda/recurrent_kda/docs/design.md)

`flash-linear-attention-npu` 为 BSD-3-Clause。[LICENSE](https://github.com/flashserve/flash-linear-attention-npu/blob/d4d2dc4b7a07316871088d5207a2aeb6d819a3db/LICENSE)

### 9.3 一一对应关系

| NPU 设计点 | CUDA/GPU 可借鉴对象 | 说明 |
| --- | --- | --- |
| raw gate + Q/K norm + beta 融入 prefill | FlashKDA K1/K2 | 最直接的数学与融合边界参考 |
| NVIDIA 官方训练 KDA forward/backward | cudnn-frontend FROST + cuTile | 官方 OSS 参考；不是 native cuDNN backend primitive，且无 decode |
| Prepare / FwdH / Finalize 分阶段 | FLA Triton、cudnn-frontend、SGLang NVIDIA KDA、FlashKDA | 共同说明阶段可拆，关键是减少大中间量和重复读写 |
| dense 快路与 varlen/tail 泛化路分开 | FlashKDA、PTX KDA、FlashInfer Cake schedule | GPU 同样按 fixed/varlen、短/长、BH 数选择不同 schedule |
| 128K correctness | FlashKDA 长测试、NPU T=131072 回归 | 两端都有长度可运行证据，但目标 H/dtype/性能仍须逐点实测 |
| prefill intermediate state | 普通 Triton、PTX KDA、FlashInfer checkpoint | Moonshot upstream FlashKDA 不支持；vLLM fork仅一个对齐 checkpoint |
| recurrent decode | FLA/FlashInfer CuTeDSL；vLLM native fused decode | 若还要融合 conv 和 norm，vLLM 是更完整模板 |

## 10. 对 8K/16K/32K/64K/128K 优化验证的直接建议

如果目标是为另一个 GPU/MLU 后端迭代 KDA prefill kernel，建议不要把任务写成“照抄 Ascend C 或做一个单 kernel”。更准确的技术约束是：

1. 以 `fla_npu` 的数学阶段、layout/state/intermediate 语义作为 correctness 参考；
2. 以普通 FLA Triton 作为可运行基线与 fallback；
3. 把 NVIDIA cudnn-frontend v1.27 的 FROST BT16 forward 和 cuTile fwd/bwd 作为官方开源训练实现参考，但不要假定它能通过 native cuDNN backend 或覆盖 decode；
4. 以 Moonshot FlashKDA 的 K1/K2 边界、chunk=16、片上 state 和 workspace 设计作为 serving CUDA prefill 高性能主参考；
5. 允许候选方案是单 kernel、双 kernel或 3～4 阶段 pipeline，以端到端延迟、带宽和 workspace 结果决定，而非 launch 数；
6. 对 8K、16K、32K、64K、128K 每点分别做 correctness、峰值显存/workspace、kernel 数、端到端 latency、吞吐和数值漂移；不能用 8K 性能外推 128K；
7. 明确区分 single long sequence、packed varlen batch、tail、initial state、final state、checkpoint/intermediate 开关；这些条件会改变 dispatcher 和最佳 schedule；
8. serving 集成要记录最终实际选择的 backend。尤其 SGLang 当前会让 8K～128K FlashKDA 请求全部回 Triton，而 vLLM 的 FlashKDA `auto` 没有同样的长度阈值。

## 11. 最终判断

回答“NVIDIA 官方有没有 KDA 算子”：**有已发布的官方开源实现，但要加限定。**`NVIDIA/cudnn-frontend` v1.27.0 已有 KDA prefill/training forward+bwd 的 Python graph engine 与 PyTorch op；它没有 native cuDNN backend lowering，不是 CUDA Toolkit/native-cuDNN 产品 primitive，且该 release 中未发现 KDA decode engine。

回答“CUDA GPU 有没有专门的 KDA 融合算子”：**有。**官方训练实现可对标 cudnn-frontend；serving prefill 可对标 MoonshotAI/FlashKDA；decode 可对标 vLLM/FlashInfer fused KDA。后两者不能统称为 CUDA Toolkit 或 native cuDNN 官方 API。

回答“CUDA 是否已经有通用、跨架构、覆盖所有 shape/state/checkpoint、完整 prefill 单 launch 的算子”：**没有。**现有高性能实现都在融合度、架构、dtype、head dim、gate、varlen、checkpoint 和可移植性之间做取舍；普通 Triton 仍承担最完整的 fallback。

回答“能否直接用于 8K～128K”：**Moonshot FlashKDA 底层没有 2K 硬上限且有 128K/1M 可运行测试，但不同 serving 框架会施加自己的长度策略；当前 SGLang 的 FlashKDA 路径只选 64～2048，vLLM 则没有长度阈值。最终必须在目标框架、目标 H/批量和 state 模式下逐点验证。**
