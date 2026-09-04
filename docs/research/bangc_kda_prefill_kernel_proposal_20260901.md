# BANG C KDA Prefill 深度融合优化立项建议

> 修订日期：2026-09-01  
> 目标代码：xLLM `feat/mlu-glm5-next`，固定 commit `04fae8b3ad00202c91fd9628568713159326b37f`  
> 目标平台：寒武纪 MLU / BANG C  
> 目标算子：GLM5-Next KDA prefill  
> 结论口径：文中用 **事实**、**本地实验**、**推断**、**建议** 区分证据与项目决策。

## 0. 修订说明

旧版报告误以 `preview/glm-5.3-flash` 为 MLU 现状基线，因而得出了“MLU 只有 scalar-GDN、缺少专用 KDA chunk prefill”的错误结论。本版已按用户指定的 `feat/mlu-glm5-next` 重新审计。

修订后的准确结论是：

- **事实：** 目标分支已经有独立的 MLU `ChunkKDA`，不是把 scalar-GDN 的 `chunk_gated_delta_rule` 强行用于 GLM5-Next。
- **事实：** 现有实现已经处理 per-key gate、BF16 Q/K/V、FP32 V-first state、packed varlen、tail，以及 prefill 到 decode 的 state 交接。
- **事实：** 现有 prefill 仍由 Torch 运算、5 个 MLU Triton kernel 和 1 个 `solve_tril` op 组成；连同 Q/K normalize，实际提交数多于 6 次，且会物化多个 O(T) 中间张量。
- **事实：** 检入仓库的单元测试覆盖 chunk 边界和 prefill→decode，但没有检入 8K～128K 的正式长上下文回归。
- **本地实验：** 目标 worktree 中有一套未提交的 38-case MLU590 实验产物，覆盖到 128K，38/38 功能通过；但高级优化工作流最终为 `failed`，`accepted_case_count=0`、`delivery_status=none`，不能当作已交付实现。
- **建议：** BANG C 项目应定位为“在保持现有 ABI、数学和 fallback 的前提下，重写/融合当前多阶段 MLU KDA，降低 launch、workspace 和长上下文延迟”，而不是“从零补齐 KDA 功能”。

## 1. 立项结论

**建议立项，但立项类型应是性能与架构优化项目。**

项目目标建议写成：

> 以 `feat/mlu-glm5-next` 已有 `ChunkKDA` 为正确性和回退基线，实现 BANG C KDA prefill backend；通过片上数据驻留、DMA 流水、阶段融合和长序列调度，优化 8K、16K、32K、64K、128K，并保持 packed-varlen、tail、segmented-prefill、非零 initial state 和 prefill→decode 状态一致性。

不应再写成：

> MLU 没有 KDA prefill，需要补一个 KDA 算子。

## 2. 证据范围与版本

| 来源 | 固定版本/状态 | 用途 | 许可证 |
| --- | --- | --- | --- |
| xLLM `feat/mlu-glm5-next` | `04fae8b3` | 当前 MLU KDA 实现、集成和检入测试 | Apache-2.0 |
| xLLM 本地实验目录 | 未提交 worktree artifact | 38-case/MLU590 实验数据；不视为分支交付物 | 仅作本地证据 |
| `fla_npu` | [`d4d2dc4b`](https://github.com/flashserve/flash-linear-attention-npu/commit/d4d2dc4b7a07316871088d5207a2aeb6d819a3db) | Ascend C KDA 对照 | BSD-3-Clause |
| NVIDIA cudnn-frontend | [`v1.27.0 / f77fbc3d`](https://github.com/NVIDIA/cudnn-frontend/commit/f77fbc3d21be3f24cd0286b9b368105f7c518b8a) | NVIDIA 官方 OSS KDA engine | Apache-2.0 |
| MoonshotAI FlashKDA | [`1ce47ea3`](https://github.com/MoonshotAI/FlashKDA/commit/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b) | CUDA prefill 融合边界 | MIT |
| SGLang | [`71cee04e`](https://github.com/sgl-project/sglang/commit/71cee04ebe8061af901e4880169f6a5e86f7c8c1) | SM103 shape-aware 调度 | Apache-2.0 |
| Cambricon `mlu-ops` | [`0390ea42`](https://github.com/Cambricon/mlu-ops/commit/0390ea42853f38ba96be3e7404f47f39b1c32989) | BANG C 存储、任务和流水依据 | MIT |

目标分支在独立 worktree `/workspace/xllm-glm5-next`。本报告引用的 xLLM 行号均对应 `04fae8b3`，不再用 preview 分支推断 MLU 现状。

## 3. `feat/mlu-glm5-next` 的真实实现

### 3.1 模型和 ABI

**事实：** 目标分支配置读取逻辑给出的默认 KDA 参数为：

| 项 | 值 |
| --- | --- |
| 最大 position 默认值 | 1,048,576 |
| 全局 linear heads | 64 |
| K/V head dim | 128 / 128 |
| depthwise conv width | 4 |
| safe gate lower bound | -5.0 |
| state dtype | FP32 |
| 本地 heads | `64 / TP` |

证据见 [glm5_next_args.h](/workspace/xllm-glm5-next/xllm/models/llm/mlu/glm5_next_args.h:105)。KDA 层数由模型 `layer_types` 决定，分支要求其长度等于 `num_hidden_layers`，不能只依据代码默认值硬编码 34 层。[配置校验](/workspace/xllm-glm5-next/xllm/models/llm/mlu/glm5_next_args.h:261)

**事实：** 当前 `ChunkKDA` 接受：

```text
q, k       BF16 [1,T,H,128]
v          BF16 [1,T,H,128]
log_gate   FP32 [T,H,128] 或 [1,T,H,128]
beta       FP32 [T,H] 或 [1,T,H]，已经 sigmoid
state      FP32 [N,H,V,K]，V-first
cu_seqlens INT32 [N+1]
chunks     INT32 [num_chunks,2]，每行 [sequence, local_chunk]
output     BF16 [1,T,H,128]
final      FP32 [N,H,V,K]
```

它固定 `chunk_size=64`、K=V=128、packed batch dimension=1，并用 `cu_seqlens/chunk_indices` 表示多序列和尾块。[接口定义](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/chunk_kda.h:25) [运行时检查](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/chunk_kda.cpp:262)

### 3.2 GLM5-Next 实际调用链

**事实：** GLM5-Next 为 KDA 单独注册 `ChunkKDA`：[模块注册](/workspace/xllm-glm5-next/xllm/core/layers/mlu/glm5_next/glm5_next_kda.cpp:318)。

Prefill 路径依次执行：

1. Q/K/V projection、raw gate、beta 和 output gate projection；
2. causal conv1d；
3. `glm5_next_safe_gate` 生成 `[T,H,K]` per-key log gate；
4. `sigmoid(beta)`；
5. 从 FP32 SSM cache 取 initial state；
6. 调用 `chunk_kda_->forward(...)`；
7. 回写 final state；
8. 在算子外执行 output norm、sigmoid gate 和 output projection。

证据见 [prefill 调用](/workspace/xllm-glm5-next/xllm/core/layers/mlu/glm5_next/glm5_next_kda.cpp:472)。Decode/spec-verify 则走 `fused_sigmoid_gating_delta_rule_update` 并设置 `is_kda=true`、safe gate=true，不复用 prefill chunk pipeline。[decode/spec-verify 路径](/workspace/xllm-glm5-next/xllm/core/layers/mlu/glm5_next/glm5_next_kda.cpp:517)

因此 scalar-GDN 与 GLM5-Next KDA 在该分支中已经正确分流：

```text
Qwen3.5 scalar-GDN prefill -> ChunkGatedDeltaRule
GLM5-Next KDA prefill      -> ChunkKDA
GLM5-Next decode/verify    -> fused update with is_kda=true
```

### 3.3 当前 KDA prefill pipeline

**事实：** `ChunkKDAImpl::forward` 当前包含以下主要阶段：[完整 orchestration](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/chunk_kda.cpp:262)

| 阶段 | 当前实现 | 主要输出 |
| --- | --- | --- |
| Q/K normalize 与 Q scale | Torch tensor op | BF16 normalized Q/K |
| per-key chunk cumsum | MLU Triton | FP32 cumulative gate `[1,T,H,128]` |
| KKT/QKT inter | MLU Triton | FP32 matrix 与 intra-output matrix |
| KKT/QKT intra | MLU Triton | 补齐同一 16-token block 内三角部分 |
| triangular solve | `tmo::torch_api::solve_tril` | BF16 solved matrix |
| recompute W/U/gated-K | MLU Triton | 三个 BF16 O(T) tensor |
| state recurrence/output | MLU Triton | BF16 output、FP32 final state |

`scaled_dot_kkt` 逻辑阶段内部拆成 inter 与 intra 两个 Triton launch。因此只计算 chunk 主体，不含 gate/conv/output norm，也至少有 5 个 Triton launch加 1 个 `solve_tril` op；Q/K normalize 的 Torch 运算还会产生额外提交。[五个 Triton kernel](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/triton_kernel/chunk_kda_fwd.py:22)

### 3.4 当前实现不是错误实现

现有代码明确检查 `gate.sizes() == q.sizes()`，即 gate 是 `[1,T,H,K]`；state 明确为 `[N,H,V,K]`。这解决了旧报告担心的 scalar gate 和 state layout 问题。[gate/state 检查](/workspace/xllm-glm5-next/xllm/core/kernels/mlu/chunk_kda.cpp:286)

所以本项目不应以“现有代码数学错误”为立项前提。是否还存在实现 bug，需要通过代码评审和硬件回归单独判断；从接口和调用链本身不能推出有 bug。

## 4. 现有测试和实验数据

### 4.1 检入仓库的正式测试

**事实：** 当前正式单测覆盖：

- safe bounded per-key gate；
- output gate 使用 sigmoid；
- eager recurrence 跨 segment 状态保持；
- packed lengths `{1,63,64,65,127}`；
- 65-token prefill state 接入已有 decode kernel。

测试见 [glm5_next_kda_test.cpp](/workspace/xllm-glm5-next/tests/core/layers/mlu/glm5_next_kda_test.cpp:104)。精度门槛为 output/state `rtol=3e-2, atol=2e-2`。测试会记录 320 packed tokens 的 eager/chunk latency，但没有性能断言，也没有检入 8K～128K case。

### 4.2 本地未提交的 38-case 实验

**本地实验，不是分支事实：** `/workspace/xllm-glm5-next/output_mlu_triton_kernel_dev/` 中存在针对 MLU590-M9DK 的 38-case 优化工作流。它覆盖：

- 边界 1/63/64/65/127/128/129/255/256/257；
- one-shot 8K/16K/32K/64K/128K，zero/warm state；
- 8K/16K segment 的 segmented continuation；
- packed varlen 与 H=1/2/4/8；
- BF16 output、FP32 final/intermediate state。

Case 定义见 [case_manifest.md](/workspace/xllm-glm5-next/output_mlu_triton_kernel_dev/CodeGen/CaseNormalizer/case_manifest.md:1)。最新合并候选 38/38 功能通过，H=8 warm one-shot p50 为：

| T | p50 | 备注 |
| ---: | ---: | --- |
| 8K | 2.675 ms | MLU590-M9DK 本地候选 |
| 16K | 5.279 ms | 同上 |
| 32K | 10.522 ms | 同上 |
| 64K | 20.965 ms | 同上 |
| 128K | 42.688 ms | 同上 |

数据见 [merged_classification.md](/workspace/xllm-glm5-next/output_mlu_triton_kernel_dev/AdvancedOptimizationWorkflow/classification/merged_classification.md:1)。

必须同时保留以下限制：

- 目录未提交，不属于 `feat/mlu-glm5-next@04fae8b3`；
- 测的是生成候选，不等于当前检入 `ChunkKDA`；
- 最终 workflow 状态为 `failed`；
- `accepted_case_count=0`、`delivery_status=none`；
- 失败原因是连续三轮未减少剩余 case，不是功能精度失败。

状态见 [artifact_index.json](/workspace/xllm-glm5-next/output_mlu_triton_kernel_dev/AdvancedOptimizationWorkflow/final/artifact_index.json:1)。因此这些数据适合用来定义 BANG C case 和参考量级，不适合宣称“现有 Triton 已完成深度优化”。

## 5. 修订后的性能缺口

### 5.1 主要缺口

> xLLM MLU 已经有正确的 KDA prefill；当前缺口是没有 BANG C 深度融合 backend，也没有进入仓库和 CI 的 8K～128K 可复现性能/显存基线与验收闭环。

具体表现为：

1. 多阶段 launch 和 Torch orchestration；
2. 多个 O(T) 中间 tensor 同时存活；
3. 固定 chunk=64，没有基于 MLU 实测的 route/chunk dispatcher；
4. 检入测试只覆盖小长度；
5. 长单序列、低 `H_local`、packed-varlen 的调度仍由同一 pipeline 承担；
6. 当前不支持 context parallelism，并要求 eager execution，这属于模型集成约束，不应由 KDA kernel 项目擅自放宽。[模型约束](/workspace/xllm-glm5-next/xllm/models/llm/mlu/glm5_next.h:49)

### 5.2 当前中间量规模

按检入 `ChunkKDA` 的实际分配，对 `T=131072,H_local=8,K=V=128` 做静态容量计算：

| Tensor | dtype/shape | 约占用 |
| --- | --- | ---: |
| cumulative gate | FP32 `[T,H,128]` | 512 MiB |
| matrix | FP32 `[T,H,64]` | 256 MiB |
| intra-output matrix | FP32 `[T,H,64]` | 256 MiB |
| solved matrix | BF16 `[T,H,64]` | 128 MiB |
| W | BF16 `[T,H,128]` | 256 MiB |
| U | BF16 `[T,H,128]` | 256 MiB |
| gated-K | BF16 `[T,H,128]` | 256 MiB |
| output | BF16 `[T,H,128]` | 256 MiB |
| 合计 | 不含输入、normalize 临时量和 state | 2,176 MiB |

这是从源码 shape 推导的逻辑分配量，不是 profiler 峰值。TP=1/H=64 时线性放大约 8 倍；scheduler segmentation 会降低单次调用 T，但增加多次 state handoff 和 launch。减少中间量及其 GDRAM traffic 是 BANG C 立项的核心价值。

## 6. Ascend C 和 CUDA 实现能借鉴什么

### 6.1 Ascend C `chunk_kda_fwd`

`fla_npu` 支持 K=128、V=128/256、chunk=64/128、dense/varlen/tail、initial/final state 和 FP32 state。[接口说明](https://github.com/flashserve/flash-linear-attention-npu/blob/d4d2dc4b7a07316871088d5207a2aeb6d819a3db/fla/ops/ascendc/kda/README.md)

其 Gate/Prepare、Post-WU、FwdH、Finalize 分层可用来检查 BANG C 的依赖图和 fallback；A5 dense aligned 快路与泛化多阶段 route 分离，说明不必强制所有 shape 走同一个融合内核。[Ascend C 设计](https://github.com/flashserve/flash-linear-attention-npu/blob/d4d2dc4b7a07316871088d5207a2aeb6d819a3db/fla/ops/ascendc/kda/chunk_kda_fwd/docs/design.md)

### 6.2 NVIDIA 官方实现

NVIDIA cudnn-frontend v1.27 提供 KDA graph op、FROST forward 和 cuTile forward/backward engine，但官方文档将其归为 Python-engine-only linear-attention op，没有 native cuDNN backend lowering。[官方 engine 说明](https://github.com/NVIDIA/cudnn-frontend/blob/f77fbc3d21be3f24cd0286b9b368105f7c518b8a/docs/python_graph_and_execution_backends.md#L64-L105)

可迁移的是 verifier、workspace query、packed metadata、unsupported fallback；不能迁移的是 TMA、CUTLASS/CuTile、SM100/103 调度和指令。

### 6.3 FlashKDA

FlashKDA 将 chunk-local prepare 与跨 chunk recurrence/output 拆成 K1+K2。其设计记录指出早期单 kernel 比双 kernel 至少慢 15%，说明“越融合越快”不是成立条件。[设计记录](https://github.com/MoonshotAI/FlashKDA/blob/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b/docs/20260420-flashkda-v1-deep-dive.md#L28-L43)

BANG C 应迁移 K1/K2 融合边界和片上 state 思路，不迁移 WGMMA、tcgen、CuTe tile。

### 6.4 SGLang PTX KDA

SGLang 的 SM103 实现按长单序列、短序列、varlen 和 high-H 选择不同 route。[wrapper](https://github.com/sgl-project/sglang/blob/71cee04ebe8061af901e4880169f6a5e86f7c8c1/python/sglang/srt/layers/attention/linear/kernels/kda_ptx.py#L1-L82) [CUDA/PTX 源码](https://github.com/sgl-project/sglang/blob/71cee04ebe8061af901e4880169f6a5e86f7c8c1/python/sglang/kernels/jit/csrc/attention/kda_prefill.cu)

可迁移的是 shape-aware dispatcher、长序列 persistent window、bucket workspace；不能迁移的是 PTX、TMEM、CTA cluster 和 SM103 资源假设。

## 7. BANG C 能力映射

Cambricon 官方 `mlu-ops` 可确认：NRAM 是 core-private 低延迟片上空间，WRAM 主要用于特殊权重布局，SRAM 在 cluster 内共享；Block task 不共享 SRAM，Union task 可以使用 SRAM 和 cluster 同步。[存储模型](https://github.com/Cambricon/mlu-ops/blob/0390ea42853f38ba96be3e7404f47f39b1c32989/docs/BANG-C-OPS-Develop-Guide.md#L639-L647) [Block/Union](https://github.com/Cambricon/mlu-ops/blob/0390ea42853f38ba96be3e7404f47f39b1c32989/docs/BANG-C-OPS-Develop-Guide.md#L891-L903)

官方指南还给出 GDRAM→SRAM→NRAM、异步 memcpy 和 ping-pong 流水。[NRAM 双缓冲](https://github.com/Cambricon/mlu-ops/blob/0390ea42853f38ba96be3e7404f47f39b1c32989/docs/BANG-C-OPS-Develop-Guide.md#L999-L1007) [Memory Core/SRAM 流水](https://github.com/Cambricon/mlu-ops/blob/0390ea42853f38ba96be3e7404f47f39b1c32989/docs/BANG-C-OPS-Develop-Guide.md#L1301-L1322)

建议映射：

| 数据/工作 | 候选位置 | 目的 |
| --- | --- | --- |
| Q/K/V/g/beta 当前 tile | NRAM ping-pong | 隐藏 GDRAM 搬运，复用 exp/cumsum 输入 |
| `[V_tile,K]` FP32 state | NRAM | 跨 chunk 保持，避免每 chunk 回写 GDRAM |
| chunk descriptor/共享 tile | SRAM | Union route 下 Memory Core 预取 |
| 矩阵 primitive 特殊布局 | WRAM 候选 | 仅在目标卡 microbench 证明有收益后使用 |
| output/final state | GDRAM | 持久结果 |
| 跨 launch 压缩 intermediate | GDRAM/LDRAM | 只保留下一阶段必须的数据 |

公开源码不能证明目标 SKU 的具体 NRAM/SRAM/WRAM 容量、DMA channel、BF16 matrix tile 或 barrier 成本。P0 必须从目标 MLU/CNToolkit 查询和实测，不得抄 Ascend/CUDA 参数。

## 8. 建议 BANG C 架构

### 8.1 兼容性原则

BANG C wrapper 首版直接兼容现有 `ChunkKDA::forward` 合约：

- BF16 Q/K/V/output；
- FP32 activated log gate、beta、initial/final state；
- `[1,T,H,128]` packed tensor；
- INT32 `cu_seqlens/chunk_indices`；
- chunk=64；
- state V-first；
- Q/K L2Norm 可开关。

现有 MLU Triton `ChunkKDA` 保留为 fallback 和 A/B 基线。首版不融合 causal conv、raw gate projection、beta sigmoid、output norm/gate，以免扩大 ABI 和数值风险。

### 8.2 三个候选

**Candidate A：阶段对齐 BANG C。** 先按现有数学阶段实现 BANG C 版本，逐阶段对比 cumulative gate、matrix、W/U/gated-K、output 和 final state。这是 bring-up 路径，不假设最终性能最好。

**Candidate B：K1+K2。**

1. K1 融合 normalize/scale、per-key cumsum、KKT/QKT、triangular prepare 与 W/U 压缩表示；
2. K2 对 `(sequence,head,V_tile)` persistent 地推进 chunk state并产生 output/final state。

目标是避免全量物化 cumulative gate、两份 FP32 `[T,H,64]` matrix 和 W/U/gated-K 中不必要的组合。若 triangular solve 无法安全并入 K1，可保留 3-kernel route。

**Candidate C：长序列 streaming。** 对 B=1、H_local 较小、T≥32K 的 route，按 chunk window 处理；window 内中间量生产后立即被 recurrence/output 消费。它必须与多阶段 fallback A/B 对比，不能以“单 grid”为验收条件。

### 8.3 并行轴

- chunk-local prepare：`(sequence, head, chunk, matrix_tile)`；
- recurrence/output：`(sequence, head, V_tile)`，在 task 内沿 chunk/window 维顺序推进；
- TP=8 时 H_local=8，必须增加 V tile、sequence、chunk-local tile 并行；
- packed-varlen 与 single-long-sequence 分 route；
- tail mask 不能拖慢 dense aligned 主路径。

## 9. `max_prefill_tokens` 的正确映射

目标分支中没有名为 `max_prefill_tokens` 的核心配置字段。xLLM 对应的实际配置是：

- `max_tokens_per_batch`：每 batch token 总预算，默认 10240；
- `max_tokens_per_chunk_for_prefill`：单次 chunked-prefill 的 token 上限，默认 -1，表示未单独限制。

见 [scheduler_config.h](/workspace/xllm-glm5-next/xllm/core/framework/config/scheduler_config.h:62) 和 [scheduler_policy.cpp](/workspace/xllm-glm5-next/xllm/core/scheduler/scheduler_policy.cpp:328)。

如果优化 skill/prompt 使用 `max_prefill_tokens`，应把它解释为测试时的“单次 prefill segment 上限”，映射到 xLLM 的 `max_tokens_per_chunk_for_prefill`，而不是 KDA kernel 的编译期最大 T。

即使线上值暂未确定，也必须同时验证：

1. one-shot 8K/16K/32K/64K/128K；
2. 同一逻辑序列按 2K/4K/8K/16K 分段，串联 FP32 final/initial state；
3. one-shot 与 segmented 的 output/final state 等价。

## 10. 实施路线

### P0：冻结现有基线

交付：

1. 固定 `feat/mlu-glm5-next@04fae8b3`；
2. 将 38-case manifest 和运行脚本整理成可提交、可复现测试；
3. 分别测检入 `ChunkKDA` 和本地生成候选，不混写结果；
4. 记录每阶段 latency、launch、峰值显存和 O(T) 中间量；
5. 冻结目标 MLU、CNToolkit/CNCC/CNRT/torch_mlu 版本及片上资源事实表；
6. 保留 eager recurrence 和 prefill→decode 测试作为 reference。

退出条件：现有基线在所有目标 case 上结果可复现；不能复现的本地实验数字不得作为 BANG C KPI 基线。

### P1：BANG C 阶段对齐

交付 Candidate A，优先覆盖 H=8、K=V=128、chunk=64、BF16/FP32、zero/warm state、dense/tail/packed。

退出条件：每个阶段和最终 output/state 对齐；8K/16K/32K 可稳定运行；不支持条件能回退到现有 `ChunkKDA`。

### P2：融合与 workspace 优化

交付 Candidate B，比较 2-kernel 与 3-kernel route，减少 O(T) 中间张量和 GDRAM round trip。

退出条件：8K～32K 端到端优于检入基线；workspace 明显下降；tail/packed/segmented 无回归。

### P3：64K/128K 和调度优化

交付 Candidate C、shape-aware dispatcher、workspace bucket、长稳测试和模型级结果。

退出条件：64K/128K one-shot 与 segmented 全部通过，无 OOM、越界、死锁；默认 route 由数据决定，未获益 route 不启用。

## 11. 8K～128K 验收矩阵

每个 `L ∈ {8192,16384,32768,65536,131072}` 至少覆盖：

| Case | 组织 | Heads | State | 目的 |
| --- | --- | --- | --- | --- |
| D1 | B=1，T=L one-shot | H_local=8 | zero | 主性能面 |
| D2 | B=1，T=L one-shot | H_local=8 | random FP32 | warm state |
| D3 | B=1，T=L | H=64 | zero/random | TP=1 扩展性与内存上界 |
| T1 | T=L-1/L+1 | H=8 | zero/random | tail |
| V1 | packed N=2，总 token≈L | H=8 | 每 seq 不同 | varlen 隔离 |
| V2 | packed N=8，含 1/63/64/65 | H=8 | cold/warm 混合 | ragged 边界 |
| S1 | 按 2K/4K/8K/16K 分段 | H=8 | state 串联 | scheduler 等价性 |
| C1 | 全模型 KDA 层链路 | 部署 TP | framework cache | 误差、峰值显存、真实收益 |

正确性至少记录：allclose、max abs、NRMSE、cosine、NaN/Inf、output、final state、每段 intermediate state。初始门槛沿用当前测试的 `rtol=3e-2, atol=2e-2`；若现有基线在同输入上能达到更严标准，则采用更严标准。

性能必须同时报告：

- 检入 MLU Triton `ChunkKDA`；
- 可复现时的本地生成候选；
- BANG C Candidate A/B/C；
- kernel-only 与完整 KDA op p50/p90；
- launch 数、tokens/s、峰值显存、静态 workspace；
- actual route 和 fallback reason。

建议目标而非上游事实：

- 所有 case 正确且无静默 fallback；
- 8K～32K 相对检入基线几何平均 ≥1.30×；
- 64K/128K ≥1.50×；
- 峰值临时显存相对检入基线下降 ≥50%；
- 默认 route 连续 1000 次无漂移，混合长短/packed/segmented 压力至少 1 小时。

## 12. 风险与缓解

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| 把旧报告的“缺功能”当立项前提 | 团队重复实现已有能力 | 以 `ChunkKDA` 为基线，立项标题明确“融合优化” |
| 把本地未提交 artifact 当正式基线 | KPI 不可复现 | P0 重跑并提交脚本/JSON；标注 commit/hardware |
| K=V=128 掩盖 state 转置 | prefill 可能过、decode 错 | V-first 固化；prefill→decode；非对称数值 pattern |
| 过度融合 | NRAM/寄存器压力抵消 launch 收益 | 同时保留 2/3-stage 候选，用 profiler 晋升 |
| H_local=8 并行度不足 | 设备利用率低 | 拆 V tile、sequence 和 chunk-local matrix tile |
| 128K 中间量/输入占用高 | OOM、带宽瓶颈 | streaming window、压缩 intermediate、scheduler segment A/B |
| packed descriptor 串序列 | state 污染 | packed 对逐 sequence reference；每 seq final state 对比 |
| exp/cumsum 漂移 | 长序列 state 误差 | FP32 gate/cumsum/state；分段等价；误差随 L 曲线 |
| BANG C硬件信息不足 | tile/Union 选择失误 | P0 查询和 microbench，不抄 CUDA/Ascend 常量 |
| fallback 使用 CHECK 终止 | 不支持 shape 无法服务 | wrapper verifier + dispatcher，回退当前 `ChunkKDA` |

## 13. 团队交付物

1. 当前 `ChunkKDA` 的可复现 38-case correctness/performance 基线；
2. BANG C ABI、workspace query、verifier、dispatcher；
3. Candidate A 阶段对齐实现和逐阶段调试工具；
4. Candidate B 2/3-kernel 融合实现；
5. Candidate C 长序列 streaming route；
6. dense/tail/packed/segmented/pre-decode 单元与集成测试；
7. 8K～128K 原始 JSON、日志、profiler 与峰值显存；
8. NRAM/SRAM/WRAM、DMA、矩阵 primitive microbench；
9. actual-route telemetry 和明确 fallback reason；
10. 全模型 KDA 层服务级正确性与性能报告；
11. 第三方来源和许可证清单；
12. 支持矩阵、目标硬件/工具链版本和回归命令。

## 14. Go/No-Go

**Go：** 当前 `ChunkKDA` 38-case 基线可复现；目标 BANG C 工具链能表达 BF16 matrix/vector 与 FP32 state；Candidate A 可对齐全部中间结果。

**进入融合：** Candidate A 在 8K～32K 正确且稳定；profiling 证明主要瓶颈来自多阶段 GDRAM traffic、launch 或中间量，而不是模型外 conv/projection。

**进入 128K：** tail、packed、segmented、prefill→decode 全部通过；64K/128K 内存预算可控。

**不晋升默认：** BANG C route 无稳定端到端收益、精度更差、workspace 更高，或 unsupported 条件不能可靠 fallback。此时保留实验实现和数据，但继续使用检入 MLU Triton `ChunkKDA`。

最终建议：**不要重新发明 KDA 数学；以现有 `ChunkKDA` 为契约和基线，把 BANG C 团队的精力集中在片上驻留、阶段融合、workspace 和 8K～128K 调度。**
