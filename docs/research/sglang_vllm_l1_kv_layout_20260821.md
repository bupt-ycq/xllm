# SGLang 与上游 vLLM L1 KV Cache 跨层 Page-first 布局调研

日期：2026-08-21  
范围：L1（GPU/加速器 HBM）KV cache 的**跨层物理布局**，不把“单层内部 block-first”或 L2 host 的 page-first 混入结论。  
来源：仅使用官方 GitHub 源码、官方文档、官方 PR/issue。

## 调研快照

- vLLM `main`：[`df3b3422b495a33fd10d0b2f06c051144d6bb2c2`](https://github.com/vllm-project/vllm/commit/df3b3422b495a33fd10d0b2f06c051144d6bb2c2)，提交日期 2026-08-20。
- SGLang `main`：[`e0cf75d9bda78fa46bdc5ff7436c5a76290663d1`](https://github.com/sgl-project/sglang/commit/e0cf75d9bda78fa46bdc5ff7436c5a76290663d1)，提交日期 2026-08-21。
- 同时复核了当前最新稳定 tag：vLLM [`v0.27.1`](https://github.com/vllm-project/vllm/tree/v0.27.1) 已包含 `use_uniform_kv_cache`/cross-layer blocks；SGLang [`v0.5.17`](https://github.com/sgl-project/sglang/tree/v0.5.17) 已包含 `--enable-page-major-kv-layout` 与 `PageMajorMHATokenToKVPool`。因此下述核心结论不是仅存在于尚未发布的 `main`。

这里的定义如下：

- **跨层 page-first**：同一个 logical block/page 的多层 KV 在一块连续或统一的 block slab 中；计算时每层通过 strided view、offset 或重编号后的 dense view 访问。
- **layer-first/general**：每层拥有独立 tensor，或一个 tensor 只代表某一层的完整 block 序列。即使单层 tensor 自身是 `[num_blocks, ...]`，也不算“跨层 page-first”。

## 总结结论

两者当前都只能说“**部分支持 L1 跨层 page-first**”，默认和适用范围不同：

| 场景 | vLLM `main` | SGLang `main` |
|---|---|---|
| 默认普通 MHA/GQA，无 KV transfer/offload | layer-first/general | layer-first |
| 普通 MHA/GQA + 合适 connector/backend | 可自动或显式进入 cross-layer page-first | 不由 HiCache 自动改变；需显式 `--enable-page-major-kv-layout` |
| 普通 MLA | 单一统一 group + connector 偏好 + 已验证 MLA backend 时可 page-first | 默认 layer-first；普通 page-major 开关不替换纯 MLA pool |
| DSA / DeepSeek-V3.2 indexer / GLM-5.2 | layer-first/general | layer-first |
| 普通 multi-group / Full+SWA | 默认 general；可通过实验配置进入 packed block slab | 默认分别 layer-first；page-major 开关使 Full、SWA 各自 page-first |
| DeepSeek-V4 | 默认 packed per-block slab | 专用 pool 仍逐层分配；unified-memory 明确排除 DSV4 |
| Mamba/SSM hybrid | NIXL 明确不启用 cross-layer uniform layout | page-major 可覆盖 MHA 与 Mamba state；更彻底的 unified-memory 仅支持部分 hybrid |
| CPU offload/HiCache | OffloadingConnector 默认偏好 cross-layer（canonical layout 除外） | `--hicache-mem-layout page_first` 只控制 L2；standalone L1 PageMajorMHA 暂不支持 CPU offload |

对当前讨论的 **GLM-5.2**，两个最新上游主干的结论仍然一致：**L1 不是跨层 page-first，而是 layer-first/general**。

## 1. 上游 vLLM

### 1.1 默认路径仍是 layer-first/general

vLLM 的 cross-layer uniform allocation 不是默认布局。`use_uniform_kv_cache()` 首先要求已创建 KV transfer group；没有 connector 时直接返回 `False`，然后 model runner 落到 general allocation。源码对 uniform layout 的定义也很明确：同一 block 的所有层数据连续，可一次传输整块。[判定逻辑](https://github.com/vllm-project/vllm/blob/df3b3422b495a33fd10d0b2f06c051144d6bb2c2/vllm/v1/worker/kv_connector_model_runner_mixin.py#L115-L162)

general allocation 中，不带 `block_stride` 的 `KVCacheTensor` 会分别 `torch.zeros()`；随后再按各层 backend shape 构造 view。[general/packed allocation 分叉](https://github.com/vllm-project/vllm/blob/df3b3422b495a33fd10d0b2f06c051144d6bb2c2/vllm/v1/worker/gpu_model_runner.py#L7444-L7585)

因此普通模型即使每层内部是：

```text
Layer 0: [block0][block1][block2]...
Layer 1: [block0][block1][block2]...
```

也仍属于跨层意义上的 layer-first。

### 1.2 单一统一 group 的 cross-layer page-first

这条路径必须同时满足：

1. 已配置 KV connector；
2. connector 的 `prefer_cross_layer_blocks` 为真；
3. 只有一个 KV cache group，且其中只有一个 AttentionGroup；
4. group spec 是 `AttentionSpec`；
5. 不是 per-token-head KV quant；
6. attention backend 声明能够按运行时 block stride 索引 KV。

以上条件直接来自 [`use_uniform_kv_cache()`](https://github.com/vllm-project/vllm/blob/df3b3422b495a33fd10d0b2f06c051144d6bb2c2/vllm/v1/worker/kv_connector_model_runner_mixin.py#L115-L162)。backend 的能力门槛是：带 layer 维度的 stride order 必须把 `num_blocks` 放在物理最外层，而不能让 `num_layers` 留在最外层。[`indexes_kv_by_block_stride()`](https://github.com/vllm-project/vllm/blob/df3b3422b495a33fd10d0b2f06c051144d6bb2c2/vllm/v1/attention/backend.py#L219-L248)

通过门槛后，vLLM 分配一个 `cross_layers_kv_cache`，再 `permute` 成每层 view；不是为每层重新复制一份。[统一 buffer 分配](https://github.com/vllm-project/vllm/blob/df3b3422b495a33fd10d0b2f06c051144d6bb2c2/vllm/v1/worker/kv_connector_model_runner_mixin.py#L191-L255)

普通 FlashAttention 的物理顺序由 cache layout 决定：

- NHD：`[block, layer, token-in-block, head, content]`；
- HND：`[block, head, layer, token-in-block, content]`。

两者都是 block/page 最外层，所以都满足跨层 page-first；区别只是 page 内 layer 与 head 的次序。[FlashAttention stride order](https://github.com/vllm-project/vllm/blob/df3b3422b495a33fd10d0b2f06c051144d6bb2c2/vllm/v1/attention/backends/flash_attn.py#L150-L175)

### 1.3 Connector 对 page-first 的影响

- `OffloadingConnector` 默认 `canonical_layout=False`，因此默认偏好 cross-layer；显式启用 canonical layout 时会关闭这一偏好。[connector 属性](https://github.com/vllm-project/vllm/blob/df3b3422b495a33fd10d0b2f06c051144d6bb2c2/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py#L49-L70) [canonical 默认值](https://github.com/vllm-project/vllm/blob/df3b3422b495a33fd10d0b2f06c051144d6bb2c2/vllm/v1/kv_offload/config.py#L75-L86)
- NIXL 还要求无 Mamba group、backend 名称属于 `FLASH_ATTN`/`FLASHINFER`/`TRITON_ATTN`、KV layout 为 HND，并显式设置 `enable_cross_layers_blocks=true`。[NIXL 门槛](https://github.com/vllm-project/vllm/blob/df3b3422b495a33fd10d0b2f06c051144d6bb2c2/vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py#L87-L114)
- Mooncake Store 默认不启用，只有 `enable_cross_layers_blocks=true` 才返回偏好。[Mooncake Store 门槛](https://github.com/vllm-project/vllm/blob/df3b3422b495a33fd10d0b2f06c051144d6bb2c2/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py#L89-L98)
- `MultiConnector` 要求所有子 connector 都偏好 cross-layer；只要一个不支持，就回退。[MultiConnector 逻辑](https://github.com/vllm-project/vllm/blob/df3b3422b495a33fd10d0b2f06c051144d6bb2c2/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py#L208-L216)

所以 vLLM 的普通 single-group 模型更接近：

```text
无 connector / connector 不偏好 / backend 不支持
    -> layer-first/general

connector 偏好 + single uniform group + stride-aware backend
    -> cross-layer page-first
```

### 1.4 MLA：已从“一律不支持”演进为 backend 选择性支持

上游 MLA common backend 默认仍用 identity layered stride order，表示不支持跨层 layout；每个 MLA backend 必须单独 opt-in。[MLA common 默认](https://github.com/vllm-project/vllm/blob/df3b3422b495a33fd10d0b2f06c051144d6bb2c2/vllm/model_executor/layers/attention/mla_attention.py#L1450-L1464)

当前已验证的 `FlashAttnMLA`、`FlashInferMLA`、`FlashMLA`、`TritonMLA`、`CutlassMLA` 都把 layered stride order 设成 `(1, 0, 2, 3)`，即物理 `[block, layer, token-in-block, dim]`。[官方验证矩阵](https://github.com/vllm-project/vllm/blob/df3b3422b495a33fd10d0b2f06c051144d6bb2c2/tests/v1/kv_connector/unit/test_kv_cache_layout.py#L27-L58) 例如 [`FlashAttnMLABackend`](https://github.com/vllm-project/vllm/blob/df3b3422b495a33fd10d0b2f06c051144d6bb2c2/vllm/v1/attention/backends/mla/flashattn_mla.py#L50-L65)。因此，**不带异构 indexer、能形成单一普通 `AttentionSpec` group 的 MLA**，现在可以在满足 connector 条件时进入 cross-layer page-first。

但 DeepSeek-V3.2/DSA indexer 明确返回 identity layered order，注释直接说明其 kernel 不支持 cross-layer KV layout。[Indexer backend](https://github.com/vllm-project/vllm/blob/df3b3422b495a33fd10d0b2f06c051144d6bb2c2/vllm/v1/attention/backends/mla/indexer.py#L185-L205)

### 1.5 GLM-5.2 / DSA 为什么仍是 layer-first

上游把 `GlmMoeDsaForCausalLM` 直接建立在 DeepSeek-V2/V3.2 路线上；当配置含 `index_topk` 时会创建 indexer cache。[GLM DSA 模型类](https://github.com/vllm-project/vllm/blob/df3b3422b495a33fd10d0b2f06c051144d6bb2c2/vllm/model_executor/models/deepseek_v2.py#L1955-L1964) [indexer 创建条件](https://github.com/vllm-project/vllm/blob/df3b3422b495a33fd10d0b2f06c051144d6bb2c2/vllm/model_executor/models/deepseek_v2.py#L1109-L1163)

Indexer cache 本身也是一个 `MLAAttentionSpec`，但 head size 与主 MLA cache 不同，并使用 `DeepseekV32IndexerBackend`。[Indexer cache spec/backend](https://github.com/vllm-project/vllm/blob/df3b3422b495a33fd10d0b2f06c051144d6bb2c2/vllm/model_executor/models/deepseek_v2.py#L633-L659)

这类“同类型但不同 hidden/page size”的层会被聚合成 `UniformTypeKVCacheSpecs`。配置生成器对“单一 `UniformTypeKVCacheSpecs` group”有更早的特殊分支：逐层生成独立 `KVCacheTensor`；它不会进入后面的 packed 分支。[单 group 异构 page-size 分配](https://github.com/vllm-project/vllm/blob/df3b3422b495a33fd10d0b2f06c051144d6bb2c2/vllm/v1/core/kv_cache_utils.py#L1405-L1424)

此外，统一 cross-layer 判定要求 spec 本身是普通 `AttentionSpec`，聚合 spec 无法通过；Indexer backend 自身也显式拒绝 layered stride。因此结论是：

```text
GLM-5.2 / DeepSeek-V3.2：
  主 MLA layer tensors     -> layer-first/general
  Indexer layer tensors    -> layer-first/general
```

配置 OffloadingConnector、NIXL 或 Mooncake 不会绕过这两个布局门槛。

### 1.6 第二种 page-first：packed per-block slab

最新 vLLM 还有一条独立于 single-group uniform tensor 的 packed layout：

```text
Block 0 slab: [group 内 layer0 page][layer1 page]...
Block 1 slab: [group 内 layer0 page][layer1 page]...
```

不同 cache group 的 block ID 在调度上互斥，因此它们可以重叠复用同一 slab；同一 group 内各层保持不同 offset。[packed layout 规划](https://github.com/vllm-project/vllm/blob/df3b3422b495a33fd10d0b2f06c051144d6bb2c2/vllm/v1/core/kv_cache_utils.py#L1300-L1374)

启用条件是：

- DeepSeek-V4 风格的多 `UniformTypeKVCacheSpecs` groups：默认启用；
- 其他 multi-group：`enable_cross_layers_blocks=true` 且 group 数大于 1，实验 API；
- 单一 `UniformTypeKVCacheSpecs` group 会被前面的逐层分配分支截获，因此 GLM-5.2 不属于这里。

对应判断和注释见 [`_use_packed_kv_cache_config()`](https://github.com/vllm-project/vllm/blob/df3b3422b495a33fd10d0b2f06c051144d6bb2c2/vllm/v1/core/kv_cache_utils.py#L1324-L1343) 与 [allocation 分叉](https://github.com/vllm-project/vllm/blob/df3b3422b495a33fd10d0b2f06c051144d6bb2c2/vllm/v1/core/kv_cache_utils.py#L1405-L1430)。该实验 API 对应官方 [issue #42082](https://github.com/vllm-project/vllm/issues/42082)。

worker 只分配一次 backing，然后给各层构造带 `(offset, block_stride)` 的 view。[model runner packed backing](https://github.com/vllm-project/vllm/blob/df3b3422b495a33fd10d0b2f06c051144d6bb2c2/vllm/v1/worker/gpu_model_runner.py#L7457-L7522) NIXL 还能识别所有 view 共用同一 storage，把整个 backing 注册为一个 region，每个 logical block 只生成一个 descriptor。[NIXL packed registration](https://github.com/vllm-project/vllm/blob/df3b3422b495a33fd10d0b2f06c051144d6bb2c2/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py#L960-L1005) [packed 检测](https://github.com/vllm-project/vllm/blob/df3b3422b495a33fd10d0b2f06c051144d6bb2c2/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py#L1044-L1059)

## 2. SGLang

SGLang 的机制与 vLLM 不同：它新加入了显式 L1 page-major 布局开关，但默认关闭；HiCache 的 L2 page-first 不会自动改写 L1。

官方引入 L1 page-major 的 PR 是 [sgl-project/sglang#29533](https://github.com/sgl-project/sglang/pull/29533)。

### 2.1 默认普通 MHA/GQA 是 per-layer

默认 `MHATokenToKVPool` 为每层分别创建 K tensor 和 V tensor。层内可以是 NHD、HND 或 ROCm vectorized-5D，但外层仍是 Python tensor list，而不是一个跨层 block slab。[默认 MHA allocation](https://github.com/sgl-project/sglang/blob/e0cf75d9bda78fa46bdc5ff7436c5a76290663d1/python/sglang/srt/mem_cache/memory_pool.py#L2049-L2113)

```text
K = [layer0 tensor, layer1 tensor, ...]
V = [layer0 tensor, layer1 tensor, ...]
```

### 2.2 普通 MHA/SWA 的 opt-in L1 page-major

参数 `--enable-page-major-kv-layout` 默认是 `False`。帮助信息明确说明：page 是最外层，page 内按 layer 排布，并要求 Triton attention/linear-attention/Mamba backend。[参数定义](https://github.com/sgl-project/sglang/blob/e0cf75d9bda78fa46bdc5ff7436c5a76290663d1/python/sglang/srt/server_args.py#L951-L968)

开启后，普通 MHA-shaped pool 会把 `MHATokenToKVPool` 替换成 `PageMajorMHATokenToKVPool`；但后续的 DSV4、DSA、纯 MLA、Ascend/OOT 专用 pool 分支会绕过这个 MHA class。[pool 选择分支](https://github.com/sgl-project/sglang/blob/e0cf75d9bda78fa46bdc5ff7436c5a76290663d1/python/sglang/srt/mem_cache/kv_cache_configurator.py#L950-L1060)

其准确物理布局是：

```text
page bytes = [L0_K * page_size | L0_V * page_size |
              L1_K * page_size | L1_V * page_size | ...]
```

即近似：

```text
[page][layer][K/V][token-in-page][head][dim]
```

所有 page 共用一个连续 `_raw` buffer，每层计算拿到的是 4-D strided view，不需要逐层复制。[page-major layout builder](https://github.com/sgl-project/sglang/blob/e0cf75d9bda78fa46bdc5ff7436c5a76290663d1/python/sglang/srt/mem_cache/layout/page_major.py#L1-L15) [view shape/stride](https://github.com/sgl-project/sglang/blob/e0cf75d9bda78fa46bdc5ff7436c5a76290663d1/python/sglang/srt/mem_cache/layout/page_major.py#L39-L108) [`PageMajorMHATokenToKVPool`](https://github.com/sgl-project/sglang/blob/e0cf75d9bda78fa46bdc5ff7436c5a76290663d1/python/sglang/srt/mem_cache/memory_pool.py#L3139-L3230)

当前 backend gate 很严格：standalone page-major MHA 只允许 Triton；unified-memory MLA 是特殊例外，可额外使用一组已做 dense-id 重映射的 MLA backend。[backend 限制](https://github.com/sgl-project/sglang/blob/e0cf75d9bda78fa46bdc5ff7436c5a76290663d1/python/sglang/srt/server_args.py#L8449-L8487)

另外，FP4 普通 MHA 明确不支持 page-major。[FP4 gate](https://github.com/sgl-project/sglang/blob/e0cf75d9bda78fa46bdc5ff7436c5a76290663d1/python/sglang/srt/mem_cache/kv_cache_configurator.py#L1047-L1059)

### 2.3 standalone PageMajorMHA 目前不适合作为 HiCache/offload 布局

`PageMajorMHATokenToKVPool` 当前明确把以下接口设为 `NotImplementedError`：

- `get_contiguous_buf_infos()`：普通 KV transfer / PD disaggregation registration 不支持；
- `get_cpu_copy()` / `load_cpu_copy()`：CPU offload 不支持；
- prefix-valid commit 不支持。

源码还说明 tiled KV copy kernel 假设 per-layer contiguous 3-D tensor，因此 page-major 只能走 native move。[限制说明与接口](https://github.com/sgl-project/sglang/blob/e0cf75d9bda78fa46bdc5ff7436c5a76290663d1/python/sglang/srt/mem_cache/memory_pool.py#L3139-L3151) [NotImplemented 路径](https://github.com/sgl-project/sglang/blob/e0cf75d9bda78fa46bdc5ff7436c5a76290663d1/python/sglang/srt/mem_cache/memory_pool.py#L3262-L3289)

因此它目前更像一个**计算侧 L1 布局实验**，不是 HiCache 打开后自动采用的 offload-friendly L1 layout。

### 2.4 纯 MLA、DSA、GLM-5.2 仍是 layer-first

`MLATokenToKVPool` 仍用 list comprehension 为每层分别创建：

```text
[size + page_size, 1, kv_cache_dim]
```

[纯 MLA allocation](https://github.com/sgl-project/sglang/blob/e0cf75d9bda78fa46bdc5ff7436c5a76290663d1/python/sglang/srt/mem_cache/memory_pool.py#L3936-L4005)

DSA 继承 MLA pool 并追加 indexer cache；configurator 的 DSA 与纯 MLA 分支都在 MHA page-major class 使用之前返回自己的 pool，因此普通 `--enable-page-major-kv-layout` 不会把它们改成跨层布局。[DSA/MLA 分支](https://github.com/sgl-project/sglang/blob/e0cf75d9bda78fa46bdc5ff7436c5a76290663d1/python/sglang/srt/mem_cache/kv_cache_configurator.py#L1016-L1029)

SGLang 的 GLM-5.2 对应 `GlmMoeDsaForCausalLM`，因此走 DSA pool。[GLM DSA 模型类](https://github.com/sgl-project/sglang/blob/e0cf75d9bda78fa46bdc5ff7436c5a76290663d1/python/sglang/srt/models/glm4_moe.py#L1447-L1453)

### 2.5 Hybrid/mixed cache 的两种 page-major 程度

第一种只开 `--enable-page-major-kv-layout`：

- Full+SWA：full pool 和 SWA pool 分别使用 PageMajorMHA，因此各自内部是跨层 page-first，但两种 cache type 仍是两个物理 pool；
- MHA+Mamba/GDN：full-attention pool 使用 PageMajorMHA，Mamba conv/temporal state 使用自己的 per-slot envelope；
- 纯 MLA/DSA 和 DSV4 不受这一 MHA class 替换影响。

第二种是 `--enable-unified-memory`：它会隐含开启 page-major，并让 hybrid Mamba 或 hybrid SWA 的两个 sub-pool 共享一块原始 byte buffer；当前只支持这两类 hybrid，明确排除 DeepSeek-V4。[unified-memory 支持范围](https://github.com/sgl-project/sglang/blob/e0cf75d9bda78fa46bdc5ff7436c5a76290663d1/python/sglang/srt/mem_cache/kv_cache_configurator.py#L386-L426)

对于 MLA-hybrid-Mamba，unified pool 的物理 page envelope 是：

```text
[L0_latent * page_size | L1_latent * page_size | ...]
```

但通过 layer offset 和 dense-id 重编号，每层可继续看到 contiguous dense view，供 MLA kernel 使用。[MLA page-major dense views](https://github.com/sgl-project/sglang/blob/e0cf75d9bda78fa46bdc5ff7436c5a76290663d1/python/sglang/srt/mem_cache/layout/page_major.py#L116-L176) 这条能力的官方 PR 是 [sgl-project/sglang#32971](https://github.com/sgl-project/sglang/pull/32971)。

unified-memory 仍有明显限制：HiCache/LMCache 不兼容、DCP 不兼容，PD 只有 Mooncake+PP1 等窄条件，speculative decoding 只支持受限 DSPARK。[运行限制](https://github.com/sgl-project/sglang/blob/e0cf75d9bda78fa46bdc5ff7436c5a76290663d1/python/sglang/srt/server_args.py#L8371-L8447)

DeepSeek-V4 的 SGLang 专用 device pool 仍为每层分别创建 `kv_buffer`，且 unified-memory 的 model-family gate明确排除 DSV4。[DSV4 per-layer allocation](https://github.com/sgl-project/sglang/blob/e0cf75d9bda78fa46bdc5ff7436c5a76290663d1/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L59-L104)

### 2.6 HiCache 的 `page_first` 是 L2，不是 L1

`--hicache-mem-layout` 的帮助文本明确写的是“host memory pool”，其默认值为 `page_first`。[HiCache 参数](https://github.com/sgl-project/sglang/blob/e0cf75d9bda78fa46bdc5ff7436c5a76290663d1/python/sglang/srt/server_args.py#L2720-L2741)

官方 HiCache 设计文档也明确说明：L2/L3 为减少对象数可采用 page-first，而 GPU 因逐层计算“inherently operates in a layer first layout”，L1↔L2 之间由转换/传输 kernel 处理。[HiCache layout 设计](https://github.com/sgl-project/sglang/blob/e0cf75d9bda78fa46bdc5ff7436c5a76290663d1/docs/docs/advanced_features/hicache_design.mdx#L92-L107)

所以正常的 SGLang HiCache 配置通常是：

```text
L1 GPU/HBM：默认 layer-first
L2 CPU：     page-first / page_first_direct / 其他 host layout
L3：         page object
```

这与“显式开启 L1 `--enable-page-major-kv-layout`”是两套不同能力。

## 3. 与内部 vLLM MLU v0.25.1 的直接对照

共同点：

- 逐层计算不妨碍底层跨层 page-first，关键是 backend 是否能使用 strided view/block stride；
- 对普通 attention，二者都已经存在可工作的跨层 page-first 路径；
- 对 GLM-5.2/DSA/indexer，两边最新上游仍没有把整个 L1 变成跨层 page-first。

区别：

1. **vLLM 是 connector-driven。** single-group layout 由 connector 偏好与 backend stride 能力共同触发；OffloadingConnector 默认倾向 cross-layer。
2. **SGLang 是显式 L1 实验开关。** HiCache 的 host `page_first` 不会自动改变 L1，且 standalone PageMajorMHA 暂时不能走 CPU offload/普通 PD 注册。
3. **vLLM 的 MLA 覆盖更广。** 多个 CUDA MLA backend 已正式声明 cross-layer stride；但 DSV3.2 indexer/GLM-5.2 仍失败。
4. **两边对 DSV4 选择相反。** vLLM 为 DSV4 默认 packed block slab；SGLang DSV4 当前仍使用专用 per-layer device pools。
5. **SGLang 的 unified-memory 目标不只是 offload。** 它允许两个 hybrid sub-pool 动态共享一块 buffer，但支持矩阵和部署限制仍很窄。

## 4. 对 xLLM/MLU 的启示

针对 GLM-5.2，不应把“最新上游已经普遍改成 L1 page-first”当作前提。两边现实上都保留 layer-first，因此 xLLM 当前的 TMO/2D/batch 多层传输优化仍有价值。

若后续考虑在 xLLM 引入 L1 page-first，建议把问题拆成三项分别验证：

1. GLM-5.2 主 MLA kernel 是否读取真实 block stride；
2. DSA indexer cache kernel 是否也读取真实 block stride；
3. layer-wise H2D、prefix cache block move、spec decode、PD/offload 注册是否都能接受 strided layer view。

只验证主 MLA attention 不够。vLLM 与 SGLang 当前都在 indexer、offload/transfer、混合 cache group 的边界上保留 fallback 或直接拒绝，说明真正困难的是整条 cache 生命周期，而不是单个 attention kernel 的逐层计算。
