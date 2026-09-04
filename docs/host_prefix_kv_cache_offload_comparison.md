<!--
Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0.
-->

# Host Prefix KV Cache Offload：vllm-ascend 与 xLLM 源码对比

## 1. 结论先行

本文对比的源码快照是：

- `vllm-ascend`: `a0cc4ce1d`
- vllm-ascend 验证的 upstream vLLM: `fe784ff22e630a31fd798f392b01e0a75c18f047`（`../vllm-ascend/.github/vllm-main-verified.commit:1`）
- `xllm`: `459658f1`

两边实现的共同目标都是：NPU/HBM 中的 prefix KV block 仍是一级缓存，主机内存保存相同 block 的副本，NPU miss 而 host hit 时将 block 恢复到新的 NPU block，从而避免重算。但它们不是同一套架构：

| 维度 | vllm-ascend 主线 | xLLM 当前主线 |
|---|---|---|
| 当前入口 | 上游 vLLM `SimpleCPUOffloadConnector`，Ascend 替换 worker/DMA | xLLM 自有 `HierarchyBlockManagerPool` + `HierarchyKVCacheTransfer` |
| 调度/索引所有权 | 复用上游 vLLM `BlockPool`、KV cache coordinator、scheduler | device/host 各有一套 block manager 和 prefix-cache 索引；`Sequence` 同时持有 `kv_state` 与 `host_kv_state` |
| D2H 触发 | 上游 scheduler 对“confirmed computed blocks”安排 store，支持 eager/lazy 策略 | 序列 deallocate 时复制完整、唯一持有的 computed prefix block；不是 active-sequence swap |
| H2D | 独立 NPU load stream，event 轮询/等待 | scheduler 建立 H2D transfer，2 个后台线程按 layer batch 搬运，forward 到相应 layer batch 时等待 |
| 低层搬运 | `aclrtMemcpyBatchAsync`，不支持时退化为多个 `aclrtMemcpyAsync` | `aclrtMemcpyBatch`，每个 layer batch 最后显式 `stream->synchronize()` |
| Host 内存 | `torch.zeros(..., pin_memory=True)`；不可 pin 时允许 pageable fallback | `mmap` + `mlock` + `aclrtHostRegister`；失败直接 fatal |
| Host 淘汰 | 上游 BlockPool/LRU | xLLM 独立 host `PrefixCache` LRU |
| 当前模型范围 | 由上游 HMA/connector 与 Ascend cache flattening 共同决定 | scheduler 数据路径仅支持普通 `BlockType::KV`；DeepSeek-V4 显式拒绝 |

因此，xLLM 不是对 vllm-ascend NPU worker 的简单 C++ 重写。它保留了更显式的“device block pool + host block pool”层次结构，并实现了 layer-wise restore pipeline；代价是状态机和错误恢复明显更复杂。

## 2. 术语和范围

### 2.1 本文所称 Host Prefix KV Cache Offload

本文只讨论“prefix cache 的主机二级副本”：

1. KV 先在 NPU 上计算；
2. 完整 block 以相同 prefix hash 在 host 保存副本；
3. 后续请求先命中 NPU prefix cache；NPU 前缀较短而 host 前缀更长时，为 host-only 部分分配 NPU block 并 H2D；
4. host 容量不足时淘汰旧 prefix block。

它不等于：

- weight CPU offload；
- P/D 实例间 KV transfer；
- Mooncake/LMCache/远端 KV store；
- 为活跃请求临时腾挪 HBM 的通用 swap。

xLLM 的 host path 与远端 store 共用过历史类名和 transfer type，但当前 `WorkerImpl` 创建 hierarchy transfer 时强制 `.enable_kvcache_store(false)`，所以本文分析的是本机 DRAM 二级缓存，而不是 Mooncake store（`xllm/core/runtime/worker_impl.cpp:1673-1696`）。

### 2.2 vllm-ascend 有新旧两条 CPU offload 路径

当前应以 `SimpleCPUOffloadConnector` 为主：

- connector 注册时，如上游模块存在，就用 `AscendSimpleCPUOffloadConnector` 覆盖同名 upstream connector（`../vllm-ascend/vllm_ascend/distributed/kv_transfer/__init__.py:67-81`）；
- Ascend connector 明确声明 scheduler 侧完全复用上游，只替换 worker handler（`../vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/simple_cpu_offload/simple_cpu_offload_connector.py:1-6,24-39`）；
- 当前 e2e 使用 `SimpleCPUOffloadConnector`、`cpu_bytes_to_use` 和 `lazy_offload`（`../vllm-ascend/tests/e2e/pull_request/one_card/test_simple_cpu_offload.py:27-38`）。

仓库现有 `kv_cache_cpu_offload.md` 仍描述旧 `OffloadingConnector + NPUOffloadingSpec`（`../vllm-ascend/docs/source/user_guide/feature_guide/kv_cache_cpu_offload.md:5-13,23-31,49-71`）。对应旧 e2e 已标成 `skip(reason="cpu offload connector is deprecated.")`（`../vllm-ascend/tests/e2e/pull_request/one_card/test_cpu_offloading.py:131-147`）。所以旧指南不能当作主线配置手册。

Git history 也支持这一判断：`a4179063b`（2026-06-05，`[Feature] Simple yet General CPU KV Cache Offloading`）引入新路径，提交说明强调它直接复用 vLLM BlockPool/KVCacheCoordinator；旧 `NPUOffloadingSpec` 路径始于 `bc30874f8`。本文只把旧路径作为演进背景，不用它代表当前设计。

## 3. vllm-ascend：完整数据流

### 3.1 配置与初始化

推荐配置形态由当前 e2e 锁定：

```python
KVTransferConfig(
    kv_connector="SimpleCPUOffloadConnector",
    kv_role="kv_both",
    kv_connector_extra_config={
        "cpu_bytes_to_use": 1 << 30,
        "lazy_offload": False,
    },
)
```

同时必须启用 prefix caching；e2e 显式传入 `enable_prefix_caching=True`（`../vllm-ascend/tests/e2e/pull_request/one_card/test_simple_cpu_offload.py:49-55,87-95`）。

固定 upstream 版本解析的完整含义是：`cpu_bytes_to_use` 是全实例预算（默认 8 GiB），通常除以 `world_size` 得 per-rank 容量；`cpu_bytes_to_use_per_rank` 可直接覆盖；`lazy_offload` 默认 `False`（`vllm@fe784ff:vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py:41-116`）。

初始化链路是：

1. 插件注册阶段将 connector 名称覆盖到 Ascend 实现（`../vllm-ascend/vllm_ascend/distributed/kv_transfer/__init__.py:67-81`）。
2. `AscendSimpleCPUOffloadConnector.__init__` 先执行 upstream 初始化，从上游 handler 读取已经解析好的 per-rank `cpu_capacity_bytes`，再换成 `SimpleCPUOffloadNPUWorker`（`../vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/simple_cpu_offload/simple_cpu_offload_connector.py:42-60`）。
3. 若 prefix caching 未启用，上游会令两个 handler 都为 `None`，Ascend connector 也保持 no-op（同文件 `:50-56`）。
4. `SimpleCPUOffloadNPUWorker` 继承上游 step-time 状态，只把 copy backend 改成 `NPUDmaCopyBackend`（`../vllm-ascend/vllm_ascend/simple_kv_offload/worker.py:1-22,55-73`）。

这里有一个源码边界：scheduler、CPU BlockPool、LRU、命中元数据和 preemption 逻辑属于固定版本的 upstream vLLM，不在 vllm-ascend plugin tree 内。Ascend 类明确列出 `bind_connector_metadata`、`get_finished`、`handle_preemptions` 等均继承上游（connector 文件 `:27-32`；worker 文件 `:18-22`）。下文以 `.github/vllm-main-verified.commit` 固定的 `fe784ff` 为控制面审计基线，upstream 引用统一写成 `vllm@fe784ff:<relative-path>:<lines>`。

### 3.2 Host cache 分配和布局

`register_kv_caches` 做如下工作：

1. 从首个 cache tensor 取得 NPU device，并从 `KVCacheConfig` 取得 device block 数（`../vllm-ascend/vllm_ascend/simple_kv_offload/worker.py:75-95`）。
2. 遍历每层所有 constituent tensor。Ascend 的 K/V 是独立 allocation；代码按 storage pointer 去重，避免别名 allocation 被重复复制，同时不会漏掉 V（同文件 `:96-113`）。
3. 将每个 allocation 变成 `[num_blocks, bytes_per_block]` 的 `int8` block view（同文件 `:160-196`）。对于 `(N, num_blocks, ...)` 的 stacked layout，拆成 N 个 keyed segment（同文件 `:198-223`）。
4. 每个 CPU block 的总字节数是所有唯一 tensor 的 `stride(0) * element_size()` 之和；`num_cpu_blocks = max(1, cpu_capacity_bytes // total_bytes_per_block)`（同文件 `:114-122`）。
5. 为每个 NPU block view 创建相同 per-block shape 的 CPU tensor；优先 pinned memory，但 pin 不可用时只警告并继续（同文件 `:124-137`）。
6. 创建独立 load/store NPU stream；由于 `torch.npu.Stream` 没有 priority API，这两条流没有 CUDA 实现中的低优先级提示（同文件 `:139-158`）。

Ascend runner 的 KV allocation 可能有 2 MiB 对齐导致的 storage 前后 padding，因此 block view 必须根据 tensor shape/stride 截取，不能按 `storage.nbytes()` 平分；源码对此有专门保护（同文件 `:168-183`）。

### 3.3 D2H：生成 host 副本

控制面由 upstream `SimpleCPUOffloadScheduler` 决定 store 哪些 block：

- eager（默认）每个 scheduling step 用 `num_computed_tokens - output_placeholders` 得到 confirmed tokens，只处理 block-aligned 的完整 blocks，并跳过 null、无 hash、已在传输或 CPU 已缓存的 block（`vllm@fe784ff:vllm/v1/simple_kv_offload/manager.py:524-630`）；
- lazy 增量扫描 device `free_block_queue` 靠近淘汰端的 hashed blocks，直到 `_target_free` 或 CPU capacity 上限，CPU 已缓存 hash 不重复 store（同文件 `:469-522`）；`_target_free` 根据每 step token、Mamba 和 sliding-window 需求估算（同文件 `:218-236`）；
- 两种策略都会分配 CPU blocks、stamp device hash 并 touch device blocks，防止 DMA 期间复用（同文件 `:634-653`）；`build_connector_meta()` 分配单调 event id 并生成 device→CPU block mapping（同文件 `:406-425,445-458`）。

Ascend 数据面的重要约束是：

- scheduler 只 store `confirmed_tokens` 对应的已计算 block，防止未计算 block 进入 store；
- 当前固定 upstream worker 还会在 compute stream 上创建并记录 compute-done event，D2H store stream 显式等待该 event；
- load 目标 NPU block 会被 upstream `BlockPool.touch` 持有到完成。

`confirmed_tokens` 和 block lifetime 契约记录在 `NPUDmaCopyBackend._copy_loop`（`../vllm-ascend/vllm_ascend/simple_kv_offload/copy_backend.py:92-100`）；compute-done event 由 upstream worker 创建/传入（`vllm@fe784ff:vllm/v1/simple_kv_offload/worker.py:235-245`），backend 在 copy 前执行 `stream.wait_event(wait_event)`（`../vllm-ascend/vllm_ascend/simple_kv_offload/copy_backend.py:121-125`）。它们共同构成正确性契约。

具体 D2H：

1. upstream worker 调用 backend `launch_copy(src_blocks, dst_blocks, is_store=True, ...)`；
2. job 放入单个 FIFO `SimpleQueue`，由 daemon thread 处理（`../vllm-ascend/vllm_ascend/simple_kv_offload/copy_backend.py:44-78`）；
3. copy thread 设置当前 NPU device，选择 `store_stream`，必要时等待上游传入的 event，然后调用 `copy_blocks`，在同一 stream 记录完成 event（同文件 `:92-128`）；
4. `copy_blocks` 用广播一次生成“所有 tensor × 所有 block pair”的源地址、目标地址、长度数组（`../vllm-ascend/vllm_ascend/simple_kv_offload/npu_mem_ops.py:71-99`）；
5. C++ op 在当前 NPU stream 上调用 `aclrtMemcpyBatchAsync`；若编译环境无该 API，则逐 descriptor 调用 `aclrtMemcpyAsync`（`../vllm-ascend/csrc/torch_binding.cpp:124-168,208-243`）。

copy thread 不同步设备；主线程通过继承自 upstream worker 的 event polling/get-finished 状态机异步收割完成结果。backend 的 event list append 在 enqueue 之后立即可见，但 event 本身只有 query/synchronize 完成后才表示 DMA 完成（`../vllm-ascend/vllm_ascend/simple_kv_offload/copy_backend.py:121-128`）。

Upstream worker 用非阻塞 event query 和 stream high-water mark 收割 load/store；store completion 通过 `{event_id: 1}` worker metadata 上报（`vllm@fe784ff:vllm/v1/simple_kv_offload/worker.py:207-277,299-313`）。scheduler 汇总 TP/PP workers，只有 completion count 达到 `world_size` 才 publish CPU blocks；随后将其送入 LRU eviction queue并释放 device touch refs（`vllm@fe784ff:vllm/v1/simple_kv_offload/manager.py:668-745`）。request finish 不必同步等待 store，引用计数继续保护在途 block（同文件 `:764-796`）。

### 3.4 Host 命中与 H2D 恢复

上游 scheduler 先计算 device prefix 命中，再从尚未命中的 suffix 查 CPU BlockPool；host-only 命中部分会：

1. CPU coordinator 执行 `find_longest_cache_hit()`；命中后立即 touch CPU blocks，防止下一调度阶段前被 LRU 淘汰，并返回 async external-KV hit（`vllm@fe784ff:vllm/v1/simple_kv_offload/manager.py:243-279`）。
2. `update_state_after_alloc()` 接收为 external tokens 新分配的 NPU blocks，按 cache group block size 对齐两侧 block table，跳过 sliding-window/Mamba null blocks，同时 touch 两侧 blocks，建立 `LoadRequestState`（同文件 `:283-404`）。
3. 下一次 `build_connector_meta()` 合并尚未发出的 request loads，并分配 load event（同文件 `:426-458`）。
4. worker 用 `launch_copy(..., is_store=False)` 投递；backend 选择 `load_stream` 和预构建 H2D descriptors（`../vllm-ascend/vllm_ascend/simple_kv_offload/copy_backend.py:52-57,67-78,121-128`）。
5. scheduler 将其视为 async external KV；load completion request id 回传后才恢复调度。完成时清除 event→request mapping，并释放 CPU/NPU touch refs（`vllm@fe784ff:vllm/v1/simple_kv_offload/manager.py:668-679,814-846`）。

e2e 的验证方式是：冷启动先生成并触发 CPU store，等待 pending store 排空，再 `reset_prefix_cache()` 清掉 device prefix，重复相同 prompt，要求 CPU reload 后的输出与冷启动一致（`../vllm-ascend/tests/e2e/pull_request/one_card/test_simple_cpu_offload.py:41-77`）。另一个参数化测试覆盖 eager/lazy 两种策略的重复运行（同文件 `:80-103`）。

### 3.5 内存管理与淘汰

新路径本身不在 Ascend 仓重复实现 LRU。scheduler 根据 CPU/NPU capacity ratio 构造第二套 CPU `KVCacheCoordinator + BlockPool`（`vllm@fe784ff:vllm/v1/simple_kv_offload/manager.py:70-134,180-216`）。CPU BlockPool 的 free queue 同时承担 eviction order；`get_new_blocks()` 从队头复用并删除旧 hash，`touch()` 将命中或 transfer 中的 block 移出 eviction queue，`free_blocks()` 将 hashed block 追加到队尾（`vllm@fe784ff:vllm/v1/core/block_pool.py:143-149,178-184,647-740`）。Ascend 只按字节预算创建物理 CPU mirrors（worker 文件 `:114-137`）。

Preemption 时 metadata 设置 `need_flush`，worker 同步在途 load/store，避免目标 device block 被抢占复用（`vllm@fe784ff:vllm/v1/simple_kv_offload/manager.py:445-457`；`vllm@fe784ff:vllm/v1/simple_kv_offload/worker.py:279-297`）。reset 会把在途 store 变成 abandoned、保留已提交 load 的引用直到完成，并在仍有 transfer 时返回 false（`vllm@fe784ff:vllm/v1/simple_kv_offload/manager.py:868-906`）。

与旧路径不同，旧 `NPUOffloadingSpec` 显式创建 `CPUOffloadingManager` 和固定 `num_cpu_blocks`（`../vllm-ascend/vllm_ascend/kv_offload/npu.py:16-43`）；旧 handler 自己创建 D2H/H2D stream 和 event queue（`../vllm-ascend/vllm_ascend/kv_offload/cpu_npu.py:54-75`）。这正是旧指南中的架构，不应与新 Simple path 混用。

## 4. xLLM：完整数据流

### 4.1 配置与启用条件

主配置入口是 `KVCacheStoreConfig`：

- `host_blocks_factor`，默认 `0.0`；
- `layers_wise_copy_batchs`，默认 `4`；
- `enable_control_h2d_block_num`，默认 `false`。

三者均可从 CLI flag 或 JSON 读取（`xllm/core/framework/config/kv_cache_store_config.cpp:21-36,54-83`；option category 位于 `xllm/core/framework/config/kv_cache_store_config.h:41-75`）。

Host Prefix KV Cache Offload 的有效启用条件是：

```text
enable_prefix_cache = true
host_blocks_factor > 1.0
平台 = NPU
模型有普通 BlockType::KV cache
```

证据如下：

- master 将 `host_blocks_factor` 传入 runtime options（`xllm/xllm.cpp:227-233`）；
- engine 按 `host_num_blocks = npu_blocks * factor` 建逻辑 host pool，并在 factor > 1 时选用 `HierarchyBlockManagerPool`（`xllm/core/distributed_runtime/llm_engine.cpp:541-558,608-624`）；
- worker 也只在 factor > 1 时创建 `HierarchyKVCacheTransfer`（`xllm/core/runtime/worker_impl.cpp:1673-1697`）；
- `create_batch_memcpy` 只有 `USE_NPU` 返回 provider，其他平台返回 `nullptr`（`xllm/core/platform/batch_memcpy.cpp:20-34`）。

当前代码没有在配置入口同时强制 `enable_prefix_cache=true` 和 NPU provider，所以“有效条件”比“语法上可设置”更严格，见风险章节。

### 4.2 初始化：两套逻辑 block pool + 一套 host 物理 KVCache

engine 侧：

1. device pool 使用普通 `BlockManagerPool` 的 composite leaves；
2. factor > 1 时外包一层 `HierarchyBlockManagerPool`；
3. hierarchy pool 为每个 DP rank 创建独立 host `BlockManagerImpl`，容量为 `host_num_blocks`，并启用相同 block size、prefix cache 和 hasher（`xllm/core/framework/block/hierarchy_block_manager_pool.cpp:26-55`）；
4. 因 D2H callback 和 scheduler 会跨线程访问 host/device leaves，相关 manager 被 `ConcurrentBlockManagerImpl` 包装（同文件 `:41-51`；device composite 的包装条件见 `xllm/core/framework/block/block_manager_pool.cpp:67-94`）。

worker 侧：

1. device KVCache 正常分配完成后，立刻初始化 hierarchy transfer（`xllm/core/runtime/worker_impl.cpp:390-415`）；
2. transfer 建 2 个 H2D load thread，并准备 `2 + 4` 条 copy stream，额外 4 条对应 RemoteWorker D2H copy thread 并发（`xllm/core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.cpp:132-169`）；
3. 扫描每层 device KVCache，按 `BlockType` 和 tensor role 建映射（同文件 `:171-194`）；
4. 每个 block type 创建一份 host `KVCache`，其 shape 是 `[host_blocks, layer_count, ...per_block_dims]`（`xllm/core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.h:51-56`；创建见 `.cpp:196-221`）。

Host tensor 不是普通 `torch::empty`。每个 tensor 都：

1. 按 OS page size 对齐；
2. `mmap`；
3. `mlock`；
4. NPU 构建下 `aclrtHostRegister`；
5. 用 `torch::from_blob` 包装。

实现位于 `xllm/core/framework/kv_cache/kv_cache_utils.cpp:369-403,415-447,476-493`。host block 数按 `max(factor, 1.0)` 缩放，layer 维插在 dim 1（同文件 `:449-473`）。普通 KV 的 K/V 分别创建 host tensor（`xllm/core/framework/kv_cache/kv_cache_impl.cpp:54-78`）；indexed KV 还会镜像 index 和 INT8 index scale（`xllm/core/framework/kv_cache/indexed_kv_cache_impl.cpp:68-123`）。

### 4.3 D2H：序列释放时生成二级副本

xLLM 的 D2H 不是在 device LRU 淘汰回调中启动，而是在 sequence deallocate 时启动：

1. `HierarchyBlockManagerPool::deallocate` 先把 device blocks publish 到 device prefix cache（`xllm/core/framework/block/hierarchy_block_manager_pool.cpp:58-68`）。
2. `collect_offload_pairs` 只处理启用 prefix cache 的普通 `BlockType::KV`（同文件 `:88-100`）。
3. 它只为“device 已计算、host 尚无副本”的范围分配 host block；若本轮所需 host blocks 无法一次全部分配，整轮跳过，不做 partial offload（同文件 `:102-140`）。
4. 只复制 `ref_count == 2` 的 device block：此时引用恰好来自 sequence 和 device prefix-cache node；beam/fork 共享 block 被跳过（同文件 `:141-152`）。
5. device/host `Block` 被 move 到 per-DP 并发 offload queue。sequence 随后释放其余 host blocks 和 device composite blocks，但排队中的 ids 保持占用直至复制完成（同文件 `:58-85`）。
6. scheduler 每 tick 在 batch 建好后调用 `transfer_blocks(batches)`；没有 batch 时也会 drain D2H queue（`xllm/core/scheduler/continuous_scheduler.cpp:234-248`）。
7. queue 被组装成 `D2H2G` transfer infos，再广播到该 DP rank 的每个 TP worker（`xllm/core/framework/block/hierarchy_block_manager_pool.cpp:431-457`；`xllm/core/distributed_runtime/llm_engine.cpp:693-704`）。
8. 所有 TP worker 返回成功数量后，callback 才把 host blocks publish 到 host prefix cache。失败时无论如何释放两侧 reserved id，但绝不 publish 未完成的 host data（`xllm/core/framework/block/hierarchy_block_manager_pool.cpp:449-487`）。

虽然 transfer type 名为 `D2H2G`，当前本机 offload 的 worker path 在 `offload()` 中只执行 D2H；`enable_kvcache_store` 被强制关闭，不会继续到 global store（`xllm/core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.cpp:324-327,376-419`；`xllm/core/runtime/worker_impl.cpp:1687-1690`）。

### 4.4 Host 命中

新 sequence 的 device allocation 先执行 device prefix match（`xllm/core/framework/block/block_manager_pool.cpp:175-205`）。`HierarchyBlockManagerPool` 随后对同一 token prefix 调用 host manager 的 `allocate_shared`，把命中的 host blocks 挂到 `Sequence::host_kv_state`（`xllm/core/framework/block/hierarchy_block_manager_pool.cpp:156-166,246-255,313-327`）。

Host prefix index 的行为是：

- chained block hash，连续查找，遇到首个 miss 停止；
- hit 会刷新 LRU；
- 返回的 `Block` 引用会 pin 住物理 host block，防止正在使用时被淘汰。

实现见 `xllm/core/framework/prefix_cache/prefix_cache.cpp:29-103`。`Sequence` 对“已缓存 token 数”的公开视图取 device 和 host 的最大值（`xllm/core/framework/request/sequence.h:165-179`），因此 scheduler 可以把 host-only prefix 视为无需重算，但必须在 forward 前完成 H2D。

### 4.5 H2D 恢复

1. `UnifiedPolicy` 计算 `host_blocks_num - device_blocks_num`，并可按 latency model 限制本 tick 的 copy block budget（`xllm/core/scheduler/unified_policy.cpp:94-99,144-161,236-243`）。预算模型入口见同文件 `:430-475`。
2. `HierarchyBlockManagerPool::allocate` 为 host-only block 分配 device blocks，并生成 `BlockTransferInfo(host_id, device_id, hash, H2D)`（`xllm/core/framework/block/hierarchy_block_manager_pool.cpp:156-203`）。
3. chunked prefill 下，恢复范围会被 clamp 到本 tick `num_tokens`、已分配 device/host block 数和当前 device capacity，避免 KV token cursor 超过 scheduler budget（同文件 `:205-241`；无显式 copy budget 的 overload 在 `:246-310`）。
4. batch 建好后，pool 为含 H2D 的 batch 设置 batch id，并把 transfer 发到所有 TP worker（同文件 `:416-428`；engine 广播见 `xllm/core/distributed_runtime/llm_engine.cpp:707-714`）。
5. worker 在返回 RPC 前同步创建 `LayerSynchronizer` 并按 batch id 注册，再把真正 H2D 投递给 2-thread load pool；这样 forward 查 synchronizer 不会和异步线程注册动作竞态（`xllm/core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.cpp:309-359`）。
6. copy plan 将每个 host block 的 `[host_id][layer_slot]` tensor slice 对应到 device `[device_id]` slice，覆盖该 block type 的每个 tensor role（同文件 `:224-306`）。
7. H2D 按 layer ranges 搬运；每个 range 完成后在 copy stream 记录 event。失败时 `abort()`，令尚未满足的 forward wait 返回 false，而不是死等或读取脏 KV（同文件 `:421-465`）。
8. `WorkerImpl::step_async` 在模型执行线程把 batch-id synchronizer 注入 `ModelInputParams`（`xllm/core/runtime/worker_impl.cpp:1011-1029`；注入实现见 hierarchy transfer `:468-480`）。
9. 各 NPU model 在逐 layer forward 前调用 `input_params.synchronize_layer(i)`；公共方法只在 layer batch 边界等待（`xllm/core/framework/model/model_input_params.h:1009-1018`，例如 `xllm/models/llm/npu/llm_model_base.h:235-246`）。

### 4.6 搬运与同步的真实语义

xLLM 的 H2D 是“对 scheduler/forward 异步”，但当前底层不是纯异步 DMA API：

- `NPUBatchMemcpy` 使用 `aclrtMemcpyBatch`，descriptor 超过 4096 时分块提交；
- 每个 layer range 提交完后调用 `stream->synchronize()`。

见 `xllm/core/platform/npu/npu_batch_memcpy.cpp:58-123` 和最大 descriptor 常量 `xllm/core/platform/npu/npu_batch_memcpy.h:47-60`。

因此 pipeline 形态是：

```text
load thread: copy+sync layers 0..k -> record event -> copy+sync layers k..2k -> ...
forward:     wait range 0 -> compute range 0 -> wait range 1 -> compute range 1 -> ...
```

它可以让“后续 layer range 的 H2D”与“前面 layer 的 compute”重叠，但每个 range 内 copy thread 会阻塞等待 DMA 完成。D2H 也在 RemoteWorker copy thread 内同步完成，只是通过 folly future 与 scheduler 解耦（`xllm/core/distributed_runtime/remote_worker.cpp:278-299`）。

NPU synchronizer 的 waiter 先 busy-spin 等待 event-record flag，再在独立 wait stream 上 `aclrtStreamWaitEvent` 并同步该 stream；abort 会释放所有 spin waiter（`xllm/core/platform/npu/npu_layer_synchronizer.cpp:63-112`）。adapter 负责把 event 记录到实际 copy stream（`xllm/core/platform/layer_synchronizer.cpp:26-60`）。

### 4.7 Host LRU 淘汰

Host manager 使用与普通 prefix cache 相同的 LRU：

- hit 将节点移到 LRU 尾部（最近使用）（`xllm/core/framework/prefix_cache/prefix_cache.cpp:50-60,90-94`）；
- host block allocation 不足时，`BlockManagerImpl::has_enough_blocks` 请求 prefix cache 淘汰所需数量（`xllm/core/framework/block/block_manager_impl.cpp:131-158`）；
- `PrefixCache::evict` 从最旧端扫描，只淘汰没有外部共享引用的 block（`xllm/core/framework/prefix_cache/prefix_cache.cpp:243-275`）；
- `Block` 最后一个引用析构时把 physical id 返还 owning manager（`xllm/core/framework/block/block.cpp:101-123`）。

这意味着 xLLM 的 host eviction 与 device eviction 是两套独立 LRU；D2H 创建副本不会立即删除 device prefix block。device block 仍由 device LRU 独立淘汰，所以它确实是二级缓存，而不是 move-only swap。

## 5. 主要实现差异

### 5.1 控制面：复用 vs 双状态

vllm-ascend 新路径将控制面留在 upstream vLLM，Ascend 代码只有 connector replacement、cache layout adaptation 和 DMA backend（connector `:24-39`；worker `:1-22`）。其设计目标就是不维护平行 block stack。

xLLM 明确维护：

- device `BlockManagerPool`；
- `host_block_managers_`；
- per-sequence `kv_state`；
- per-sequence `host_kv_state`；
- H2D metadata queue；
- D2H offload pair queue。

核心字段见 `xllm/core/framework/block/hierarchy_block_manager_pool.h:40-82` 和 `xllm/core/framework/request/sequence.h:165-179,355,519`。它更容易做 xLLM 自有的 layer-wise pipeline，但需要手工维持两个 token cursor、两套 block refcount 与异步生命周期的一致性。

### 5.2 Offload 触发时机

vllm-ascend/上游 Simple path 以 scheduler 已确认 block 为单位安排 store，且暴露 `lazy_offload` 策略；安全性依赖 `confirmed_tokens` 契约（copy backend `:92-100`）。

xLLM 只在 sequence deallocate 时收集完整 block（hierarchy pool `:58-85,88-154`）。所以当前 xLLM：

- 适合“请求结束后保留可复用 prompt”；
- 不会主动把仍在运行的长请求 KV 搬到 host 以扩展 active context；
- D2H 与 device LRU eviction 不是同一个动作。

### 5.3 物理内存

vllm-ascend 使用 PyTorch pinned allocator，pin 不可用时降级；容量以 bytes 配置并根据真实 per-block bytes 换算（worker `:114-137`）。

xLLM 用 factor 配置逻辑/物理 block 数，并强制 page-lock + NPU register；任何 `mlock` 或 register 失败都会终止（KV utils `:369-403`）。这给 DMA 提供稳定 registered memory，但部署对 `RLIMIT_MEMLOCK` 更敏感。

### 5.4 DMA 与计算重叠

vllm-ascend 调用真正 async 的 CANN API并通过 event query 收割；load/store 使用不同 NPU stream（binding `:165-243`；worker `:139-158`）。

xLLM 在后台线程里用同步 `aclrtMemcpyBatch`，显式同步每个 layer range；它通过 layer pipeline 获得跨 range 的 copy/compute overlap，而不是通过一次完全异步 copy job 获得 overlap（NPU batch memcpy `:78-123`；hierarchy transfer `:421-465`）。

### 5.5 多 rank 成功语义

xLLM scheduler 同一个 logical block id 映射会广播到所有 TP worker，只有所有 future 都返回完整 block count 才 publish host prefix（hierarchy pool `:449-483`）。这是显式的 all-rank commit。

vllm-ascend connector 是 vLLM worker/connector 生命周期的一部分；该仓只负责 per-rank CPU capacity 和 NPU DMA，跨 rank 聚合逻辑由 upstream connector 持有（connector `:42-60`）。因此不能只看 Ascend plugin 推断其全局 commit 语义。

### 5.6 模型/cache layout

vllm-ascend worker按 storage 遍历任意 tuple/list constituent，处理 K/V 独立 allocation、别名去重、stacked segments 和对齐 padding（worker `:39-52,96-113,160-223`）。

xLLM worker transfer 能为 KV、SINGLE、SWA、C4、C128 构造 tensor-role map（hierarchy transfer `:58-127`），物理 host `KVCache` 也能创建这些形态；但 scheduler 的 host block manager 和 offload-pair collector 当前只走 `BlockType::KV`，engine 因而对 DeepSeek-V4 直接 fail loud（`xllm/core/distributed_runtime/llm_engine.cpp:608-618`）。即“物理 tensor 支持”领先于“端到端调度支持”。

## 6. 限制与风险

### 6.1 vllm-ascend

1. **文档滞后。** 当前 feature guide 仍指导用户启用已弃用的旧 connector，而新 e2e 已使用另一组配置（旧指南 `:23-71`；deprecated test `:131-147`；新 test `:27-38`）。
2. **对 upstream 内部 API 强耦合。** connector 继承并替换上游私有 handler 字段；注册还以 upstream module 是否存在作为条件（connector registration `:67-81`）。vLLM 版本漂移可能让 feature 静默不注册或字段契约失配。
3. **正确性依赖跨模块不变量和全局 wrapper。** `confirmed_tokens` 防止未计算 block store；D2H 同时显式等待 upstream compute-done event；H2D 依赖 `BlockPool.touch` 保证目标 block 生命周期（copy backend `:92-100,121-125`；`vllm@fe784ff:vllm/v1/simple_kv_offload/worker.py:235-245`）。upstream 仍使用 `torch.Event`/`torch.cuda.current_stream()`，Ascend 靠 `_torch_cuda_wrapper` 将 Event、Stream、current_stream 全局映射到 NPU，并在 `finally` 保留 NPU event/current-stream 入口（`../vllm-ascend/vllm_ascend/worker/model_runner_v1.py:4766-4812`）。这不是当前已确认 bug，但属于版本耦合和回归测试风险。
4. **NPU transfer stream 无低优先级。** copy 和 compute 可能争用执行/带宽资源（worker `:139-149`）。
5. **pin fallback 是性能而非正确性降级。** host 不支持 pin 时服务继续运行，但吞吐可能明显下降（worker `:124-126`）。
6. **copy daemon 缺少显式异常恢复。** `_copy_loop` 没有 `try/except`；C++ `TORCH_CHECK` 若抛异常，daemon 线程可能退出而 pending event 永远不完成（copy backend `:92-128`；C++ error path `:214-216,237-242`）。这是从源码控制流推导出的风险。
7. **当前测试偏 e2e，且容忍度较宽。** accuracy test 只要求 5 次中至少 50% 成功（test `:63-77`），没有在本仓逐项验证 CPU LRU、block reuse、preemption race 和 copy failure。

### 6.2 xLLM

1. **当前是 NPU-only。** 非 NPU 构建的 `create_batch_memcpy` 返回 `nullptr`，但 factor > 1 的 engine/worker guard 没有平台检查；最终会创建 host state，却无法执行正确 copy（batch memcpy factory `:26-34`；hierarchy transfer `:165-168,382-384`）。
2. **必须启用 prefix cache，但配置层没有 fail-fast。** factor > 1 且 prefix cache=false 时仍会分配大块 registered host memory，`collect_offload_pairs` 却立即 no-op（hierarchy pool `:88-92`）。
3. **不是 active KV swap。** 只在 sequence deallocate 时 D2H；不能靠它让单个活跃长上下文突破 device KV capacity（hierarchy pool `:58-85`）。
4. **DeepSeek-V4 端到端不支持。** 虽然 host tensor/transfer code 有 SWA/C4/C128 支持，scheduler 只有 flat KV host pool，engine 明确拒绝（LLM engine `:608-618`）。
5. **Host memory 部署要求苛刻。** `mlock` 失败提示必须 `ulimit -l unlimited` 并 fatal；大 factor 也没有 graceful fallback（KV utils `:369-403`）。
6. **Host pool 满时整轮放弃。** `needed_block_num` 必须一次分配成功；没有 partial D2H，会降低高压力时的副本保留率（hierarchy pool `:116-130`）。
7. **H2D 状态先行、失败不回滚。** pool 在 copy 完成前就推进 device KV token cursor；源码只通过 synchronizer abort 让 forward 失败，没有把 block/cursor 回滚到 pre-load 状态（pool `:205-241`；transfer `:328-359,458-465`）。
8. **batch-id synchronizer 可能残留。** batch 取消或 id 重用时旧 entry 直到 collision 才被发现并覆盖；源码已有 error log（transfer `:341-351`）。
9. **waiter busy-spin。** `synchronize_layer` 等 event-record flag 时没有 sleep/yield，copy thread延迟时会消耗 CPU core（NPU synchronizer `:63-68`）。
10. **底层 copy 每 range 同步。** `aclrtMemcpyBatch` + `stream->synchronize()` 让 copy thread 阻塞，layer batch 过小会增加同步次数，过大又会削弱流水重叠（NPU batch memcpy `:78-123`）。
11. **D2H 期间同时 pin device/host ids。** copy 完成 callback 前两侧 block 都不能回池，会形成额外瞬时压力（hierarchy pool `:68-84,449-484`）。
12. **reset 语义不对称。** `HierarchyBlockManagerPool` 没有 override `reset_prefix_cache`，继承实现只清 device `block_managers_`；host prefix cache 不在该 vector 中。调用 reset 后 host 二级缓存可能仍保留，这是源码结构推导出的行为（`xllm/core/framework/block/block_manager_pool.cpp:338-340`；hierarchy header `:40-67,76-82`）。
13. **测试覆盖不足。** 当前 host tests 只验证 host tensor shape/layout（`tests/core/framework/kv_cache/kv_cache_test.cpp:759-864`）；没有 hierarchy manager 的 D2H→LRU→host hit→H2D 端到端测试，也没有 copy failure、TP partial failure、chunked restore 或 cancellation race 测试。
14. **功能仍处于近期重构阶段。** Git history 中 `a01233b3` 曾删除旧 host upload 并禁用 hierarchy host mode，`5638849c`（2026-07-20）才以新框架恢复；当前快照距核心重构很近，应按新功能而非长期稳定路径评估。

## 7. 建议的对齐方向

如果目标是让 xLLM 行为更接近 vllm-ascend 当前 Simple path，优先级建议如下：

1. 先补 manager/transfer 端到端测试，覆盖：
   - device+host 双命中，device miss/host hit；
   - host LRU；
   - chunked H2D；
   - TP 单 rank copy failure；
   - batch cancellation；
   - beam shared block；
   - reset 同时清两级 cache。
2. 在配置解析时 fail-fast：
   - `host_blocks_factor > 1` 必须 `enable_prefix_cache=true`；
   - 必须存在有效 `BatchMemcpy` provider；
   - 明确拒绝未支持的 block types/models。
3. 将 xLLM 的 D2H store 从“sequence deallocate 专属”抽象成“computed full block 发布事件”，再选择 eager/lazy store policy；这样才可能覆盖更广的 reuse window，并向 vLLM `confirmed_tokens` 模型靠拢。
4. 将 `NPUBatchMemcpy` 改为 `aclrtMemcpyBatchAsync` + event completion，保留当前 layer ranges 作为流水粒度，避免每 range 在 copy thread 内做全 stream synchronize。
5. 为 H2D 建立事务语义：copy 成功后再 commit KV cursor，或失败时可靠 rollback block/cursor/synchronizer map。
6. 为 host memory 提供 bytes 上限与可选 pageable fallback。factor 易配置，但无法直接表达每个 TP rank 的实际 DRAM 占用。

## 8. 源码审计边界

- 全部结论来自本机仓库源码、测试、文档和 Git history，没有引用二手文章。
- vllm-ascend 主线 Simple path 的 scheduler/LRU 实现在 upstream vLLM 包中，不在 vllm-ascend 仓；本文对其控制面只陈述 Ascend connector 源码和引入提交明确声明的契约，不伪造缺失的上游逐行引用。
- 未在 NPU 机器运行服务或性能测试；“异步/同步”结论来自具体 API、threadpool、stream 和 event 控制流。

## 9. 三方 H2D 时机与粒度

比较 H2D 不能只说“按 block”或“按 layer”，因为这里有两个正交维度：

1. **block/token 选择**：一次恢复哪些 host-only token，以及预算不足时能否只接收一部分；
2. **layer 时序**：已选 block 的所有层是形成一个完成点，还是分层/分段产生完成点并与同一请求的 forward 重叠。

| 实现 | H2D 选择与触发时机 | 搬运提交粒度 | forward 可见的完成粒度 |
|---|---|---|---|
| xLLM | `HierarchyBlockManagerPool::allocate` 取 device prefix 之后的 host-only suffix，并可受 `UnifiedPolicy` latency block budget 限制；scheduler 在 forward 前调用 `transfer_blocks` | 2-thread load pool；默认 `requested_batches=4`，按 `layers_per_batch=max(num_layers/requested_batches,1)` 切成约 4 个 **layer ranges**，不是每批 4 层；每个 range 覆盖全部已选 blocks × 该 range 的 layers × tensor roles | 每个 range 一个 event，forward 在 layer 边界等待，因此后续 range H2D 可与前面 layers 的计算重叠 |
| vLLM + Ascend | 先取 device hit，再在 CPU 中匹配 suffix；只接受 scheduler 对齐后的完整 blocks，预算不足可接收部分 block；同一步全部 pending requests 合成一个 load event | Ascend backend 把所有 requests 对应的 tensors × blocks 合成一次异步 batch copy | worker 只有在整个 event 完成后才把 request 报为完成；同一请求没有 layer-wise H2D/forward overlap |
| SGLang | 请求 admission 后、batch 构造前调用 `init_load_back`；只装载连续的 host-only evicted chain，受至少 10 slots 的门槛和本次 Device/HBM memory quota 约束，并采用 all-or-none | scheduler 启动 load；controller 合并 batch load ops，再逐 layer 复制该层全部 indices | 每层记录 event，forward 每层等待对应 event，因此支持同一请求的 H2D/compute overlap |

xLLM 的选择逻辑位于 `xllm/core/framework/block/hierarchy_block_manager_pool.cpp:156-310`，可选 latency budget 位于 `xllm/core/scheduler/unified_policy.cpp:430-475`。layer range 公式和默认值见 `xllm/core/framework/kv_cache_transfer/hierarchy_kv_cache_transfer.cpp:33-51`；每个 range 的 copy plan 包含全部所选 blocks 和 tensor roles，执行由 2 个 load 线程承接。底层 `copy_h2d` 使用同步 `aclrtMemcpyBatch` 后再同步 stream（`xllm/core/platform/npu/npu_batch_memcpy.cpp:58-123`），完成后按 range 记录 event；forward 在边界调用等待（`xllm/core/framework/model/model_input_params.h:1009-1017`）。所以 xLLM 的并发来自“range 之间”的流水，而不是单个 range 内的异步 batch API。

vLLM scheduler 的 device hit → CPU suffix hit、whole-block 对齐、预算截断和 event metadata 合并见官方 [`manager.py:242-457`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/simple_kv_offload/manager.py#L242-L457)。worker 在 `get_finished` 路径启动 H2D，并仅在完整 load event 完成后报告 request completion（官方 [`worker.py:198-267`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/simple_kv_offload/worker.py#L198-L267)）。Ascend 实现将全部 tensors × blocks 展开为 batch descriptors（官方 [`npu_mem_ops.py:71-99`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/simple_kv_offload/npu_mem_ops.py#L71-L99)），并在独立 stream 上异步提交整个 job（官方 [`copy_backend.py:67-128`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/simple_kv_offload/copy_backend.py#L67-L128)）。

SGLang 在 admission 之后、batch 构造之前触发 `init_load_back`（官方 [`schedule_policy.py:1156-1165`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py#L1156-L1165)）；连续 host-only evicted chain、10-slot 门槛、quota 和 all-or-none 选择见 [`hiradix_cache.py:1297-1390`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/hiradix_cache.py#L1297-L1390)，实际启动点见 [`scheduler.py:3144-3148`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L3144-L3148)。batch load ops 合并与逐层复制/记录 event 见 [`hybrid_cache_controller.py:485-540`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py#L485-L540)；逐层 forward wait 见 [`cache_controller.py:70-112`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/cache_controller.py#L70-L112) 和 [`memory_pool.py:2231-2261`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/memory_pool.py#L2231-L2261)。
