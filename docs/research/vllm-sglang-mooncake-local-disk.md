# vLLM / SGLang 对 Mooncake `LOCAL_DISK` SSD 回读的支持核查

核查日期：2026-08-24

核查范围：只使用 vLLM、SGLang、Mooncake 官方仓库中的源码、PR、测试和文档。本文所说的“支持 SSD 回读”，特指：Mooncake Store 对象的 `MEMORY` 副本已经不存在、元数据只剩 `LOCAL_DISK` 副本时，读取能够进入 RealClient 的 SSD/offload RPC 路径并恢复数据；不把 Mooncake Transfer Engine 的内存搬运能力算作 SSD 回读。

## 结论

| 框架 | Mooncake Store 内存读写 | `LOCAL_DISK` 识别 | RealClient SSD 回读入口 | 真实 SSD-only 验证 |
|---|---|---|---|---|
| vLLM | 已实现 | 已识别；框架侧主要用于层级统计，实际分派由 Mooncake 完成 | 已通过 `MooncakeDistributedStore.batch_get_into_multi_buffers()` 接入 | PR 有真实 SSD E2E 数据；仓库内未发现自动化的真实 `LOCAL_DISK`-only E2E |
| SGLang | 已实现 | SGLang 本身不解析副本类型；由 Mooncake RealClient 透明识别 | 已通过 `batch_get_into()` / `batch_get_into_multi_buffers()` 接入 | Mooncake 官方有 SGLang + SSD 实机 benchmark；SGLang 仓库内未发现自动化的真实 `LOCAL_DISK`-only E2E |

因此，简短答案是：**vLLM 和 SGLang 当前都实现了“只剩 Mooncake 本地 SSD 副本时的回读链路”**。二者不是在框架里直接调用一个名为 `RealClient` 的类，而是使用 Python 的 `MooncakeDistributedStore` 统一接口；该绑定内部创建 RealClient，再由 RealClient 按副本类型选择内存传输或 `LOCAL_DISK` offload RPC。

这与单纯支持 Mooncake Transfer Engine 不同。Transfer Engine 解决的是已知内存地址之间的 RDMA/TCP/设备传输；`LOCAL_DISK` 恢复还需要 Store 元数据查询、RealClient 读取 SSD、临时/固定缓冲区以及后续数据传输。

## Mooncake 底层能力边界

Mooncake 官方文档明确说明 SSD offload 需要 RealClient，且无论是进程内 RealClient 还是独立 `mooncake_client` + DummyClient，SSD 的读写都发生在 RealClient 内部：[SSD Offload 文档](https://github.com/kvcache-ai/Mooncake/blob/9e9a989d13662eefa40640500a7981559a17b478/docs/source/deployment/ssd/ssd-offload.md#overview)。

官方当前源码把 Python 层调用链写得很清楚：

1. `MooncakeDistributedStore` 的包装对象持有 `PyClient`，`init_real_client()` 创建并保存 `RealClient`：[store_py.cpp L400-L418](https://github.com/kvcache-ai/Mooncake/blob/9e9a989d13662eefa40640500a7981559a17b478/mooncake-integration/store/store_py.cpp#L400-L418)。
2. `setup(..., enable_ssd_offload, ssd_offload_path)` 转发给 `RealClient::setup_real()`：[store_py.cpp L2200-L2236](https://github.com/kvcache-ai/Mooncake/blob/9e9a989d13662eefa40640500a7981559a17b478/mooncake-integration/store/store_py.cpp#L2200-L2236)。
3. Python 的 `batch_get_into_multi_buffers()` 转发给底层 `PyClient`，实际对象为 RealClient：[store_py.cpp L3012-L3024](https://github.com/kvcache-ai/Mooncake/blob/9e9a989d13662eefa40640500a7981559a17b478/mooncake-integration/store/store_py.cpp#L3012-L3024)。
4. RealClient 的多缓冲区读取显式区分 `MEMORY` 与 `LOCAL_DISK`：[real_client.cpp L6119-L6239](https://github.com/kvcache-ai/Mooncake/blob/9e9a989d13662eefa40640500a7981559a17b478/mooncake-store/src/real_client.cpp#L6119-L6239)；对 `LOCAL_DISK` 调用 `batch_get_into_offload_object_internal()`：[real_client.cpp L6288-L6346](https://github.com/kvcache-ai/Mooncake/blob/9e9a989d13662eefa40640500a7981559a17b478/mooncake-store/src/real_client.cpp#L6288-L6346)。
5. 该内部函数向持有 SSD 副本的 RealClient 发起 offload RPC，或在本机直接从 FileStorage 读取，然后调用 `BatchGetOffloadObject` 把数据送入目标 slices：[real_client.cpp L6815-L6901](https://github.com/kvcache-ai/Mooncake/blob/9e9a989d13662eefa40640500a7981559a17b478/mooncake-store/src/real_client.cpp#L6815-L6901)。

Mooncake 自己还有真正构造 `LOCAL_DISK`-only 状态并校验返回字节的测试：溢出 DRAM 后筛出“无 MEMORY、但有 LOCAL_DISK”的 key，再多次 `store.get()` 并做 bit-exact 比较；测试注释明确说它经过 offload RPC：[test_promotion_on_hit.py L124-L204](https://github.com/kvcache-ai/Mooncake/blob/9e9a989d13662eefa40640500a7981559a17b478/mooncake-wheel/tests/test_promotion_on_hit.py#L124-L204)。这证明统一 Store 接口下的 RealClient 路径本身已经闭环。

## vLLM

### 是否已经实现

已实现。SSD offload 支持于 2026-05-16 通过 [vLLM PR #42689](https://github.com/vllm-project/vllm/pull/42689) 合入，对应提交 [`36e74c9`](https://github.com/vllm-project/vllm/commit/36e74c9ea4feb5ade38ffa1ea96f24dd73316e02)，进入 v0.22.0 开发线。

当前源码基准为 `e8888b2d68bd7c6cce0aada7f0e214e55020e20d`：

- vLLM 创建的是 `mooncake.store.MooncakeDistributedStore`，支持进程内 Store 和独立 Store/RealClient 拓扑：[worker.py L1338-L1404](https://github.com/vllm-project/vllm/blob/e8888b2d68bd7c6cce0aada7f0e214e55020e20d/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py#L1338-L1404)。
- 读取会按 DirectIO staging budget 分批，最终统一调用 `batch_get_into_multi_buffers()`：[worker.py L1212-L1273](https://github.com/vllm-project/vllm/blob/e8888b2d68bd7c6cce0aada7f0e214e55020e20d/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py#L1212-L1273)。结合上一节的 Mooncake 源码，这个调用会在只有 `LOCAL_DISK` 副本时进入 RealClient offload RPC。
- vLLM 能识别 `is_disk_replica()` 和 `is_local_disk_replica()`，但该框架侧判断主要用于把命中记为 disk tier；真正的副本选择和 SSD RPC 分派仍在 Mooncake 内部：[worker.py L303-L316](https://github.com/vllm-project/vllm/blob/e8888b2d68bd7c6cce0aada7f0e214e55020e20d/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py#L303-L316)。
- 官方部署文档给出了带 SSD offload、独立 Mooncake Store 进程的拓扑和配置：[mooncake_store_connector_usage.md L162-L204](https://github.com/vllm-project/vllm/blob/e8888b2d68bd7c6cce0aada7f0e214e55020e20d/docs/features/mooncake_store_connector_usage.md#L162-L204)。

### 测试证据与边界

PR #42689 报告了真实 SSD E2E：约 49.6 GB 回读、21,027 个 disk keys、0 次失败。这是实机验证，但不是仓库自动化测试。

当前仓库的自动化测试覆盖了：

- Mock 的 `LOCAL_DISK` 描述符：[test_mooncake_store_worker.py L390-L401](https://github.com/vllm-project/vllm/blob/e8888b2d68bd7c6cce0aada7f0e214e55020e20d/tests/v1/kv_connector/unit/test_mooncake_store_worker.py#L390-L401)；
- Mock 的内存/磁盘层级识别和批量 get：[L1436-L1477](https://github.com/vllm-project/vllm/blob/e8888b2d68bd7c6cce0aada7f0e214e55020e20d/tests/v1/kv_connector/unit/test_mooncake_store_worker.py#L1436-L1477)；
- SSD staging budget 分批：[L1504-L1574](https://github.com/vllm-project/vllm/blob/e8888b2d68bd7c6cce0aada7f0e214e55020e20d/tests/v1/kv_connector/unit/test_mooncake_store_worker.py#L1504-L1574)。

未发现一个自动化测试会真实填满 Mooncake DRAM、确认 MEMORY 副本消失、只剩 `LOCAL_DISK`，再通过 vLLM 完成恢复。因此 vLLM 的状态应描述为：**功能已合入且有真实手工 E2E，自动化 SSD-only 回归保护仍不完整**。

## SGLang

### 是否已经实现

已实现。SGLang 于 2026-05-14 合入 [PR #24277](https://github.com/sgl-project/sglang/pull/24277)，对应提交 [`be156d6`](https://github.com/sgl-project/sglang/commit/be156d68043171d2d9b9f6fbd7b9f7d90dde7b21)，增加 Mooncake Store SSD offload 配置。

当前源码基准为 `230c052ebcf3968bb1b8d1be793c2c07deb9ca0d`：

- `enable_ssd_offload` 和 `ssd_offload_path` 被传给 `MooncakeDistributedStore.setup()`；若安装的旧 Mooncake 不支持这些参数，SGLang 会警告并回退到无 SSD 的旧行为：[mooncake_store.py L500-L542](https://github.com/sgl-project/sglang/blob/230c052ebcf3968bb1b8d1be793c2c07deb9ca0d/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L500-L542)。所以“支持”以安装了兼容版本的 Mooncake 为前提。
- 普通布局使用 `batch_get_into()`，多缓冲区布局使用 `batch_get_into_multi_buffers()`：[mooncake_store.py L1323-L1330](https://github.com/sgl-project/sglang/blob/230c052ebcf3968bb1b8d1be793c2c07deb9ca0d/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L1323-L1330)。SGLang 不在这一层自行读取副本描述符，而是把 `MEMORY` / `LOCAL_DISK` 的选择交给 Mooncake RealClient。
- SGLang 官方 README 明确说明该配置会让 Mooncake 把 DRAM 溢出数据 spill 到本地 SSD，并将路径转发给 `MooncakeDistributedStore.setup()`：[README L263-L300](https://github.com/sgl-project/sglang/blob/230c052ebcf3968bb1b8d1be793c2c07deb9ca0d/python/sglang/srt/mem_cache/storage/mooncake_store/README.md#L263-L300)。

### 测试证据与边界

SGLang 仓库的 Mooncake HiCache 单测和注册 E2E 使用普通内存 Store 配置，未发现显式开启 SSD、制造真实 `LOCAL_DISK`-only 状态并回读的自动化测试。

不过 Mooncake 官方给出了使用 SGLang 多轮 benchmark 的实机结果：80 GB Mooncake 内存耗尽后，无 SSD 的命中率从 83% 降至 36%；启用 SSD 后，已淘汰数据仍可读取，命中率保持在 84% 以上，并观察到 NVMe 回读带来的延迟差异：[benchmark 结果 L18-L30](https://github.com/kvcache-ai/Mooncake/blob/9e9a989d13662eefa40640500a7981559a17b478/docs/source/performance/mooncake/ssd-offload-benchmark-results.md#L18-L30)。文档也给出了 SGLang、Mooncake master、独立 RealClient 和 SSD offload 的完整启动方式：[L86-L157](https://github.com/kvcache-ai/Mooncake/blob/9e9a989d13662eefa40640500a7981559a17b478/docs/source/performance/mooncake/ssd-offload-benchmark-results.md#L86-L157)。

因此 SGLang 的状态应描述为：**功能已合入，Mooncake 官方有真实 SGLang + SSD 回读 benchmark，但 SGLang 自己的自动化 SSD-only 回归测试仍缺失**。

## 对 xLLM 当前补丁的含义

不能因为 vLLM/SGLang 已支持，就推导出“只要 xLLM 已经接入 Mooncake 内存双向复制，SSD 也会自动可用”。关键差异是调用层级：

```text
仅 Transfer Engine / 普通 Client 路径
    MEMORY 地址 -> Host/设备内存

vLLM / SGLang 当前 Store 路径
    MooncakeDistributedStore
      -> RealClient 查询副本类型
      -> MEMORY: BatchGet
      -> LOCAL_DISK: offload RPC / FileStorage read
      -> 目标 buffers
```

如果 xLLM 当前使用的是不能透明处理 `LOCAL_DISK` 的较低层 `Client::BatchGet`，那么显式接入 RealClient 或与 `MooncakeDistributedStore.batch_get_into[_multi_buffers]` 等价的路径仍然是必要工作。vLLM/SGLang 的实现恰好说明：完整支持的判断标准不是“HBM 与 Mooncake 内存能双向搬运”，而是“SSD-only 元数据命中后，确实进入 RealClient offload RPC 并把字节恢复到框架目标缓冲区”。
