# 基于多密钥同态加密的联邦学习安全聚合
# 后端架构、训练流程、实验记录与 Flower 对接说明

## 1. 文档目的

本文档用于说明当前 `backend/` 的实际实现状态，回答以下问题：

- 现在的整体架构是什么。
- 训练流程每个阶段做了什么。
- 训练过程中各部分当前状态如何（已实现内容、边界与约束）。
- 实验数据如何记录、记录了哪些字段、如何导出与画图。
- 如何与 Flower 分布式训练框架对接。
- 分布式训练过程中如何与该后端通信。

说明基于当前代码实现（`backend/main.go`、`backend/service.go`、`backend/types.go`、`backend/experiment.go`）。

---

## 2. 当前整体架构

### 2.1 系统组件

系统可以拆成四层：

1. Flower Orchestration 层
- `Flower Server`：负责联邦轮次调度、客户端采样、参数广播与收集。
- `Flower Clients`：本地训练模型，产生本地权重更新。

2. 安全聚合后端层（本仓库 `backend/`）
- HTTP API 层：接收 Flower 侧请求（初始化、加密、聚合、解密、实验追踪）。
- HE Service 层：执行 MKCKKS 加密、同态加法聚合、解密协议。
- Experiment Tracker 层：记录实验事件、汇总统计、导出 CSV 和绘图数据。

3. 密码学实现层
- `mkckks` / `mkrlwe`：多密钥 CKKS 与 RLWE 相关操作。
- `lattigo/v2`：底层环、参数、PRNG、CKKS 原语实现。

4. 数据持久层
- 训练密文：内存中的 `CipherStore`（进程重启后丢失）。
- 实验日志：磁盘目录 `./experiment_logs`（可配置为 `MKHE_EXPERIMENT_DIR`）。

### 2.2 后端内部结构

后端主入口由 `main.go` 启动：

- 创建 `HEService`：维护密钥集、加密器、解密器、聚合器、轮次状态、密文存储。
- 创建 `ExperimentTracker`：维护实验清单、active 实验、事件文件、汇总统计。
- 挂载 API 路由：
  - 系统管理：`/system/init`, `/system/status`, `/health`
  - 客户端管理：`/clients/register`
  - 核心链路：`/encrypt`, `/aggregate`, `/decrypt`
  - 分布式解密：`/decrypt/partial`, `/decrypt/final`
  - 密文管理：`/cipher/{id}`, `/cipher/cleanup`
  - 轮次管理：`/round/advance`
  - 实验追踪：`/experiments/*`

中间件：
- CORS：允许跨域，便于 Python 侧调用。
- Logging：记录方法、路径、远端地址、耗时。

### 2.3 关键运行状态

`HEService` 维护的状态：

- `Initialized`：系统是否完成初始化。
- `CurrentRound`：当前训练轮次（初始 1）。
- `ClientList`：已注册参与方 ID。
- `CipherStore`：`cipher_id -> CipherEntry`，包含密文与元信息（轮次、层标签、chunk 序号）。

`ExperimentTracker` 维护的状态：

- `experiments`：所有实验运行实例。
- `activeID`：当前默认实验 ID。
- 每个实验的事件列表与累积统计：按操作和按轮次聚合。

---

## 3. 训练流程（端到端）

## 3.1 训练前初始化阶段

1. 可选：启动实验
- `POST /api/v1/experiments/start`
- 产出 `experiment_id`，后续所有请求可带上该 ID（请求体字段 / Header / query）。

2. 初始化 HE 系统
- `POST /api/v1/system/init`
- 输入：`logN`（14 或 15）、`client_ids`
- 输出：`slots`, `max_level`, `scale`, `registered_count`

初始化完成后：
- 生成并加载所有参与方的密钥材料（pk/sk/relin key）。
- 重置轮次为 1。
- 清空旧密文。

## 3.2 每轮训练主流程（集中式解密）

1. Flower Server 下发全局模型给客户端。
2. Flower Client 本地训练，得到本地权重 `weights`。
3. 客户端调用 `POST /encrypt`：
- 后端使用客户端公钥加密。
- 超过 `slots` 自动分 chunk，返回多个 `cipher_ids`。
4. Flower Server 收集所有客户端 `cipher_ids`，按 chunk 对齐，构造 `cipher_groups`。
5. Flower Server 调用 `POST /aggregate`：
- 同态逐项相加。
- 若 `average=true`，乘以 `1/N` 做 FedAvg（会消耗 1 层 level）。
6. Flower Server 调用 `POST /decrypt` 得到聚合后明文权重。
7. Flower Server 用解密结果更新全局模型。
8. 轮次收尾：
- `POST /cipher/cleanup` 清理当轮密文。
- `POST /round/advance` 推进轮次。

## 3.3 每轮训练主流程（分布式解密）

前半流程（加密、构造分组、聚合）一致。聚合后使用：

1. `POST /decrypt/partial`（client_1）
2. `POST /decrypt/partial`（client_2）
3. ...直到 `remaining_parties=0`
4. `POST /decrypt/final` 得到明文

这一路径符合“多方逐步消除密钥贡献”的分布式解密模式，避免单点持钥的解密方式。

---

## 4. 训练过程各部分当前情况

### 4.1 已实现并可用

1. 多密钥加密与同态聚合
- 支持多客户端注册与多密钥场景。
- 支持自动分块加密（大模型参数超过 slots）。
- 支持同态求和与 FedAvg（常数乘 + rescale）。

2. 两种解密路径
- 集中式解密：便于调试与基准对比。
- 分布式解密：逐客户端部分解密 + 最终提取。

3. 密文生命周期管理
- 查询密文元信息。
- 单个删除与按轮次批量清理。

4. 实验追踪与统计
- 自动埋点核心操作耗时和体量。
- 支持手动记录任务指标（accuracy/loss/通信量）。
- 支持 summary、events、CSV、plot-data。

### 4.2 运行健康情况（当前）

当前 `backend` 测试可通过：

- 命令：`go test ./backend -count=1`
- 状态：`ok`

### 4.3 当前边界与注意事项

1. 密文存储在内存
- `CipherStore` 非持久化，进程重启会丢失。
- 长时间实验务必按轮次 cleanup，避免内存增长。

2. 初始化会重置状态
- 再次调用 `/system/init` 会重建密钥并清空密文。

3. CKKS 近似误差
- 解密结果是近似值，存在小数误差。

4. 分布式解密顺序由外部控制
- 后端不负责调度“谁先部分解密”，由 Flower 侧编排。

---

## 5. 实验数据如何记录

## 5.1 自动记录机制

后端在以下接口处理过程中自动调用 `recordServiceEvent(...)`：

- `system_init`
- `client_register`
- `encrypt`
- `aggregate`
- `decrypt`
- `partial_decrypt`
- `final_decrypt`
- `cipher_cleanup`
- `round_advance`

每次事件写入 `events.jsonl`（一行一条 JSON），并更新内存统计与 `summary.json`。

## 5.2 实验 ID 解析优先级

后端按以下顺序确定事件归属实验：

1. 请求体 `experiment_id`
2. Header `X-Experiment-ID`
3. Query `?experiment_id=...`
4. 当前 active 实验（`/experiments/active`）

如果最终没有实验 ID，则该次操作不会写入实验事件。

## 5.3 单条事件字段（核心）

自动/手动事件统一结构，关键字段：

- `event_id`: 事件唯一 ID
- `experiment_id`: 实验 ID
- `timestamp`: 时间戳（RFC3339Nano）
- `round`: 训练轮次
- `operation`: 操作名
- `client_id`: 相关客户端
- `layer_tag`: 模型层标签
- `status`: `success` / `failed`
- `duration_ms`: 操作耗时（毫秒）
- `input_cipher_count`: 输入密文数量
- `output_cipher_count`: 输出密文数量
- `weight_count`: 相关权重数量
- `error`: 错误消息（失败时）
- `metrics`: 自定义数值指标（手动事件常用）
- `metadata`: 自定义字符串标签

## 5.4 聚合统计（summary）

`summary` 提供两类视图：

1. 按 operation 聚合
- `count`
- `success_count`
- `failure_count`
- `avg_duration_ms`
- `max_duration_ms`

2. 按 round 聚合
- `event_count`
- `avg_duration_ms`
- `output_cipher_count`
- `weight_count`

### 5.5 文件落盘结构

每个实验目录：

- `meta.json`: 实验元信息（名称、描述、标签、起止时间、是否 active）
- `events.jsonl`: 原始事件流
- `summary.json`: 实时汇总统计
- `events.csv`: 导出文件（按需生成）

---

## 6. 记录了哪些数据（可用于论文/实验图）

系统当前可稳定提供下列数据维度：

1. 系统配置与实验元数据
- logN、客户端数、实验名称、标签。

2. 每轮/每阶段时延
- 加密、聚合、解密、部分解密等耗时。

3. 密文规模与工作量
- 输入/输出密文数量。
- 处理权重数量。

4. 成功率与失败信息
- 各操作成功失败计数。
- 错误消息。

5. 可选精度指标
- 通过 `/experiments/event` 手动上报 `metrics.accuracy`、`metrics.loss` 等。

6. 可绘图数据
- `latency_by_operation`
- `output_cipher_by_round`
- `throughput_by_round`
- `event_count_by_operation`

注：模型精度、网络字节量等“业务侧指标”需要由 Flower 侧主动上报为手动事件。

---

## 7. 如何与 Flower 框架对接

## 7.1 对接角色分工

推荐分工：

1. Flower Client 负责
- 本地训练。
- 将权重展平为 `float64[]`。
- 调用 `/encrypt` 获取 `cipher_ids`。

2. Flower Server 负责
- 收集所有客户端 `cipher_ids`。
- 构造 `cipher_groups` 并调用 `/aggregate`。
- 调用 `/decrypt` 或编排 `/decrypt/partial + /decrypt/final`。
- 更新全局模型。
- 调用 `/cipher/cleanup` 与 `/round/advance`。

3. Flower Server 或独立实验控制器负责
- 启停实验 `/experiments/start|stop`。
- 上报 accuracy/loss/通信量 `/experiments/event`。
- 拉取 summary/csv/plot-data。

## 7.2 与 Flower 训练循环映射

典型映射关系：

1. `on_fit_config_fn` 或训练前钩子
- 调用 `/system/init`
- 可选调用 `/experiments/start`

2. Client `fit(...)`
- 本地训练后调用 `/encrypt`
- 向 Server 返回 `cipher_ids`（而非明文权重）

3. Server `aggregate_fit(...)`
- 按 chunk 组装 `cipher_groups`
- 调 `/aggregate`
- 调 `/decrypt` 获取明文全局权重
- 写回全局模型参数

4. 轮次结束
- `/cipher/cleanup`
- `/round/advance`
- 可选 `/experiments/event` 记录 round accuracy/loss

## 7.3 对接时的请求规范建议

1. 全链路传递 `experiment_id`
- 统一在 body 带 `experiment_id`，避免事件归属混乱。

2. 明确 round 语义
- 所有本轮请求都带同一 `round`。

3. 固定客户端 ID
- 同一客户端在整个实验中 ID 不变。

4. 大模型注意 chunk 对齐
- 所有客户端 chunk 数应一致。
- 组装 `cipher_groups` 时必须按 chunk 索引对齐。

---

## 8. 分布式训练过程中如何与后端通信

## 8.1 通信协议

- 协议：HTTP/1.1
- 数据：`application/json`（CSV 导出接口为 `text/csv`）
- 统一响应：`{code, message, data}`
- 跨域：已开启 CORS

## 8.2 通信路径（推荐）

推荐路径 A（最常见）：

- Flower Client <-> Flower Server（gRPC，Flower 内部）
- Flower Server <-> MKHE Backend（HTTP JSON）

说明：
- Client 可以只把必要信息交给 Server，再由 Server 统一调后端。
- 或者 Client 直接调用后端 `/encrypt`，再把 `cipher_ids` 回传 Server；两种都可行。

推荐路径 B（更集中控制）：

- 所有后端调用统一由 Flower Server 发起。
- 优点：实验记录、错误处理、重试策略集中。

## 8.3 请求频率与时序建议

每个 round 建议顺序：

1. 并发阶段：多个客户端并发加密 `POST /encrypt`
2. 汇总阶段：Server 一次 `POST /aggregate`
3. 解密阶段：
- 集中式：一次 `POST /decrypt`
- 分布式：多次 `POST /decrypt/partial` + 一次 `POST /decrypt/final`
4. 收尾阶段：`POST /cipher/cleanup` + `POST /round/advance`

## 8.4 失败与重试建议

1. 如果 `/encrypt` 某客户端失败
- 该客户端本轮可视为掉队，Server 按策略剔除或重试。

2. 如果 `/aggregate` 失败
- 多为 `cipher_id` 缺失或分组错误，先检查 `cipher_groups` 对齐。

3. 如果 `/partial` 失败
- 检查该客户端是否已注册且密钥存在。

4. 重试原则
- `GET` 类查询可直接重试。
- `POST` 写入类重试前需确认幂等语义，防止重复写入事件或状态推进。

---

## 9. 可直接落地的实验记录方案

建议每轮至少记录：

1. 自动事件（后端已自动完成）
- encrypt / aggregate / decrypt 时延与规模。

2. 手动事件（Flower 侧补充）
- `accuracy_eval`: `metrics.accuracy`, `metrics.loss`
- `comm_stats`: `metrics.upload_bytes`, `metrics.download_bytes`
- `client_train`: `metrics.local_train_ms`

3. 轮次结束导出
- 训练中定期拉 `/experiments/{id}/summary`
- 训练后导出 `/experiments/{id}/events.csv`
- 画图使用 `/experiments/{id}/plot-data` 或 `backend/plot_experiment.py`

---

## 10. 当前实现结论（给实验负责人）

当前后端已经具备用于“基于多密钥同态加密的联邦学习安全聚合”实验的核心能力：

- 安全聚合主链路完整（初始化、加密、聚合、集中/分布式解密）。
- 实验追踪闭环完整（自动事件、手动事件、汇总、CSV、绘图数据）。
- 与 Flower 的集成边界清晰（客户端训练与服务器编排可分离）。

因此可以直接开展以下类型实验：

- 不同 `logN`、不同客户端数量下的时延对比。
- 集中式解密 vs 分布式解密开销对比。
- 加密训练轮次中吞吐与精度联合分析。
- 同态安全聚合带来的通信/计算成本评估。
