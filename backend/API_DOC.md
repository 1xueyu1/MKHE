# MKHE 联邦学习后端 API 接口说明

## 概述

这是一个基于多密钥同态加密(MKHE)的联邦学习安全聚合 Go 后端服务。使用 MKCKKS 方案（基于 lattigo 库），为 Flower 框架提供加密、同态聚合、解密能力。

- **监听地址**: `http://localhost:8082`
- **API 前缀**: `/api/v1`
- **数据格式**: JSON
- **已启用 CORS**，Python 可直接跨域调用

---

## 统一响应格式

所有接口返回以下 JSON 结构：

```json
{
  "code": 0,
  "message": "描述信息",
  "data": { ... }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| code | int | 0=成功，非0=失败 |
| message | string | 操作描述 |
| data | object/null | 成功时返回业务数据，失败时无此字段 |

---

## 核心概念

### 1. Slots（槽位数）
每个 CKKS 密文可以编码的浮点数数量。LogN=14 时 slots=8192，LogN=15 时 slots=16384。当模型权重数 > slots 时，后端会自动拆分为多个密文（chunk）。

### 2. Chunk（分块）
当 flatten 后的模型权重数组长度超过 slots 时，权重被切分为多个 chunk，每个 chunk 对应一个密文。加密返回 `cipher_ids[]` 数组，数组长度即 chunk 数。

### 3. CipherID（密文ID）
后端为每个密文生成的 32 字符十六进制唯一标识。所有后续操作（聚合、解密、查询、删除）都通过 CipherID 引用密文。密文本身保存在 Go 后端内存中，不会通过网络传输。

### 4. CipherGroups（密文分组）
聚合时的参数。`cipher_groups[i]` 是第 i 个 chunk，来自所有参与方的密文 ID 列表。例如 3 个客户端、2 个 chunk：
```
cipher_groups = [
  [clientA_chunk0, clientB_chunk0, clientC_chunk0],  // chunk 0
  [clientA_chunk1, clientB_chunk1, clientC_chunk1],  // chunk 1
]
```

### 5. 两种解密模式
- **集中式解密** (`/decrypt`): 后端持有所有密钥，一步解密。适合开发调试。
- **分布式解密** (`/decrypt/partial` + `/decrypt/final`): 各参与方依次用自己的密钥执行部分解密，所有人解密完毕后提取明文。安全性更高。

---

## 接口列表

### ① POST /api/v1/system/init — 初始化加密系统

**调用方**: Flower Server，训练开始前调用一次。

**请求体**:
```json
{
  "logN": 14,
  "client_ids": ["client_0", "client_1", "client_2"]
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| logN | int | 是 | CKKS 参数等级。14=快速/测试（slots=8192），15=生产（slots=16384） |
| client_ids | string[] | 是 | 参与联邦学习的客户端 ID 列表 |

**成功响应** `data`:
```json
{
  "status": "initialized",
  "slots": 8192,
  "max_level": 5,
  "scale": 4503599627370496,
  "registered_count": 3
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| status | string | 固定 "initialized" |
| slots | int | 每个密文可编码的浮点数数量 |
| max_level | int | 密文乘法深度上限。FedAvg 均值聚合会消耗 1 层（仅求和不消耗） |
| scale | float64 | CKKS 编码缩放因子 |
| registered_count | int | 已注册客户端数 |

**注意**: 初始化会重置所有状态（密文、轮次、密钥）。重复调用会覆盖之前的状态。初始化耗时约 0.5~1 秒。

---

### ② GET /api/v1/system/status — 获取系统状态

**调用方**: Flower Server / 管理员。

**请求**: 无参数。

**成功响应** `data`:
```json
{
  "initialized": true,
  "current_round": 1,
  "registered_ids": ["client_0", "client_1", "client_2"],
  "cipher_count": 6,
  "slots": 8192,
  "max_level": 5
}
```

---

### ③ GET /api/v1/health — 健康检查

**请求**: 无参数。

**成功响应** `data`:
```json
{
  "status": "ok",
  "uptime": "2h30m15s"
}
```

---

### ④ POST /api/v1/clients/register — 动态注册客户端

**调用方**: Flower Server，新客户端加入时调用。

**请求体**:
```json
{
  "client_id": "client_3"
}
```

**成功响应** `data`:
```json
{
  "client_id": "client_3",
  "has_public_key": true,
  "has_secret_key": true
}
```

**注意**: 如果客户端已在 init 时注册，无需再调此接口。重复注册同一 ID 会返回错误。

---

### ⑤ POST /api/v1/encrypt — 加密模型权重（核心）

**调用方**: Flower Client（通过 Server 中转），本地训练后上传权重。

**请求体**:
```json
{
  "client_id": "client_0",
  "weights": [0.1, 0.2, 0.3, ..., 0.9],
  "round": 1,
  "layer_tag": "fc1",
  "weight_count": 1000
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| client_id | string | 是 | 客户端 ID（必须已注册） |
| weights | float64[] | 是 | flatten 后的模型权重数组 |
| round | int | 是 | 当前训练轮次 |
| layer_tag | string | 否 | 层标签，用于分层聚合场景 |
| weight_count | int | 否 | 原始权重数（解密时截断零填充用） |

**成功响应** `data`:
```json
{
  "cipher_ids": ["a1b2c3d4...", "e5f6g7h8..."],
  "chunk_count": 2,
  "round": 1
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| cipher_ids | string[] | 密文 ID 列表，长度 = chunk_count。按 chunk 顺序排列 |
| chunk_count | int | 密文 chunk 数。权重数 ≤ slots 时为 1，否则 = ceil(权重数 / slots) |
| round | int | 回显当前轮次 |

**自动分 chunk 规则**: 若 weights 长度为 20000、slots=8192，则拆为 3 个 chunk：chunk0 编码 weights[0:8192]，chunk1 编码 weights[8192:16384]，chunk2 编码 weights[16384:20000]（不足 slots 的部分零填充）。

---

### ⑥ POST /api/v1/aggregate — 同态聚合（FedAvg 核心）

**调用方**: Flower Server，收集完所有客户端密文后调用。

**请求体**:
```json
{
  "round": 1,
  "cipher_groups": [
    ["cid_client0_chunk0", "cid_client1_chunk0", "cid_client2_chunk0"],
    ["cid_client0_chunk1", "cid_client1_chunk1", "cid_client2_chunk1"]
  ],
  "average": true,
  "client_count": 3
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| round | int | 是 | 当前训练轮次 |
| cipher_groups | string[][] | 是 | 按 chunk 分组的密文 ID。groups[i] 是第 i 个 chunk 来自各客户端的密文 |
| average | bool | 否 | true=FedAvg 求均值（÷N），false=仅同态求和。默认 false |
| client_count | int | 条件 | average=true 时必填，参与方数量 N |

**成功响应** `data`:
```json
{
  "aggregated_cipher_ids": ["agg_chunk0_id", "agg_chunk1_id"],
  "round": 1
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| aggregated_cipher_ids | string[] | 聚合结果密文 ID 列表，每个 chunk 一个 |
| round | int | 回显当前轮次 |

**cipher_groups 构造方法**:
```python
# Python 侧: 假设 3 个客户端，每个返回 2 个 chunk
all_cipher_ids = {
    "client_0": ["c0_chunk0", "c0_chunk1"],
    "client_1": ["c1_chunk0", "c1_chunk1"],
    "client_2": ["c2_chunk0", "c2_chunk1"],
}
chunk_count = 2
cipher_groups = []
for i in range(chunk_count):
    group = [all_cipher_ids[cid][i] for cid in all_cipher_ids]
    cipher_groups.append(group)
```

**注意**: average=true 时会消耗密文 1 层 level（执行常数乘法 + rescale）。仅求和不消耗 level。

---

### ⑦ POST /api/v1/decrypt — 集中式解密

**调用方**: Flower Server，聚合后获取明文全局模型。

**请求体**:
```json
{
  "cipher_ids": ["agg_chunk0_id", "agg_chunk1_id"],
  "weight_count": 1000
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| cipher_ids | string[] | 是 | 待解密的密文 ID 列表（按 chunk 顺序） |
| weight_count | int | 否 | 原始权重数量。提供后会截断零填充，不提供则返回完整 slots 长度 |

**成功响应** `data`:
```json
{
  "weights": [0.1, 0.2, ...],
  "length": 1000
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| weights | float64[] | 解密后的模型权重（所有 chunk 拼接，已截断）|
| length | int | 返回的权重数量 |

**注意**: 集中式解密使用后端持有的全部密钥，安全性较低，主要用于开发调试。所有 chunk 的解密结果会自动按顺序拼接成完整权重数组。

---

### ⑧ POST /api/v1/decrypt/partial — 分布式部分解密

**调用方**: 各 Flower Client（通过 Server 中转），依次执行。

**请求体**:
```json
{
  "cipher_ids": ["agg_chunk0_id", "agg_chunk1_id"],
  "client_id": "client_0"
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| cipher_ids | string[] | 是 | 待部分解密的密文 ID 列表 |
| client_id | string | 是 | 当前执行部分解密的客户端 ID |

**成功响应** `data`:
```json
{
  "cipher_ids": ["new_chunk0_id", "new_chunk1_id"],
  "remaining_parties": 2
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| cipher_ids | string[] | 部分解密后生成的新密文 ID（原密文不变） |
| remaining_parties | int | 还需要几方执行部分解密。为 0 时可调用 final |

**分布式解密流程**:
```
聚合结果 cipher_ids → client_0 partial → 新 ids → client_1 partial → 新 ids → client_2 partial → 新 ids (remaining=0) → final
```
每次调用返回的 `cipher_ids` 作为下一个客户端的输入。原始聚合密文不被修改。

---

### ⑨ POST /api/v1/decrypt/final — 分布式最终解密

**调用方**: Flower Server，所有参与方部分解密完成后调用（remaining_parties=0）。

**请求体**:
```json
{
  "cipher_ids": ["final_chunk0_id", "final_chunk1_id"],
  "weight_count": 1000
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| cipher_ids | string[] | 是 | 最后一轮部分解密返回的密文 ID |
| weight_count | int | 否 | 同 decrypt 接口 |

**成功响应** `data`: 同 `/decrypt` 的 `DecryptResponse`。

---

### ⑩ GET /api/v1/cipher/{cipher_id} — 查询密文元信息

**请求**: 无 body，cipher_id 在 URL 路径中。

**成功响应** `data`:
```json
{
  "cipher_id": "a1b2c3d4...",
  "id_set": ["client_0"],
  "level": 5,
  "scale": 4503599627370496,
  "round": 1,
  "client_id": "client_0",
  "layer_tag": "fc1",
  "chunk_index": 0
}
```

---

### ⑪ DELETE /api/v1/cipher/{cipher_id} — 删除单个密文

**请求**: 无 body，cipher_id 在 URL 路径中。

**成功响应**:
```json
{ "code": 0, "message": "密文已删除" }
```

---

### ⑫ POST /api/v1/cipher/cleanup — 批量清理密文

**调用方**: Flower Server，每轮结束后清理旧密文释放内存。

**请求体**:
```json
{
  "round": 1
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| round | int | 是 | 要清理的轮次。传 0 则清理所有密文 |

**成功响应** `data`:
```json
{
  "deleted_count": 12
}
```

---

### ⑬ POST /api/v1/round/advance — 推进训练轮次

**调用方**: Flower Server，每轮训练结束后调用。

**请求体**: 空 `{}`

**成功响应** `data`:
```json
{
  "previous_round": 1,
  "current_round": 2
}
```

---

## Flower 集成完整调用流程

下面是 Python Flower 侧调用 Go 后端的完整伪代码：

```python
import requests
import numpy as np

GO_BACKEND = "http://localhost:8082/api/v1"

# =========================================================
# 阶段 0: 训练开始前 — Server 初始化 MKHE 系统
# =========================================================
def init_mkhe(client_ids: list[str], logN: int = 14):
    resp = requests.post(f"{GO_BACKEND}/system/init", json={
        "logN": logN,
        "client_ids": client_ids
    })
    data = resp.json()["data"]
    slots = data["slots"]      # 记住 slots，后续判断 chunk
    return data

# =========================================================
# 阶段 1: 每轮训练 — Client 侧（在 Strategy 或 Client 中）
# =========================================================

# Step 1: Client 本地训练后，将权重加密上传
def encrypt_weights(client_id: str, weights: np.ndarray, round_num: int) -> dict:
    flat_weights = weights.flatten().tolist()
    resp = requests.post(f"{GO_BACKEND}/encrypt", json={
        "client_id": client_id,
        "weights": flat_weights,
        "round": round_num,
        "weight_count": len(flat_weights)
    })
    return resp.json()["data"]
    # 返回: {"cipher_ids": [...], "chunk_count": N, "round": R}

# =========================================================
# 阶段 2: 每轮训练 — Server 侧聚合
# =========================================================

# Step 2: Server 收集所有客户端的 cipher_ids，构造 cipher_groups
def build_cipher_groups(all_client_cipher_ids: dict) -> list:
    """
    输入: {"client_0": ["id0", "id1"], "client_1": ["id0", "id1"], ...}
    输出: [["c0_chunk0", "c1_chunk0", ...], ["c0_chunk1", "c1_chunk1", ...]]
    """
    clients = list(all_client_cipher_ids.keys())
    chunk_count = len(all_client_cipher_ids[clients[0]])
    cipher_groups = []
    for i in range(chunk_count):
        group = [all_client_cipher_ids[cid][i] for cid in clients]
        cipher_groups.append(group)
    return cipher_groups

# Step 3: Server 发起同态聚合
def aggregate(cipher_groups: list, round_num: int, num_clients: int, average: bool = True) -> dict:
    resp = requests.post(f"{GO_BACKEND}/aggregate", json={
        "round": round_num,
        "cipher_groups": cipher_groups,
        "average": average,
        "client_count": num_clients
    })
    return resp.json()["data"]
    # 返回: {"aggregated_cipher_ids": [...], "round": R}

# Step 4A: 集中式解密（开发调试用）
def decrypt_centralized(cipher_ids: list, weight_count: int) -> np.ndarray:
    resp = requests.post(f"{GO_BACKEND}/decrypt", json={
        "cipher_ids": cipher_ids,
        "weight_count": weight_count
    })
    return np.array(resp.json()["data"]["weights"])

# Step 4B: 分布式解密（安全协议）
def decrypt_distributed(cipher_ids: list, client_ids: list, weight_count: int) -> np.ndarray:
    current_ids = cipher_ids
    for cid in client_ids:
        resp = requests.post(f"{GO_BACKEND}/decrypt/partial", json={
            "cipher_ids": current_ids,
            "client_id": cid
        })
        result = resp.json()["data"]
        current_ids = result["cipher_ids"]
        # result["remaining_parties"] 可用来确认进度
    
    # 所有人解密完毕 → 提取明文
    resp = requests.post(f"{GO_BACKEND}/decrypt/final", json={
        "cipher_ids": current_ids,
        "weight_count": weight_count
    })
    return np.array(resp.json()["data"]["weights"])

# =========================================================
# 阶段 3: 每轮训练结束 — Server 清理和推进
# =========================================================

def finish_round(round_num: int):
    # 清理本轮密文
    requests.post(f"{GO_BACKEND}/cipher/cleanup", json={"round": round_num})
    # 推进轮次
    requests.post(f"{GO_BACKEND}/round/advance", json={})
```

---

## FedAvg 完整一轮时序（集中式解密）

```
时间轴   Flower Server                          Go Backend(:8082)
  │
  │  ──POST /system/init────────────────────►  初始化密钥(~1s)
  │  ◄──{slots:8192, max_level:5}────────────
  │
  │  [下发全局模型给各 Client]
  │
  │  Client_0 本地训练完成
  │  ──POST /encrypt {client_0, weights}────►  加密 → cipher_ids_0
  │  ◄──{cipher_ids:[...], chunk_count:1}────
  │
  │  Client_1 本地训练完成
  │  ──POST /encrypt {client_1, weights}────►  加密 → cipher_ids_1
  │  ◄──{cipher_ids:[...], chunk_count:1}────
  │
  │  Client_2 本地训练完成
  │  ──POST /encrypt {client_2, weights}────►  加密 → cipher_ids_2
  │  ◄──{cipher_ids:[...], chunk_count:1}────
  │
  │  构造 cipher_groups
  │  ──POST /aggregate {groups, avg=true}───►  同态加法 + ÷N
  │  ◄──{aggregated_cipher_ids:[...]}────────
  │
  │  ──POST /decrypt {agg_ids, weight_count}►  集中解密
  │  ◄──{weights:[...], length:N}────────────
  │
  │  [用 weights 更新全局模型]
  │
  │  ──POST /cipher/cleanup {round:1}───────►  释放密文内存
  │  ──POST /round/advance──────────────────►  轮次 1→2
  │
  │  [开始下一轮...]
```

---

## 关键约束和注意事项

| 项目 | 说明 |
|------|------|
| **client_id 规则** | 不能为 "0"（库保留字），建议用 "client_0"、"client_1" 格式 |
| **权重精度** | CKKS 是近似计算，解密后有 ~1e-10 级别误差，实际 FL 训练误差可忽略 |
| **单次编码上限** | weights 长度无上限，超过 slots 时自动分 chunk |
| **聚合 level 消耗** | 仅求和(average=false)不消耗 level；FedAvg(average=true)消耗 1 层 |
| **内存管理** | 每个密文约占 MB 级内存，每轮结束务必调用 cleanup 释放 |
| **并发安全** | 后端内部使用读写锁，支持多客户端并发加密，聚合/解密自动互斥 |
| **初始化耗时** | LogN=14 约 0.5 秒，LogN=15 约 2-3 秒（密钥生成） |
| **加密耗时** | 单次加密（≤slots 权重）约 50-100ms |
| **聚合耗时** | N 个客户端同态加法约 N×10ms，FedAvg 额外 +50ms（常数乘+rescale）|
| **状态重置** | 重新调用 /system/init 会清除所有密文和密钥，重新开始 |
