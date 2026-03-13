package main

// ================================================================
//  通用响应结构 — 所有接口统一返回格式
// ================================================================

type Response struct {
	Code    int         `json:"code"`           // 业务状态码 0=成功, 非0=失败
	Message string      `json:"message"`        // 描述信息
	Data    interface{} `json:"data,omitempty"` // 返回数据（成功时）
}

// ================================================================
//  POST /api/v1/system/init — 初始化 MKHE 加密系统
//  调用方: Flower Server（训练开始前调用一次）
// ================================================================

type InitRequest struct {
	LogN         int      `json:"logN"`                    // CKKS 参数等级: 14 (快速/测试) 或 15 (生产)
	ClientIDs    []string `json:"client_ids"`              // 参与方 ID 列表（首次注册）
	ExperimentID string   `json:"experiment_id,omitempty"` // 可选: 归属实验 ID
}

type InitResponse struct {
	Status          string  `json:"status"`           // "initialized"
	Slots           int     `json:"slots"`            // 每个密文可编码的槽位数
	MaxLevel        int     `json:"max_level"`        // 最大乘法深度
	Scale           float64 `json:"scale"`            // 默认缩放因子
	RegisteredCount int     `json:"registered_count"` // 已注册参与方数
}

// ================================================================
//  GET /api/v1/system/status — 获取系统状态
//  调用方: Flower Server / 管理员
// ================================================================

type SystemStatusResponse struct {
	Initialized   bool     `json:"initialized"`
	CurrentRound  int      `json:"current_round"`
	RegisteredIDs []string `json:"registered_ids"`
	CipherCount   int      `json:"cipher_count"`
	Slots         int      `json:"slots"`
	MaxLevel      int      `json:"max_level"`
}

// ================================================================
//  POST /api/v1/clients/register — 动态注册新的 FL 客户端
//  调用方: Flower Server（新客户端加入时）
// ================================================================

type RegisterClientRequest struct {
	ClientID     string `json:"client_id"`
	ExperimentID string `json:"experiment_id,omitempty"` // 可选: 归属实验 ID
}

type RegisterClientResponse struct {
	ClientID     string `json:"client_id"`
	HasPublicKey bool   `json:"has_public_key"`
	HasSecretKey bool   `json:"has_secret_key"`
}

// ================================================================
//  POST /api/v1/encrypt — 加密客户端模型参数
//  调用方: Flower Client（本地训练后上传权重）
//  说明: 大模型自动拆分为多个密文(chunk)
// ================================================================

type EncryptModelRequest struct {
	ClientID     string    `json:"client_id"`               // 客户端 ID
	Weights      []float64 `json:"weights"`                 // flatten 后的模型参数
	Round        int       `json:"round"`                   // 当前训练轮次
	LayerTag     string    `json:"layer_tag"`               // 层标签（可选，用于分层聚合）
	WeightCount  int       `json:"weight_count"`            // 原始权重数量（用于解密时截断填充）
	ExperimentID string    `json:"experiment_id,omitempty"` // 可选: 归属实验 ID
}

type EncryptModelResponse struct {
	CipherIDs  []string `json:"cipher_ids"`  // 密文 ID 列表（多chunk时有多个）
	ChunkCount int      `json:"chunk_count"` // chunk 数量
	Round      int      `json:"round"`
}

// ================================================================
//  POST /api/v1/aggregate — 同态聚合（FedAvg 核心）
//  调用方: Flower Server（收集完所有客户端密文后）
//  说明: cipher_groups[i] 是第 i 个 chunk 来自各参与方的密文 ID 列表
// ================================================================

type AggregateRequest struct {
	Round        int        `json:"round"`
	CipherGroups [][]string `json:"cipher_groups"`           // 按 chunk 分组, groups[i] = [client1_chunk_i, client2_chunk_i, ...]
	Average      bool       `json:"average"`                 // true=求均值(÷N), false=仅求和
	ClientCount  int        `json:"client_count"`            // 参与方数量（average=true 时必填）
	ExperimentID string     `json:"experiment_id,omitempty"` // 可选: 归属实验 ID
}

type AggregateResponse struct {
	AggregatedCipherIDs []string `json:"aggregated_cipher_ids"` // 聚合结果密文 ID 列表（每 chunk 一个）
	Round               int      `json:"round"`
}

// ================================================================
//  POST /api/v1/decrypt — 完整解密（集中式，需要所有密钥）
//  调用方: Flower Server（聚合后获取明文全局模型）
// ================================================================

type DecryptRequest struct {
	CipherIDs    []string `json:"cipher_ids"`              // 待解密密文 ID 列表（按 chunk 顺序）
	WeightCount  int      `json:"weight_count"`            // 原始权重数量（用于截断填充的零）
	ExperimentID string   `json:"experiment_id,omitempty"` // 可选: 归属实验 ID
}

type DecryptResponse struct {
	Weights []float64 `json:"weights"` // 解密后的模型参数
	Length  int       `json:"length"`  // 实际权重数量
}

// ================================================================
//  POST /api/v1/decrypt/partial — 部分解密（分布式安全协议）
//  调用方: 各 Flower Client 依次调用
//  说明: MKHE 分布式解密 — 每个参与方用自己的密钥部分解密
// ================================================================

type PartialDecryptRequest struct {
	CipherIDs    []string `json:"cipher_ids"`              // 待部分解密的密文 ID 列表
	ClientID     string   `json:"client_id"`               // 当前执行部分解密的客户端 ID
	ExperimentID string   `json:"experiment_id,omitempty"` // 可选: 归属实验 ID
}

type PartialDecryptResponse struct {
	CipherIDs        []string `json:"cipher_ids"`        // 部分解密后的密文 ID（新生成）
	RemainingParties int      `json:"remaining_parties"` // 还需要几方解密
}

// ================================================================
//  POST /api/v1/decrypt/final — 完成分布式解密（取回明文）
//  调用方: Flower Server（所有部分解密完成后）
// ================================================================

type FinalDecryptRequest struct {
	CipherIDs    []string `json:"cipher_ids"`
	WeightCount  int      `json:"weight_count"`
	ExperimentID string   `json:"experiment_id,omitempty"` // 可选: 归属实验 ID
}

// 复用 DecryptResponse

// ================================================================
//  GET    /api/v1/cipher/{cipher_id} — 查询密文元信息
//  DELETE /api/v1/cipher/{cipher_id} — 删除单个密文
//  调用方: Flower Server / 管理员
// ================================================================

type CipherInfoResponse struct {
	CipherID   string   `json:"cipher_id"`
	IDSet      []string `json:"id_set"` // 密文关联的参与方 ID
	Level      int      `json:"level"`  // 当前密文等级
	Scale      float64  `json:"scale"`  // 缩放因子
	Round      int      `json:"round"`
	ClientID   string   `json:"client_id"`
	LayerTag   string   `json:"layer_tag"`
	ChunkIndex int      `json:"chunk_index"`
}

// ================================================================
//  POST /api/v1/cipher/cleanup — 批量清理密文（释放内存）
//  调用方: Flower Server（每轮结束后清理旧密文）
// ================================================================

type CleanupRequest struct {
	Round        int    `json:"round"`                   // 清理指定轮次的密文, 0=全部清理
	ExperimentID string `json:"experiment_id,omitempty"` // 可选: 归属实验 ID
}

type CleanupResponse struct {
	DeletedCount int `json:"deleted_count"`
}

// ================================================================
//  POST /api/v1/round/advance — 推进到下一训练轮次
//  调用方: Flower Server
// ================================================================

type AdvanceRoundResponse struct {
	PreviousRound int `json:"previous_round"`
	CurrentRound  int `json:"current_round"`
}

// ================================================================
//  GET /api/v1/health — 健康检查
// ================================================================

type HealthResponse struct {
	Status string `json:"status"` // "ok"
	Uptime string `json:"uptime"` // 运行时长
}

// ================================================================
//  实验追踪 API
// ================================================================

type ExperimentStartRequest struct {
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Tags        map[string]string `json:"tags"`
	SetActive   bool              `json:"set_active"`
}

type ExperimentStartResponse struct {
	ExperimentID string `json:"experiment_id"`
	Name         string `json:"name"`
	StartedAt    string `json:"started_at"`
	LogDir       string `json:"log_dir"`
	Active       bool   `json:"active"`
}

type ExperimentStopRequest struct {
	ExperimentID string `json:"experiment_id"`
}

type ExperimentStopResponse struct {
	ExperimentID string  `json:"experiment_id"`
	StoppedAt    string  `json:"stopped_at"`
	DurationSec  float64 `json:"duration_sec"`
	TotalEvents  int     `json:"total_events"`
}

type ExperimentEventRequest struct {
	ExperimentID      string             `json:"experiment_id"`
	Round             int                `json:"round"`
	Operation         string             `json:"operation"`
	ClientID          string             `json:"client_id"`
	LayerTag          string             `json:"layer_tag"`
	Status            string             `json:"status"`
	DurationMS        float64            `json:"duration_ms"`
	InputCipherCount  int                `json:"input_cipher_count"`
	OutputCipherCount int                `json:"output_cipher_count"`
	WeightCount       int                `json:"weight_count"`
	Error             string             `json:"error"`
	Metrics           map[string]float64 `json:"metrics"`
	Metadata          map[string]string  `json:"metadata"`
}

type ExperimentEvent struct {
	EventID           string             `json:"event_id"`
	ExperimentID      string             `json:"experiment_id"`
	Timestamp         string             `json:"timestamp"`
	Round             int                `json:"round"`
	Operation         string             `json:"operation"`
	ClientID          string             `json:"client_id"`
	LayerTag          string             `json:"layer_tag"`
	Status            string             `json:"status"`
	DurationMS        float64            `json:"duration_ms"`
	InputCipherCount  int                `json:"input_cipher_count"`
	OutputCipherCount int                `json:"output_cipher_count"`
	WeightCount       int                `json:"weight_count"`
	Error             string             `json:"error,omitempty"`
	Metrics           map[string]float64 `json:"metrics,omitempty"`
	Metadata          map[string]string  `json:"metadata,omitempty"`
}

type ExperimentOperationMetric struct {
	Operation     string  `json:"operation"`
	Count         int     `json:"count"`
	SuccessCount  int     `json:"success_count"`
	FailureCount  int     `json:"failure_count"`
	AvgDurationMS float64 `json:"avg_duration_ms"`
	MaxDurationMS float64 `json:"max_duration_ms"`
}

type ExperimentRoundMetric struct {
	Round             int     `json:"round"`
	EventCount        int     `json:"event_count"`
	AvgDurationMS     float64 `json:"avg_duration_ms"`
	OutputCipherCount int     `json:"output_cipher_count"`
	WeightCount       int     `json:"weight_count"`
}

type ExperimentSummaryResponse struct {
	ExperimentID     string                      `json:"experiment_id"`
	Name             string                      `json:"name"`
	Description      string                      `json:"description"`
	Tags             map[string]string           `json:"tags,omitempty"`
	StartedAt        string                      `json:"started_at"`
	StoppedAt        string                      `json:"stopped_at,omitempty"`
	Active           bool                        `json:"active"`
	TotalEvents      int                         `json:"total_events"`
	SuccessEvents    int                         `json:"success_events"`
	FailureEvents    int                         `json:"failure_events"`
	AvgDurationMS    float64                     `json:"avg_duration_ms"`
	OperationMetrics []ExperimentOperationMetric `json:"operation_metrics"`
	RoundMetrics     []ExperimentRoundMetric     `json:"round_metrics"`
	EventLogPath     string                      `json:"event_log_path"`
	CSVExportPath    string                      `json:"csv_export_path,omitempty"`
}

type ExperimentListItem struct {
	ExperimentID string `json:"experiment_id"`
	Name         string `json:"name"`
	StartedAt    string `json:"started_at"`
	StoppedAt    string `json:"stopped_at,omitempty"`
	Active       bool   `json:"active"`
	TotalEvents  int    `json:"total_events"`
}

type ExperimentEventsResponse struct {
	ExperimentID string            `json:"experiment_id"`
	Total        int               `json:"total"`
	Limit        int               `json:"limit"`
	Events       []ExperimentEvent `json:"events"`
}

type MetricPoint struct {
	Round int     `json:"round"`
	Value float64 `json:"value"`
}

type PlotSeries struct {
	Name   string        `json:"name"`
	Points []MetricPoint `json:"points"`
}

type ExperimentPlotDataResponse struct {
	ExperimentID          string         `json:"experiment_id"`
	LatencyByOperation    []PlotSeries   `json:"latency_by_operation"`
	OutputCipherByRound   []MetricPoint  `json:"output_cipher_by_round"`
	ThroughputByRound     []MetricPoint  `json:"throughput_by_round"`
	EventCountByOperation map[string]int `json:"event_count_by_operation"`
}
