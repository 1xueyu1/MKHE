package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"
)

// 全局 HE 服务实例
var service *HEService

// 全局实验追踪器
var experimentTracker *ExperimentTracker

// 服务启动时间（健康检查用）
var startTime time.Time

// ================================================================
//  工具函数
// ================================================================

func writeJSON(w http.ResponseWriter, httpCode int, resp Response) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(httpCode)
	_ = json.NewEncoder(w).Encode(resp)
}

func success(w http.ResponseWriter, msg string, data interface{}) {
	writeJSON(w, http.StatusOK, Response{Code: 0, Message: msg, Data: data})
}

func fail(w http.ResponseWriter, httpCode int, msg string) {
	writeJSON(w, httpCode, Response{Code: 1, Message: msg})
}

func totalCipherInputs(groups [][]string) int {
	total := 0
	for _, group := range groups {
		total += len(group)
	}
	return total
}

// ================================================================
//  CORS 中间件（Python Flower 调用需要跨域）
// ================================================================

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// ================================================================
//  日志中间件
// ================================================================

func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("[%s] %s %s (%v)", r.Method, r.URL.Path, r.RemoteAddr, time.Since(start))
	})
}

// ================================================================
//  Handler: POST /api/v1/system/init
// ================================================================

func initHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		fail(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var req InitRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		fail(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	if len(req.ClientIDs) == 0 {
		fail(w, http.StatusBadRequest, "client_ids 不能为空")
		return
	}

	start := time.Now()
	resp, err := service.Init(req.LogN, req.ClientIDs)
	if err != nil {
		recordServiceEvent(r, req.ExperimentID, "system_init", start, 0, "", "", 0, 0, 0, err)
		fail(w, http.StatusInternalServerError, err.Error())
		return
	}
	recordServiceEvent(r, req.ExperimentID, "system_init", start, 0, "", "", 0, 0, 0, nil)

	success(w, "MKHE 系统初始化成功", resp)
}

// ================================================================
//  Handler: GET /api/v1/system/status
// ================================================================

func statusHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		fail(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	success(w, "ok", service.GetStatus())
}

// ================================================================
//  Handler: GET /api/v1/health
// ================================================================

func healthHandler(w http.ResponseWriter, r *http.Request) {
	success(w, "ok", HealthResponse{
		Status: "ok",
		Uptime: time.Since(startTime).String(),
	})
}

// ================================================================
//  Handler: POST /api/v1/clients/register
// ================================================================

func registerClientHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		fail(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var req RegisterClientRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		fail(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	start := time.Now()
	resp, err := service.RegisterClient(req.ClientID)
	if err != nil {
		recordServiceEvent(r, req.ExperimentID, "client_register", start, 0, req.ClientID, "", 0, 0, 0, err)
		fail(w, http.StatusBadRequest, err.Error())
		return
	}
	recordServiceEvent(r, req.ExperimentID, "client_register", start, 0, req.ClientID, "", 0, 0, 0, nil)

	success(w, "客户端注册成功", resp)
}

// ================================================================
//  Handler: POST /api/v1/encrypt
// ================================================================

func encryptModelHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		fail(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var req EncryptModelRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		fail(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	if req.ClientID == "" || len(req.Weights) == 0 {
		fail(w, http.StatusBadRequest, "client_id 和 weights 为必填项")
		return
	}

	start := time.Now()
	resp, err := service.EncryptModel(req.ClientID, req.Weights, req.Round, req.LayerTag)
	if err != nil {
		recordServiceEvent(r, req.ExperimentID, "encrypt", start, req.Round, req.ClientID, req.LayerTag, 0, 0, len(req.Weights), err)
		fail(w, http.StatusInternalServerError, err.Error())
		return
	}
	recordServiceEvent(r, req.ExperimentID, "encrypt", start, req.Round, req.ClientID, req.LayerTag, 0, len(resp.CipherIDs), len(req.Weights), nil)

	success(w, "模型加密完成", resp)
}

// ================================================================
//  Handler: POST /api/v1/aggregate
// ================================================================

func aggregateHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		fail(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var req AggregateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		fail(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	if len(req.CipherGroups) == 0 {
		fail(w, http.StatusBadRequest, "cipher_groups 不能为空")
		return
	}

	start := time.Now()
	resp, err := service.Aggregate(req.CipherGroups, req.Round, req.Average, req.ClientCount)
	if err != nil {
		recordServiceEvent(r, req.ExperimentID, "aggregate", start, req.Round, "", "", totalCipherInputs(req.CipherGroups), 0, 0, err)
		fail(w, http.StatusInternalServerError, err.Error())
		return
	}
	recordServiceEvent(r, req.ExperimentID, "aggregate", start, req.Round, "", "", totalCipherInputs(req.CipherGroups), len(resp.AggregatedCipherIDs), 0, nil)

	success(w, "聚合完成", resp)
}

// ================================================================
//  Handler: POST /api/v1/decrypt
// ================================================================

func decryptHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		fail(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var req DecryptRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		fail(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	start := time.Now()
	resp, err := service.Decrypt(req.CipherIDs, req.WeightCount)
	if err != nil {
		recordServiceEvent(r, req.ExperimentID, "decrypt", start, 0, "", "", len(req.CipherIDs), 0, req.WeightCount, err)
		fail(w, http.StatusInternalServerError, err.Error())
		return
	}
	recordServiceEvent(r, req.ExperimentID, "decrypt", start, 0, "", "", len(req.CipherIDs), 0, resp.Length, nil)

	success(w, "解密成功", resp)
}

// ================================================================
//  Handler: POST /api/v1/decrypt/partial
// ================================================================

func partialDecryptHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		fail(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var req PartialDecryptRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		fail(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	start := time.Now()
	resp, err := service.PartialDecrypt(req.CipherIDs, req.ClientID)
	if err != nil {
		recordServiceEvent(r, req.ExperimentID, "partial_decrypt", start, 0, req.ClientID, "", len(req.CipherIDs), 0, 0, err)
		fail(w, http.StatusInternalServerError, err.Error())
		return
	}
	recordServiceEvent(r, req.ExperimentID, "partial_decrypt", start, 0, req.ClientID, "", len(req.CipherIDs), len(resp.CipherIDs), 0, nil)

	success(w, "部分解密完成", resp)
}

// ================================================================
//  Handler: POST /api/v1/decrypt/final
// ================================================================

func finalDecryptHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		fail(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var req FinalDecryptRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		fail(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	start := time.Now()
	resp, err := service.FinalDecrypt(req.CipherIDs, req.WeightCount)
	if err != nil {
		recordServiceEvent(r, req.ExperimentID, "final_decrypt", start, 0, "", "", len(req.CipherIDs), 0, req.WeightCount, err)
		fail(w, http.StatusInternalServerError, err.Error())
		return
	}
	recordServiceEvent(r, req.ExperimentID, "final_decrypt", start, 0, "", "", len(req.CipherIDs), 0, resp.Length, nil)

	success(w, "最终解密成功", resp)
}

// ================================================================
//  Handler: GET/DELETE /api/v1/cipher/{cipher_id}
// ================================================================

func cipherHandler(w http.ResponseWriter, r *http.Request) {
	// 从路径中提取 cipher_id: /api/v1/cipher/{cipher_id}
	parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/v1/cipher/"), "/")
	if len(parts) == 0 || parts[0] == "" {
		fail(w, http.StatusBadRequest, "缺少 cipher_id")
		return
	}
	cipherID := parts[0]

	switch r.Method {
	case http.MethodGet:
		resp, err := service.GetCipherInfo(cipherID)
		if err != nil {
			fail(w, http.StatusNotFound, err.Error())
			return
		}
		success(w, "ok", resp)

	case http.MethodDelete:
		if err := service.DeleteCipher(cipherID); err != nil {
			fail(w, http.StatusNotFound, err.Error())
			return
		}
		success(w, "密文已删除", nil)

	default:
		fail(w, http.StatusMethodNotAllowed, "method not allowed")
	}
}

// ================================================================
//  Handler: POST /api/v1/cipher/cleanup
// ================================================================

func cleanupHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		fail(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var req CleanupRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		fail(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	start := time.Now()
	resp := service.Cleanup(req.Round)
	recordServiceEvent(r, req.ExperimentID, "cipher_cleanup", start, req.Round, "", "", 0, 0, resp.DeletedCount, nil)
	success(w, "清理完成", resp)
}

// ================================================================
//  Handler: POST /api/v1/round/advance
// ================================================================

func advanceRoundHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		fail(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	start := time.Now()
	resp := service.AdvanceRound()
	recordServiceEvent(r, "", "round_advance", start, resp.CurrentRound, "", "", 0, 0, 0, nil)
	success(w, "轮次已推进", resp)
}

// ================================================================
//  主函数
// ================================================================

func main() {
	startTime = time.Now()
	service = NewHEService()

	tracker, err := NewExperimentTracker(strings.TrimSpace(os.Getenv("MKHE_EXPERIMENT_DIR")))
	if err != nil {
		log.Fatalf("experiment tracker init failed: %v", err)
	}
	experimentTracker = tracker

	mux := http.NewServeMux()

	// ---- 系统管理 ----
	mux.HandleFunc("/api/v1/system/init", initHandler)
	mux.HandleFunc("/api/v1/system/status", statusHandler)
	mux.HandleFunc("/api/v1/health", healthHandler)

	// ---- 客户端管理 ----
	mux.HandleFunc("/api/v1/clients/register", registerClientHandler)

	// ---- 加密 / 聚合 / 解密（核心链路） ----
	mux.HandleFunc("/api/v1/encrypt", encryptModelHandler)
	mux.HandleFunc("/api/v1/aggregate", aggregateHandler)
	mux.HandleFunc("/api/v1/decrypt", decryptHandler)

	// ---- 分布式解密协议 ----
	mux.HandleFunc("/api/v1/decrypt/partial", partialDecryptHandler)
	mux.HandleFunc("/api/v1/decrypt/final", finalDecryptHandler)

	// ---- 密文管理 ----
	mux.HandleFunc("/api/v1/cipher/cleanup", cleanupHandler)
	mux.HandleFunc("/api/v1/cipher/", cipherHandler) // 通配: /api/v1/cipher/{id}

	// ---- 轮次管理 ----
	mux.HandleFunc("/api/v1/round/advance", advanceRoundHandler)

	// ---- 实验追踪 ----
	mux.HandleFunc("/api/v1/experiments/start", experimentsStartHandler)
	mux.HandleFunc("/api/v1/experiments/stop", experimentsStopHandler)
	mux.HandleFunc("/api/v1/experiments/active", experimentsActiveHandler)
	mux.HandleFunc("/api/v1/experiments/event", experimentsManualEventHandler)
	mux.HandleFunc("/api/v1/experiments", experimentsListHandler)
	mux.HandleFunc("/api/v1/experiments/", experimentsDetailHandler)

	// 组装中间件
	handler := loggingMiddleware(corsMiddleware(mux))

	server := &http.Server{
		Addr:         ":8082",
		Handler:      handler,
		ReadTimeout:  60 * time.Second,
		WriteTimeout: 120 * time.Second,
	}

	// 优雅退出
	go func() {
		log.Println("========================================")
		log.Println("  Federated MKCKKS Backend  :8082")
		log.Println("========================================")
		log.Println("API Base: http://localhost:8082/api/v1")
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("listen error: %v\n", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("Shutting down server...")
}
