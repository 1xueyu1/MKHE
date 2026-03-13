package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const experimentIDHeader = "X-Experiment-ID"

type operationAccumulator struct {
	Count         int
	SuccessCount  int
	FailureCount  int
	TotalDuration float64
	MaxDuration   float64
}

type roundAccumulator struct {
	EventCount        int
	TotalDuration     float64
	OutputCipherCount int
	WeightCount       int
}

type experimentRun struct {
	ID            string
	Name          string
	Description   string
	Tags          map[string]string
	StartedAt     time.Time
	StoppedAt     time.Time
	HasStopped    bool
	EventLogPath  string
	CSVExportPath string
	Events        []ExperimentEvent
	TotalEvents   int
	SuccessEvents int
	FailureEvents int
	TotalDuration float64
	OperationAcc  map[string]*operationAccumulator
	RoundAcc      map[int]*roundAccumulator
}

type experimentMetaFile struct {
	ExperimentID string            `json:"experiment_id"`
	Name         string            `json:"name"`
	Description  string            `json:"description"`
	Tags         map[string]string `json:"tags,omitempty"`
	StartedAt    string            `json:"started_at"`
	StoppedAt    string            `json:"stopped_at,omitempty"`
	Active       bool              `json:"active"`
}

type ExperimentTracker struct {
	mu          sync.RWMutex
	baseDir     string
	experiments map[string]*experimentRun
	activeID    string
}

func NewExperimentTracker(baseDir string) (*ExperimentTracker, error) {
	if strings.TrimSpace(baseDir) == "" {
		baseDir = "./experiment_logs"
	}
	if err := os.MkdirAll(baseDir, 0o755); err != nil {
		return nil, fmt.Errorf("创建实验日志目录失败: %w", err)
	}

	tracker := &ExperimentTracker{
		baseDir:     baseDir,
		experiments: make(map[string]*experimentRun),
	}

	if err := tracker.loadExisting(); err != nil {
		return nil, err
	}

	return tracker, nil
}

func (t *ExperimentTracker) Start(req ExperimentStartRequest) (*ExperimentStartResponse, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	now := time.Now()
	name := strings.TrimSpace(req.Name)
	if name == "" {
		name = "mkhe-exp-" + now.Format("20060102-150405")
	}

	experimentID := fmt.Sprintf("exp-%s-%s", now.Format("20060102-150405"), generateID()[:8])
	dir := filepath.Join(t.baseDir, experimentID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("创建实验目录失败: %w", err)
	}

	run := &experimentRun{
		ID:           experimentID,
		Name:         name,
		Description:  strings.TrimSpace(req.Description),
		Tags:         cloneStringMap(req.Tags),
		StartedAt:    now,
		EventLogPath: filepath.Join(dir, "events.jsonl"),
		OperationAcc: make(map[string]*operationAccumulator),
		RoundAcc:     make(map[int]*roundAccumulator),
	}

	t.experiments[experimentID] = run
	setActive := req.SetActive || t.activeID == ""
	if setActive {
		t.activeID = experimentID
	}

	if err := t.persistMetaLocked(run); err != nil {
		return nil, err
	}
	if err := t.persistSummaryLocked(run); err != nil {
		return nil, err
	}

	return &ExperimentStartResponse{
		ExperimentID: experimentID,
		Name:         run.Name,
		StartedAt:    run.StartedAt.Format(time.RFC3339Nano),
		LogDir:       dir,
		Active:       t.activeID == experimentID,
	}, nil
}

func (t *ExperimentTracker) Stop(experimentID string) (*ExperimentStopResponse, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	experimentID = strings.TrimSpace(experimentID)
	if experimentID == "" {
		experimentID = t.activeID
	}
	if experimentID == "" {
		return nil, fmt.Errorf("没有正在运行的实验")
	}

	run, ok := t.experiments[experimentID]
	if !ok {
		return nil, fmt.Errorf("实验 %s 不存在", experimentID)
	}

	if !run.HasStopped {
		run.HasStopped = true
		run.StoppedAt = time.Now()
	}
	if t.activeID == experimentID {
		t.activeID = ""
	}

	if err := t.persistMetaLocked(run); err != nil {
		return nil, err
	}
	if err := t.persistSummaryLocked(run); err != nil {
		return nil, err
	}

	durationSec := run.StoppedAt.Sub(run.StartedAt).Seconds()
	if durationSec < 0 {
		durationSec = 0
	}

	return &ExperimentStopResponse{
		ExperimentID: experimentID,
		StoppedAt:    run.StoppedAt.Format(time.RFC3339Nano),
		DurationSec:  durationSec,
		TotalEvents:  run.TotalEvents,
	}, nil
}

func (t *ExperimentTracker) ActiveExperimentID() string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.activeID
}

func (t *ExperimentTracker) List() []ExperimentListItem {
	t.mu.RLock()
	defer t.mu.RUnlock()

	items := make([]ExperimentListItem, 0, len(t.experiments))
	for _, run := range t.experiments {
		item := ExperimentListItem{
			ExperimentID: run.ID,
			Name:         run.Name,
			StartedAt:    run.StartedAt.Format(time.RFC3339Nano),
			Active:       t.activeID == run.ID,
			TotalEvents:  run.TotalEvents,
		}
		if run.HasStopped {
			item.StoppedAt = run.StoppedAt.Format(time.RFC3339Nano)
		}
		items = append(items, item)
	}

	sort.Slice(items, func(i, j int) bool {
		return items[i].StartedAt > items[j].StartedAt
	})
	return items
}

func (t *ExperimentTracker) Summary(experimentID string) (*ExperimentSummaryResponse, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	run, ok := t.experiments[experimentID]
	if !ok {
		return nil, fmt.Errorf("实验 %s 不存在", experimentID)
	}
	summary := t.buildSummaryLocked(run)
	return &summary, nil
}

func (t *ExperimentTracker) GetEvents(experimentID string, limit int) ([]ExperimentEvent, int, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	run, ok := t.experiments[experimentID]
	if !ok {
		return nil, 0, fmt.Errorf("实验 %s 不存在", experimentID)
	}

	total := len(run.Events)
	if limit <= 0 || limit > total {
		limit = total
	}
	start := total - limit
	if start < 0 {
		start = 0
	}

	result := make([]ExperimentEvent, 0, limit)
	for i := total - 1; i >= start; i-- {
		result = append(result, cloneEvent(run.Events[i]))
	}
	return result, total, nil
}

func (t *ExperimentTracker) RecordEvent(experimentID string, event ExperimentEvent) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	experimentID = strings.TrimSpace(experimentID)
	if experimentID == "" {
		experimentID = t.activeID
	}
	if experimentID == "" {
		return fmt.Errorf("未提供 experiment_id 且当前没有激活实验")
	}

	run, ok := t.experiments[experimentID]
	if !ok {
		return fmt.Errorf("实验 %s 不存在", experimentID)
	}

	now := time.Now()
	if strings.TrimSpace(event.EventID) == "" {
		event.EventID = generateID()
	}
	if strings.TrimSpace(event.Timestamp) == "" {
		event.Timestamp = now.Format(time.RFC3339Nano)
	}
	event.ExperimentID = experimentID
	event.Operation = strings.TrimSpace(event.Operation)
	if event.Operation == "" {
		event.Operation = "custom"
	}
	event.Status = strings.TrimSpace(strings.ToLower(event.Status))
	if event.Status == "" {
		event.Status = "success"
	}
	event.ClientID = strings.TrimSpace(event.ClientID)
	event.LayerTag = strings.TrimSpace(event.LayerTag)
	event.Metadata = cloneStringMap(event.Metadata)
	event.Metrics = cloneFloatMap(event.Metrics)

	run.Events = append(run.Events, event)
	t.applyEventLocked(run, event)

	if err := t.appendEventLocked(run, event); err != nil {
		return err
	}
	if err := t.persistSummaryLocked(run); err != nil {
		return err
	}

	return nil
}

func (t *ExperimentTracker) ExportEventsCSV(experimentID string) (string, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	run, ok := t.experiments[experimentID]
	if !ok {
		return "", fmt.Errorf("实验 %s 不存在", experimentID)
	}

	csvPath := filepath.Join(t.baseDir, experimentID, "events.csv")
	fp, err := os.Create(csvPath)
	if err != nil {
		return "", fmt.Errorf("创建 CSV 失败: %w", err)
	}
	defer fp.Close()

	writer := csv.NewWriter(fp)
	header := []string{
		"event_id",
		"experiment_id",
		"timestamp",
		"round",
		"operation",
		"client_id",
		"layer_tag",
		"status",
		"duration_ms",
		"input_cipher_count",
		"output_cipher_count",
		"weight_count",
		"error",
		"metrics_json",
		"metadata_json",
	}
	if err := writer.Write(header); err != nil {
		return "", err
	}

	for _, e := range run.Events {
		metricsJSON, _ := json.Marshal(e.Metrics)
		metaJSON, _ := json.Marshal(e.Metadata)
		row := []string{
			e.EventID,
			e.ExperimentID,
			e.Timestamp,
			strconv.Itoa(e.Round),
			e.Operation,
			e.ClientID,
			e.LayerTag,
			e.Status,
			fmt.Sprintf("%.6f", e.DurationMS),
			strconv.Itoa(e.InputCipherCount),
			strconv.Itoa(e.OutputCipherCount),
			strconv.Itoa(e.WeightCount),
			e.Error,
			string(metricsJSON),
			string(metaJSON),
		}
		if err := writer.Write(row); err != nil {
			return "", err
		}
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		return "", err
	}

	run.CSVExportPath = csvPath
	if err := t.persistSummaryLocked(run); err != nil {
		return "", err
	}

	return csvPath, nil
}

func (t *ExperimentTracker) BuildPlotData(experimentID string) (*ExperimentPlotDataResponse, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	run, ok := t.experiments[experimentID]
	if !ok {
		return nil, fmt.Errorf("实验 %s 不存在", experimentID)
	}

	type avgStat struct {
		Count int
		Sum   float64
	}
	latency := make(map[string]map[int]*avgStat)
	cipherByRound := make(map[int]float64)
	weightByRound := make(map[int]float64)
	durationByRound := make(map[int]float64)
	eventCountByOperation := make(map[string]int)

	for _, e := range run.Events {
		round := e.Round
		if round < 0 {
			round = 0
		}

		eventCountByOperation[e.Operation]++

		opMap, ok := latency[e.Operation]
		if !ok {
			opMap = make(map[int]*avgStat)
			latency[e.Operation] = opMap
		}
		ls, ok := opMap[round]
		if !ok {
			ls = &avgStat{}
			opMap[round] = ls
		}
		ls.Count++
		ls.Sum += e.DurationMS

		cipherByRound[round] += float64(e.OutputCipherCount)
		weightByRound[round] += float64(e.WeightCount)
		durationByRound[round] += e.DurationMS
	}

	latencySeries := make([]PlotSeries, 0, len(latency))
	for op, byRound := range latency {
		rounds := make([]int, 0, len(byRound))
		for r := range byRound {
			rounds = append(rounds, r)
		}
		sort.Ints(rounds)

		series := PlotSeries{Name: op, Points: make([]MetricPoint, 0, len(rounds))}
		for _, r := range rounds {
			stat := byRound[r]
			avg := 0.0
			if stat.Count > 0 {
				avg = stat.Sum / float64(stat.Count)
			}
			series.Points = append(series.Points, MetricPoint{Round: r, Value: avg})
		}
		latencySeries = append(latencySeries, series)
	}
	sort.Slice(latencySeries, func(i, j int) bool { return latencySeries[i].Name < latencySeries[j].Name })

	rounds := make([]int, 0, len(cipherByRound))
	for r := range cipherByRound {
		rounds = append(rounds, r)
	}
	sort.Ints(rounds)

	cipherPoints := make([]MetricPoint, 0, len(rounds))
	throughputPoints := make([]MetricPoint, 0, len(rounds))
	for _, r := range rounds {
		cipherPoints = append(cipherPoints, MetricPoint{Round: r, Value: cipherByRound[r]})
		throughput := 0.0
		if durationByRound[r] > 0 {
			throughput = (weightByRound[r] / durationByRound[r]) * 1000.0
		}
		if math.IsNaN(throughput) || math.IsInf(throughput, 0) {
			throughput = 0
		}
		throughputPoints = append(throughputPoints, MetricPoint{Round: r, Value: throughput})
	}

	resp := &ExperimentPlotDataResponse{
		ExperimentID:          experimentID,
		LatencyByOperation:    latencySeries,
		OutputCipherByRound:   cipherPoints,
		ThroughputByRound:     throughputPoints,
		EventCountByOperation: eventCountByOperation,
	}

	return resp, nil
}

func (t *ExperimentTracker) loadExisting() error {
	entries, err := ioutil.ReadDir(t.baseDir)
	if err != nil {
		return fmt.Errorf("读取实验目录失败: %w", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		expID := entry.Name()
		dir := filepath.Join(t.baseDir, expID)
		metaPath := filepath.Join(dir, "meta.json")
		metaBytes, err := ioutil.ReadFile(metaPath)
		if err != nil {
			continue
		}

		var meta experimentMetaFile
		if err := json.Unmarshal(metaBytes, &meta); err != nil {
			continue
		}

		startedAt := parseTime(meta.StartedAt)
		if startedAt.IsZero() {
			startedAt = entry.ModTime()
		}

		run := &experimentRun{
			ID:           meta.ExperimentID,
			Name:         meta.Name,
			Description:  meta.Description,
			Tags:         cloneStringMap(meta.Tags),
			StartedAt:    startedAt,
			EventLogPath: filepath.Join(dir, "events.jsonl"),
			OperationAcc: make(map[string]*operationAccumulator),
			RoundAcc:     make(map[int]*roundAccumulator),
		}
		if run.ID == "" {
			run.ID = expID
		}
		if ts := parseTime(meta.StoppedAt); !ts.IsZero() {
			run.HasStopped = true
			run.StoppedAt = ts
		}

		if err := t.loadEventsLocked(run); err != nil {
			log.Printf("加载实验事件失败: %s (%v)", run.ID, err)
		}

		t.experiments[run.ID] = run
		if meta.Active && t.activeID == "" && !run.HasStopped {
			t.activeID = run.ID
		}
	}

	if t.activeID == "" {
		var latest *experimentRun
		for _, run := range t.experiments {
			if run.HasStopped {
				continue
			}
			if latest == nil || run.StartedAt.After(latest.StartedAt) {
				latest = run
			}
		}
		if latest != nil {
			t.activeID = latest.ID
		}
	}

	return nil
}

func (t *ExperimentTracker) loadEventsLocked(run *experimentRun) error {
	fp, err := os.Open(run.EventLogPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer fp.Close()

	scanner := bufio.NewScanner(fp)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var event ExperimentEvent
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			continue
		}
		run.Events = append(run.Events, cloneEvent(event))
		t.applyEventLocked(run, event)
	}
	return scanner.Err()
}

func (t *ExperimentTracker) appendEventLocked(run *experimentRun, event ExperimentEvent) error {
	if err := os.MkdirAll(filepath.Dir(run.EventLogPath), 0o755); err != nil {
		return err
	}
	fp, err := os.OpenFile(run.EventLogPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer fp.Close()

	payload, err := json.Marshal(event)
	if err != nil {
		return err
	}
	if _, err := fp.Write(append(payload, '\n')); err != nil {
		return err
	}
	return nil
}

func (t *ExperimentTracker) applyEventLocked(run *experimentRun, event ExperimentEvent) {
	run.TotalEvents++
	run.TotalDuration += event.DurationMS
	if event.Status == "failed" {
		run.FailureEvents++
	} else {
		run.SuccessEvents++
	}

	op := event.Operation
	if op == "" {
		op = "custom"
	}
	acc, ok := run.OperationAcc[op]
	if !ok {
		acc = &operationAccumulator{}
		run.OperationAcc[op] = acc
	}
	acc.Count++
	if event.Status == "failed" {
		acc.FailureCount++
	} else {
		acc.SuccessCount++
	}
	acc.TotalDuration += event.DurationMS
	if event.DurationMS > acc.MaxDuration {
		acc.MaxDuration = event.DurationMS
	}

	round := event.Round
	if round < 0 {
		round = 0
	}
	rAcc, ok := run.RoundAcc[round]
	if !ok {
		rAcc = &roundAccumulator{}
		run.RoundAcc[round] = rAcc
	}
	rAcc.EventCount++
	rAcc.TotalDuration += event.DurationMS
	rAcc.OutputCipherCount += event.OutputCipherCount
	rAcc.WeightCount += event.WeightCount
}

func (t *ExperimentTracker) buildSummaryLocked(run *experimentRun) ExperimentSummaryResponse {
	summary := ExperimentSummaryResponse{
		ExperimentID:  run.ID,
		Name:          run.Name,
		Description:   run.Description,
		Tags:          cloneStringMap(run.Tags),
		StartedAt:     run.StartedAt.Format(time.RFC3339Nano),
		Active:        t.activeID == run.ID,
		TotalEvents:   run.TotalEvents,
		SuccessEvents: run.SuccessEvents,
		FailureEvents: run.FailureEvents,
		EventLogPath:  run.EventLogPath,
		CSVExportPath: run.CSVExportPath,
	}
	if run.HasStopped {
		summary.StoppedAt = run.StoppedAt.Format(time.RFC3339Nano)
	}
	if run.TotalEvents > 0 {
		summary.AvgDurationMS = run.TotalDuration / float64(run.TotalEvents)
	}

	opKeys := make([]string, 0, len(run.OperationAcc))
	for op := range run.OperationAcc {
		opKeys = append(opKeys, op)
	}
	sort.Strings(opKeys)
	for _, op := range opKeys {
		acc := run.OperationAcc[op]
		avg := 0.0
		if acc.Count > 0 {
			avg = acc.TotalDuration / float64(acc.Count)
		}
		summary.OperationMetrics = append(summary.OperationMetrics, ExperimentOperationMetric{
			Operation:     op,
			Count:         acc.Count,
			SuccessCount:  acc.SuccessCount,
			FailureCount:  acc.FailureCount,
			AvgDurationMS: avg,
			MaxDurationMS: acc.MaxDuration,
		})
	}

	rounds := make([]int, 0, len(run.RoundAcc))
	for round := range run.RoundAcc {
		rounds = append(rounds, round)
	}
	sort.Ints(rounds)
	for _, round := range rounds {
		acc := run.RoundAcc[round]
		avg := 0.0
		if acc.EventCount > 0 {
			avg = acc.TotalDuration / float64(acc.EventCount)
		}
		summary.RoundMetrics = append(summary.RoundMetrics, ExperimentRoundMetric{
			Round:             round,
			EventCount:        acc.EventCount,
			AvgDurationMS:     avg,
			OutputCipherCount: acc.OutputCipherCount,
			WeightCount:       acc.WeightCount,
		})
	}

	return summary
}

func (t *ExperimentTracker) persistMetaLocked(run *experimentRun) error {
	meta := experimentMetaFile{
		ExperimentID: run.ID,
		Name:         run.Name,
		Description:  run.Description,
		Tags:         cloneStringMap(run.Tags),
		StartedAt:    run.StartedAt.Format(time.RFC3339Nano),
		Active:       t.activeID == run.ID,
	}
	if run.HasStopped {
		meta.StoppedAt = run.StoppedAt.Format(time.RFC3339Nano)
	}

	payload, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return err
	}
	metaPath := filepath.Join(t.baseDir, run.ID, "meta.json")
	return ioutil.WriteFile(metaPath, payload, 0o644)
}

func (t *ExperimentTracker) persistSummaryLocked(run *experimentRun) error {
	summary := t.buildSummaryLocked(run)
	payload, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		return err
	}
	summaryPath := filepath.Join(t.baseDir, run.ID, "summary.json")
	return ioutil.WriteFile(summaryPath, payload, 0o644)
}

func resolveExperimentID(r *http.Request, requestExperimentID string) string {
	if v := strings.TrimSpace(requestExperimentID); v != "" {
		return v
	}
	if r != nil {
		if v := strings.TrimSpace(r.Header.Get(experimentIDHeader)); v != "" {
			return v
		}
		if v := strings.TrimSpace(r.URL.Query().Get("experiment_id")); v != "" {
			return v
		}
	}
	if experimentTracker != nil {
		return experimentTracker.ActiveExperimentID()
	}
	return ""
}

func recordServiceEvent(r *http.Request, requestExperimentID, operation string, start time.Time, round int, clientID, layerTag string, inputCipherCount, outputCipherCount, weightCount int, opErr error) {
	if experimentTracker == nil {
		return
	}
	experimentID := resolveExperimentID(r, requestExperimentID)
	if experimentID == "" {
		return
	}

	status := "success"
	errMsg := ""
	if opErr != nil {
		status = "failed"
		errMsg = opErr.Error()
	}

	event := ExperimentEvent{
		ExperimentID:      experimentID,
		Timestamp:         time.Now().Format(time.RFC3339Nano),
		Round:             round,
		Operation:         operation,
		ClientID:          clientID,
		LayerTag:          layerTag,
		Status:            status,
		DurationMS:        time.Since(start).Seconds() * 1000.0,
		InputCipherCount:  inputCipherCount,
		OutputCipherCount: outputCipherCount,
		WeightCount:       weightCount,
		Error:             errMsg,
	}

	if err := experimentTracker.RecordEvent(experimentID, event); err != nil {
		log.Printf("record experiment event failed: %v", err)
	}
}

func experimentsStartHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		fail(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if experimentTracker == nil {
		fail(w, http.StatusInternalServerError, "experiment tracker not initialized")
		return
	}

	var req ExperimentStartRequest
	if err := decodeJSONBody(r, &req); err != nil {
		fail(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	resp, err := experimentTracker.Start(req)
	if err != nil {
		fail(w, http.StatusInternalServerError, err.Error())
		return
	}

	success(w, "实验已启动", resp)
}

func experimentsStopHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		fail(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if experimentTracker == nil {
		fail(w, http.StatusInternalServerError, "experiment tracker not initialized")
		return
	}

	var req ExperimentStopRequest
	if err := decodeJSONBody(r, &req); err != nil {
		fail(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	experimentID := resolveExperimentID(r, req.ExperimentID)
	if experimentID == "" {
		fail(w, http.StatusBadRequest, "缺少 experiment_id 且无激活实验")
		return
	}

	resp, err := experimentTracker.Stop(experimentID)
	if err != nil {
		fail(w, http.StatusBadRequest, err.Error())
		return
	}

	success(w, "实验已停止", resp)
}

func experimentsListHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		fail(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if experimentTracker == nil {
		fail(w, http.StatusInternalServerError, "experiment tracker not initialized")
		return
	}
	success(w, "ok", experimentTracker.List())
}

func experimentsActiveHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		fail(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if experimentTracker == nil {
		fail(w, http.StatusInternalServerError, "experiment tracker not initialized")
		return
	}

	activeID := experimentTracker.ActiveExperimentID()
	if activeID == "" {
		success(w, "ok", map[string]interface{}{"active": false})
		return
	}

	summary, err := experimentTracker.Summary(activeID)
	if err != nil {
		fail(w, http.StatusInternalServerError, err.Error())
		return
	}
	success(w, "ok", summary)
}

func experimentsManualEventHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		fail(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if experimentTracker == nil {
		fail(w, http.StatusInternalServerError, "experiment tracker not initialized")
		return
	}

	var req ExperimentEventRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		fail(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	experimentID := resolveExperimentID(r, req.ExperimentID)
	if experimentID == "" {
		fail(w, http.StatusBadRequest, "缺少 experiment_id 且无激活实验")
		return
	}

	event := ExperimentEvent{
		Round:             req.Round,
		Operation:         req.Operation,
		ClientID:          req.ClientID,
		LayerTag:          req.LayerTag,
		Status:            req.Status,
		DurationMS:        req.DurationMS,
		InputCipherCount:  req.InputCipherCount,
		OutputCipherCount: req.OutputCipherCount,
		WeightCount:       req.WeightCount,
		Error:             req.Error,
		Metrics:           req.Metrics,
		Metadata:          req.Metadata,
	}

	if err := experimentTracker.RecordEvent(experimentID, event); err != nil {
		fail(w, http.StatusBadRequest, err.Error())
		return
	}

	success(w, "实验事件已记录", map[string]string{"experiment_id": experimentID})
}

func experimentsDetailHandler(w http.ResponseWriter, r *http.Request) {
	if experimentTracker == nil {
		fail(w, http.StatusInternalServerError, "experiment tracker not initialized")
		return
	}
	if r.Method != http.MethodGet {
		fail(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	trimmed := strings.TrimPrefix(r.URL.Path, "/api/v1/experiments/")
	parts := strings.Split(strings.Trim(trimmed, "/"), "/")
	if len(parts) == 0 || parts[0] == "" {
		fail(w, http.StatusBadRequest, "缺少 experiment_id")
		return
	}

	experimentID := parts[0]
	action := "summary"
	if len(parts) > 1 && parts[1] != "" {
		action = parts[1]
	}

	switch action {
	case "summary":
		summary, err := experimentTracker.Summary(experimentID)
		if err != nil {
			fail(w, http.StatusNotFound, err.Error())
			return
		}
		success(w, "ok", summary)

	case "events":
		limit := 200
		if val := strings.TrimSpace(r.URL.Query().Get("limit")); val != "" {
			if v, err := strconv.Atoi(val); err == nil && v > 0 {
				limit = v
			}
		}
		events, total, err := experimentTracker.GetEvents(experimentID, limit)
		if err != nil {
			fail(w, http.StatusNotFound, err.Error())
			return
		}
		success(w, "ok", ExperimentEventsResponse{
			ExperimentID: experimentID,
			Total:        total,
			Limit:        limit,
			Events:       events,
		})

	case "events.csv":
		csvPath, err := experimentTracker.ExportEventsCSV(experimentID)
		if err != nil {
			fail(w, http.StatusNotFound, err.Error())
			return
		}
		content, err := ioutil.ReadFile(csvPath)
		if err != nil {
			fail(w, http.StatusInternalServerError, err.Error())
			return
		}
		w.Header().Set("Content-Type", "text/csv; charset=utf-8")
		w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s-events.csv\"", experimentID))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(content)

	case "plot-data":
		plotData, err := experimentTracker.BuildPlotData(experimentID)
		if err != nil {
			fail(w, http.StatusNotFound, err.Error())
			return
		}
		success(w, "ok", plotData)

	default:
		fail(w, http.StatusNotFound, "unknown experiment action")
	}
}

func parseTime(raw string) time.Time {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return time.Time{}
	}
	t, err := time.Parse(time.RFC3339Nano, raw)
	if err != nil {
		return time.Time{}
	}
	return t
}

func cloneStringMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func cloneFloatMap(in map[string]float64) map[string]float64 {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]float64, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func cloneEvent(e ExperimentEvent) ExperimentEvent {
	e.Metadata = cloneStringMap(e.Metadata)
	e.Metrics = cloneFloatMap(e.Metrics)
	return e
}

func decodeJSONBody(r *http.Request, dst interface{}) error {
	if r == nil || r.Body == nil {
		return nil
	}
	dec := json.NewDecoder(r.Body)
	if err := dec.Decode(dst); err != nil {
		if err == io.EOF {
			return nil
		}
		return err
	}
	return nil
}
