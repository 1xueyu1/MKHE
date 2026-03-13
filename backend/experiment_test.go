package main

import (
	"io/ioutil"
	"os"
	"testing"
)

func TestExperimentTrackerLifecycle(t *testing.T) {
	dir, err := ioutil.TempDir("", "mkhe-exp-")
	if err != nil {
		t.Fatalf("创建临时目录失败: %v", err)
	}
	defer os.RemoveAll(dir)

	tracker, err := NewExperimentTracker(dir)
	if err != nil {
		t.Fatalf("创建 tracker 失败: %v", err)
	}

	startResp, err := tracker.Start(ExperimentStartRequest{
		Name:      "lifecycle-test",
		SetActive: true,
	})
	if err != nil {
		t.Fatalf("启动实验失败: %v", err)
	}
	if startResp.ExperimentID == "" {
		t.Fatal("experiment_id 不能为空")
	}

	experimentID := startResp.ExperimentID

	err = tracker.RecordEvent(experimentID, ExperimentEvent{
		Round:             1,
		Operation:         "encrypt",
		Status:            "success",
		DurationMS:        12.5,
		OutputCipherCount: 2,
		WeightCount:       128,
	})
	if err != nil {
		t.Fatalf("记录事件失败: %v", err)
	}

	err = tracker.RecordEvent(experimentID, ExperimentEvent{
		Round:      1,
		Operation:  "aggregate",
		Status:     "failed",
		DurationMS: 8.3,
		Error:      "mock error",
	})
	if err != nil {
		t.Fatalf("记录失败事件失败: %v", err)
	}

	summary, err := tracker.Summary(experimentID)
	if err != nil {
		t.Fatalf("读取 summary 失败: %v", err)
	}
	if summary.TotalEvents != 2 {
		t.Fatalf("summary total events 异常: %d", summary.TotalEvents)
	}
	if summary.SuccessEvents != 1 || summary.FailureEvents != 1 {
		t.Fatalf("success/failure 统计异常: success=%d failure=%d", summary.SuccessEvents, summary.FailureEvents)
	}

	events, total, err := tracker.GetEvents(experimentID, 100)
	if err != nil {
		t.Fatalf("读取事件失败: %v", err)
	}
	if total != 2 || len(events) != 2 {
		t.Fatalf("events 数量异常: total=%d len=%d", total, len(events))
	}

	csvPath, err := tracker.ExportEventsCSV(experimentID)
	if err != nil {
		t.Fatalf("导出 CSV 失败: %v", err)
	}
	if _, err := os.Stat(csvPath); err != nil {
		t.Fatalf("CSV 文件不存在: %v", err)
	}

	plotData, err := tracker.BuildPlotData(experimentID)
	if err != nil {
		t.Fatalf("构建 plot-data 失败: %v", err)
	}
	if len(plotData.LatencyByOperation) == 0 {
		t.Fatalf("plot-data 为空: %+v", plotData)
	}

	stopResp, err := tracker.Stop(experimentID)
	if err != nil {
		t.Fatalf("停止实验失败: %v", err)
	}
	if stopResp.TotalEvents != 2 {
		t.Fatalf("停止时统计异常: %d", stopResp.TotalEvents)
	}
}

func TestExperimentTrackerReload(t *testing.T) {
	dir, err := ioutil.TempDir("", "mkhe-exp-reload-")
	if err != nil {
		t.Fatalf("创建临时目录失败: %v", err)
	}
	defer os.RemoveAll(dir)

	tracker, err := NewExperimentTracker(dir)
	if err != nil {
		t.Fatalf("创建 tracker 失败: %v", err)
	}

	startResp, err := tracker.Start(ExperimentStartRequest{Name: "reload-test", SetActive: true})
	if err != nil {
		t.Fatalf("启动实验失败: %v", err)
	}

	experimentID := startResp.ExperimentID
	err = tracker.RecordEvent(experimentID, ExperimentEvent{Operation: "decrypt", Status: "success", DurationMS: 3.1})
	if err != nil {
		t.Fatalf("记录事件失败: %v", err)
	}

	reloaded, err := NewExperimentTracker(dir)
	if err != nil {
		t.Fatalf("重载 tracker 失败: %v", err)
	}

	summary, err := reloaded.Summary(experimentID)
	if err != nil {
		t.Fatalf("重载后读取 summary 失败: %v", err)
	}
	if summary.TotalEvents != 1 {
		t.Fatalf("重载后事件数量异常: %d", summary.TotalEvents)
	}
}
