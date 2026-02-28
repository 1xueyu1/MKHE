package main

import (
	"fmt"
	"math"
	"testing"
)

// ================================================================
//  辅助函数
// ================================================================

// initTestService 创建并初始化一个测试用 HEService（3个客户端, LogN=14）
func initTestService(t *testing.T) *HEService {
	t.Helper()
	svc := NewHEService()
	resp, err := svc.Init(14, []string{"client_a", "client_b", "client_c"})
	if err != nil {
		t.Fatalf("Init 失败: %v", err)
	}
	t.Logf("初始化成功: slots=%d, maxLevel=%d, scale=%.0f, 参与方=%d",
		resp.Slots, resp.MaxLevel, resp.Scale, resp.RegisteredCount)
	return svc
}

// almostEqual 判断两个 float64 是否在误差范围内相等
func almostEqual(a, b, epsilon float64) bool {
	return math.Abs(a-b) < epsilon
}

// ================================================================
//  Test 1: 系统初始化
// ================================================================

func TestInit(t *testing.T) {
	svc := NewHEService()

	// 未初始化时状态检查
	status := svc.GetStatus()
	if status.Initialized {
		t.Fatal("新创建的服务不应标记为已初始化")
	}

	// 正常初始化
	resp, err := svc.Init(14, []string{"alice", "bob"})
	if err != nil {
		t.Fatalf("Init 失败: %v", err)
	}

	if resp.Status != "initialized" {
		t.Errorf("期望 status=initialized, 实际=%s", resp.Status)
	}
	if resp.Slots <= 0 {
		t.Errorf("slots 应该 > 0, 实际=%d", resp.Slots)
	}
	if resp.RegisteredCount != 2 {
		t.Errorf("期望注册 2 个参与方, 实际=%d", resp.RegisteredCount)
	}

	t.Logf("✓ 初始化: slots=%d, maxLevel=%d, scale=%.0f", resp.Slots, resp.MaxLevel, resp.Scale)
}

// ================================================================
//  Test 2: 客户端注册
// ================================================================

func TestRegisterClient(t *testing.T) {
	svc := initTestService(t)

	// 注册新客户端
	resp, err := svc.RegisterClient("client_d")
	if err != nil {
		t.Fatalf("注册失败: %v", err)
	}
	if resp.ClientID != "client_d" || !resp.HasPublicKey || !resp.HasSecretKey {
		t.Errorf("注册响应异常: %+v", resp)
	}
	t.Logf("✓ 注册新客户端: %s", resp.ClientID)

	// 重复注册应失败
	_, err = svc.RegisterClient("client_a")
	if err == nil {
		t.Error("重复注册应返回错误")
	} else {
		t.Logf("✓ 重复注册被正确拒绝: %v", err)
	}

	// 系统状态
	status := svc.GetStatus()
	if len(status.RegisteredIDs) != 4 {
		t.Errorf("期望 4 个注册客户端, 实际=%d", len(status.RegisteredIDs))
	}
	t.Logf("✓ 当前注册客户端: %v", status.RegisteredIDs)
}

// ================================================================
//  Test 3: 加密模型权重
// ================================================================

func TestEncryptModel(t *testing.T) {
	svc := initTestService(t)

	// 模拟一个小模型（10个权重）
	weights := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}

	resp, err := svc.EncryptModel("client_a", weights, 1, "fc1")
	if err != nil {
		t.Fatalf("加密失败: %v", err)
	}

	if resp.ChunkCount != 1 {
		t.Errorf("10个权重应只需 1 个 chunk, 实际=%d", resp.ChunkCount)
	}
	if len(resp.CipherIDs) != 1 {
		t.Errorf("期望 1 个密文ID, 实际=%d", len(resp.CipherIDs))
	}

	t.Logf("✓ 加密完成: round=%d, chunkCount=%d, cipherIDs=%v", resp.Round, resp.ChunkCount, resp.CipherIDs)

	// 查询密文元信息
	info, err := svc.GetCipherInfo(resp.CipherIDs[0])
	if err != nil {
		t.Fatalf("查询密文信息失败: %v", err)
	}
	t.Logf("✓ 密文信息: clientID=%s, level=%d, scale=%.0f, idSet=%v",
		info.ClientID, info.Level, info.Scale, info.IDSet)
}

// ================================================================
//  Test 4: 加密不存在的客户端 → 应报错
// ================================================================

func TestEncryptModel_InvalidClient(t *testing.T) {
	svc := initTestService(t)

	_, err := svc.EncryptModel("unknown_client", []float64{1.0}, 1, "")
	if err == nil {
		t.Error("不存在的客户端加密应返回错误")
	} else {
		t.Logf("✓ 正确报错: %v", err)
	}
}

// ================================================================
//  Test 5: 集中式解密（加密→解密 回环测试）
// ================================================================

func TestEncryptDecryptRoundTrip(t *testing.T) {
	svc := initTestService(t)

	original := []float64{1.1, 2.2, 3.3, 4.4, 5.5}
	encResp, err := svc.EncryptModel("client_a", original, 1, "test")
	if err != nil {
		t.Fatalf("加密失败: %v", err)
	}

	decResp, err := svc.Decrypt(encResp.CipherIDs, len(original))
	if err != nil {
		t.Fatalf("解密失败: %v", err)
	}

	if decResp.Length != len(original) {
		t.Errorf("解密长度不匹配: 期望=%d, 实际=%d", len(original), decResp.Length)
	}

	epsilon := 0.001
	for i, v := range original {
		if !almostEqual(decResp.Weights[i], v, epsilon) {
			t.Errorf("权重[%d] 不匹配: 期望=%.4f, 实际=%.4f", i, v, decResp.Weights[i])
		}
	}
	t.Logf("✓ 加密→解密回环测试通过（误差<%.4f）", epsilon)
	t.Logf("  原始: %v", original)
	t.Logf("  解密: %v", decResp.Weights)
}

// ================================================================
//  Test 6: 同态聚合（求和）
// ================================================================

func TestAggregate_Sum(t *testing.T) {
	svc := initTestService(t)

	weightsA := []float64{1.0, 2.0, 3.0, 4.0}
	weightsB := []float64{5.0, 6.0, 7.0, 8.0}
	weightsC := []float64{9.0, 10.0, 11.0, 12.0}

	encA, _ := svc.EncryptModel("client_a", weightsA, 1, "")
	encB, _ := svc.EncryptModel("client_b", weightsB, 1, "")
	encC, _ := svc.EncryptModel("client_c", weightsC, 1, "")

	// 按 chunk 分组（都只有 1 个 chunk）
	groups := [][]string{
		{encA.CipherIDs[0], encB.CipherIDs[0], encC.CipherIDs[0]},
	}

	aggResp, err := svc.Aggregate(groups, 1, false, 3)
	if err != nil {
		t.Fatalf("聚合失败: %v", err)
	}

	// 解密验证
	decResp, err := svc.Decrypt(aggResp.AggregatedCipherIDs, len(weightsA))
	if err != nil {
		t.Fatalf("解密聚合结果失败: %v", err)
	}

	expected := []float64{15.0, 18.0, 21.0, 24.0} // 1+5+9, 2+6+10, ...
	epsilon := 0.01
	for i, v := range expected {
		if !almostEqual(decResp.Weights[i], v, epsilon) {
			t.Errorf("聚合结果[%d] 不匹配: 期望=%.4f, 实际=%.4f", i, v, decResp.Weights[i])
		}
	}

	t.Logf("✓ 同态求和聚合通过")
	t.Logf("  A=%v + B=%v + C=%v", weightsA, weightsB, weightsC)
	t.Logf("  期望: %v", expected)
	t.Logf("  实际: %v", decResp.Weights)
}

// ================================================================
//  Test 7: 同态聚合（FedAvg 求均值）
// ================================================================

func TestAggregate_Average(t *testing.T) {
	svc := initTestService(t)

	weightsA := []float64{2.0, 4.0, 6.0}
	weightsB := []float64{4.0, 8.0, 12.0}

	encA, _ := svc.EncryptModel("client_a", weightsA, 1, "")
	encB, _ := svc.EncryptModel("client_b", weightsB, 1, "")

	groups := [][]string{
		{encA.CipherIDs[0], encB.CipherIDs[0]},
	}

	aggResp, err := svc.Aggregate(groups, 1, true, 2)
	if err != nil {
		t.Fatalf("均值聚合失败: %v", err)
	}

	decResp, err := svc.Decrypt(aggResp.AggregatedCipherIDs, len(weightsA))
	if err != nil {
		t.Fatalf("解密聚合结果失败: %v", err)
	}

	expected := []float64{3.0, 6.0, 9.0} // (2+4)/2, (4+8)/2, (6+12)/2
	epsilon := 0.01
	for i, v := range expected {
		if !almostEqual(decResp.Weights[i], v, epsilon) {
			t.Errorf("均值结果[%d] 不匹配: 期望=%.4f, 实际=%.4f", i, v, decResp.Weights[i])
		}
	}

	t.Logf("✓ FedAvg 均值聚合通过")
	t.Logf("  期望: %v", expected)
	t.Logf("  实际: %v", decResp.Weights)
}

// ================================================================
//  Test 8: 分布式部分解密
// ================================================================

func TestPartialDecrypt(t *testing.T) {
	svc := initTestService(t)

	// 3个客户端各自加密
	weightsA := []float64{1.0, 2.0, 3.0}
	weightsB := []float64{4.0, 5.0, 6.0}
	weightsC := []float64{7.0, 8.0, 9.0}

	encA, _ := svc.EncryptModel("client_a", weightsA, 1, "")
	encB, _ := svc.EncryptModel("client_b", weightsB, 1, "")
	encC, _ := svc.EncryptModel("client_c", weightsC, 1, "")

	// 同态聚合
	groups := [][]string{
		{encA.CipherIDs[0], encB.CipherIDs[0], encC.CipherIDs[0]},
	}
	aggResp, _ := svc.Aggregate(groups, 1, false, 3)
	currentIDs := aggResp.AggregatedCipherIDs

	t.Logf("聚合完成，开始分布式解密...")

	// 第1轮部分解密: client_a
	pd1, err := svc.PartialDecrypt(currentIDs, "client_a")
	if err != nil {
		t.Fatalf("client_a 部分解密失败: %v", err)
	}
	t.Logf("  client_a 部分解密完成, 剩余 %d 方", pd1.RemainingParties)
	currentIDs = pd1.CipherIDs

	// 第2轮部分解密: client_b
	pd2, err := svc.PartialDecrypt(currentIDs, "client_b")
	if err != nil {
		t.Fatalf("client_b 部分解密失败: %v", err)
	}
	t.Logf("  client_b 部分解密完成, 剩余 %d 方", pd2.RemainingParties)
	currentIDs = pd2.CipherIDs

	// 第3轮部分解密: client_c
	pd3, err := svc.PartialDecrypt(currentIDs, "client_c")
	if err != nil {
		t.Fatalf("client_c 部分解密失败: %v", err)
	}
	t.Logf("  client_c 部分解密完成, 剩余 %d 方", pd3.RemainingParties)
	currentIDs = pd3.CipherIDs

	if pd3.RemainingParties != 0 {
		t.Errorf("所有参与方解密后 remaining 应为 0, 实际=%d", pd3.RemainingParties)
	}

	// 最终解密
	decResp, err := svc.FinalDecrypt(currentIDs, len(weightsA))
	if err != nil {
		t.Fatalf("最终解密失败: %v", err)
	}

	expected := []float64{12.0, 15.0, 18.0}
	epsilon := 0.01
	for i, v := range expected {
		if !almostEqual(decResp.Weights[i], v, epsilon) {
			t.Errorf("分布式解密结果[%d] 不匹配: 期望=%.4f, 实际=%.4f", i, v, decResp.Weights[i])
		}
	}

	t.Logf("✓ 分布式部分解密通过")
	t.Logf("  期望: %v", expected)
	t.Logf("  实际: %v", decResp.Weights)
}

// ================================================================
//  Test 9: 密文管理（查询/删除/清理）
// ================================================================

func TestCipherManagement(t *testing.T) {
	svc := initTestService(t)

	// 加密一些数据
	enc1, _ := svc.EncryptModel("client_a", []float64{1.0, 2.0}, 1, "layer1")
	enc2, _ := svc.EncryptModel("client_b", []float64{3.0, 4.0}, 1, "layer1")
	enc3, _ := svc.EncryptModel("client_a", []float64{5.0, 6.0}, 2, "layer1")

	// 查询密文信息
	info, err := svc.GetCipherInfo(enc1.CipherIDs[0])
	if err != nil {
		t.Fatalf("查询密文失败: %v", err)
	}
	if info.ClientID != "client_a" || info.Round != 1 || info.LayerTag != "layer1" {
		t.Errorf("密文元信息不正确: %+v", info)
	}
	t.Logf("✓ 密文查询: clientID=%s, round=%d, layerTag=%s", info.ClientID, info.Round, info.LayerTag)

	// 删除单个密文
	err = svc.DeleteCipher(enc2.CipherIDs[0])
	if err != nil {
		t.Fatalf("删除失败: %v", err)
	}
	_, err = svc.GetCipherInfo(enc2.CipherIDs[0])
	if err == nil {
		t.Error("已删除的密文不应被查到")
	}
	t.Logf("✓ 单个密文删除成功")

	// 清理 Round 1 的密文
	cleanResp := svc.Cleanup(1)
	t.Logf("✓ 清理 round=1: 删除了 %d 个密文", cleanResp.DeletedCount)

	// Round 2 的还在
	_, err = svc.GetCipherInfo(enc3.CipherIDs[0])
	if err != nil {
		t.Error("Round 2 的密文不应被清理")
	}
	t.Logf("✓ Round 2 的密文未受影响")

	// 全部清理
	cleanAll := svc.Cleanup(0)
	t.Logf("✓ 全部清理: 删除了 %d 个密文", cleanAll.DeletedCount)

	status := svc.GetStatus()
	if status.CipherCount != 0 {
		t.Errorf("全部清理后密文数应为 0, 实际=%d", status.CipherCount)
	}
}

// ================================================================
//  Test 10: 轮次管理
// ================================================================

func TestRoundManagement(t *testing.T) {
	svc := initTestService(t)

	status := svc.GetStatus()
	if status.CurrentRound != 1 {
		t.Errorf("初始轮次应为 1, 实际=%d", status.CurrentRound)
	}

	resp := svc.AdvanceRound()
	if resp.PreviousRound != 1 || resp.CurrentRound != 2 {
		t.Errorf("轮次推进异常: prev=%d, curr=%d", resp.PreviousRound, resp.CurrentRound)
	}

	resp2 := svc.AdvanceRound()
	if resp2.CurrentRound != 3 {
		t.Errorf("期望轮次 3, 实际=%d", resp2.CurrentRound)
	}

	t.Logf("✓ 轮次推进: 1 → 2 → 3")
}

// ================================================================
//  Test 11: 完整联邦学习一轮流程（端到端）
// ================================================================

func TestFullFederatedRound(t *testing.T) {
	svc := initTestService(t)

	numClients := 3
	clients := []string{"client_a", "client_b", "client_c"}
	numWeights := 10

	t.Log("======= 联邦学习完整一轮端到端测试 =======")

	// Step 1: 每个客户端生成模拟权重并加密
	allCipherIDs := make([][]string, numClients)
	allWeights := make([][]float64, numClients)

	for i, cid := range clients {
		weights := make([]float64, numWeights)
		for j := range weights {
			weights[j] = float64(i*numWeights+j) * 0.1
		}
		allWeights[i] = weights

		encResp, err := svc.EncryptModel(cid, weights, 1, "full_model")
		if err != nil {
			t.Fatalf("客户端 %s 加密失败: %v", cid, err)
		}
		allCipherIDs[i] = encResp.CipherIDs
		t.Logf("  [加密] %s: %d 个权重 → %d 个 chunk", cid, len(weights), encResp.ChunkCount)
	}

	// Step 2: 组装 cipher_groups（按 chunk 分组）
	chunkCount := len(allCipherIDs[0])
	groups := make([][]string, chunkCount)
	for ci := 0; ci < chunkCount; ci++ {
		groups[ci] = make([]string, numClients)
		for cli := 0; cli < numClients; cli++ {
			groups[ci][cli] = allCipherIDs[cli][ci]
		}
	}
	t.Logf("  [分组] %d 个 chunk, 每组 %d 个密文", chunkCount, numClients)

	// Step 3: 同态聚合（FedAvg）
	aggResp, err := svc.Aggregate(groups, 1, true, numClients)
	if err != nil {
		t.Fatalf("聚合失败: %v", err)
	}
	t.Logf("  [聚合] FedAvg 完成, %d 个聚合密文", len(aggResp.AggregatedCipherIDs))

	// Step 4: 集中式解密
	decResp, err := svc.Decrypt(aggResp.AggregatedCipherIDs, numWeights)
	if err != nil {
		t.Fatalf("解密失败: %v", err)
	}

	// 验证: 均值应为 (w_a + w_b + w_c) / 3
	epsilon := 0.05
	for j := 0; j < numWeights; j++ {
		expected := (allWeights[0][j] + allWeights[1][j] + allWeights[2][j]) / 3.0
		if !almostEqual(decResp.Weights[j], expected, epsilon) {
			t.Errorf("  FedAvg 权重[%d] 不匹配: 期望=%.4f, 实际=%.4f", j, expected, decResp.Weights[j])
		}
	}
	t.Logf("  [解密] 聚合结果验证通过 (误差<%.4f)", epsilon)

	// Step 5: 清理 & 推进轮次
	cleanResp := svc.Cleanup(1)
	t.Logf("  [清理] 删除 %d 个密文", cleanResp.DeletedCount)

	roundResp := svc.AdvanceRound()
	t.Logf("  [轮次] %d → %d", roundResp.PreviousRound, roundResp.CurrentRound)

	t.Log("======= ✓ 联邦学习完整一轮端到端测试通过 =======")
}

// ================================================================
//  Test 12: 完整分布式解密流程（端到端）
// ================================================================

func TestFullDistributedDecrypt(t *testing.T) {
	svc := initTestService(t)

	clients := []string{"client_a", "client_b", "client_c"}
	numWeights := 5

	t.Log("======= 分布式解密完整流程测试 =======")

	// 各客户端加密
	var allCipherIDs [][]string
	var allWeights [][]float64
	for i, cid := range clients {
		w := make([]float64, numWeights)
		for j := range w {
			w[j] = float64((i+1)*(j+1)) * 0.5
		}
		allWeights = append(allWeights, w)
		enc, err := svc.EncryptModel(cid, w, 1, "")
		if err != nil {
			t.Fatalf("加密失败: %v", err)
		}
		allCipherIDs = append(allCipherIDs, enc.CipherIDs)
		t.Logf("  [加密] %s weights=%v", cid, w)
	}

	// 聚合（仅求和）
	groups := [][]string{
		{allCipherIDs[0][0], allCipherIDs[1][0], allCipherIDs[2][0]},
	}
	aggResp, _ := svc.Aggregate(groups, 1, false, 3)

	// 分布式解密 — 每个参与方依次部分解密
	currentIDs := aggResp.AggregatedCipherIDs
	for _, cid := range clients {
		pdResp, err := svc.PartialDecrypt(currentIDs, cid)
		if err != nil {
			t.Fatalf("%s 部分解密失败: %v", cid, err)
		}
		t.Logf("  [部分解密] %s 完成, 剩余 %d 方", cid, pdResp.RemainingParties)
		currentIDs = pdResp.CipherIDs
	}

	// 最终解密
	decResp, err := svc.FinalDecrypt(currentIDs, numWeights)
	if err != nil {
		t.Fatalf("最终解密失败: %v", err)
	}

	// 验证求和
	epsilon := 0.01
	for j := 0; j < numWeights; j++ {
		expected := allWeights[0][j] + allWeights[1][j] + allWeights[2][j]
		if !almostEqual(decResp.Weights[j], expected, epsilon) {
			t.Errorf("  求和结果[%d]: 期望=%.4f, 实际=%.4f", j, expected, decResp.Weights[j])
		}
	}

	t.Logf("  [验证] 分布式解密结果: %v", decResp.Weights)
	t.Log("======= ✓ 分布式解密完整流程测试通过 =======")
}

// ================================================================
//  Test 13: 多 Chunk 大模型测试
// ================================================================

func TestLargeModelMultiChunk(t *testing.T) {
	svc := initTestService(t)

	slots := svc.GetStatus().Slots
	if slots == 0 {
		// 先初始化才能获取 slots
		initResp, _ := svc.Init(14, []string{"client_a", "client_b"})
		slots = initResp.Slots
	}

	// 生成超过 slots 的权重，强制分 chunk
	numWeights := slots + 100
	weightsA := make([]float64, numWeights)
	weightsB := make([]float64, numWeights)
	for i := range weightsA {
		weightsA[i] = float64(i) * 0.001
		weightsB[i] = float64(i) * 0.002
	}

	t.Logf("模型大小=%d, slots=%d, 预期 chunk 数=%d", numWeights, slots, (numWeights+slots-1)/slots)

	encA, err := svc.EncryptModel("client_a", weightsA, 1, "")
	if err != nil {
		t.Fatalf("client_a 大模型加密失败: %v", err)
	}
	encB, err := svc.EncryptModel("client_b", weightsB, 1, "")
	if err != nil {
		t.Fatalf("client_b 大模型加密失败: %v", err)
	}

	if encA.ChunkCount < 2 {
		t.Errorf("预期 >= 2 个 chunk, 实际=%d", encA.ChunkCount)
	}
	t.Logf("✓ client_a: %d chunks, client_b: %d chunks", encA.ChunkCount, encB.ChunkCount)

	// 按 chunk 分组聚合
	groups := make([][]string, encA.ChunkCount)
	for i := 0; i < encA.ChunkCount; i++ {
		groups[i] = []string{encA.CipherIDs[i], encB.CipherIDs[i]}
	}

	aggResp, err := svc.Aggregate(groups, 1, false, 2)
	if err != nil {
		t.Fatalf("多chunk聚合失败: %v", err)
	}

	// 解密验证前 10 个和最后 10 个权重
	decResp, err := svc.Decrypt(aggResp.AggregatedCipherIDs, numWeights)
	if err != nil {
		t.Fatalf("解密失败: %v", err)
	}

	epsilon := 0.05
	checkIndices := []int{0, 1, 2, 3, 4, numWeights - 5, numWeights - 4, numWeights - 3, numWeights - 2, numWeights - 1}
	for _, idx := range checkIndices {
		expected := weightsA[idx] + weightsB[idx]
		if !almostEqual(decResp.Weights[idx], expected, epsilon) {
			t.Errorf("  权重[%d]: 期望=%.6f, 实际=%.6f", idx, expected, decResp.Weights[idx])
		}
	}

	t.Logf("✓ 多 chunk 大模型测试通过 (总权重=%d, chunks=%d)", numWeights, encA.ChunkCount)
}

// ================================================================
//  Test 14: 系统状态查询
// ================================================================

func TestGetStatus(t *testing.T) {
	svc := initTestService(t)

	status := svc.GetStatus()
	if !status.Initialized {
		t.Error("初始化后 Initialized 应为 true")
	}
	if status.CurrentRound != 1 {
		t.Errorf("初始轮次应为 1, 实际=%d", status.CurrentRound)
	}
	if len(status.RegisteredIDs) != 3 {
		t.Errorf("期望 3 个注册客户端, 实际=%d", len(status.RegisteredIDs))
	}
	if status.Slots <= 0 {
		t.Errorf("slots 应 > 0, 实际=%d", status.Slots)
	}

	// 加密几个密文后检查 CipherCount
	svc.EncryptModel("client_a", []float64{1.0}, 1, "")
	svc.EncryptModel("client_b", []float64{2.0}, 1, "")

	status2 := svc.GetStatus()
	if status2.CipherCount != 2 {
		t.Errorf("期望 2 个密文, 实际=%d", status2.CipherCount)
	}

	t.Logf("✓ 状态查询: initialized=%v, round=%d, clients=%v, ciphers=%d, slots=%d",
		status2.Initialized, status2.CurrentRound, status2.RegisteredIDs,
		status2.CipherCount, status2.Slots)
}

// ================================================================
//  Test 15: 未初始化时调用各接口 → 应全部报错
// ================================================================

func TestUninitializedErrors(t *testing.T) {
	svc := NewHEService()

	_, err := svc.EncryptModel("x", []float64{1.0}, 1, "")
	if err == nil {
		t.Error("未初始化时加密应报错")
	}

	_, err = svc.Aggregate([][]string{{"x"}}, 1, false, 1)
	if err == nil {
		t.Error("未初始化时聚合应报错")
	}

	_, err = svc.Decrypt([]string{"x"}, 1)
	if err == nil {
		t.Error("未初始化时解密应报错")
	}

	_, err = svc.RegisterClient("x")
	if err == nil {
		t.Error("未初始化时注册应报错")
	}

	_, err = svc.PartialDecrypt([]string{"x"}, "x")
	if err == nil {
		t.Error("未初始化时部分解密应报错")
	}

	t.Logf("✓ 未初始化时所有操作均正确报错")
}

// ================================================================
//  Benchmark: 加密性能
// ================================================================

func BenchmarkEncrypt(b *testing.B) {
	svc := NewHEService()
	svc.Init(14, []string{"bench_client"})
	weights := make([]float64, 1000)
	for i := range weights {
		weights[i] = float64(i) * 0.01
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = svc.EncryptModel("bench_client", weights, 1, "")
	}
}

// ================================================================
//  Benchmark: 聚合性能（2方）
// ================================================================

func BenchmarkAggregate2(b *testing.B) {
	svc := NewHEService()
	svc.Init(14, []string{"c1", "c2"})
	weights := make([]float64, 1000)
	for i := range weights {
		weights[i] = float64(i) * 0.01
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e1, _ := svc.EncryptModel("c1", weights, 1, "")
		e2, _ := svc.EncryptModel("c2", weights, 1, "")
		groups := [][]string{{e1.CipherIDs[0], e2.CipherIDs[0]}}
		_, _ = svc.Aggregate(groups, 1, false, 2)
	}
}

// ================================================================
//  打印所有测试汇总（可选，go test -v 时会自动打印）
// ================================================================

func TestPrintAPISummary(t *testing.T) {
	apis := []struct {
		method string
		path   string
		desc   string
	}{
		{"POST", "/api/v1/system/init", "初始化 MKHE 系统"},
		{"GET", "/api/v1/system/status", "获取系统状态"},
		{"GET", "/api/v1/health", "健康检查"},
		{"POST", "/api/v1/clients/register", "注册新客户端"},
		{"POST", "/api/v1/encrypt", "加密模型权重"},
		{"POST", "/api/v1/aggregate", "同态聚合"},
		{"POST", "/api/v1/decrypt", "集中式解密"},
		{"POST", "/api/v1/decrypt/partial", "分布式部分解密"},
		{"POST", "/api/v1/decrypt/final", "分布式最终解密"},
		{"GET", "/api/v1/cipher/{id}", "查询密文信息"},
		{"DELETE", "/api/v1/cipher/{id}", "删除密文"},
		{"POST", "/api/v1/cipher/cleanup", "批量清理密文"},
		{"POST", "/api/v1/round/advance", "推进训练轮次"},
	}

	fmt.Println("\n╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║            MKHE 联邦学习后端 API 接口一览                    ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════╣")
	for _, api := range apis {
		fmt.Printf("║  %-7s %-30s %s\n", api.method, api.path, api.desc)
	}
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
}
