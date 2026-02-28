package main

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"math"
	"sync"

	"mk-lattigo/mkckks"
	"mk-lattigo/mkrlwe"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
)

// ================================================================
//  CKKS 参数预设
// ================================================================

var (
	paramPN15QP880 = ckks.ParametersLiteral{
		LogN:     15,
		LogSlots: 14,
		Q: []uint64{
			0xfffffffff6a0001,
			0x3fffffffd60001, 0x3fffffffca0001,
			0x3fffffff6d0001, 0x3fffffff5d0001,
			0x3fffffff550001, 0x3fffffff390001,
			0x3fffffff360001, 0x3fffffff2a0001,
			0x3fffffff000001, 0x3ffffffefa0001,
			0x3ffffffef40001, 0x3ffffffed70001,
			0x3ffffffed30001,
		},
		P:     []uint64{0x7ffffffffe70001, 0x7ffffffffe10001},
		Scale: 1 << 54,
		Sigma: rlwe.DefaultSigma,
	}
	paramPN14QP439 = ckks.ParametersLiteral{
		LogN:     14,
		LogSlots: 13,
		Q: []uint64{
			0x7ffffffffe70001,
			0xffffffff00001, 0xfffffffe40001,
			0xfffffffe20001, 0xfffffffbe0001,
			0xfffffffa60001,
		},
		P:     []uint64{0xffffffffffc0001, 0xfffffffff840001},
		Scale: 1 << 52,
		Sigma: rlwe.DefaultSigma,
	}
)

// getParamLiteral 根据 logN 返回对应的预设参数
func getParamLiteral(logN int) ckks.ParametersLiteral {
	switch logN {
	case 15:
		return paramPN15QP880
	default:
		return paramPN14QP439
	}
}

// ================================================================
//  密文存储条目
// ================================================================

// CipherEntry 在内存中保存密文及其元数据
type CipherEntry struct {
	Ciphertext *mkckks.Ciphertext
	ClientID   string
	Round      int
	LayerTag   string
	ChunkIndex int
}

// ================================================================
//  HEService — 核心服务
// ================================================================

type HEService struct {
	mu sync.RWMutex

	// HE 基础组件
	Params    mkckks.Parameters
	RingQ     *ring.Ring
	PRNG      utils.PRNG
	Kgen      *mkrlwe.KeyGenerator
	SkSet     *mkrlwe.SecretKeySet
	PkSet     *mkrlwe.PublicKeySet
	RlkSet    *mkrlwe.RelinearizationKeySet
	RtkSet    *mkrlwe.RotationKeySet
	Encryptor *mkckks.Encryptor
	Decryptor *mkckks.Decryptor
	Evaluator *mkckks.Evaluator
	IdSet     *mkrlwe.IDSet

	// 联邦学习状态
	Initialized  bool
	CurrentRound int
	CipherStore  map[string]*CipherEntry
	ClientList   []string
}

// NewHEService 创建空的 HE 服务实例
func NewHEService() *HEService {
	return &HEService{
		CipherStore:  make(map[string]*CipherEntry),
		CurrentRound: 1,
	}
}

// generateID 生成 32 字符十六进制随机 ID
func generateID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// ================================================================
//  系统初始化
// ================================================================

// Init 初始化 MKHE 加密环境，生成所有参与方的密钥
func (s *HEService) Init(logN int, clientIDs []string) (*InitResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	paramLit := getParamLiteral(logN)
	ckksParams, err := ckks.NewParametersFromLiteral(paramLit)
	if err != nil {
		return nil, fmt.Errorf("创建 CKKS 参数失败: %w", err)
	}

	params := mkckks.NewParameters(ckksParams)
	s.Params = params
	s.IdSet = mkrlwe.NewIDSet()
	s.Kgen = mkckks.NewKeyGenerator(params)
	s.SkSet = mkrlwe.NewSecretKeySet()
	s.PkSet = mkrlwe.NewPublicKeyKeySet()
	s.RlkSet = mkrlwe.NewRelinearizationKeyKeySet(params.Parameters)
	s.RtkSet = mkrlwe.NewRotationKeySet()
	s.Encryptor = mkckks.NewEncryptor(params)
	s.Decryptor = mkckks.NewDecryptor(params)
	s.Evaluator = mkckks.NewEvaluator(params)
	s.RingQ = params.RingQ()
	s.PRNG, _ = utils.NewPRNG()

	// 注册初始参与方
	for _, id := range clientIDs {
		s.registerClientUnsafe(id)
	}

	s.Initialized = true
	s.CurrentRound = 1
	s.CipherStore = make(map[string]*CipherEntry)
	s.ClientList = append([]string{}, clientIDs...)

	return &InitResponse{
		Status:          "initialized",
		Slots:           params.Slots(),
		MaxLevel:        params.MaxLevel(),
		Scale:           params.Scale(),
		RegisteredCount: len(clientIDs),
	}, nil
}

// registerClientUnsafe 注册客户端（无锁，内部调用）
func (s *HEService) registerClientUnsafe(id string) {
	s.IdSet.Add(id)
	sk, pk := s.Kgen.GenKeyPair(id)
	r := s.Kgen.GenSecretKey(id)
	rlk := s.Kgen.GenRelinearizationKey(sk, r)
	s.SkSet.AddSecretKey(sk)
	s.PkSet.AddPublicKey(pk)
	s.RlkSet.AddRelinearizationKey(rlk)
}

// ================================================================
//  客户端注册
// ================================================================

// RegisterClient 动态注册一个新的 FL 客户端
func (s *HEService) RegisterClient(clientID string) (*RegisterClientResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.Initialized {
		return nil, fmt.Errorf("系统未初始化")
	}

	for _, id := range s.ClientList {
		if id == clientID {
			return nil, fmt.Errorf("客户端 %s 已注册", clientID)
		}
	}

	s.registerClientUnsafe(clientID)
	s.ClientList = append(s.ClientList, clientID)

	return &RegisterClientResponse{
		ClientID:     clientID,
		HasPublicKey: true,
		HasSecretKey: true,
	}, nil
}

// ================================================================
//  模型加密（自动分 chunk）
// ================================================================

// EncryptModel 对客户端的明文权重进行 MKHE 加密
// 当权重数量超过 slots 时自动拆分为多个密文
func (s *HEService) EncryptModel(clientID string, weights []float64, round int, layerTag string) (*EncryptModelResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.Initialized {
		return nil, fmt.Errorf("系统未初始化")
	}

	pk, hasPk := s.PkSet.Value[clientID]
	if !hasPk {
		return nil, fmt.Errorf("未找到客户端 %s 的公钥", clientID)
	}

	slots := s.Params.Slots()
	chunkCount := int(math.Ceil(float64(len(weights)) / float64(slots)))
	if chunkCount == 0 {
		chunkCount = 1
	}

	cipherIDs := make([]string, chunkCount)

	for i := 0; i < chunkCount; i++ {
		start := i * slots
		end := start + slots
		if end > len(weights) {
			end = len(weights)
		}

		// 将 float64 转为 complex128（虚部为 0）
		values := make([]complex128, slots)
		for j := 0; j < end-start; j++ {
			values[j] = complex(weights[start+j], 0)
		}

		msg := mkckks.NewMessage(s.Params)
		copy(msg.Value, values)
		ct := s.Encryptor.EncryptMsgNew(msg, pk)

		cid := generateID()
		s.CipherStore[cid] = &CipherEntry{
			Ciphertext: ct,
			ClientID:   clientID,
			Round:      round,
			LayerTag:   layerTag,
			ChunkIndex: i,
		}
		cipherIDs[i] = cid
	}

	return &EncryptModelResponse{
		CipherIDs:  cipherIDs,
		ChunkCount: chunkCount,
		Round:      round,
	}, nil
}

// ================================================================
//  同态聚合
// ================================================================

// Aggregate 对多个客户端的密文执行同态加法聚合
// cipherGroups[i] = 第 i 个 chunk 来自各参与方的密文 ID
func (s *HEService) Aggregate(cipherGroups [][]string, round int, average bool, clientCount int) (*AggregateResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.Initialized {
		return nil, fmt.Errorf("系统未初始化")
	}

	resultIDs := make([]string, len(cipherGroups))

	for gi, group := range cipherGroups {
		if len(group) == 0 {
			return nil, fmt.Errorf("第 %d 组密文列表为空", gi)
		}

		first, ok := s.CipherStore[group[0]]
		if !ok {
			return nil, fmt.Errorf("密文 %s 不存在", group[0])
		}

		result := first.Ciphertext.CopyNew()

		// 逐一同态相加
		for _, cid := range group[1:] {
			entry, ok := s.CipherStore[cid]
			if !ok {
				return nil, fmt.Errorf("密文 %s 不存在", cid)
			}
			result = s.Evaluator.AddNew(result, entry.Ciphertext)
		}

		// 可选: 乘以 1/N 求均值（消耗一层 level）
		if average && clientCount > 1 {
			scaleFactor := 1.0 / float64(clientCount)
			s.Evaluator.MultByConst(result, scaleFactor, result)
			if err := s.Evaluator.Rescale(result, s.Params.Scale(), result); err != nil {
				return nil, fmt.Errorf("rescale 失败: %w", err)
			}
		}

		rid := generateID()
		s.CipherStore[rid] = &CipherEntry{
			Ciphertext: result,
			ClientID:   "aggregated",
			Round:      round,
			LayerTag:   first.LayerTag,
			ChunkIndex: first.ChunkIndex,
		}
		resultIDs[gi] = rid
	}

	return &AggregateResponse{
		AggregatedCipherIDs: resultIDs,
		Round:               round,
	}, nil
}

// ================================================================
//  集中式解密（需全部密钥）
// ================================================================

// Decrypt 使用所有参与方的密钥集合完整解密
func (s *HEService) Decrypt(cipherIDs []string, weightCount int) (*DecryptResponse, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if !s.Initialized {
		return nil, fmt.Errorf("系统未初始化")
	}

	var allWeights []float64

	for _, cid := range cipherIDs {
		entry, ok := s.CipherStore[cid]
		if !ok {
			return nil, fmt.Errorf("密文 %s 不存在", cid)
		}
		msg := s.Decryptor.Decrypt(entry.Ciphertext, s.SkSet)
		for _, v := range msg.Value {
			allWeights = append(allWeights, real(v))
		}
	}

	// 截断到原始权重长度（去除 slot 填充的零）
	if weightCount > 0 && weightCount < len(allWeights) {
		allWeights = allWeights[:weightCount]
	}

	return &DecryptResponse{
		Weights: allWeights,
		Length:  len(allWeights),
	}, nil
}

// ================================================================
//  分布式部分解密（安全协议）
// ================================================================

// PartialDecrypt 使用指定客户端的密钥执行部分解密
// 返回部分解密后的新密文 ID（原密文不变）
func (s *HEService) PartialDecrypt(cipherIDs []string, clientID string) (*PartialDecryptResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.Initialized {
		return nil, fmt.Errorf("系统未初始化")
	}

	sk, hasSk := s.SkSet.Value[clientID]
	if !hasSk {
		return nil, fmt.Errorf("未找到客户端 %s 的密钥", clientID)
	}

	newIDs := make([]string, len(cipherIDs))

	for i, cid := range cipherIDs {
		entry, ok := s.CipherStore[cid]
		if !ok {
			return nil, fmt.Errorf("密文 %s 不存在", cid)
		}

		// 拷贝密文后执行部分解密（不修改原密文）
		ctCopy := entry.Ciphertext.CopyNew()
		s.Decryptor.PartialDecrypt(ctCopy, sk)

		nid := generateID()
		s.CipherStore[nid] = &CipherEntry{
			Ciphertext: ctCopy,
			ClientID:   "partial_" + clientID,
			Round:      entry.Round,
			LayerTag:   entry.LayerTag,
			ChunkIndex: entry.ChunkIndex,
		}
		newIDs[i] = nid
	}

	// 计算剩余需要解密的参与方数量
	remaining := 0
	if len(cipherIDs) > 0 {
		if entry, ok := s.CipherStore[newIDs[0]]; ok {
			idset := entry.Ciphertext.IDSet()
			remaining = idset.Size()
		}
	}

	return &PartialDecryptResponse{
		CipherIDs:        newIDs,
		RemainingParties: remaining,
	}, nil
}

// FinalDecrypt 在所有部分解密完成后提取明文
// 此时密文 IDSet 应为空（所有参与方密钥已消除）
func (s *HEService) FinalDecrypt(cipherIDs []string, weightCount int) (*DecryptResponse, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if !s.Initialized {
		return nil, fmt.Errorf("系统未初始化")
	}

	var allWeights []float64

	for _, cid := range cipherIDs {
		entry, ok := s.CipherStore[cid]
		if !ok {
			return nil, fmt.Errorf("密文 %s 不存在", cid)
		}

		// 部分解密完毕后直接用空 SkSet 解密（所有 sk 已消除）
		msg := s.Decryptor.Decrypt(entry.Ciphertext, mkrlwe.NewSecretKeySet())
		for _, v := range msg.Value {
			allWeights = append(allWeights, real(v))
		}
	}

	if weightCount > 0 && weightCount < len(allWeights) {
		allWeights = allWeights[:weightCount]
	}

	return &DecryptResponse{
		Weights: allWeights,
		Length:  len(allWeights),
	}, nil
}

// ================================================================
//  密文管理
// ================================================================

// GetCipherInfo 获取单个密文元信息
func (s *HEService) GetCipherInfo(cipherID string) (*CipherInfoResponse, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	entry, ok := s.CipherStore[cipherID]
	if !ok {
		return nil, fmt.Errorf("密文 %s 不存在", cipherID)
	}

	// 提取 IDSet
	var ids []string
	idset := entry.Ciphertext.IDSet()
	for id := range idset.Value {
		ids = append(ids, id)
	}

	return &CipherInfoResponse{
		CipherID:   cipherID,
		IDSet:      ids,
		Level:      entry.Ciphertext.Level(),
		Scale:      entry.Ciphertext.Scale,
		Round:      entry.Round,
		ClientID:   entry.ClientID,
		LayerTag:   entry.LayerTag,
		ChunkIndex: entry.ChunkIndex,
	}, nil
}

// DeleteCipher 删除单个密文
func (s *HEService) DeleteCipher(cipherID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.CipherStore[cipherID]; !ok {
		return fmt.Errorf("密文 %s 不存在", cipherID)
	}
	delete(s.CipherStore, cipherID)
	return nil
}

// Cleanup 批量清理密文
func (s *HEService) Cleanup(round int) *CleanupResponse {
	s.mu.Lock()
	defer s.mu.Unlock()

	count := 0
	for id, entry := range s.CipherStore {
		if round == 0 || entry.Round == round {
			delete(s.CipherStore, id)
			count++
		}
	}
	return &CleanupResponse{DeletedCount: count}
}

// ================================================================
//  轮次管理
// ================================================================

// AdvanceRound 推进训练轮次
func (s *HEService) AdvanceRound() *AdvanceRoundResponse {
	s.mu.Lock()
	defer s.mu.Unlock()

	prev := s.CurrentRound
	s.CurrentRound++
	return &AdvanceRoundResponse{
		PreviousRound: prev,
		CurrentRound:  s.CurrentRound,
	}
}

// GetStatus 获取系统状态
func (s *HEService) GetStatus() *SystemStatusResponse {
	s.mu.RLock()
	defer s.mu.RUnlock()

	resp := &SystemStatusResponse{
		Initialized:   s.Initialized,
		CurrentRound:  s.CurrentRound,
		RegisteredIDs: append([]string{}, s.ClientList...),
		CipherCount:   len(s.CipherStore),
	}

	if s.Initialized {
		resp.Slots = s.Params.Slots()
		resp.MaxLevel = s.Params.MaxLevel()
	}

	return resp
}
