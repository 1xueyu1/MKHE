package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"mk-lattigo/mkckks"
	"mk-lattigo/mkrlwe"
	"os"
	"strconv"
	"time"

	"github.com/ldsec/lattigo/v2/ckks"
)

type DataPoint struct {
	Features []float64
	Label    int
}

type TreeNode struct {
	Label        mkckks.Ciphertext
	FeatureIndex int
	Threshold    mkckks.Ciphertext
	Left, Right  *TreeNode
	IsLeaf       bool
}

type DataPointCt struct {
	Feature []mkckks.Ciphertext
	Label   mkckks.Ciphertext
}

func main() {

	// 写入文件中保存
	file, _ := os.OpenFile("./result/desion_tree_MKHE.txt", os.O_APPEND|os.O_CREATE|os.O_RDWR, 0664)

	beginTime := time.Now()

	ckksParams, err := ckks.NewParametersFromLiteral(ckks.PN15QP880)
	if err != nil {
		panic(err)
	}
	params := mkckks.NewParameters(ckksParams)

	genParaTime := time.Since(beginTime)
	fmt.Fprintf(file, "参数 PN15QP880\n")
	fmt.Fprintf(file, "生成参数耗时：%s \n", genParaTime.String())

	// generate keys
	kgen := mkckks.NewKeyGenerator(params)
	skSet := mkrlwe.NewSecretKeySet()
	sk1 := kgen.GenSecretKey("data")
	sk2 := kgen.GenSecretKey("model")
	pk1 := kgen.GenPublicKey(sk1)
	pk2 := kgen.GenPublicKey(sk2)
	skSet.AddSecretKey(sk1)
	skSet.AddSecretKey(sk2)

	genKeyTime := time.Since(beginTime)
	fmt.Fprintf(file, "生成密钥耗时：%s \n", (genKeyTime - genParaTime).String())

	// create encryptor and decryptor
	encryptor := mkckks.NewEncryptor(params)
	decryptor := mkckks.NewDecryptor(params)
	evaluator := mkckks.NewEvaluator(params)

	// 读取CSV文件
	data, err := readCSV("./iris/iris.csv")
	if err != nil {
		panic(err)
	}

	// shuffle the data
	rand.Shuffle(len(data), func(i, j int) { data[i], data[j] = data[j], data[i] })
	train_data := data[0:125]
	test_data := data[125:]

	tree := buildTree(train_data, 0, 10, encryptor, pk2)
	encrypt_test_data := encryptDataPoint(test_data, *encryptor, pk1)

	cnt := 0.0
	// 预测加密的 test_data 数据
	for i, encrypt_data_item := range encrypt_test_data {
		// 得到预测结果的密文
		predict_ct := predict(tree, &encrypt_data_item, evaluator, decryptor, skSet)
		// 将预测结果解密
		predict_msgOut := decryptor.Decrypt(&predict_ct, skSet)
		// 比对结果是否正确
		str := strconv.FormatFloat(real(predict_msgOut.Value[0]), 'f', 0, 64)
		target, _ := strconv.ParseInt(str, 10, 64)

		if target == int64(test_data[i].Label) {
			cnt += 1
		}
		s := "Actual: " + strconv.Itoa(test_data[i].Label) + "  predict: " + str + "\n"
		fmt.Print(s)
	}
	testDataSize := len(encrypt_test_data)
	accuracy := cnt / float64(testDataSize)
	fmt.Printf("\n%f\n", accuracy)

	endTime := time.Since(beginTime)
	fmt.Fprintf(file, "总耗时：%s \n", endTime.String())
	fmt.Fprintf(file, "正确率：%f \n\n", accuracy)

	file.Close()
}

// 将 DataPoint 的数据加密为 DataPointCt
func encryptDataPoint(data []DataPoint, encryptor mkckks.Encryptor, pk *mkrlwe.PublicKey) (dataCt []DataPointCt) {
	for _, item := range data {
		dataPointCt := new(DataPointCt)
		featureMsg := make([]mkckks.Message, 0)
		for _, num := range item.Features {
			msg := new(mkckks.Message)
			msg.Value = append(msg.Value, complex(num, 0.0))
			featureMsg = append(featureMsg, *msg)
		}
		for _, msg := range featureMsg {
			ct1 := encryptor.EncryptMsgNew(&msg, pk)
			dataPointCt.Feature = append(dataPointCt.Feature, *ct1.CopyNew())
		}

		labelMsg := new(mkckks.Message)
		labelMsg.Value = append(labelMsg.Value, complex(float64(item.Label), 0.0))
		ct2 := encryptor.EncryptMsgNew(labelMsg, pk)
		dataPointCt.Label = *ct2.CopyNew()

		dataCt = append(dataCt, *dataPointCt)
	}
	return
}

// 预测
func predict(tree *TreeNode, point *DataPointCt, evaluator *mkckks.Evaluator, decryptor *mkckks.Decryptor, skSet *mkrlwe.SecretKeySet) mkckks.Ciphertext {
	if tree.IsLeaf {
		return tree.Label
	}

	// 计算差值
	delta_ct := evaluator.SubNew(&point.Feature[tree.FeatureIndex], &tree.Threshold)
	delta_msgOut := decryptor.Decrypt(delta_ct, skSet)

	if real(delta_msgOut.Value[0]) < 0.0 && math.Abs(real(delta_msgOut.Value[0])) > 0.0005 {
		return predict(tree.Left, point, evaluator, decryptor, skSet)
	} else {
		return predict(tree.Right, point, evaluator, decryptor, skSet)
	}
}

// 构建决策树
func buildTree(data []DataPoint, depth int, maxDepth int, encryptor *mkckks.Encryptor, pk *mkrlwe.PublicKey) *TreeNode {
	if len(data) == 0 {
		return nil
	}

	labelCounts := make(map[int]int)
	for _, point := range data {
		labelCounts[point.Label]++
	}

	var majorityLabel int
	maxCount := 0
	for label, count := range labelCounts {
		if count > maxCount {
			maxCount = count
			majorityLabel = label
		}
	}

	majorityLabel_msg := new(mkckks.Message)
	threshold_msg := new(mkckks.Message)
	majorityLabel_msg.Value = append(majorityLabel_msg.Value, complex(float64(majorityLabel), 0))
	majorityLabel_ct := encryptor.EncryptMsgNew(majorityLabel_msg, pk)

	if len(labelCounts) == 1 || depth == maxDepth {
		return &TreeNode{Label: *majorityLabel_ct, IsLeaf: true}
	}

	feature, threshold, gain := bestSplit(data)
	if gain == 0 {
		return &TreeNode{Label: *majorityLabel_ct, IsLeaf: true}
	}

	threshold_msg.Value = append(threshold_msg.Value, complex(threshold, 0))
	threshold_ct := encryptor.EncryptMsgNew(threshold_msg, pk)

	leftData, rightData := splitData(data, feature, threshold)
	return &TreeNode{
		FeatureIndex: feature,
		Threshold:    *threshold_ct,
		Left:         buildTree(leftData, depth+1, maxDepth, encryptor, pk),
		Right:        buildTree(rightData, depth+1, maxDepth, encryptor, pk),
	}
}

// 读取CSV文件
func readCSV(filename string) ([]DataPoint, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var data []DataPoint
	for _, record := range records {
		var features []float64
		for _, value := range record[:len(record)-1] {
			feature, err := strconv.ParseFloat(value, 64)
			if err != nil {
				return nil, err
			}
			features = append(features, feature)
		}
		label := int(0)
		name := record[len(record)-1]
		if name == "Iris-setosa" {
			label = 1
		} else if name == "Iris-versicolor" {
			label = 2
		} else {
			label = 3
		}
		data = append(data, DataPoint{Features: features, Label: label})
	}

	return data, nil
}

// 计算熵
func entropy(data []DataPoint) float64 {
	labelCounts := make(map[int]int)
	for _, point := range data {
		labelCounts[point.Label]++
	}

	var ent float64
	total := float64(len(data))
	for _, count := range labelCounts {
		p := float64(count) / total
		ent -= p * math.Log2(p)
	}
	return ent
}

// 根据特征和阈值分割数据
func splitData(data []DataPoint, featureIndex int, threshold float64) ([]DataPoint, []DataPoint) {
	var left, right []DataPoint
	for _, point := range data {
		if point.Features[featureIndex] < threshold {
			left = append(left, point)
		} else {
			right = append(right, point)
		}
	}
	return left, right
}

// 选择最佳分割点
func bestSplit(data []DataPoint) (int, float64, float64) {
	bestFeature := -1
	bestThreshold := 0.0
	bestGain := 0.0
	baseEntropy := entropy(data)

	for i := range data[0].Features {
		thresholds := make(map[float64]struct{})
		for _, point := range data {
			thresholds[point.Features[i]] = struct{}{}
		}

		for threshold := range thresholds {
			left, right := splitData(data, i, threshold)
			if len(left) == 0 || len(right) == 0 {
				continue
			}

			pLeft := float64(len(left)) / float64(len(data))
			pRight := float64(len(right)) / float64(len(data))
			gain := baseEntropy - pLeft*entropy(left) - pRight*entropy(right)

			if gain > bestGain {
				bestGain = gain
				bestFeature = i
				bestThreshold = threshold
			}
		}
	}

	return bestFeature, bestThreshold, bestGain
}

// // generateRandomComplexArray 生成包含随机 complex128 数组
// func generateRandomComplexArray(size int, lowerBound, upperBound float64) []complex128 {
// 	rand.Seed(time.Now().UnixNano()) // 设置随机种子
// 	array := make([]complex128, size)
// 	for i := 0; i < size; i++ {
// 		realPart := lowerBound + rand.Float64()*(upperBound-lowerBound)
// 		array[i] = complex(realPart, 0.0)
// 	}
// 	return array
// }
