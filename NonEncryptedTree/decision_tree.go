package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

// 数据点结构
type DataPoint struct {
	Features []float64
	Label    int
}

// 节点结构
type TreeNode struct {
	Label        int
	FeatureIndex int
	Threshold    float64
	Left, Right  *TreeNode
	IsLeaf       bool
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
		label, err := strconv.Atoi(record[0])
		var features []float64
		for _, value := range record[1:] {
			feature, err := strconv.ParseFloat(value, 64)
			if err != nil {
				return nil, err
			}
			features = append(features, feature)
		}
		if err != nil {
			panic(err)
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

// 构建决策树
func buildTree(data []DataPoint, depth int, maxDepth int) *TreeNode {
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

	if len(labelCounts) == 1 || depth == maxDepth {
		return &TreeNode{Label: majorityLabel, IsLeaf: true}
	}

	feature, threshold, gain := bestSplit(data)
	if gain == 0 {
		return &TreeNode{Label: majorityLabel, IsLeaf: true}
	}

	leftData, rightData := splitData(data, feature, threshold)
	return &TreeNode{
		FeatureIndex: feature,
		Threshold:    threshold,
		Left:         buildTree(leftData, depth+1, maxDepth),
		Right:        buildTree(rightData, depth+1, maxDepth),
	}
}

// 预测
func predict(tree *TreeNode, point DataPoint) int {
	if tree.IsLeaf {
		return tree.Label
	}

	if point.Features[tree.FeatureIndex] < tree.Threshold {
		return predict(tree.Left, point)
	} else {
		return predict(tree.Right, point)
	}
}

// PrintDecisionTree 递归打印决策树
func PrintDecisionTree(node *TreeNode, depth int, branch string) {
	if node == nil {
		return
	}

	indent := strings.Repeat("    ", depth) // 根据深度生成缩进，四个空格
	if node.IsLeaf {
		fmt.Printf("%s%s└── Leaf Node: Label = %d\n", indent, branch, node.Label)
	} else {
		fmt.Printf("%s%s├── Decision Node: Feature[%d] <= %.2f\n", indent, branch, node.FeatureIndex, node.Threshold)
		// 递归打印左子树
		PrintDecisionTree(node.Left, depth+1, "L ")
		// 递归打印右子树
		PrintDecisionTree(node.Right, depth+1, "R ")
	}
}

func main() {

	file, _ := os.OpenFile("../decision_tree/result/decision_tree_NonEncrypted.txt", os.O_RDWR|os.O_APPEND|os.O_CREATE, 0644)
	defer file.Close()

	beginTime := time.Now()

	// 读取数据集
	data, err := readCSV("../decision_tree/iris/iris.data")
	if err != nil {
		panic(err)
	}

	// shuffle the data
	rand.Shuffle(len(data), func(i, j int) { data[i], data[j] = data[j], data[i] })
	train_data := data[0:125]
	test_data := data[125:]

	// 构建决策树
	tree := buildTree(train_data, 0, 10)

	// 打印决策树
	fmt.Println("决策树结构：")
	PrintDecisionTree(tree, 0, "")

	cnt := 0.0
	testDataSize := len(test_data)
	// 测试预测
	for _, point := range test_data {
		prediction := predict(tree, point)
		if prediction == point.Label {
			cnt++
		}
		// fmt.Printf("Prediction: %d, Actual: %d\n", prediction, point.Label)
	}
	accuracy := cnt / float64(testDataSize)
	fmt.Printf("正确率：%f", accuracy)
	fmt.Fprintf(file, "正确率：%f\n", accuracy)

	endTime := time.Since(beginTime)
	fmt.Println("\nRun time: ", endTime)
	fmt.Fprintf(file, "\nRun time: "+endTime.String()+"\n\n")
}
