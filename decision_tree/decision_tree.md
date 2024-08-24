普通的决策树（ID3算法）

```go
// 数据点结构
type DataPoint struct {
    // 存鸢尾花的4个特征
	Features []float64
    // 品种
	Label    string
}

// 节点结构
type TreeNode struct {
	Label        string
	FeatureIndex int
	Threshold    float64
	Left, Right  *TreeNode
	IsLeaf       bool
}

// 读取CSV文件，返回一个存储鸢尾花信息的数组
func readCSV(filename string) ([]DataPoint, error) {}

// 计算当前节点的熵（根据ID3算法）
func entropy(data []DataPoint) float64 {}

/*
 * 根据每个特征的阈值进行数据的分割
 * data 当前数组中的鸢尾花信息
 * featureIndex 当前选取的特征(鸢尾花的4个特征之一)
 * threshold 根据该阈值将当前数组分割为左右两个数组，将样例中对应特征值大于该阈值的放入右侧，否则放入左侧
 * 返回 left, right 两个数组
 */
func splitData(data []DataPoint, featureIndex int, threshold float64) ([]DataPoint, []DataPoint) {}

// 在当前节点找到最佳分割的特征
/*
 * 对于当前节点中的所有数据，通过ID3算法计算出具有最大信息增益的特征
 * 返回值: 
 * 		bestFeature 在当前节点选取的最佳分割特征
 * 		bestThreshold 选取的特征中的最佳分配阈值
 *		bestGain 得到的最大信息增益		
 */
func bestSplit(data []DataPoint) (int, float64, float64) {
    ...
	return bestFeature, bestThreshold, bestGain
}

// 递归构建决策树
// 在当前节点中根据bestSplit函数分割为左右子树
/*
 * 参数：
 *		data 当前节点还存在的鸢尾花数据数组
 *		depth 当前节点所在的深度
 *		maxDepth 决策树的最大深度
 */
func buildTree(data []DataPoint, depth int, maxDepth int) *TreeNode {
	...
	return &TreeNode{
		FeatureIndex: feature,
		Threshold:    threshold,
		Left:         buildTree(leftData, depth+1, maxDepth),
		Right:        buildTree(rightData, depth+1, maxDepth),
	}
}

// 预测函数
func predict(tree *TreeNode, point DataPoint) string {
	if tree.IsLeaf {
		return tree.Label
	}

	if point.Features[tree.FeatureIndex] < tree.Threshold {
		return predict(tree.Left, point)
	} else {
		return predict(tree.Right, point)
	}
}

// 主函数
func main() {
	// 读取数据集
	data, err := readCSV(".\\iris\\iris.csv")
	if err != nil {
		panic(err)
	}

	// shuffle the data
	rand.Shuffle(len(data), func(i, j int) { data[i], data[j] = data[j], data[i] })

	// 构建决策树
	tree := buildTree(data, 0, 5)

	// 测试预测
	for _, point := range data[:5] {
		prediction := predict(tree, point)
		fmt.Printf("Prediction: %s, Actual: %s\n", prediction, point.Label)
	}
}


```

