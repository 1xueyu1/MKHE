package cnn

import (
	"fmt"
	"io/ioutil"
	"strconv"
	"strings"

	"testing"

	"github.com/stretchr/testify/require"

	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"

	"mk-lattigo/mkckks"
	"mk-lattigo/mkrlwe"
)

var (
	imageSize = 28    // 输入图像的尺寸（宽度和高度），即图像是 28x28 像素
	numClass  = 10    // 分类的类别数量，通常用于 MNIST 数据集，该数据集有 10 类（数字 0-9）
	numImage  = 10000 // 数据集中图像的总数量

	numKernels  = 5  // 卷积层中使用的卷积核（过滤器）的数量
	kernelSize  = 4  // 卷积核的尺寸（宽度和高度），即每个卷积核是 4x4 像素
	stride      = 2  // 卷积操作的步幅，表示卷积核每次移动的像素数
	blockSize   = 14 // 每个块的尺寸，用于将图像分块处理
	convOutSize = 13 // 卷积层输出的尺寸，表示卷积操作后得到的特征图的大小
	numFCUnit   = 64 // 全连接层中的单元（神经元）数量

	gap = 128 // 一个间隔值，可能用于图像处理或数据组织

	dataOwner  = "dataOwner"  // 数据所有者的标识符，用于标识拥有输入数据的实体
	modelOwner = "modelOwner" // 模型所有者的标识符，用于标识拥有训练模型的实体
)

func GetTestName(params mkckks.Parameters, opname string) string {
	return fmt.Sprintf("%sLogN=%d/LogSlots=%d/LogQP=%d/Levels=%d/",
		opname,
		params.LogN(),
		params.LogSlots(),
		params.LogQP(),
		params.MaxLevel()+1)
}

type testParams struct {
	params mkckks.Parameters
	ringQ  *ring.Ring
	ringP  *ring.Ring
	prng   utils.PRNG
	kgen   *mkrlwe.KeyGenerator
	skSet  *mkrlwe.SecretKeySet
	pkSet  *mkrlwe.PublicKeySet
	rlkSet *mkrlwe.RelinearizationKeySet
	rtkSet *mkrlwe.RotationKeySet

	encryptor *mkckks.Encryptor
	decryptor *mkckks.Decryptor
	evaluator *mkckks.Evaluator
	idset     *mkrlwe.IDSet
}

const imagefile = "./data/mnist_test.csv"
const k0file = "./data/k1.txt"
const FC1file = "./data/FC1.txt"
const FC2file = "./data/FC2.txt"
const B1file = "./data/B1.txt"
const B2file = "./data/B2.txt"

var (
	classes   []complex128     // 分类标签，共有 10000 个元素
	images    [][][]complex128 // 输入图像数据，尺寸为 10000 x 28 x 28
	kernels   [][][]complex128 // 卷积核数据，尺寸为 5 x 4 x 4
	FC1       [][]complex128   // 全连接层 1 的权重矩阵，尺寸为 845 x 64
	FC2       [][]complex128   // 全连接层 2 的权重矩阵，尺寸为 64 x 10
	B1        []complex128     // 全连接层 1 的偏置，长度为 64
	B2        []complex128     // 全连接层 2 的偏置，长度为 10
	PN14QP433 = ckks.ParametersLiteral{
		LogN:     14,
		LogSlots: 13,
		Q: []uint64{
			//57 + 47 x 6
			0x2000000002b0001,
			0x800000020001, 0x800000280001,
			0x800000520001, 0x800000770001,
			0x800000aa0001, 0x800000ad0001,
		},
		P: []uint64{
			//47 x 2
			0x800000df0001, 0x800000f80001,
		},
		Scale: 1 << 47,
		Sigma: rlwe.DefaultSigma,
	}
)

func TestCNN(t *testing.T) {
	// 定义默认参数
	defaultParam := PN14QP433
	// 从默认参数创建 CKKS 参数
	ckksParams, err := ckks.NewParametersFromLiteral(defaultParam)
	// 创建 MKCKKS 参数
	params := mkckks.NewParameters(ckksParams)

	// 检查是否有错误
	if err != nil {
		panic(err)
	}

	// 定义测试上下文变量
	var testContext *testParams
	// 创建 ID 集合
	idset := mkrlwe.NewIDSet()
	// 添加模型所有者到 ID 集合
	idset.Add(modelOwner)
	// 添加数据所有者到 ID 集合
	idset.Add(dataOwner)

	// 生成测试参数
	if testContext, err = genTestParams(params, idset); err != nil {
		panic(err)
	}

	// 读取数据(images, 卷积核, 全连接层1、2的偏置值和权重矩阵)
	readData()

	// 预计算部分
	imageIndex := 2 // 选择要加密的图像索引
	// 加密图像
	ctImage := encryptImage(testContext, dataOwner, imageIndex)
	// 加密卷积核
	ctKernels := encryptKernels(testContext, modelOwner)
	// 加密全连接层1的权重
	ctFC1 := encryptFC1(testContext, modelOwner)
	// 加密全连接层2的权重
	ctFC2 := encryptFC2(testContext, modelOwner)
	// 加密偏置1
	ctB1 := encryptB1(testContext, modelOwner)
	// 加密偏置2
	ctB2 := encryptB2(testContext, modelOwner)

	// 获取评估器
	eval := testContext.evaluator
	// 获取重新线性化密钥集(降低密文阶数、减少噪声积累)
	rlkSet := testContext.rlkSet
	// 获取旋转密钥集(使得在加密域内能够进行复杂的矩阵运算和向量操作)
	rtkSet := testContext.rtkSet

	// 将加密图像转换为提升形式
	// 卷积操作需要对输入图像的密文进行多次旋转和乘法，使用提升形式可以显著提高这些操作的效率
	// 全连接层中的矩阵乘法操作也会从提升形式中受益，使得矩阵向量乘法更高效
	ctImageHoisted := eval.HoistedForm(ctImage)
	// 初始化提升形式的卷积核数组
	ctKernelsHoisted := make([]*mkrlwe.HoistedCiphertext, len(ctKernels))
	for i := 0; i < len(ctKernelsHoisted); i++ {
		// 将每个卷积核转换为提升形式
		ctKernelsHoisted[i] = eval.HoistedForm(ctKernels[i])
	}
	// 初始化提升形式的全连接层1权重数组
	ctFC1Hoisted := make([]*mkrlwe.HoistedCiphertext, len(ctFC1))
	for i := 0; i < len(ctFC1Hoisted); i++ {
		// 将每个全连接层1的权重转换为提升形式
		ctFC1Hoisted[i] = eval.HoistedForm(ctFC1[i])
	}

	// 获取加密参数的插槽数
	numSlots := testContext.params.Slots()
	// 创建掩码数组=============?(应用偏置, 特定位置累加值->加权求和, 卷积操作中的选择性权重应用)
	mask := make([]complex128, numSlots)
	for i := 0; i < numSlots; i += 128 {
		// 每隔128个插槽设置掩码为1
		mask[i] = 1
	}
	// 创建消息
	msg := mkckks.NewMessage(testContext.params)
	// 设置消息的值为掩码
	msg.Value = mask
	// 编码消息为明文
	ptMask := testContext.encryptor.EncodeMsgNew(msg)

	// 评估部分
	// 执行卷积操作==========================？
	convOut := Convolution(eval, rlkSet, rtkSet, ctImage, ctImageHoisted, ctKernels, ctKernelsHoisted)

	// 将卷积输出转换为提升形式
	convOutHoisted := eval.HoistedForm(convOut)
	// 对卷积输出进行平方操作
	square1Out := eval.MulRelinHoistedNew(convOut, convOut, convOutHoisted, convOutHoisted, rlkSet)
	// 将平方输出转换为提升形式
	square1OutHoisted := eval.HoistedForm(square1Out)

	// 执行全连接层1操作 实现矩阵相乘
	fc1Out := FC1Layer(eval, rlkSet, rtkSet, square1Out, square1OutHoisted, ctFC1, ctFC1Hoisted, ctB1)
	// 将全连接层1输出转换为提升形式
	fc1OutHoisted := eval.HoistedForm(fc1Out)
	// 对全连接层1的输出进行平方操作
	square2Out := eval.MulRelinHoistedNew(fc1Out, fc1Out, fc1OutHoisted, fc1OutHoisted, rlkSet)

	// 执行全连接层2操作
	fc2Out := FC2Layer(eval, rlkSet, rtkSet, square2Out, ctFC2, ctB2, ptMask)

	// 解密部分
	// 解密最终输出
	ptResult := testContext.decryptor.Decrypt(fc2Out, testContext.skSet)
	// 获取解密结果的值
	value := ptResult.Value
	// 初始化最大值索引和最大值
	maxIndex := -1
	maxValue := -100.0
	for i := 0; i < 10; i++ {
		// 找到最大值的索引
		if real(value[i]) > maxValue {
			maxValue = real(value[i])
			maxIndex = i
		}
	}

	// 获取图像的真实分类标签
	answer := int(real(classes[imageIndex]))
	// 断言预测结果是否等于真实结果
	require.Equal(t, answer, maxIndex)
}

func genTestParams(defaultParam mkckks.Parameters, idset *mkrlwe.IDSet) (testContext *testParams, err error) {
	testContext = new(testParams)

	testContext.params = defaultParam

	rots := []int{14, 15, 384, 512, 640, 768, 896, 8191, 8190, 8188, 8184}

	for _, rot := range rots {
		testContext.params.AddCRS(rot)
	}

	testContext.kgen = mkckks.NewKeyGenerator(testContext.params)

	testContext.skSet = mkrlwe.NewSecretKeySet()
	testContext.pkSet = mkrlwe.NewPublicKeyKeySet()
	testContext.rlkSet = mkrlwe.NewRelinearizationKeyKeySet(testContext.params.Parameters)
	testContext.rtkSet = mkrlwe.NewRotationKeySet()

	for i := 0; i < testContext.params.LogN()-1; i++ {
		rots = append(rots, 1<<i)
	}

	for id := range idset.Value {
		sk, pk := testContext.kgen.GenKeyPair(id)
		r := testContext.kgen.GenSecretKey(id)
		rlk := testContext.kgen.GenRelinearizationKey(sk, r)

		for _, rot := range rots {
			rk := testContext.kgen.GenRotationKey(rot, sk)
			testContext.rtkSet.AddRotationKey(rk)
		}

		testContext.skSet.AddSecretKey(sk)
		testContext.pkSet.AddPublicKey(pk)
		testContext.rlkSet.AddRelinearizationKey(rlk)
	}

	testContext.ringQ = defaultParam.RingQ()

	if testContext.prng, err = utils.NewPRNG(); err != nil {
		return nil, err
	}

	testContext.encryptor = mkckks.NewEncryptor(testContext.params)
	testContext.decryptor = mkckks.NewDecryptor(testContext.params)
	testContext.evaluator = mkckks.NewEvaluator(testContext.params)

	return testContext, nil
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Data Reading Functions //////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

func readData() {
	// 读取10000张图片的分类和数据
	readTestData(imagefile)
	// 读取5个卷积核数据
	readKData(k0file)
	// 读取全连接层1中的偏置值
	B1 = readBData(B1file, numFCUnit)
	// 读取全连接层2中的偏置值
	B2 = readBData(B2file, numClass)
	// 读取全连接层FC的权重数据convOutSize*convOutSize*numKernels(13x13x5), numFCUnit	845x64
	FC1 = readFCData(FC1file, convOutSize*convOutSize*numKernels, numFCUnit)
	// numFCUnit, numClass	64x10
	FC2 = readFCData(FC2file, numFCUnit, numClass)
}

// 读取10000张图片的分类和数据
func readTestData(filename string) {
	classes = make([]complex128, numImage)
	images = make([][][]complex128, numImage)
	for i := 0; i < numImage; i++ {
		images[i] = make([][]complex128, imageSize)
		for j := 0; j < imageSize; j++ {
			images[i][j] = make([]complex128, imageSize)
		}
	}

	f, error := ioutil.ReadFile(filename)
	if error != nil {
		panic(error)
	}

	lines := strings.Split(string(f), "\r\n")
	// 遍历10000张图片数据，每张图片都是第一个数字代表分类，后面28 x 28大小
	for i, l := range lines {
		values := strings.Split(l, ",")
		// 用images存下每张图片的数据
		for j, v := range values {
			value, error := strconv.ParseFloat(v, 64)

			if error != nil {
				panic(error)
			}
			if j == 0 {
				classes[i] = complex(value, 0)
			} else {
				images[i][(j-1)/imageSize][(j-1)%imageSize] = complex(value/255, 0)
			}
		}
	}
}

// 读取5个 4 x 4 卷积核数据
func readKData(filename string) {
	kernels = make([][][]complex128, numKernels)
	for i := 0; i < numKernels; i++ {
		kernels[i] = make([][]complex128, kernelSize)
		for j := 0; j < kernelSize; j++ {
			kernels[i][j] = make([]complex128, kernelSize)
		}
	}

	f, error := ioutil.ReadFile(filename)
	if error != nil {
		panic(error)
	}

	lines := strings.Split(string(f), "\r\n")
	for i, l := range lines {
		values := strings.Split(l, " ")
		for j, v := range values {
			value, error := strconv.ParseFloat(v, 64)
			if error != nil {
				panic(error)
			}
			kernels[i][j/kernelSize][j%kernelSize] = complex(float64(value), 0)
		}
	}
}

// 读取全连接层FC的权重数据
func readFCData(filename string, insize int, outsize int) (FCData [][]complex128) {
	FCData = make([][]complex128, insize)
	for i := 0; i < insize; i++ {
		FCData[i] = make([]complex128, outsize)
	}

	f, error := ioutil.ReadFile(filename)
	if error != nil {
		panic(error)
	}

	lines := strings.Split(string(f), "\r\n")
	for i, l := range lines {
		values := strings.Split(l, " ")
		for j, v := range values {
			value, error := strconv.ParseFloat(v, 64)
			if error != nil {
				panic(error)
			}
			FCData[i][j] = complex(float64(value), 0)
		}
	}

	return FCData
}

// 读取全连接层中的偏置值
func readBData(filename string, size int) (BData []complex128) {
	BData = make([]complex128, size)

	f, error := ioutil.ReadFile(filename)
	if error != nil {
		panic(error)
	}

	line := strings.Split(string(f), " ")
	for i, v := range line {
		value, error := strconv.ParseFloat(v, 64)
		if error != nil {
			panic(error)
		}
		BData[i] = complex(float64(value), 0)
	}

	return BData
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Encryption Functions ////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// 将28*28的图片分成2*2的像素块，将4个像素点间隔1024存储
func encryptImage(testContext *testParams, id string, imageIndex int) (ctOut *mkckks.Ciphertext) {
	if testContext.encryptor != nil {
		encodedImage := make([]complex128, 8192)

		// 卷积核数量
		for k := 0; k < numKernels; k++ {
			for i := 0; i < blockSize; i++ {
				for j := 0; j < blockSize; j++ {
					// max = 14x14x4 + 14x13 + 13 = 14x14x5 = 980
					index := blockSize*blockSize*k + blockSize*i + j
					encodedImage[index] = images[imageIndex][2*i][2*j]

					index += 1024
					encodedImage[index] = images[imageIndex][2*i][2*j+1]

					index += 1024
					encodedImage[index] = images[imageIndex][2*i+1][2*j]

					index += 1024
					encodedImage[index] = images[imageIndex][2*i+1][2*j+1]
				}
			}
		}

		for i := 0; i < 4096; i++ {
			encodedImage[i+4096] = encodedImage[i]
		}

		msg := mkckks.NewMessage(testContext.params)
		msg.Value = encodedImage
		ctOut = testContext.encryptor.EncryptMsgNew(msg, testContext.pkSet.GetPublicKey(id))
	} else {
		panic("cannot encrypt image: encryptor is not initialized")
	}
	return ctOut
}

// 将4x4的卷积核分成2x2的像素块
func encryptKernels(testContext *testParams, id string) (ctOut []*mkckks.Ciphertext) {
	if testContext.encryptor != nil {
		// encededKernels[0:4]4个数组存放2x2像素块中的4组像素点
		encodedKernels := make([][]complex128, kernelSize)
		for i := 0; i < kernelSize; i++ {
			encodedKernels[i] = make([]complex128, 8192)
		}

		// 因为共享参数所以和重复多次 和image大小相同？
		for i := 0; i < numKernels; i++ {
			// 卷积核	共享参数？
			for j := 0; j < convOutSize; j++ {
				for k := 0; k < convOutSize; k++ {
					index := blockSize*blockSize*i + blockSize*j + k
					// 2x2中左上
					encodedKernels[0][index] = kernels[i][0][0]
					encodedKernels[1][index] = kernels[i][0][2]
					encodedKernels[2][index] = kernels[i][2][0]
					encodedKernels[3][index] = kernels[i][2][2]

					// 右上
					index += 1024
					encodedKernels[0][index] = kernels[i][0][1]
					encodedKernels[1][index] = kernels[i][0][3]
					encodedKernels[2][index] = kernels[i][2][1]
					encodedKernels[3][index] = kernels[i][2][3]

					// 左下
					index += 1024
					encodedKernels[0][index] = kernels[i][1][0]
					encodedKernels[1][index] = kernels[i][1][2]
					encodedKernels[2][index] = kernels[i][3][0]
					encodedKernels[3][index] = kernels[i][3][2]

					// 右下
					index += 1024
					encodedKernels[0][index] = kernels[i][1][1]
					encodedKernels[1][index] = kernels[i][1][3]
					encodedKernels[2][index] = kernels[i][3][1]
					encodedKernels[3][index] = kernels[i][3][3]
				}
			}
		}

		for i := 0; i < 4; i++ {
			for j := 0; j < 4096; j++ {
				encodedKernels[i][j+4096] = encodedKernels[i][j]
			}
		}

		ctOut = make([]*mkckks.Ciphertext, kernelSize)
		for i := 0; i < kernelSize; i++ {
			msg := mkckks.NewMessage(testContext.params)
			msg.Value = encodedKernels[i]
			ctOut[i] = testContext.encryptor.EncryptMsgNew(msg, testContext.pkSet.GetPublicKey(id))
		}
	} else {
		panic("cannot encrypt image: encryptor is not initialized")
	}
	return ctOut
}

func encryptFC1(testContext *testParams, id string) (ctOut []*mkckks.Ciphertext) {
	// 检查是否初始化了加密器
	if testContext.encryptor != nil {
		// 创建一个临时的全连接层数据矩阵
		tempFC1 := make([][]complex128, numFCUnit)
		for i := 0; i < numFCUnit; i++ {
			tempFC1[i] = make([]complex128, 1024)
		}

		// 创建一个用于存储编码后的全连接层数据的矩阵
		encodedFC1 := make([][]complex128, 8)
		for i := 0; i < 8; i++ {
			encodedFC1[i] = make([]complex128, 8192)
		}

		// 填充 tempFC1 矩阵
		// FC1: 5*13*13 x 64
		// tempFC1: 64 x 1024
		// 卷积核索引
		for i := 0; i < numKernels; i++ {
			// 卷积输出行索引
			for j := 0; j < convOutSize; j++ {
				// 卷积输出列索引
				for k := 0; k < convOutSize; k++ {
					// 全连接层单元索引
					for l := 0; l < numFCUnit; l++ {
						tempFC1[l][blockSize*blockSize*i+blockSize*j+k] = FC1[i+numKernels*(j*convOutSize+k)][l]
					}
				}
			}
		}

		// 将 tempFC1 数据重新排列并填充到 encodedFC1 矩阵中
		// encodedFC1: 8 x 8192(64*128)
		for i := 0; i < 8; i++ {
			for j := 0; j < 64; j++ {
				for k := 0; k < 128; k++ {
					encodedFC1[i][128*j+k] = tempFC1[j][128*((i+j)%8)+k]
				}
			}
		}

		// 初始化密文输出切片
		ctOut = make([]*mkckks.Ciphertext, 8)
		for i := 0; i < 8; i++ {
			// 创建一个新的消息对象，并将 encodedFC1[i] 赋值给消息
			msg := mkckks.NewMessage(testContext.params)
			msg.Value = encodedFC1[i]
			// 使用公共密钥加密消息
			ctOut[i] = testContext.encryptor.EncryptMsgNew(msg, testContext.pkSet.GetPublicKey(id))
		}
	} else {
		// 如果未初始化加密器，则抛出异常
		panic("cannot encrypt image: encryptor is not initialized")
	}
	return ctOut
}

// 64 x 10 权重矩阵加密
func encryptFC2(testContext *testParams, id string) (ctOut *mkckks.Ciphertext) {
	// 检查加密器是否被初始化
	if testContext.encryptor != nil {
		// 获取槽数（每个密文可以容纳的复杂数数量）
		numSlots := testContext.params.Slots()
		// 创建一个长度为槽数的复数数组来存储编码后的 FC2 数据
		encodedFC2 := make([]complex128, numSlots)

		// 定义全连接层的行数和列数
		numRows := 10
		numColumns := 64

		// 遍历所有的槽位
		for i := 0; i < numSlots; i++ {
			// 计算当前槽位对应的行（x）和列（y）
			x := i / gap
			y := i % gap
			// 如果当前槽位对应的索引在全连接层矩阵的范围内，则将对应的值赋给 encodedFC2
			if y < numRows && x < numColumns {
				encodedFC2[i] = FC2[x][y]
			}
		}

		// 创建一个新消息并将编码后的 FC2 数据赋给消息的值
		msg := mkckks.NewMessage(testContext.params)
		msg.Value = encodedFC2
		// 使用公共密钥加密消息并返回加密后的密文
		ctOut = testContext.encryptor.EncryptMsgNew(msg, testContext.pkSet.GetPublicKey(id))
	} else {
		// 如果加密器未初始化，则抛出异常
		panic("cannot encrypt image: encryptor is not initialized")
	}
	return ctOut
}

func encryptB1(testContext *testParams, id string) (ctOut *mkckks.Ciphertext) {
	// 检查加密器是否被初始化
	if testContext.encryptor != nil {
		// 获取槽数（每个密文可以容纳的复杂数数量）
		numSlots := testContext.params.Slots()
		// 创建一个长度为槽数的复数数组来存储编码后的 B1 数据
		encodedB1 := make([]complex128, numSlots)
		// 将偏置 B1 的每个值填充到 encodedB1 的适当位置
		for i := 0; i < numFCUnit; i++ {
			encodedB1[i*gap] = B1[i]
		}

		// 创建一个新消息并将编码后的 B1 数据赋给消息的值
		msg := mkckks.NewMessage(testContext.params)
		msg.Value = encodedB1
		// 使用公共密钥加密消息并返回加密后的密文
		ctOut = testContext.encryptor.EncryptMsgNew(msg, testContext.pkSet.GetPublicKey(id))
	} else {
		// 如果加密器未初始化，则抛出异常
		panic("cannot encrypt image: encryptor is not initialized")
	}
	return ctOut
}

func encryptB2(testContext *testParams, id string) (ctOut *mkckks.Ciphertext) {
	// 检查加密器是否被初始化
	if testContext.encryptor != nil {
		// 获取槽数（每个密文可以容纳的复杂数数量）
		numSlots := testContext.params.Slots()
		// 创建一个长度为槽数的复数数组来存储编码后的 B2 数据
		encodedB2 := make([]complex128, numSlots)
		// 将偏置 B2 的每个值填充到 encodedB2 的适当位置
		for i := 0; i < numClass; i++ {
			encodedB2[i] = B2[i]
		}

		// 创建一个新消息并将编码后的 B2 数据赋给消息的值
		msg := mkckks.NewMessage(testContext.params)
		msg.Value = encodedB2
		// 使用公共密钥加密消息并返回加密后的密文
		ctOut = testContext.encryptor.EncryptMsgNew(msg, testContext.pkSet.GetPublicKey(id))
	} else {
		// 如果加密器未初始化，则抛出异常
		panic("cannot encrypt image: encryptor is not initialized")
	}
	return ctOut
}
