package cnn

import (
	"math"
	"mk-lattigo/mkckks"
	"mk-lattigo/mkrlwe"

	"github.com/ldsec/lattigo/v2/ckks"
)

// 卷积操作
func Convolution(eval *mkckks.Evaluator, rlkSet *mkrlwe.RelinearizationKeySet, rtkSet *mkrlwe.RotationKeySet,
	ctImage *mkckks.Ciphertext, ctImageHoisted *mkrlwe.HoistedCiphertext,
	ctKernels []*mkckks.Ciphertext, ctKernelsHoisted []*mkrlwe.HoistedCiphertext) (convOut *mkckks.Ciphertext) {

	/*
		kernel提取特征，为什么要旋转(移动卷积核？)？为什么选这几个位置旋转？
	*/

	// 初始化卷积输出，通过第一个内核与图像进行乘法和重新线性化
	convOut = eval.MulRelinHoistedNew(ctImage, ctKernels[0], ctImageHoisted, ctKernelsHoisted[0], rlkSet)

	// 将图像旋转1个位置并转换为提升形式
	temp := eval.RotateHoistedNew(ctImage, 1, ctImageHoisted, rtkSet)
	tempHoisted := eval.HoistedForm(temp)
	// 将旋转后的图像与第二个内核进行乘法和重新线性化，然后将结果添加到卷积输出中
	temp = eval.MulRelinHoistedNew(temp, ctKernels[1], tempHoisted, ctKernelsHoisted[1], rlkSet)
	convOut = eval.AddNew(convOut, temp)

	// 将图像旋转14个位置并转换为提升形式
	temp = eval.RotateHoistedNew(ctImage, 14, ctImageHoisted, rtkSet)
	tempHoisted = eval.HoistedForm(temp)
	// 将旋转后的图像与第三个内核进行乘法和重新线性化，然后将结果添加到卷积输出中
	temp = eval.MulRelinHoistedNew(temp, ctKernels[2], tempHoisted, ctKernelsHoisted[2], rlkSet)
	convOut = eval.AddNew(convOut, temp)

	// 将图像旋转15个位置并转换为提升形式
	temp = eval.RotateHoistedNew(ctImage, 15, ctImageHoisted, rtkSet)
	tempHoisted = eval.HoistedForm(temp)
	// 将旋转后的图像与第四个内核进行乘法和重新线性化，然后将结果添加到卷积输出中
	temp = eval.MulRelinHoistedNew(temp, ctKernels[3], tempHoisted, ctKernelsHoisted[3], rlkSet)
	convOut = eval.AddNew(convOut, temp)

	// 将卷积输出旋转2048个位置并将结果添加回卷积输出
	temp = eval.RotateNew(convOut, 2048, rtkSet)
	convOut = eval.AddNew(convOut, temp)

	// 将卷积输出旋转1024个位置并将结果添加回卷积输出
	temp = eval.RotateNew(convOut, 1024, rtkSet)
	convOut = eval.AddNew(convOut, temp)

	// 返回最终的卷积输出
	return convOut
}

// 全连接层1
func FC1Layer(eval *mkckks.Evaluator, rlkSet *mkrlwe.RelinearizationKeySet, rtkSet *mkrlwe.RotationKeySet,
	ctVec *mkckks.Ciphertext, ctVecHoisted *mkrlwe.HoistedCiphertext,
	ctMat []*mkckks.Ciphertext, ctMatHoisted []*mkrlwe.HoistedCiphertext,
	ctBias *mkckks.Ciphertext) (fc1Out *mkckks.Ciphertext) {

	var temp *mkckks.Ciphertext
	var tempHoisted *mkrlwe.HoistedCiphertext

	// 实现矩阵乘法
	for i := 0; i < len(ctMat); i++ {
		// 复制向量值
		temp = ctVec.CopyNew()
		// 将向量旋转i*128个位置并转换为提升形式
		temp = eval.RotateHoistedNew(ctVec, i*128, ctVecHoisted, rtkSet)
		tempHoisted = eval.HoistedForm(temp)
		// 将旋转后的向量与矩阵的第i行进行乘法和重新线性化
		temp = eval.MulRelinHoistedNew(temp, ctMat[i], tempHoisted, ctMatHoisted[i], rlkSet)
		if i == 0 {
			// 初始化fc1Out为第一次的乘法结果
			fc1Out = temp.CopyNew()
		} else {
			// 将后续的乘法结果累加到fc1Out
			fc1Out = eval.AddNew(fc1Out, temp)
		}
	}

	// logn为128的对数，用于后续的旋转累加
	logn := int(math.Log2(float64(128)))
	for i := 0; i < logn; i++ {
		// 将fc1Out旋转(1 << i)(步长)个位置并将结果累加到fc1Out
		temp = eval.RotateNew(fc1Out, (1 << i), rtkSet)
		fc1Out = eval.AddNew(fc1Out, temp)
	}
	// 将偏置添加到fc1Out
	fc1Out = eval.AddNew(fc1Out, ctBias)

	return
}

// 全连接层2
func FC2Layer(eval *mkckks.Evaluator, rlkSet *mkrlwe.RelinearizationKeySet, rtkSet *mkrlwe.RotationKeySet,
	ctVec *mkckks.Ciphertext, ctMat *mkckks.Ciphertext,
	ctBias *mkckks.Ciphertext, ptMask *ckks.Plaintext) (fc2Out *mkckks.Ciphertext) {

	// 使用ptMask对ctVec进行乘法, 实现密文和明文之间的乘法
	fc2Out = eval.MulPtxtNew(ctVec, ptMask)

	var temp *mkckks.Ciphertext
	// logn为16的对数，用于后续的旋转累加
	logn := int(math.Log2(float64(16)))
	for i := 0; i < logn; i++ {
		// 将fc2Out旋转-1*(1<<i)个位置并将结果累加到fc2Out
		temp = eval.RotateNew(fc2Out, -1*(1<<i), rtkSet)
		fc2Out = eval.AddNew(fc2Out, temp)
	}

	// 将fc2Out与ctMat进行乘法和重新线性化
	fc2Out = eval.MulRelinNew(fc2Out, ctMat, rlkSet)

	// logn为64的对数，用于后续的旋转累加
	logn = int(math.Log2(float64(64)))
	for i := 0; i < logn; i++ {
		// 将fc2Out旋转128*(1<<i)个位置并将结果累加到fc2Out
		temp = eval.RotateNew(fc2Out, 128*(1<<i), rtkSet)
		fc2Out = eval.AddNew(fc2Out, temp)
	}

	// 将偏置添加到fc2Out
	fc2Out = eval.AddNew(fc2Out, ctBias)
	return
}
