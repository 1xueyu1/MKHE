package mkckks

import (
	"strconv"
	"testing"

	"github.com/ldsec/lattigo/v2/ckks"
	"mk-lattigo/mkrlwe"
)

func BenchmarkMKCKKS(b *testing.B) {

	defaultParams := []ckks.ParametersLiteral{PN13QP213, PN14QP438, PN15QP830}

	for _, defaultParam := range defaultParams {
		ckksParams, err := ckks.NewParametersFromLiteral(defaultParam)

		if ckksParams.PCount() < 2 {
			continue
		}

		if err != nil {
			panic(err)
		}

		params := NewParameters(ckksParams)
		maxUsers := 64
		userList := make([]string, maxUsers)
		idset := mkrlwe.NewIDSet()

		for i := range userList {
			userList[i] = "user" + strconv.Itoa(i)
			idset.Add(userList[i])
		}

		var testContext *testParams
		if testContext, err = genTestParams(params, idset); err != nil {
			panic(err)
		}

		benchMulAndRelin(testContext, userList[:2], b)
		benchMulAndRelin(testContext, userList[:4], b)
		benchMulAndRelin(testContext, userList[:8], b)
		benchMulAndRelin(testContext, userList[:16], b)
		benchMulAndRelin(testContext, userList[:32], b)
		benchMulAndRelin(testContext, userList[:64], b)
	}
}

func benchMulAndRelin(testContext *testParams, userList []string, b *testing.B) {

	numUsers := len(userList)
	msgList := make([]*Message, numUsers)
	ctList := make([]*Ciphertext, numUsers)

	rlkSet := testContext.rlkSet
	eval := testContext.evaluator

	for i := range userList {
		msgList[i], ctList[i] = newTestVectors(testContext, userList[i], complex(-1, 1), complex(-1, 1))
	}

	ct := ctList[0]

	for i := range userList {
		ct = eval.AddNew(ct, ctList[i])
	}

	b.Run(GetTestName(testContext.params, "MKMulAndRelin: "+strconv.Itoa(numUsers)+"/ "), func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			eval.MulRelinNew(ct, ct, rlkSet)
		}

	})
}