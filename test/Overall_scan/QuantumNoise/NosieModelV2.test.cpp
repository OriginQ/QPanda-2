
#include "QPanda.h"
#include "gtest/gtest.h"


TEST(NoiseModelInface,test) {
	NoiseQVM qvm;
	qvm.init();
	auto qvec = qvm.qAllocMany(4);
	auto cvec = qvm.cAllocMany(4);

	NoiseModel noisemodel;
	noisemodel.add_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, GateType::PAULI_X_GATE, 0.1);
	
	EXPECT_EQ(0, noisemodel.rotation_error());
	EXPECT_TRUE(noisemodel.enabled());
	EXPECT_FALSE(noisemodel.readout_error_enabled());

	NoisyQuantum nq = noisemodel.quantum_noise();
	
	auto cpu_qvm = CPUQVM();
	cpu_qvm.init();
	NoiseModel noise;

	auto prog1 = QProg();
	prog1 << X(qvec[0]) << P(qvec[0], PI) << MeasureAll(qvec, cvec);

	noise.add_noise_model(NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR, GateType::P_GATE, 0.3);
	noise.add_noise_model(NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR, GateType::S_GATE, 0.5);

	auto r1 = qvm.runWithConfiguration(prog1, cvec, 5000);
	auto r2 = cpu_qvm.runWithConfiguration(prog1, cvec, 5000, noise);

	for (auto &i : r1) {
		//std::cout << i.first << "," << i.second << std::endl;
	}

	for (auto& i : r2) {
		//std::cout << i.first << "," << i.second << std::endl;
	}

}