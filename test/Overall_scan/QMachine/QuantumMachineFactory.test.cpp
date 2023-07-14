/// QuantMachineFactory Inface test

#include "QPanda.h"
#include "gtest/gtest.h"

USING_QPANDA

TEST(QuantMachineFactoryTest, test) {
	QuantumMachineFactory qmfac = QuantumMachineFactory::GetFactoryInstance();
//	QuantumMachine* cpuqm = qmfac.CreateByName("CPU");
	QuantumMachine* cpuqm = qmfac.CreateByType(CPU);
	cpuqm->init();
	QVec qv = cpuqm->allocateQubits(4);
	Qubit* qb = cpuqm->allocateQubitThroughPhyAddress(1);
	EXPECT_EQ(1,qb->get_phy_addr());

	QuantumMachineFactory::constructor_t a;
	a = std::bind([](QuantumMachineFactory m_qm)->QuantumMachine* {
		return m_qm.CreateByType(NOISE);
		}, qmfac) ;
	qmfac.registerclass("CPU1", a);

}