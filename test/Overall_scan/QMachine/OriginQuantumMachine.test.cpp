/// OriginQuantMachine Inface test

#include "QPanda.h"
#include "gtest/gtest.h"


TEST(OriginPhysicalQubitInface, test) {
	OriginPhysicalQubit oqubit = OriginPhysicalQubit();
	oqubit.setQubitAddr(1);
	oqubit.setOccupancy(true);
	EXPECT_EQ(1, oqubit.getQubitAddr());
	EXPECT_TRUE(oqubit.getOccupancy());
}


TEST(OriginQubitInface, test) {
	OriginPhysicalQubit oqubit = OriginPhysicalQubit();
	oqubit.setQubitAddr(1);
	oqubit.setOccupancy(true);
	OriginQubit oq = OriginQubit(&oqubit);
	PhysicalQubit* oqu = oq.getPhysicalQubitPtr();

	EXPECT_EQ(1, oq.get_phy_addr());
	EXPECT_TRUE(oq.getOccupancy());
	EXPECT_EQ(1, oqu->getQubitAddr());
	EXPECT_TRUE(oqu->getOccupancy());

}

TEST(OriginQubitPoolV1Inface,test){

	OriginQubitPoolv1 oqpoolv = OriginQubitPoolv1(10);	
	EXPECT_EQ(10, oqpoolv.getMaxQubit());
	EXPECT_EQ(10, oqpoolv.getIdleQubit());
	EXPECT_EQ(0, oqpoolv.get_max_usedqubit_addr());

	Qubit* q = oqpoolv.allocateQubit();
	EXPECT_EQ(0, q->get_phy_addr());

	Qubit* q1 = oqpoolv.allocateQubitThroughPhyAddress(8);
	EXPECT_EQ(8, q1->get_phy_addr());
	EXPECT_EQ(8, oqpoolv.getVirtualQubitAddress(q1));
	EXPECT_TRUE(q1->getOccupancy());

	Qubit* q2 = oqpoolv.allocateQubitThroughPhyAddress(5);
	EXPECT_EQ(5, q2->get_phy_addr());
	EXPECT_EQ(5, oqpoolv.getPhysicalQubitAddr(q2));
	EXPECT_TRUE(q2->getOccupancy());

	oqpoolv.Free_Qubit(q2);

	std::vector<Qubit*> vqbit ;
	EXPECT_EQ(2, oqpoolv.get_allocate_qubits(vqbit));

	oqpoolv.clearAll(); 
	
}

TEST(OriginQubitPoolV2Inface, test) {

	OriginQubitPoolv2 oqpoolv = OriginQubitPoolv2(10);
	EXPECT_EQ(10, oqpoolv.getMaxQubit());
	EXPECT_EQ(10, oqpoolv.getIdleQubit());
	EXPECT_EQ(0, oqpoolv.get_max_usedqubit_addr());

	Qubit* q = oqpoolv.allocateQubit();
	EXPECT_EQ(0, q->get_phy_addr());

	Qubit* q1 = oqpoolv.allocateQubitThroughPhyAddress(8);
	EXPECT_EQ(8, q1->get_phy_addr());
	EXPECT_EQ(8, oqpoolv.getVirtualQubitAddress(q1));
	EXPECT_TRUE(q1->getOccupancy());

	Qubit* q2 = oqpoolv.allocateQubitThroughPhyAddress(5);
	EXPECT_EQ(5, q2->get_phy_addr());
	EXPECT_EQ(5, oqpoolv.getPhysicalQubitAddr(q2));
	EXPECT_TRUE(q2->getOccupancy());

	oqpoolv.Free_Qubit(q2);

	std::vector<Qubit*> vqbit;
	EXPECT_EQ(2, oqpoolv.get_allocate_qubits(vqbit));

	oqpoolv.clearAll();	

}

TEST(OriginCBitInface, test) {
	OriginCBit qcb =  OriginCBit("c0");
	qcb.setOccupancy(true);
	qcb.set_val(3);
	EXPECT_STREQ("c0", qcb.getName().c_str());
	EXPECT_TRUE(qcb.getOccupancy());
	EXPECT_EQ(0, qcb.get_addr());
}

TEST(OriginCMemv2Inface, test) {
	return;
	OriginCMemv2 ocm2 = OriginCMemv2(10);
	CBit* cb = ocm2.Allocate_CBit();
	CBit* cb1 = ocm2.Allocate_CBit(1);

	EXPECT_STREQ("c0",cb->getName().c_str());
	EXPECT_TRUE(cb->getOccupancy());

	EXPECT_STREQ("c1", cb1->getName().c_str());
	EXPECT_TRUE(cb1->getOccupancy());

	EXPECT_EQ(10, ocm2.getMaxMem());
	EXPECT_EQ(8, ocm2.getIdleMem());

	ocm2.Free_CBit(cb1);

	std::vector<CBit*> vcbit;
	EXPECT_EQ(1, ocm2.get_allocate_cbits(vcbit));

	ocm2.clearAll();		

}


TEST(OriginQResultInface, test) {
	return;
	OriginQResult qres = OriginQResult();

	
//	qres.append(std::make_pair<"c1", false>);
	
//	EXPECT_FALSE(qres.getResultMap()["c1"]);

}

TEST(OriginQMachineStatusInface, test) {
	OriginQMachineStatus qms = OriginQMachineStatus();
	qms.setStatusCode(1);
	EXPECT_EQ(1, qms.getStatusCode());
}

TEST(CPUQVMInfaceTest,test) {
	CPUQVM qvm;
	qvm.init();
	auto qubits = qvm.qAllocMany(4);
	auto cbits = qvm.cAllocMany(4);

	QProg prog;
	prog << H(qubits[0])
		<< CNOT(qubits[0], qubits[1])
		<< CNOT(qubits[1], qubits[2])
		<< CNOT(qubits[2], qubits[3])
		<< Measure(qubits[0], cbits[0]);

	auto result = qvm.runWithConfiguration(prog, cbits, 1000);

	for (auto& val : result)
	{
		//std::cout << val.first << ", " << val.second << std::endl;
	}

	qvm.finalize();

}


TEST(NoiseQVMInfaceTest, test) {

	NoiseQVM qvm;
	qvm.init();
	auto qubits = qvm.qAllocMany(4);
	auto cbits = qvm.cAllocMany(4);

	QProg prog;
	prog << H(qubits[0])
		<< CNOT(qubits[0], qubits[1])
		<< CNOT(qubits[1], qubits[2])
		<< CNOT(qubits[2], qubits[3])
		<< Measure(qubits[0], cbits[0]);

	auto result = qvm.runWithConfiguration(prog, cbits, 1000);
	auto re = qvm.directlyRun(prog);

	for (auto& val : result)
	{
		//std::cout << val.first << ", " << val.second << std::endl;
	}

	for (auto& val : re)
	{
		//std::cout << val.first << ", " << val.second << std::endl;
	}

	qvm.finalize();

}


TEST(GPUQVMInfaceTest, test) {
	return;
	GPUQVM qvm;
	qvm.init();	/// unused cudr error
	auto qubits = qvm.qAllocMany(4);
	auto cbits = qvm.cAllocMany(4);

	QProg prog;
	prog << H(qubits[0])
		<< CNOT(qubits[0], qubits[1])
		<< CNOT(qubits[1], qubits[2])
		<< CNOT(qubits[2], qubits[3])
		<< Measure(qubits[0], cbits[0]);

	auto result = qvm.runWithConfiguration(prog, cbits, 1000);

	for (auto& val : result)
	{
		//std::cout << val.first << ", " << val.second << std::endl;
	}

	qvm.finalize();

}