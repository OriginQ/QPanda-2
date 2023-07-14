/// OriginQubitPool Inface test

#include "QPanda.h"
#include "gtest/gtest.h"

USING_QPANDA

TEST(OriginQubitPoolTest, test) {
	auto qpool = OriginQubitPool::get_instance();
	qpool->set_capacity(20);
	EXPECT_EQ(20, qpool->get_capacity());

	Qubit* q = qpool->allocateQubit();
	EXPECT_TRUE(q->getOccupancy());

	Qubit* q1 = qpool->allocateQubitThroughPhyAddress(1);
	EXPECT_EQ(1, q1->get_phy_addr());
	PhysicalQubit* py =  q1->getPhysicalQubitPtr();
	py->setQubitAddr(1);
	EXPECT_EQ(1, py->getQubitAddr());
	py->setOccupancy(true);
	EXPECT_TRUE(py->getOccupancy());

	Qubit* q2 = qpool->allocateQubitThroughVirAddress(2);
	EXPECT_TRUE(q2->getOccupancy());
	PhysicalQubit* py1 = q2->getPhysicalQubitPtr();
	py1->setQubitAddr(1);
	EXPECT_EQ(1, py1->getQubitAddr());
	py1->setOccupancy(true);
	EXPECT_TRUE(py1->getOccupancy());
    qpool->qFreeAll();
}


TEST(QubitPoolCase,test) {
    MPSQVM qvm;

    qvm.init();
    auto qlist = qvm.qAllocMany(10);
    auto clist = qvm.cAllocMany(10);

    QProg prog;
    prog << HadamardQCircuit(qlist)
        << CZ(qlist[1], qlist[5])
        << CZ(qlist[3], qlist[5])
        << CZ(qlist[2], qlist[4])
        << CZ(qlist[3], qlist[7])
        << CZ(qlist[0], qlist[4])
        << RY(qlist[7], PI / 2)
        << RX(qlist[8], PI / 2)
        << RX(qlist[9], PI / 2)
        << CR(qlist[0], qlist[1], PI)
        << CR(qlist[2], qlist[3], PI)
        << RY(qlist[4], PI / 2)
        << RZ(qlist[5], PI / 4)
        << RX(qlist[6], PI / 2)
        << RZ(qlist[7], PI / 4)
        << CR(qlist[8], qlist[9], PI)
        << CR(qlist[1], qlist[2], PI)
        << RY(qlist[3], PI / 2)
        << RX(qlist[4], PI / 2)
        << RX(qlist[5], PI / 2)
        << CR(qlist[9], qlist[1], PI)
        << RY(qlist[1], PI / 2)
        << RY(qlist[2], PI / 2)
        << RZ(qlist[3], PI / 4)
        << CR(qlist[7], qlist[8], PI)
        << MeasureAll(qlist, clist);

    auto measure_result = qvm.runWithConfiguration(prog, clist, 1000);
    for (auto val : measure_result)
    {
        //cout << val.first << " : " << val.second << endl;
    }

    auto pmeasure_result = qvm.probRunDict(prog, qlist, -1);
    for (auto val : pmeasure_result)
    {
        //cout << val.first << " : " << val.second << endl;
    }

    qvm.finalize();
}


TEST(QubitPoolCase1,test) {
  
    auto qpool = OriginQubitPool::get_instance();
    auto cmem = OriginCMem::get_instance();
    qpool->clearAll();
    //cout << "set qubit pool capacity  before: " << qpool->get_capacity() << endl;
    qpool->set_capacity(20);
   // cout << "set qubit pool capacity  after: " << qpool->get_capacity() << endl;

    auto qv = qpool->qAllocMany(6);
    auto cv = cmem->cAllocMany(6);

    QVec used_qv;
    auto used_qv_size = qpool->get_allocate_qubits(used_qv);
    //cout << "allocate qubits number: " << used_qv_size << endl;

    auto qvm = new CPUQVM();
    qvm->init();
    auto prog = QProg();
    prog << H(0) << H(1)
        << H(2)
        << H(4)
        << X(5)
        << X1(2)
        << CZ(2, 3)
        << RX(3, PI / 4)
        << CR(4, 5, PI / 2)
        << SWAP(3, 5)
        << CU(1, 3, PI / 2, PI / 3, PI / 4, PI / 5)
        << U4(4, 2.1, 2.2, 2.3, 2.4)
        << BARRIER({ 0, 1,2,3,4,5 });

    auto res_0 = qvm->probRunDict(prog, { 0,1,2,3,4,5 });
    prog << Measure(0, 0)
        << Measure(1, 1)
        << Measure(2, 2)
        << Measure(3, 3)
        << Measure(4, 4)
        << Measure(5, 5);

    vector<int> cbit_addrs = { 0,1,2,3,4,5 };
    auto res_2 = qvm->runWithConfiguration(prog, cbit_addrs, 5000);

    qvm->finalize();
    delete(qvm);

    auto qvm_noise = new NoiseQVM();
    qvm_noise->init();
    auto res_4 = qvm_noise->runWithConfiguration(prog, cbit_addrs, 5000);
    qvm_noise->finalize();
    qpool->qFreeAll();
    cmem->cFreeAll();
    delete(qvm_noise);

}