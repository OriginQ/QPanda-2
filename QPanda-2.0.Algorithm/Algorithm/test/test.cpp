#include "test.h"
#include "../../Utility/QOptimizer/AbstractQOptimizer.h"
#include "../../Utility/QPauliOperator/QPauliOperator.h"
#include "../QAOA/QAOA.h"

void ifwhile()
{
    OriginQVM qvm;
    qvm.init();
    QProg  prog = CreateEmptyQProg();
    CBit * cbit0 = qvm.Allocate_CBit();
    CBit * cbit1 = qvm.Allocate_CBit();
    auto q0 = qvm.Allocate_Qubit();
    auto q1 = qvm.Allocate_Qubit();
    ClassicalCondition cc1(cbit1);
    QProg  ifprog = CreateEmptyQProg();
    ifprog << H(q1);
    QIfProg  ifnode = CreateIfProg(cc1, &ifprog);
    prog << H(q0) << Measure(q0, cbit0) << ifnode << Measure(q1, cbit1);
    qvm.load(prog);
    qvm.run();
    auto temp = qvm.getResult();
    auto amap = temp->getResultMap();
    qvm.finalize();
}

bool HelloWorld()
{
	// This program is about introducing the basic
	// procedures to use a quantum computer by the
	// Q-Panda system.
	// The program has used only 1 qubit.
	// Firstly we make a Hadamard gate to the qubit,
	// then measure it by mapping the result to a 
	// CBit. Finally we readout the answer.

	init(); 
	// initialize the environment
    
	QProg  A_Hello_World_Program = CreateEmptyQProg();
	// Create an empty program

	CBit * cbit1 = cAlloc();
	// allocate a cbit

	ClassicalCondition cc1 = bind_a_cbit(cbit1);
	// bind the cbit onto a classicalcondition variable
	// But not used in this sample

	auto qb = qAlloc();
	// allocate a qubit

	A_Hello_World_Program
		<< H(qb) 
		<< Measure(qb, cbit1);
	// insert a Hadamard gate and a Measurement operation
	// after the empty program sequently

	load(A_Hello_World_Program);
	// And then load it into the quantum computer

    run();
	// simply run it

    map<string,bool> resultMap = getResultMap();
	// you can get the result map, which save all the
	// measurement results in the classical register(CBit)

    for (auto aiter : resultMap)
    {
        cout << aiter.first << " ";
        cout << aiter.second << endl;
    }
	// Let's iterate over the map to see whether
	// the result is correct

    finalize();
	// Use finalize() to tell the quantum computer to stop

    return resultMap["c0"];
}

bool DJalgorithm()
{
    OriginQVM qvm;
    qvm.init();
    QProg  dj = CreateEmptyQProg();
    CBit * cbit1 = qvm.Allocate_CBit();
    auto qb = qvm.Allocate_Qubit();
    ClassicalCondition cc1(cbit1);
    auto anc = qvm.Allocate_Qubit();
    dj << X(anc) << H(qb) << H(anc) << CNOT(qb, anc) << H(qb) << Measure(qb, cbit1);
    qvm.load(dj);
    qvm.run();
    auto temp = qvm.getResult();
    auto amap = temp->getResultMap();
    qvm.finalize();
    return amap["c0"];
}

int Grover(int target)  //target is 0,1,2 or 3
{
    OriginQVM qvm;
    qvm.init();
    CBit * cbit1 = qvm.Allocate_CBit();
    CBit * cbit2 = qvm.Allocate_CBit();
    CBit * cbit3 = qvm.Allocate_CBit();
    ClassicalCondition cc1(cbit1);
    ClassicalCondition cc2(cbit2);
    ClassicalCondition cc3(cbit3);
    auto q = qvm.Allocate_Qubit();
    auto q1 = qvm.Allocate_Qubit();
    auto anc = qvm.Allocate_Qubit();
    QProg  grover = CreateEmptyQProg();
    QCircuit  init = CreateEmptyCircuit();
    QCircuit  oracle = CreateEmptyCircuit();
    QCircuit  reverse = CreateEmptyCircuit();
    init << H(q) << H(q1) << X(anc) << H(anc);
    vector<Qubit *> controlVector;
    controlVector.push_back(q);
    controlVector.push_back(q1);
    //U4  sqrtH(0.5*PI, 0, 0.25*PI, PI);
    QGate  toff = X(anc);

    toff.setControl(controlVector);
    switch (target)
    {
    case 0:
        oracle << X(q) << X(q1) << toff << X(q) << X(q1);
        break;
    case 1:
        oracle << X(q) << toff << X(q);
        break;
    case 2:
        oracle << X(q1) << toff << X(q1);
        break;
    case 3:
        oracle << toff;
        break;
    }
    reverse << H(q) << H(q1) << X(q) << X(q1) << H(q1) << CNOT(q, q1);
    reverse << H(q1) << X(q) << X(q1) << H(q) << H(q1) << X(anc);
    grover << init << oracle << reverse << Measure(q, cbit1) << Measure(q1, cbit2);
    QProg  grover1 = CreateEmptyQProg();
    //grover1<<H(q)<<H(q1)<<toff<< Measure(q, cbit1) << Measure(q1, cbit2) << Measure(anc, cbit3);
    qvm.load(grover);
    qvm.run();
    auto temp = qvm.getResult();
    auto amap = temp->getResultMap();
    for (auto aiter : amap)
    {
        std::cout << aiter.first << " ";
        std::cout << aiter.second << std::endl;

    }
    qvm.finalize();
    return amap["c1"] + amap["c2"] * 2;
}//checked

void controlandDagger()
{
    OriginQVM qvm;
    qvm.init();
    CBit * cbit1 = qvm.Allocate_CBit();
    CBit * cbit2 = qvm.Allocate_CBit();
    ClassicalCondition cc1(cbit1);
    ClassicalCondition cc2(cbit2);
    auto q = qvm.Allocate_Qubit();
    auto q1 = qvm.Allocate_Qubit();
    auto anc = qvm.Allocate_Qubit();
    QProg  aaa = CreateEmptyQProg();
    vector<Qubit *> controlVector;
    controlVector.push_back(q);
    //U4  sqrtH(0.5*PI, 0, 0.25*PI, PI);
    QGate  toff = X(q1);
    toff.setControl(controlVector);
    aaa << H(q) << toff << Measure(q, cbit1) << Measure(q1, cbit2);
    qvm.load(aaa);
    qvm.run();
    auto temp = qvm.getResult();
    auto amap = temp->getResultMap();
    for (auto aiter : amap)
    {
        std::cout << aiter.first << " ";
        std::cout << aiter.second << std::endl;

    }
    qvm.finalize();
    return;
}

QProg bell(Qubit* a, Qubit * b)
{
    auto bb = CreateEmptyQProg();
    bb << H(a) << CNOT(a, b);
    return bb;
}

void entangle()
{
    OriginQVM qvm;
    qvm.init();
    auto q0 = qvm.Allocate_Qubit();
    auto q1 = qvm.Allocate_Qubit();
    auto cbit0 = qvm.Allocate_CBit();
    auto cbit1 = qvm.Allocate_CBit();
    QProg  entangle = CreateEmptyQProg();
    entangle << H(q0) << CNOT(q0, q1);
    entangle << Measure(q0, cbit0) << Measure(q1, cbit1);
    qvm.load(entangle);
    qvm.run();
    entangle << bell(q0, q1) << Measure(q0, cbit0) << Measure(q1, cbit1);


    auto temp = qvm.getResult();
    auto amap = temp->getResultMap();
}

void HHL_Algorithm1()
{

    OriginQVM qvm;
    qvm.init();
    QProg  hhlProg = CreateEmptyQProg();

    auto q0 = qvm.Allocate_Qubit();
    auto q1 = qvm.Allocate_Qubit();
    auto q2 = qvm.Allocate_Qubit();
    auto q3 = qvm.Allocate_Qubit();
    auto ancbit = qvm.Allocate_CBit();
    auto cbit0 = qvm.Allocate_CBit();
    auto cbit1 = qvm.Allocate_CBit();
    auto cbit2 = qvm.Allocate_CBit();
    ClassicalCondition cc0(ancbit);
    ClassicalCondition cc1(cbit0);
    QCircuit  ifcircuit = CreateEmptyCircuit();
    QCircuit  PSEcircuit = CreateEmptyCircuit();
    PSEcircuit << H(q1) << H(q2) << RZ(q2, 0.75*PI);
    QGate  gat1 = CU(PI, 1.5*PI, -0.5*PI, PI / 2, q2, q3);
    QGate   gat2 = CU(PI, 1.5*PI, -PI, PI / 2, q1, q3);
    PSEcircuit << gat1 << RZ(q1, 1.5*PI) << gat2 << CNOT(q1, q2) << CNOT(q2, q1) << CNOT(q1, q2);
    //PSEcircuit << gat1 << RZ_GATE(q1, 1.5*PI)<<gat2 ;
    QGate  gat3 = CU(-0.25*PI, -0.5*PI, 0, 0, q2, q1);
    PSEcircuit << H(q2) << gat3 << H(q1);     //PSE over


                                              //control-lambda
    QCircuit  CRotate = CreateEmptyCircuit();
    vector<Qubit *> controlVector;
    controlVector.push_back(q1);
    controlVector.push_back(q2);
    QGate  gat4 = RY(q0, PI);
    gat4.setControl(controlVector);
    QGate  gat5 = RY(q0, PI / 3);
    gat5.setControl(controlVector);
    QGate  gat6 = RY(q0, 0.6796738);  //arcsin(1/3)
    gat6.setControl(controlVector);
    CRotate << X(q1) << gat4 << X(q1) << X(q2) << gat5 << X(q2) << gat6;
    //hhl circuit
    QProg  prog = CreateEmptyQProg();
    QProg  prog1 = CreateEmptyQProg();
    QProg  prog2 = CreateEmptyQProg();

    prog << prog1 << prog2;


    QProg  PSEdagger = CreateEmptyQProg();
    // PSEdagger << PSEcircuit.dagger() << Measure(q2, cbit2);
    QIfProg ifnode = CreateIfProg(cc0, &PSEdagger);
    hhlProg << PSEcircuit << CRotate << Measure(q0, ancbit);
    //hhlProg << PSEcircuit << Measure(anc, ancbit) ;
    qvm.load(hhlProg);
    qvm.run();
    auto temp = qvm.getResult();
    auto amap = temp->getResultMap();
    for (auto aiter : amap)
    {
        std::cout << aiter.first << " ";
        std::cout << aiter.second << std::endl;

    }
    qvm.finalize();

    return;

}

void testQPauliOperator()
{
	QPanda::QPauliMap pauli_map{
		{ "X0 X1 X2",{ 2,1 } },
	{ "Y0 Y1 Y2",{ 3,2 } },
	{ "Z0 Z1 Z2",{ 5, 3 } },
	{ "X0 Y1 Z2",{ 7,4 } } };

	QPanda::QPauliOperator pauli(pauli_map);

	std::cout << pauli << std::endl;
	pauli += 1;
	std::cout << pauli << std::endl;
	pauli -= 1;
	pauli *= pauli;
	std::cout << pauli << std::endl;
}

double myFunc(const std::string &key, const QPanda::QPauliMap &pauli_map)
{
	double sum = 0;

	QPanda::QPauliOperator pauli_op(pauli_map);
	QPanda::QHamiltonian hamiltonian = pauli_op.toHamiltonian();
	std::map<size_t, size_t> index_map = pauli_op.getIndexMap();

	for_each(hamiltonian.begin(),
		hamiltonian.end(),
		[&](const QPanda::QHamiltonianItem &item)
	{
		std::vector<size_t> index_vec;
		for (auto iter = item.first.begin();
			iter != item.first.end();
			iter++)
		{
			index_vec.push_back(iter->first);
		}

		double value = item.second;
		size_t i = index_vec.front();
		size_t j = index_vec.back();
		if (key[i] != key[j])
		{
			sum += item.second;
		}
	});

	return sum;
}

void testQAOA()
{
	QPanda::QPauliMap pauli_map;
	pauli_map.insert(std::make_pair("Z0 Z6", 0.49));
	pauli_map.insert(std::make_pair("Z6 Z1", 0.59));
	pauli_map.insert(std::make_pair("Z1 Z7", 0.44));
	pauli_map.insert(std::make_pair("Z7 Z2", 0.36));

	pauli_map.insert(std::make_pair("Z2 Z8", 0.63));
	pauli_map.insert(std::make_pair("Z8 Z13", 0.36));
	pauli_map.insert(std::make_pair("Z13 Z19", 0.81));
	pauli_map.insert(std::make_pair("Z19 Z14", 0.29));

	pauli_map.insert(std::make_pair("Z14 Z9", 0.52));
	pauli_map.insert(std::make_pair("Z9 Z4", 0.43));
	pauli_map.insert(std::make_pair("Z13 Z18", 0.72));
	pauli_map.insert(std::make_pair("Z18 Z12", 0.40));

	pauli_map.insert(std::make_pair("Z12 Z7", 0.60));
	pauli_map.insert(std::make_pair("Z12 Z17", 0.71));
	pauli_map.insert(std::make_pair("Z17 Z11", 0.50));
	pauli_map.insert(std::make_pair("Z11 Z6", 0.64));

	pauli_map.insert(std::make_pair("Z11 Z16", 0.57));
	pauli_map.insert(std::make_pair("Z16 Z10", 0.41));
	pauli_map.insert(std::make_pair("Z10 Z5", 0.23));
	pauli_map.insert(std::make_pair("Z10 Z15", 0.40));

	pauli_map.insert(std::make_pair("Z5 Z0", 0.18));

	QPanda::vector_d optimize_para_vec(2, 0);
	QPanda::QAOA qaoa;
	qaoa.setHamiltonian(pauli_map);
	qaoa.setShots(2000);
	qaoa.regiestUserDefinedFunc(std::bind(&myFunc,
		std::placeholders::_1,
		pauli_map));

	QPanda::AbstractQOptimizer* optimizer = qaoa.getOptimizer();
	optimizer->setMaxFCalls(2000);
	optimizer->setMaxIter(2000);
	optimizer->setDisp(true);

	QPanda::OptimizationResult result = qaoa.exec();
}

//
//bool test()
//{
//    OriginQVM qvm;
//    qvm.init();
//
//    CBit * cbit1 = qvm.Allocate_CBit();
//    CBit * cbit2 = qvm.Allocate_CBit();
//    Qubit * qb = qvm.Allocate_Qubit();
//    Qubit * qb2 = qvm.Allocate_Qubit();
//    Qubit * qb3 = qvm.Allocate_Qubit();
//
//    QCircuit & whilecircuit = CreateEmptyCircuit();
//    whilecircuit << H(qb2) << Measure(qb2, cbit2);
//
//    QuantumGate & gat = X(qb, PI / 2);
//    QuantumGate & gat1 = H(qb);
//    QuantumGate & gat2 = QSingle(PI, 1.5*PI, -0.5*PI, PI / 2, qb);
//    QuantumGate & gat3 = CU(PI, 1.5*PI, -0.5*PI, PI / 2, qb, qb2);
//
//    gat1.setDagger(1);
//    vector<Qubit *> controlVector;
//    controlVector.push_back(qb2);
//    gat2.setControl(controlVector);
//
//    QCircuit & c = CreateEmptyCircuit();
//    c << RZ_GATE(qb) << RY_GATE(qb, 23.456) << gat1;
//
//    c.dagger();
//    c.control(controlVector);
//    QProg & prog = CreateEmptyQProg();
//
//    ClassicalCondition cc1(cbit1);
//    ClassicalCondition cc2(cbit2);
//    ClassicalCondition cc3 = cc1 + cc2;
//    QIfProg & ifnode = CreateIfProg(&cc3, &c);
//    QWhileProg & whileNode = CreateWhileProg(&cc2, &ifnode);
//    prog << ifnode << whileNode;
//
//    QProg & prog1 = CreateEmptyQProg();
//    QProg & prog2 = CreateEmptyQProg();
//
//    QIfProg & ifnode = CreateIfProg(&cc3, &prog1, &prog2);
//
//    QProg & prog = CreateEmptyQProg();
//    prog << H(qb) << c.control(controlVector) << H(qb2);
//    prog << Measure(qb, cbit1) << Measure(qb2, cbit2);
//
//    prog << ifnode;
//
//    QProg & whileprog = CreateEmptyQProg();
//    whileprog << H(qb) << gat2 << ifnode;
//    //QWhileProg & whileNode = CreateWhileProg(&cc2, &whileprog);
//    prog << whileNode;
//
//
//
//    qvm.load(prog);
//    qvm.run();
//    auto temp = qvm.getResult();
//    auto amap = temp->getResultMap();
//    qvm.finalize();
//
//    int time1 = 0;
//    int time0 = 0;
//
//    for (auto aiter : amap)
//    {
//        std::cout << aiter.first << " ";
//        std::cout << aiter.second << std::endl;
//
//    }
//    return 0;
//}
