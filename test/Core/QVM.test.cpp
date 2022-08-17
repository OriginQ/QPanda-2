#include <time.h>
#include <iostream>
#include <numeric>
#include "QPanda.h"
#include <functional>
#include "gtest/gtest.h"
#include "Core/Utilities/Tools/OriginCollection.h"
USING_QPANDA
using namespace std;
using namespace Base64;
using namespace rapidjson;

QHamiltonian get_test_hamiltonian()
{

    /*  test data:

        "" : -0.097066,
        "X0 X1 Y2 Y3" : -0.045303,
        "X0 Y1 Y2 X3" : 0.045303,
        "Y0 X1 X2 Y3" : 0.045303,
        "Y0 Y1 X2 X3" : -0.045303,
        "Z0" : 0.171413,
        "Z0 Z1" : 0.168689,
        "Z0 Z2" : 0.120625,
        "Z0 Z3" : 0.165928,
        "Z1" : 0.171413,
        "Z1 Z2" : 0.165928,
        "Z1 Z3" : 0.120625,
        "Z2" : -0.223432,
        "Z2 Z3" : 0.174413,
        "Z3" : -0.223432
    */

    QHamiltonian hamiltonian;

    //"" : -0.097066
    QTerm q0;
    hamiltonian.emplace_back(make_pair(q0, -0.097066));

    // "X0 X1 Y2 Y3" : -0.045303
    QTerm q1;
    q1[0] = 'X';
    q1[1] = 'X';
    q1[2] = 'Y';
    q1[3] = 'Y';
    hamiltonian.emplace_back(make_pair(q1, -0.045303));

    // "X0 Y1 Y2 X3" : 0.045303
    QTerm q2;
    q2[0] = 'X';
    q2[1] = 'Y';
    q2[2] = 'Y';
    q2[3] = 'X';
    hamiltonian.emplace_back(make_pair(q2, 0.045303));

    // "Y0 X1 X2 Y3" : 0.045303
    QTerm q3;
    q3[0] = 'Y';
    q3[1] = 'X';
    q3[2] = 'X';
    q3[3] = 'Y';
    hamiltonian.emplace_back(make_pair(q3, 0.045303));

    // "Y0 Y1 X2 X3" : -0.045303
    QTerm q4;
    q4[0] = 'Y';
    q4[1] = 'Y';
    q4[2] = 'X';
    q4[3] = 'X';
    hamiltonian.emplace_back(make_pair(q4, -0.045303));

    //"Z0" : 0.171413
    QTerm q5;
    q5[0] = 'Z';
    hamiltonian.emplace_back(make_pair(q5, 0.171413));

    //"Z0 Z1" : 0.168689
    QTerm q6;
    q6[0] = 'Z';
    q6[1] = 'Z';
    hamiltonian.emplace_back(make_pair(q6, 0.168689));

    //"Z0 Z2" : 0.120625
    QTerm q7;
    q7[0] = 'Z';
    q7[2] = 'Z';
    hamiltonian.emplace_back(make_pair(q7, 0.120625));

    //"Z0 Z3" : 0.165928
    QTerm q8;
    q8[0] = 'Z';
    q8[3] = 'Z';
    hamiltonian.emplace_back(make_pair(q8, 0.165928));

    //"Z1" : 0.171413
    QTerm q9;
    q9[1] = 'Z';
    hamiltonian.emplace_back(make_pair(q9, 0.171413));

    //"Z1 Z2" : 0.165928
    QTerm q10;
    q10[1] = 'Z';
    q10[2] = 'Z';
    hamiltonian.emplace_back(make_pair(q10, 0.165928));

    //"Z1 Z3" : 0.120625
    QTerm q11;
    q11[1] = 'Z';
    q11[3] = 'Z';
    hamiltonian.emplace_back(make_pair(q11, 0.120625));

    //"Z2" : -0.223432
    QTerm q12;
    q12[2] = 'Z';
    hamiltonian.emplace_back(make_pair(q12, -0.223432));

    //"Z2 Z3" : 0.174413
    QTerm q13;
    q13[2] = 'Z';
    q13[3] = 'Z';
    hamiltonian.emplace_back(make_pair(q13, 0.174413));

    //"Z3" : -0.223432
    QTerm q14;
    q14[3] = 'Z';
    hamiltonian.emplace_back(make_pair(q14, -0.223432));

    return hamiltonian;
}
#ifdef USE_CUDA
TEST(QVM, GPUQVM)
{
    GPUQVM machine;
    machine.init();

    QVec qv;
    std::vector<ClassicalCondition> cv;

	std::string originir = R"(QINIT 4
                            CREG 0
                            X q[0]
                            X q[1]
                            H q[0]
                            H q[1]
                            H q[2]
                            RX q[3],(1.5707963)
                            CNOT q[0],q[3]
                            CNOT q[1],q[3]
                            CNOT q[2],q[3]
                            RZ q[3],(-0.0060418116)
                            CNOT q[0],q[3]
                            CNOT q[1],q[3]
                            CNOT q[2],q[3]
                            DAGGER
                            H q[0]
                            H q[1]
                            H q[2]
                            RX q[3],(1.5707963)
                            ENDDAGGER
                            H q[0]
                            H q[1]
                            RX q[2],(1.5707963)
                            H q[3]
                            CNOT q[0],q[3]
                            CNOT q[1],q[3]
                            CNOT q[2],q[3]
                            RZ q[3],(-0.0060418116)
                            CNOT q[0],q[3]
                            CNOT q[1],q[3]
                            CNOT q[2],q[3]
                            DAGGER
                            H q[0]
                            H q[1]
                            RX q[2],(1.5707963)
                            H q[3]
                            ENDDAGGER
                            H q[0]
                            RX q[1],(1.5707963)
                            H q[2]
                            H q[3]
                            CNOT q[0],q[3]
                            CNOT q[1],q[3]
                            CNOT q[2],q[3]
                            RZ q[3],(0.0060418116)
                            CNOT q[0],q[3]
                            CNOT q[1],q[3]
                            CNOT q[2],q[3])";

    auto prog = convert_originir_string_to_qprog(originir, &machine, qv, cv);
    auto expectation = machine.get_expectation(prog, get_test_hamiltonian(), qv);
    EXPECT_NEAR(expectation, -0.097066, 1e-7);
}
#endif

void GHZ(int a)
{
    CPUQVM qvm;
    qvm.setConfigure({ 64,64 });
    qvm.init();

    auto q = qvm.qAllocMany(a);
    auto c = qvm.cAllocMany(a);

    auto prog = QProg();
    prog << H(q[0]);

    for (auto i = 0; i < a - 1; ++i)
    {
        prog << CNOT(q[i], q[i + 1]);
    }
    prog << MeasureAll(q, c);


    const string ss = "GHZ_" + to_string(a);
    write_to_originir_file(prog, &qvm, ss);
}

static bool compare_map_result(map<string, size_t> result1, map<string, size_t> result2 , int shots)
{
    for (auto val : result1)
    {
        auto iter = result2.find(val.first);

        if (iter == result2.end())
            return false;

        if (1e-2 < std::fabs((double)(((int)val.second - (int)iter->second) / shots)))
            return false;
    }

    return true;
}

static bool compare_probs_result(const prob_vec& result1,const prob_vec& result2)
{
    QPANDA_RETURN(result1.size() != result2.size(), false);

    for (auto i = 0; i < result1.size(); ++i)
    {
        auto prob1 = result1[i];
        auto prob2 = result2[i];

        if (1e-6 < std::fabs(prob1 - prob2))
            return false;
    }

    return true;
}

static bool matrix_compare(const QStat& mat1, const QStat& mat2, const double precision /*= 0.000001*/)
{
    if (mat1.size() != mat2.size())
        return false;

    qcomplex_t ratio; // constant value
    for (size_t i = 0; i < mat1.size(); ++i)
    {
        if ((abs(mat2.at(i).real() - 0.0) > precision) || (abs(mat2.at(i).imag() - 0.0) > precision))
        {
            ratio = mat1.at(i) / mat2.at(i);
            if (precision < abs(sqrt(ratio.real()*ratio.real() + ratio.imag()*ratio.imag()) - 1.0))
                return false;
            break;
        }
    }

    qcomplex_t tmp_val;
    for (size_t i = 0; i < mat1.size(); ++i)
    {
        tmp_val = ratio * mat2.at(i);
        if ((abs(mat1.at(i).real() - tmp_val.real()) > precision) ||
            (abs(mat1.at(i).imag() - tmp_val.imag()) > precision))
            return false;
    }

    return true;
}

TEST(QVM, cpu_run_with_no_cbits_args)
{
    CPUQVM machine;
    machine.setConfigure({ 64,64 });
    machine.init();

    auto q = machine.qAllocMany(4);
    auto c = machine.cAllocMany(4);

    auto prog = QProg();
    prog << H(q[0]);

    for (auto i = 0; i < 3; ++i)
    {
        prog << CNOT(q[i], q[i + 1]);
    }
    prog << Measure(q[0], c[0]);
    prog << Measure(q[1], c[3]);
    prog << Measure(q[2], c[2]);
    prog << Measure(q[3], c[1]);

    auto result1 = machine.runWithConfiguration(prog, 100000);
    auto result2 = machine.runWithConfiguration(prog, c, 100000);

    ASSERT_EQ(compare_map_result(result1, result2, 100000), true);
}

TEST(QVM, noise_run_with_no_cbits_args)
{
    CPUQVM machine;
    machine.setConfigure({ 64,64 });
    machine.init();

    auto q = machine.qAllocMany(4);
    auto c = machine.cAllocMany(4);

    auto prog = QProg();
    prog << H(q[0]);

    for (auto i = 0; i < 3; ++i)
    {
        prog << CNOT(q[i], q[i + 1]);
    }
    prog << MeasureAll(q, c);

    auto result1 = machine.runWithConfiguration(prog, 100000);
    auto result2 = machine.runWithConfiguration(prog, c, 100000);

    ASSERT_EQ(compare_map_result(result1, result2, 100000), true);
}

TEST(QVM, mps_run_with_no_cbits_args)
{
        MPSQVM machine;
        machine.setConfigure({ 64,64 });
        machine.init();

        auto qubits = machine.qAllocMany(6);
        auto c = machine.cAllocMany(6);

        QStat matrix = { 1.0,0,0,0,
            0,1,0,0,
            0,0,1,0,
            0,0,0,1 };

        auto prog = QProg();
        prog //<< H(qubits[0])
            /*<< S(qubits[3])
            << X(qubits[1]).control({ qubits[3], qubits[2], qubits[0] })
            << T(qubits[0])
            << Y(qubits[2])
            << Z(qubits[3])
            << X1(qubits[0])
            << Z1(qubits[2]).control({ qubits[1] })
            << Y1(qubits[3])
            << U1(qubits[0], 1.570796).control({ qubits[3], qubits[2], qubits[1] })
            << U3(qubits[0], 1.570796, 4.712389, 1.570796).control({ qubits[2] })
            << RX(qubits[0], 0.785398)
            << U2(qubits[3], 1.570796, -3.141593).control({ qubits[2], qubits[1] })
            << RY(qubits[1], 0.785398)
            << RZ(qubits[3], 0.785398)
            << CNOT(qubits[0], qubits[3])
            << iSWAP(qubits[2], qubits[1])
            << SqiSWAP(qubits[0], qubits[3])
            << SWAP(qubits[2], qubits[3])
            << Toffoli(qubits[1], qubits[0], qubits[2])
            << CZ(qubits[1], qubits[0])
            << CR(qubits[2], qubits[3], 1.570796)*/
            << RZX(qubits[4], qubits[5], 20)
            //<< RXX(qubits[2], qubits[5], 20)
            //<< RYY(qubits[7], qubits[0], 20)
            //<< RZZ(qubits[6], qubits[1], 20)
            //<< QOracle({ qubits[0], qubits[2] }, matrix)
            << Measure(qubits[0], c[0])
            << Measure(qubits[1], c[2])
            << Measure(qubits[5], c[1])
            << Measure(qubits[1], c[1]);

        auto result2 = machine.runWithConfiguration(prog, c, 1000);
        auto result1 = machine.runWithConfiguration(prog, 1000);

        for (auto val : result2)
        {
            cout << val.first << " : " << val.second << endl;
        }

        cout << " ************* " << endl;


        for (auto val : result1)
        {
            cout << val.first << " : " << val.second << endl;
        }

        cout << " ==================== " << endl;


        //ASSERT_EQ(compare_map_result(result1, result2, 1000), true);

    //getchar();
}

TEST(QVM, global_run_with_no_cbits_args)
{
    auto machine = initQuantumMachine();
    machine->setConfigure({ 64,64 });
    machine->init();

    auto q = machine->qAllocMany(4);
    auto c = machine->cAllocMany(4);

    auto prog = QProg();
    prog << H(q[0]);

    for (auto i = 0; i < 3; ++i)
    {
        prog << CNOT(q[i], q[i + 1]);
    }
    prog << MeasureAll(q, c);

    auto result1 = machine->runWithConfiguration(prog, 100000);
    auto result2 = machine->runWithConfiguration(prog, c, 100000);

    ASSERT_EQ(compare_map_result(result1, result2, 100000), true);
}

TEST(QVM, partial_amplitude_with_rxx_ryy_rzz_rzx)
{
    PartialAmplitudeQVM machine;
    machine.setConfigure({ 64,64 });
    machine.init();

    auto q = machine.qAllocMany(2);
    auto c = machine.cAllocMany(2);

    prob_vec params(4);
    for (auto &val : params)
    {
        val = random_generator19937(0.0, PI);
    }

    auto prog = QProg();
    prog << RXX(q[0], q[1], params[0]);
    prog << RYY(q[0], q[1], params[1]);
    prog << RZZ(q[0], q[1], params[2]);
    prog << RZX(q[0], q[1], params[3]);

    cout << convert_qprog_to_originir(prog, &machine);

    auto circuit = QCircuit();
    circuit << H(q[0]) << H(q[1]) << CNOT(q[0], q[1])
        << RZ(q[1], 20) << CNOT(q[0], q[1]) << H(q[1]) << H(q[0]);

    cout << getCircuitMatrix(prog) << endl;
    cout << getCircuitMatrix(circuit) << endl;

    machine.run(prog);

    vector_s indices = { "0","1","2","3" };

    auto result1 = machine.pmeasure_subset(indices);

    QStat result;
    for (auto &key : indices)
    {
        for (auto val : result1)
        {
            if (key == val.first)
            {
                result.emplace_back(val.second);
            }
        }
    }

    CPUQVM qvm;
    qvm.setConfigure({ 64,64 });
    qvm.init();

    auto qlist = qvm.qAllocMany(2);
    auto clist = qvm.cAllocMany(2);

    auto cpu_prog = QProg();
    cpu_prog << RXX(qlist[0], qlist[1], params[0]);
    cpu_prog << RYY(qlist[0], qlist[1], params[1]);
    cpu_prog << RZZ(qlist[0], qlist[1], params[2]);
    cpu_prog << RZX(qlist[0], qlist[1], params[3]);

    qvm.directlyRun(prog);
    auto result_state = qvm.getQState();

    ASSERT_EQ(matrix_compare(result, result_state, 1e-6), true);
}

TEST(QVM, single_amplitude_with_rxx_ryy_rzz_rzx)
{
    SingleAmplitudeQVM machine;
    machine.setConfigure({ 64,64 });
    machine.init();

    auto q = machine.qAllocMany(3);
    auto c = machine.cAllocMany(3);

    prob_vec params(4);
    for (auto &val : params)
    {
        val = random_generator19937(0.0, PI);
    }

    auto prog = QProg();
    prog << RXX(q[0], q[1], params[0]);
    prog << RYY(q[0], q[1], params[1]);
    prog << RZZ(q[1], q[2], params[2]);
    prog << RZX(q[1], q[2], params[3]);

    cout << getCircuitMatrix(prog) << endl;

    vector_s indices = { "0","1","2","3","4","5","6","7" };

    prob_vec result;
    for (auto val : indices)
    {
        machine.run(prog, q);
        result.emplace_back(machine.pMeasureDecindex(val));
    }

    const double cost = std::cos(0.5 * 20);
    const double sint = std::sin(0.5 * 20);

    QStat rxx_matrix =
    {
        qcomplex_t(cost, 0), qcomplex_t(0,0), qcomplex_t(0,0), qcomplex_t(0,-sint),
        qcomplex_t(0,0), qcomplex_t(cost, 0), qcomplex_t(0,-sint), qcomplex_t(0,0),
        qcomplex_t(0,0), qcomplex_t(0,-sint), qcomplex_t(cost, 0), qcomplex_t(0,0),
        qcomplex_t(0,-sint), qcomplex_t(0,0), qcomplex_t(0,0), qcomplex_t(cost, 0)
    };

    CPUQVM qvm;
    qvm.init();

    auto qlist = qvm.qAllocMany(3);
    auto clist = qvm.cAllocMany(3);

    auto cpu_prog = QProg();
    cpu_prog << RXX(qlist[0], qlist[1], params[0]);
    cpu_prog << RYY(qlist[0], qlist[1], params[1]);
    cpu_prog << RZZ(qlist[1], qlist[2], params[2]);
    cpu_prog << RZX(qlist[1], qlist[2], params[3]);

    qvm.directlyRun(prog);
    auto result_state = qvm.getQState();

    prob_vec probs;
    for (auto state : result_state)
    {
        probs.emplace_back(std::norm(state));
    }

    ASSERT_EQ(compare_probs_result(result, probs), true);
}


TEST(QVM, QHamiltonian)
{
    CPUQVM machine;
    machine.init();

    QVec qv;
    std::vector<ClassicalCondition> cv;

    std::string originir = R"(QINIT 4
                            CREG 0
                            H q[0]
                            CNOT q[0],q[1]
                            CNOT q[1],q[3]
                            CNOT q[2],q[3]
                            RZ q[3],(-0.0060418116)
                            )";

    auto prog = convert_originir_string_to_qprog(originir, &machine, qv, cv);
    auto expectation = machine.get_expectation(prog, get_test_hamiltonian(), qv);
    EXPECT_NEAR(expectation, 0.134744, 1e-7);
}

TEST(CPUQVMTest, testInit)
{
	return;
	CPUQVM qvm;
	ASSERT_THROW(auto qvec = qvm.allocateQubits(2), qvm_attributes_error);
	ASSERT_THROW(auto cvec = qvm.allocateCBits(2), qvm_attributes_error);

	qvm.init();
	ASSERT_NO_THROW(auto qvec = qvm.allocateQubits(2));
	ASSERT_NO_THROW(auto cvec = qvm.allocateCBits(2));

	ASSERT_THROW(auto qvec = qvm.allocateQubits(26), qalloc_fail);
	ASSERT_THROW(auto cvec = qvm.allocateCBits(257), calloc_fail);

	qvm.finalize();
	ASSERT_THROW(auto qvec = qvm.allocateQubits(2), qvm_attributes_error);
	ASSERT_THROW(auto cvec = qvm.allocateCBits(2), qvm_attributes_error);
	ASSERT_THROW(auto qvec = qvm.getAllocateQubit(), qvm_attributes_error);
	ASSERT_THROW(auto qvec = qvm.getAllocateCMem(), qvm_attributes_error);
	ASSERT_THROW(auto qvec = qvm.getResultMap(), qvm_attributes_error);
}

TEST(NoiseMachineTest, test)
{
	//return;
	rapidjson::Document doc;
	doc.Parse("{}");
	Value value(rapidjson::kObjectType);
	Value value_h(rapidjson::kArrayType);
	value_h.PushBack(DAMPING_KRAUS_OPERATOR, doc.GetAllocator());
	value_h.PushBack(0.5, doc.GetAllocator());
	value.AddMember("H", value_h, doc.GetAllocator());

	Value value_rz(rapidjson::kArrayType);
	value_rz.PushBack(DAMPING_KRAUS_OPERATOR, doc.GetAllocator());
	value_rz.PushBack(0.5, doc.GetAllocator());
	value.AddMember("RZ", value_rz, doc.GetAllocator());

	Value value_cnot(rapidjson::kArrayType);
	value_cnot.PushBack(DAMPING_KRAUS_OPERATOR, doc.GetAllocator());
	value_cnot.PushBack(0.5, doc.GetAllocator());
	value.AddMember("CPHASE", value_cnot, doc.GetAllocator());
	doc.AddMember("noisemodel", value, doc.GetAllocator());

	NoiseQVM qvm;
	qvm.init();
	auto qvec = qvm.allocateQubits(16);
	auto cvec = qvm.allocateCBits(16);
	auto prog = QProg();

	QCircuit  qft = CreateEmptyCircuit();
	for (auto i = 0; i < qvec.size(); i++)
	{
		qft << H(qvec[qvec.size() - 1 - i]);
		for (auto j = i + 1; j < qvec.size(); j++)
		{
			qft << CR(qvec[qvec.size() - 1 - j], qvec[qvec.size() - 1 - i], 2 * PI / (1 << (j - i + 1)));
		}
	}

	prog << qft << qft.dagger()
		<< MeasureAll(qvec, cvec);

	rapidjson::Document doc1;
	doc1.Parse("{}");
	auto& alloc = doc1.GetAllocator();
	doc1.AddMember("shots", 10, alloc);

	clock_t start = clock();
	auto result = qvm.runWithConfiguration(prog, cvec, doc1);
	clock_t end = clock();
	std::cout << end - start << endl;

	/*for (auto& aiter : result)
	{
		std::cout << aiter.first << " : " << aiter.second << endl;
	}*/

	ASSERT_EQ(result.begin()->second, 10);

	//auto state = qvm.getQState();
	//for (auto &aiter : state)
	//{
	//    std::cout << aiter << endl;
	//}
	qvm.finalize();

	//std::cout << "NoiseMachineTest.test  tests over!" << endl;
}

double getStateProb(complex<double> val)
{
	return val.real() * val.real() + val.imag() * val.imag();
}

TEST(QVM, PartialAmplitudeQVM)
{
	auto machine = new PartialAmplitudeQVM();
	machine->init();
	auto qv = machine->allocateQubits(11);
	auto cv = machine->allocateCBits(11);

	auto prog = QProg();
	prog << CZ(qv[3], qv[5])
        << CZ(qv[2], qv[4])
        << CZ(qv[3], qv[7])


        << CR(qv[0], qv[1], PI)
        << CR(qv[2], qv[3], PI)

        << CR(qv[8], qv[9], PI)
        << CR(qv[1], qv[2], PI)
        << CR(qv[9], qv[1], PI)

        << CR(qv[7], qv[8], PI);
	//<< Toffoli;

    cout << machine->get_split_num(prog) << endl;;

	std::vector<string> subSet = { "0000000000000000000001000000000000000000" ,
								   "0000000000000000000010000000000000000000" ,
								   "0000000000000000000011000000000000000000" ,
								   "0000000000000000000100000000000000000000" ,
								   "0000000000000000000101000000000000000000" ,
								   "0000000000000000000110000000000000000000" ,
								   "0000000000000000000111000000000000000000" ,
								   "1000000000000000000000000000000000000000" };

	/*for (int i = 0; i < subSet.size(); ++i)
	{
		auto result = machine->PMeasure_bin_index(subSet[i]);
		std::cout << result << std::endl;
	}*/

	ASSERT_EQ(subSet.size(), 8);

	//std::cout << val.first << " : " << val.second << std::endl;


	//getchar();
}

TEST(QVM, SingleAmplitudeQVM)
{
	auto qvm = new SingleAmplitudeQVM();
	qvm->init();
	auto qv = qvm->qAllocMany(11);
	auto cv = qvm->cAllocMany(11);

	auto prog = QProg();
	for_each(qv.begin(), qv.end(), [&](Qubit* val) { prog << H(val); });
	prog << CZ(qv[1], qv[5])
		<< CZ(qv[3], qv[5])
		<< CZ(qv[2], qv[4])
		<< CZ(qv[3], qv[7])
		<< CZ(qv[0], qv[4])
		<< RY(qv[7], PI / 2)
		<< RX(qv[8], PI / 2)
		<< RX(qv[9], PI / 2)
		<< CR(qv[0], qv[1], PI)
		<< CR(qv[2], qv[3], PI)
		<< RY(qv[4], PI / 2)
		<< RZ(qv[5], PI / 4)
		<< RX(qv[6], PI / 2)
		<< RZ(qv[7], PI / 4)
		<< CR(qv[8], qv[9], PI)
		<< CR(qv[1], qv[2], PI)
		<< RY(qv[3], PI / 2)
		<< RX(qv[4], PI / 2)
		<< RX(qv[5], PI / 2)
		<< CR(qv[9], qv[1], PI)
		<< RY(qv[1], PI / 2)
		<< RY(qv[2], PI / 2)
		<< RZ(qv[3], PI / 4)
		<< CR(qv[7], qv[8], PI);

	qvm->run(prog, qv);
	//cout << qvm->pMeasureBinindex("00001100000") << endl;

	qvm->run(prog, qv);
	//cout << qvm->pMeasureDecindex("2") << endl;

	qvm->run(prog, qv);
	auto res_1 = qvm->getProbDict(qv);

	auto res = qvm->probRunDict(prog, qv);
	/*for (auto val : res)
	{
		std::cout << val.first << " : " << val.second << std::endl;
	}*/

	ASSERT_EQ(res.size(), 2048);
	qvm->finalize();
	delete(qvm);
	//getchar();
}

TEST(QubitAddr, test_0)
{
	auto qpool = OriginQubitPool::get_instance();
	auto cmem = OriginCMem::get_instance();

	qpool->set_capacity(20);
	//std::cout << "set qubit pool capacity  after: " << qpool->get_capacity() << std::endl;

	auto qvm = new CPUQVM();
	qvm->init();
	auto qv = qpool->qAllocMany(6);
	auto cv = cmem->cAllocMany(6);

	QVec used_qv;
	auto used_qv_size = qpool->get_allocate_qubits(used_qv);
	//std::cout << "allocate qubits number: " << used_qv_size << std::endl;

	auto prog = QProg();
	prog << H(0)
		<< H(1)
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
		<< BARRIER({ 0, 1,2,3,4,5 })
		<< BARRIER(0)
		;

	auto res_0 = qvm->probRunDict(prog, { 0,1,2,3,4,5 });

	prog << Measure(0, 0)
		<< Measure(1, 1)
		<< Measure(2, 2)
		<< Measure(3, 3)
		<< Measure(4, 4)
		<< Measure(5, 5)
		;

	vector<int> cbit_addrs = { 0,1,2,3,4,5 };
	auto res_2 = qvm->runWithConfiguration(prog, cbit_addrs, 5000);
	qvm->finalize();
	delete(qvm);

	auto qvm_noise = new NoiseQVM();
	qvm_noise->init();
	auto res_4 = qvm_noise->runWithConfiguration(prog, cbit_addrs, 5000);
	qvm_noise->finalize();
	delete(qvm_noise);

	//ASSERT_EQ(res_2.size(), 48);

	//getchar();
}

static double state_probs(QStat& state)
{
	double result = 0.;
	for (auto val : state)
	{
		result += std::norm(val);
	}

	return result;
}

static prob_vec state_to_probs(QStat& state)
{
	prob_vec probs;
	for (auto val : state)
	{
		probs.emplace_back(std::norm(val));
	}

	return probs;
}


