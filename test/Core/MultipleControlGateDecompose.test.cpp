#include <iostream>
#include "QPanda.h"
#include "gtest/gtest.h"
#include "Core/Utilities/Tools/MultiControlGateDecomposition.h"
#include "Core/Utilities/UnitaryDecomposer/UniformlyControlledGates.h"
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <QPandaConfig.h>


#ifdef USE_EXTENSION
#include "Extensions/VirtualZTransfer/AddBarrier.h"
#include "Extensions/VirtualZTransfer/VirtualZTransfer.h"
#endif


using namespace Eigen;
using  qmatrix_t = Matrix<qcomplex_t, Dynamic, Dynamic, RowMajor>;

USING_QPANDA
using namespace std;

bool state_compare(const QStat& state1, const QStat& state2)
{
    QPANDA_RETURN(state1.size() != state2.size(), false);

    for (auto i = 0; i < state1.size(); ++i)
    {
        if (std::fabs(state1[i].real() - state2[i].real()) > 1e-6)
            return false;
        if (std::fabs(state1[i].imag() - state2[i].imag()) > 1e-6)
            return false;
    }

    return true;
}

bool decompose_compare()
{
    CPUQVM machine;
    machine.init();

    auto q = machine.qAllocMany(6);
    auto c = machine.cAllocMany(6);

    QProg prog;
    //prog << H(q[1]) << H(q[2]) << CR(q[0], q[1], PI / 5);
    prog << CR(q[0], q[1], PI / 2).control({ q[2] });

    auto ldd = ldd_decompose(prog);
    //decompose_multiple_control_qgate(prog, &machine, "QPandaConfig.json", true);

    //machine.directlyRun(prog);
    //auto state = machine.getQState();

    //machine.directlyRun(ldd);
    //auto ldd_state = machine.getQState();

    //for (auto val : state)
    //    std::cout << val << endl;

    //std::cout << "================" << endl;

    //for (auto val : ldd_state)
     //   std::cout << val << endl;

    std::cout << convert_qprog_to_originir(prog, &machine) << endl;
    std::cout << convert_qprog_to_originir(ldd, &machine) << endl;
    std::cout << "================" << endl;

    std::cout << ldd << endl;


    std::cout << "getQGateNum(prog) " << getQGateNum(prog) << endl;
    std::cout << "getQGateNum(ldd) " << getQGateNum(ldd) << endl;
    return true;
    //return state_compare(state, ldd_state);
}

bool ldd_decompose_test_1()
{
    CPUQVM machine;
    machine.init();

    auto q = machine.qAllocMany(6);
    auto c = machine.cAllocMany(6);

    QProg prog;
    //prog << H(q[1]) << H(q[2]) << CR(q[0], q[1], PI / 5);
    prog << CR(q[0], q[1], PI / 5);

    auto ldd = ldd_decompose(prog);
    decompose_multiple_control_qgate(prog, &machine, "QPandaConfig.json", false);

    machine.directlyRun(prog);
    auto state = machine.getQState();

    machine.directlyRun(ldd);
    auto ldd_state = machine.getQState();

    for (auto val : state)
        std::cout << val << endl;

    std::cout << "================" << endl;

    for (auto val : ldd_state)
        std::cout << val << endl;

    std::cout << prog << endl;
    std::cout << ldd << endl;

    return state_compare(state, ldd_state);
}


bool ucry_decompose_test_1()
{
    CPUQVM machine;
    machine.init();

    auto q = machine.qAllocMany(3);
    auto c = machine.cAllocMany(3);

    QProg prog;

    prob_vec params;
    for (auto i = 0; i < (1ull << 2); ++i)
    {
        params.emplace_back(random_generator19937());
    }

    auto ucry = ucry_circuit({ q[0], q[1] }, q[2], params);

    cout << ucry << endl;

    auto ucry_result = ucry_decomposition({ q[0], q[1] }, q[2], params);

    cout << ucry_result << endl;

    const auto mat_1 = getCircuitMatrix(ucry);
    const auto mat_2 = getCircuitMatrix(ucry_result);

    cout << mat_1 << endl;
    cout << mat_2 << endl;

    if (mat_1 == mat_2)
    {
        std::cout << "The multi-control gate was successfully decomposed." << std::endl;
        return true;
    }

    std::cout << "Decompose error !" << std::endl;
    return false;
}

void sqrt_not_test()
{
    QStat unitary = { 0,1,1,0 };
    qmatrix_t matrix = qmatrix_t::Map(&unitary[0], 2, 2);

    Eigen::ComplexEigenSolver<qmatrix_t> solver(matrix);
    auto evecs = solver.eigenvectors();
    auto evals = solver.eigenvalues();

    for (auto i = 0; i < evals.size(); ++i)
        evals[i] = std::pow(evals[i], (double)(1. / 2));

    auto root_unitary = evecs * evals.asDiagonal() * evecs.adjoint();

    cout << root_unitary << endl;
    cout << root_unitary* root_unitary << endl;

    QStat unitary1 = { 1/SQ2,qcomplex_t(0,-1 / SQ2),qcomplex_t(0,-1 / SQ2),1 / SQ2 };

    cout << unitary1 << endl;
    cout << unitary1 * unitary1 << endl;

     unitary.clear();
     for (size_t i = 0; i < root_unitary.rows(); ++i)
     {
         for (size_t j = 0; j < root_unitary.cols(); ++j)
         {
             unitary.emplace_back((qcomplex_t)(root_unitary(i, j)));
         }
     }

     PartialAmplitudeQVM QVM;
     QVM.init();
     auto q = QVM.qAllocMany(1);

     QStat unitary12 = { 0,1,1,0 };

     QProg prog;
     prog << H(q[0]) << S(q[0]) << S(q[0]) << Z(q[0]);

     QVM.run(prog);
    
     
     auto a1 = QVM.pmeasure_bin_index("0");
     auto b1 = QVM.pmeasure_bin_index("1");
     QProg prog11;
     prog11 << U3(q[0], PI / 6, PI / 2, PI / 3).control(q[1]);

     //decompose_multiple_control_qgate(prog, &QVM);
     cout << getCircuitMatrix(prog) << endl;
     cout << getCircuitMatrix(prog11) << endl;
     //cout << QVM.getQState() << endl;



    return;
}

void pauli_hamiltonian_test()
{
    CPUQVM machine;
    machine.init();

    auto q = machine.qAllocMany(2);

    QProg prog;
    prog << H(q[0]) 
        << RX(q[0], 0.231)
        << RY(q[1], 2.51) 
        << RX(q[1], 9.831) 
        << RZ(q[0], 12.231) 
        << RX(q[1], 20.231)
        << RX(q[0], 9.831)
        << RZ(q[0], 152.231)
        << RY(q[1], 72.51)
        << RX(q[1], 98.831)
        << S(q[0]) 
        << S(q[0]) 
        << Z(q[0]);

    auto matrix = getCircuitMatrix(prog);

    PauliOperator opt;

    auto n = std::sqrt(matrix.size());

    QMatrixXd eigen_matrix = QMatrixXd::Zero(n, n);
    for (auto rdx = 0; rdx < n; ++rdx)
    {
        for (auto cdx = 0; cdx < n; ++cdx)
        {
            eigen_matrix(rdx, cdx) = matrix[rdx*n + cdx].real();
        }
    }

    //matrix_decompose_hamiltonian(&machine, eigen_matrix, opt);

    return;
}


bool ucry_decompose_test_2()
{
    CPUQVM machine;
    machine.init();

    auto q = machine.qAllocMany(4);
    auto c = machine.cAllocMany(4);

    QProg prog;

    prob_vec params;
    for (auto i = 0; i < (1ull << 3); ++i)
    {
        params.emplace_back(random_generator19937());
    }

    auto ucry = ucry_circuit({ q[0], q[1], q[2] }, q[3], params);

    cout << ucry << endl;

    auto ucry_result = ucry_decomposition({ q[0], q[1], q[2] }, q[3], params);

    cout << ucry_result << endl;

    const auto mat_1 = getCircuitMatrix(ucry);
    const auto mat_2 = getCircuitMatrix(ucry_result);

    cout << mat_1 << endl;
    cout << mat_2 << endl;


    if (mat_1 == mat_2)
    {
        std::cout << "The multi-control gate was successfully decomposed." << std::endl;
        return true;
    }

    std::cout << "Decompose error !" << std::endl;
    return false;
}

bool ucry_decompose_test_3(int start, int end)
{
    for (auto i = start; i < end; ++i)
    {
        CPUQVM machine;
        machine.init();

        auto q = machine.qAllocMany(i);
        auto c = machine.cAllocMany(i);

        QProg prog;

        prob_vec params;
        for (auto i = 0; i < (1ull << i); ++i)
            params.emplace_back(random_generator19937());

        QVec controls;
        for (auto i = 0; i < i - 1; i++)
        {
            controls.emplace_back(q[i]);
        }

        auto ucry = ucry_circuit(controls, q.back(), params);

        cout << ucry << endl;

        auto ucry_result = ucry_decomposition(controls, q.back(), params);

        cout << ucry_result << endl;

        const auto mat_1 = getCircuitMatrix(ucry);
        const auto mat_2 = getCircuitMatrix(ucry_result);

        cout << mat_1 << endl;
        cout << mat_2 << endl;

        if (mat_1 == mat_2)
        {
            std::cout << "The multi-control gate was successfully decomposed." << std::endl;
            return true;
        }

        std::cout << "Decompose error !" << std::endl;
        return false;
    }

}

bool two_qubit_rotation_gate_decompose()
{
    CPUQVM machine;
    machine.init();

    auto q = machine.qAllocMany(3);
    auto c = machine.cAllocMany(3);

    QProg prog;
    //prog << H(q[0]) << H(q[2]);
    prog << RXX(q[0], q[1], 123);
    //    << Z(q[0]).control({ q[1],q[2] });

    //prog << H(q[0]) << iSWAP(q[1], q[0]) << MeasureAll(q, c);
    //auto result = machine.runWithConfiguration(prog, c, 1000);

    std::cout << "src prog:" << prog << std::endl;

    const auto mat_1 = getCircuitMatrix(prog);

    std::cout << "src prog mat:" << getCircuitMatrix(prog) << std::endl;

    //auto ldd_prog = ldd_decompose(prog);
    decompose_multiple_control_qgate(prog, &machine, "QPandaConfig.json", false);

    cout << prog << endl;

    transform_to_base_qgate(prog, &machine);

    cout << prog << endl;
    //std::cout << "after ldd decompose_multiple_control_gate prog:" << ldd_prog << std::endl;

    //const auto mat_2 = getCircuitMatrix(ldd_prog);

    //if (mat_1 == mat_2)
    //{
        //std::cout << "The multi-control gate was successfully decomposed." << std::endl;
        //return true;
    //}

    std::cout << "Decompose error !" << std::endl;
    return false;
}

static bool control_Z_decompose_test_1()
{
	CPUQVM machine;
	//NoiseQVM machine;
	machine.init();

	std::string prog_str = "QINIT 4\r\nCREG 4\r\nH q[0]\nH q[1]\nX q[1]\nCONTROL q[0]\nZ q[1]\nENDCONTROL\nBARRIER q[0]\nX q[1]\nH q[0]\nH q[1]\nX q[0]\nX q[1]\nCONTROL q[0]\nZ q[1]\nENDCONTROL\nX q[0]\nX q[1]\nH q[0]\nH q[1]\nMEASURE q[0],c[0]\nMEASURE q[1],c[1]";
	//std::string prog_str = "QINIT 4\r\nCREG 4\r\nCONTROL q[0]\nZ q[1]\nENDCONTROL\nX q[1]\nH q[1]\nX q[1]\nMEASURE q[0],c[0]\nMEASURE q[1],c[1]";
	//std::string prog_str = "QINIT 4\r\nCREG 4\r\nX q[1]\nH q[1]\nX q[1]\nMEASURE q[0],c[0]\nMEASURE q[1],c[1]";
	//std::string prog_str = "QINIT 4\r\nCREG 4\r\nH q[0]\nX q[1]\nMEASURE q[0],c[1]\nMEASURE q[1],c[0]";
	
	const size_t shots = 1000;
	QProg prog;
	QVec q;
	vector<ClassicalCondition> c;
	prog = convert_originir_string_to_qprog(prog_str, &machine, q, c);
	std::cout << "src prog:" << prog << endl;

	const auto src_result = machine.runWithConfiguration(prog, shots);

#ifdef USE_EXTENSION
	auto_add_barrier_before_mul_qubit_gate(prog);
#endif

	decompose_multiple_control_qgate(prog, &machine, "QPandaConfig.json", true);
	std::cout << "decomposed prog:" << prog << endl;
#ifdef USE_EXTENSION
	transfer_to_u3_gate(prog, &machine);
	cout << "U3 prog: " << prog << endl;
#endif

	const auto dst_result_1 = machine.runWithConfiguration(prog, shots);
#ifdef USE_EXTENSION
	// VirtualZ transfer
	decompose_U3(prog, "QPandaConfig.json");
#endif
	const auto dst_result_2 = machine.runWithConfiguration(prog, shots);

	for (const auto& _src_result_item : src_result)
	{
		auto _itr = dst_result_2.find(_src_result_item.first);
		if (dst_result_2.end() == _itr){
			return false;
		}

		if (_itr->second != _src_result_item.second){
			return false;
		}
	}

	return true;
}


TEST(MultipleControlGateDecompose, test)
{
	bool test_val = true;
	try
	{
        test_val = test_val && ucry_decompose_test_1();
		test_val = test_val && control_Z_decompose_test_1();
        //test_val = test_val && ucry_decompose_test_2();
        //test_val = test_val && ucry_decompose_test_3(4, 9);
        //test_val = test_val && decompose_compare();
	}
	catch (const std::exception& e)
	{
        QCERR(e.what());
    }
    catch (...)
    {
        QCERR("unknow exception");
	}

	ASSERT_TRUE(test_val);
    std::cout << "MultipleControlGateDecompose test pass" << std::endl;
}