#include "gtest/gtest.h"
#include "QPanda.h"
#include <iostream>
#include <vector>
#include <stdio.h>
#include <memory>
#include "QAlg/Base_QCircuit/base_circuit.h"

#define PRINT_TRACE 0

USING_QPANDA
using namespace std;

static bool test_layer_1()
{
    CPUQVM qvm;
	qvm.init();
	auto qv = qvm.qAllocMany(8);
	auto c = qvm.cAllocMany(8);
	
	QProg prog_dj;
	prog_dj << CZ(qv[2], qv[3]) << CZ(qv[4], qv[6]) << H(qv[0]) << X(qv[1]) << H(qv[1]) << CNOT(qv[0], qv[1]) << H(qv[0])/*<<Measure(qv[0],c[0])*/;

	QProg prog_grover;
	prog_grover << SWAP(qv[5], qv[0]) << H(qv[2]) << X(qv[2]) << SWAP(qv[1], qv[3]) << H(qv[3]) << X(qv[3]) << CZ(qv[2], qv[3])
		<< X(qv[2]) << SWAP(qv[2], qv[0]) << CZ(qv[3], qv[7]) << H(qv[2]) << Z(qv[2]) << H(qv[7])
		<< X(qv[3]) << CZ(qv[5], qv[7]) << H(qv[3]) << Z(qv[3]) << CZ(qv[2], qv[3]) << X(qv[5]) << SWAP(qv[2], qv[4])
		<< H(qv[2]) << H(qv[3]) << H(qv[6]) << Measure(qv[2], c[2]) << Measure(qv[3], c[3]);

	/*cout << "The src prog_grover:" << prog_grover << endl;
	auto_add_barrier_before_mul_qubit_gate(prog_grover);
	cout << "after add_barrier prog_grover:" << prog_grover << endl;*/

	QProg prog;
    prog << QFT(qv);
    auto layer_info = prog_layer(prog);
    std::cout << layer_info.size() << std::endl;

    //auto layer_info1 = get_chip_layer(prog, ChipID::WUYUAN_1, &qvm);
    //std::cout << layer_info1.size() << std::endl;
    
    std::map<GateType, size_t> gate_time_map;
    gate_time_map.insert(std::make_pair<GateType, size_t>(GateType::U3_GATE, 3));
    gate_time_map.insert(std::make_pair<GateType, size_t>(GateType::RPHI_GATE, 1));
    gate_time_map.insert(std::make_pair<GateType, size_t>(GateType::CNOT_GATE, 6));
    gate_time_map.insert(std::make_pair<GateType, size_t>(GateType::CZ_GATE, 2));
    auto size = get_qprog_clock_cycle_chip(layer_info, gate_time_map);
    auto size1 = get_qprog_clock_cycle(prog, &qvm);
    std::cout << "The src prog:" << size << std::endl;
    {
        /* output layer info */
        auto _layer_text = draw_qprog(prog, layer_info);

#if defined(WIN32) || defined(_WIN32)
        _layer_text = fit_to_gbk(_layer_text);
        _layer_text = Utf8ToGbkOnWin32(_layer_text.c_str());
#endif

        std::cout << _layer_text << std::endl;
    }
    
    return true;
}

TEST(Layer, test1)
{
    bool test_val = false;
    try
    {
        test_val = test_layer_1();
    }
    catch (const std::exception& e)
    {
        cout << "Got a exception: " << e.what() << endl;
    }
    catch (...)
    {
        cout << "Got an unknow exception: " << endl;
    }

    ASSERT_TRUE(test_val);
}