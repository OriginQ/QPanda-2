#include "QPanda.h"
#include "gtest/gtest.h"
#include <functional>
#include <thread>

USING_QPANDA
using namespace std;
TEST(CPUQVMTest, testAsyncRunAndGetProcess)
{
    auto qvm = CPUQVM();
    qvm.init();

    auto qv = qvm.qAllocMany(3);
    auto cv = qvm.cAllocMany(3);

    auto qprog = QProg();

    for (size_t i = 0; i < 1001; i++)
    {
        qprog << H(qv);
    }
    qprog << MeasureAll(qv, cv);
    size_t total_num = qprog.get_qgate_num();

    ASSERT_EQ(qvm.get_processed_qgate_num(), 0);

    // qprog run by new thread at background
    qvm.async_run(qprog);

    // main thread accese progress
    // wait for qprog start
    size_t processed_gate_num = 0;
    while (!qvm.is_async_finished())
    {
        processed_gate_num = qvm.get_processed_qgate_num();
        std::cout << "processed_gate_num : " << processed_gate_num << "/" << total_num << std::endl;
        ASSERT_TRUE(processed_gate_num >= 0);
        ASSERT_TRUE(processed_gate_num <= total_num);
    }

    ASSERT_EQ(qvm.get_processed_qgate_num(), 0);

    auto result = qvm.get_async_result();

    for (auto &item : result)
    {
        std::cout << item.first << " " << item.second << std::endl;
    }

    qvm.finalize();
}