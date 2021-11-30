#include "gtest/gtest.h"
#include "QAlg/ArithmeticUnit/ArithmeticUnit_V2.h"
#include "QPanda.h"

USING_QPANDA
// use DDT test Adder

void checkVBE_RIPPLEAdder(int bit_len, int a_value, int b_value)
{
    auto qvm = initQuantumMachine(QMachineType::CPU);

    QVec a = qvm->qAllocMany(bit_len);
    QVec b = qvm->qAllocMany(bit_len);
    QVec ancil = qvm->qAllocMany(bit_len + 2);
    QVec result_ancil_V2(b);
    for (size_t i = 0; i < ancil.size(); i++)
    {
        result_ancil_V2.push_back(ancil[i]);
    }

    QProg draper_qprog = createEmptyQProg();

    draper_qprog << bind_data(a_value, a);
    draper_qprog << bind_data(b_value, b);
    draper_qprog << QAdd_V2(a, b, ancil, ADDER::VBE_RIPPLE);

    auto result_vbe = dynamic_cast<CPUQVM *>(qvm)->probRunDict(draper_qprog, result_ancil_V2);

    // original adder
    QProg qprog = createEmptyQProg();
    qprog << bind_data(a_value, a);
    qprog << bind_data(b_value, b);
    qprog << QAdd(a, b, ancil);

    QVec result_ancil(a);
    for (size_t i = 0; i < ancil.size(); i++)
    {
        result_ancil.push_back(ancil[i]);
    }

    auto result = dynamic_cast<CPUQVM *>(qvm)->probRunDict(qprog, result_ancil);

    ASSERT_EQ(result.size(), result_vbe.size());
    for (auto it : result)
    {
        std::cout << it.first << " " << result_vbe.at(it.first) << " " << it.second << std::endl;
        // ASSERT_DOUBLE_EQ(result_vbe.at(it.first), it.second);
    }

    destroyQuantumMachine(qvm);
}

void checkDRAPER_QFTAdder(int bit_len, int a_value, int b_value)
{

    auto qvm = initQuantumMachine(QMachineType::CPU);

    QVec a = qvm->qAllocMany(bit_len);
    QVec b = qvm->qAllocMany(bit_len);
    QVec ancil = qvm->qAllocMany(bit_len + 2);
    QVec result_ancil_V2(b);
    for (size_t i = 0; i < ancil.size(); i++)
    {
        result_ancil_V2.push_back(ancil[i]);
    }
    QProg draper_qprog = createEmptyQProg();

    draper_qprog << bind_data(a_value, a);
    draper_qprog << bind_data(b_value, b);
    draper_qprog << QAdd_V2(a, b, ancil, ADDER::DRAPER_QFT);

    auto result_draper = dynamic_cast<CPUQVM *>(qvm)->probRunDict(draper_qprog, result_ancil_V2);

    // original adder
    QProg qprog = createEmptyQProg();
    qprog << bind_data(a_value, a);
    qprog << bind_data(b_value, b);
    qprog << QAdd(a, b, ancil);

    QVec result_ancil(a);
    for (size_t i = 0; i < ancil.size(); i++)
    {
        result_ancil.push_back(ancil[i]);
    }

    auto result = dynamic_cast<CPUQVM *>(qvm)->probRunDict(qprog, result_ancil);

    ASSERT_EQ(result.size(), result_draper.size());
    for (auto it : result)
    {
        std::cout << it.first << " " << result_draper.at(it.first) << " " << it.second << std::endl;
        // ASSERT_TRUE(result_draper.at(it.first) - it.second > -1e-10 && result_draper.at(it.first) - it.second < 1e-10);
    }

    destroyQuantumMachine(qvm);
}

void checkDRAPER_QCLAAdder(int bit_len, int a_value, int b_value)
{
    auto qvm = initQuantumMachine(QMachineType::CPU);

    QVec a = qvm->qAllocMany(bit_len);
    QVec b = qvm->qAllocMany(bit_len);
    int ancil_capacity = int(2 * bit_len + 1 - std::floor(std::log2(bit_len)));
    QVec ancil = qvm->qAllocMany(ancil_capacity);
    QProg draper_qprog = createEmptyQProg();

    draper_qprog << bind_data(a_value, a);
    draper_qprog << bind_data(b_value, b);
    draper_qprog << QAdd_V2(a, b, ancil, ADDER::DRAPER_QCLA);

    QVec result_ancil_V2(b);
    for (size_t i = 0; i < ancil.size(); i++)
    {
        result_ancil_V2.push_back(ancil[i]);
    }

    auto result_draper = dynamic_cast<CPUQVM *>(qvm)->probRunDict(draper_qprog, result_ancil_V2);

    // original adder
    QProg qprog = createEmptyQProg();
    qprog << bind_data(a_value, a);
    qprog << bind_data(b_value, b);
    qprog << QAdd(a, b, ancil);

    QVec result_ancil(a);
    for (size_t i = 0; i < ancil.size(); i++)
    {
        result_ancil.push_back(ancil[i]);
    }

    auto result = dynamic_cast<CPUQVM *>(qvm)->probRunDict(qprog, result_ancil);

    ASSERT_EQ(result.size(), result_draper.size());
    for (auto it : result)
    {
        std::cout << it.first << " " << result_draper.at(it.first) << " " << it.second << std::endl;
        // ASSERT_DOUBLE_EQ(result_draper.at(it.first), it.second);
    }

    destroyQuantumMachine(qvm);
}

TEST(QAdd_V2, DDT_test_compare_with_original_QAdd)
{
    int bit_len = 2;
    for (; bit_len <= 5; bit_len++)
    {
        int max_value = std::pow(2, bit_len - 1);

        for (int a = 1 - max_value; a < max_value; a++)
        {
            if (a == 0)
                continue;
            for (int b = 1 - max_value; b < max_value; b++)
            {
                if (b == 0)
                    continue;
                std::cout << "bit: " << bit_len+1 << "  a + b " << a << " " << b << std::endl;
                std::cout << "VBE---------------------------------------" << std::endl;
                checkVBE_RIPPLEAdder(bit_len+1, a, b);
                std::cout << "QFT---------------------------------------" << std::endl;
                checkDRAPER_QFTAdder(bit_len+1, a, b);
                std::cout << "QCLA---------------------------------------" << std::endl;
                checkDRAPER_QCLAAdder(bit_len+1, a, b);
            }
        }
    }
}

// -------------------------------------------------------------------------------------------
// use DDT test Complement

void checkQComplement(int bit_len, int value)
{
    auto qvm = initQuantumMachine(QMachineType::CPU);

    QVec a = qvm->qAllocMany(bit_len);
    int ancil_capacity = int(2 * bit_len + 2 - std::floor(std::log2(bit_len)));
    QVec ancil = qvm->qAllocMany(ancil_capacity);
    QVec all(a);
    for (size_t i = 0; i < ancil.size(); i++)
    {
        all.push_back(ancil[i]);
    }

    QProg draper_qprog = createEmptyQProg();

    draper_qprog << bind_data(value, a);
    draper_qprog << QComplement_V2(a, ancil);

    auto result_v2 = dynamic_cast<CPUQVM *>(qvm)->probRunDict(draper_qprog, all);

    // original adder
    QProg qprog = createEmptyQProg();
    qprog << bind_data(value, a);
    qprog << QComplement(a, ancil);

    auto result = dynamic_cast<CPUQVM *>(qvm)->probRunDict(qprog, all);

    ASSERT_EQ(result.size(), result_v2.size());

    for (auto it : result)
    {
        ASSERT_DOUBLE_EQ(result_v2.at(it.first), it.second);
    }

    destroyQuantumMachine(qvm);
}

TEST(QComplement_V2, DDT_test_compare_with_original_QComplement)
{
    int bit_len = 2;
    for (; bit_len <= 5; bit_len++)
    {
        int max_value = std::pow(2, bit_len - 1);

        for (int i = 1 - max_value; i < max_value; i++)
        {
            if (i == 0)
                continue;
            checkQComplement(bit_len, i);
        }
    }
}