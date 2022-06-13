#include "gtest/gtest.h"
#include "QAlg/ArithmeticUnit/ArithmeticUnit_V2.h"
#include "QPanda.h"

USING_QPANDA
// use DDT test Adder

void checkAdder(int bit_len, int a_value, int b_value)
{
    auto qvm = initQuantumMachine(QMachineType::CPU);
    // std::cout << bit_len << " " << a_value << " " << b_value << std::endl;

    QVec a = qvm->qAllocMany(bit_len);
    QVec b = qvm->qAllocMany(bit_len);
    int ancil_size = std::max({CDKMRippleAdder().auxBitSize(bit_len),
                               VBERippleAdder().auxBitSize(bit_len),
                               DraperQFTAdder().auxBitSize(bit_len),
                               DraperQCLAAdder().auxBitSize(bit_len)});
    QVec ancil = qvm->qAllocMany(ancil_size);

    // cdkm
    QProg qc_cdkm = createEmptyQProg();
    if (0 != a_value)
    {
        qc_cdkm << bind_data(a_value, a);
    }
    if (0 != b_value)
    {
        qc_cdkm << bind_data(b_value, b);
    }
    qc_cdkm << QAdd_V2(a, b, ancil, ADDER::CDKM_RIPPLE);

    auto r_cdkm = dynamic_cast<CPUQVM *>(qvm)->probRunDict(qc_cdkm, a);
    auto r_cdkm_anc = dynamic_cast<CPUQVM *>(qvm)->probRunDict(qc_cdkm, ancil);

    // qft
    QProg qc_qft = createEmptyQProg();
    if (0 != a_value)
    {
        qc_qft << bind_data(a_value, a);
    }
    if (0 != b_value)
    {
        qc_qft << bind_data(b_value, b);
    }

    qc_qft << QAdd_V2(a, b, ancil, ADDER::DRAPER_QFT);

    auto r_qft = dynamic_cast<CPUQVM *>(qvm)->probRunDict(qc_qft, a);
    auto r_qft_anc = dynamic_cast<CPUQVM *>(qvm)->probRunDict(qc_qft, ancil);

    // qlc
    QProg qc_qlc = createEmptyQProg();
    if (0 != a_value)
    {
        qc_qlc << bind_data(a_value, a);
    }
    if (0 != b_value)
    {
        qc_qlc << bind_data(b_value, b);
    }
    qc_qlc << QAdd_V2(a, b, ancil, ADDER::DRAPER_QCLA);

    auto r_qlc = dynamic_cast<CPUQVM *>(qvm)->probRunDict(qc_qlc, a);
    auto r_qlc_anc = dynamic_cast<CPUQVM *>(qvm)->probRunDict(qc_qlc, ancil);

    // vbe
    QProg qc_vbe = createEmptyQProg();
    if (0 != a_value)
    {
        qc_vbe << bind_data(a_value, a);
    }
    if (0 != b_value)
    {
        qc_vbe << bind_data(b_value, b);
    }
    qc_vbe << QAdd_V2(a, b, ancil, ADDER::VBE_RIPPLE);

    auto r_vbe = dynamic_cast<CPUQVM *>(qvm)->probRunDict(qc_vbe, a);
    auto r_vbe_anc = dynamic_cast<CPUQVM *>(qvm)->probRunDict(qc_vbe, ancil);

    // original adder
    QProg qc = createEmptyQProg();
    if (0 != a_value)
    {
        qc << bind_data(a_value, a);
    }
    if (0 != b_value)
    {
        qc << bind_data(b_value, b);
    }
    qc << QAdd(a, b, ancil);

    auto r = dynamic_cast<CPUQVM *>(qvm)->probRunDict(qc, a);
    auto r_anc = dynamic_cast<CPUQVM *>(qvm)->probRunDict(qc, ancil);

    for (auto it : r)
    {
        double total = it.second + r_cdkm.at(it.first) + r_qft.at(it.first) + r_vbe.at(it.first) + r_qlc.at(it.first);
        if (total <= 0.01)
            continue;

        // std::cout << it.first << " : " << it.second
        //           << " " << r_cdkm.at(it.first)
        //           << " " << r_vbe.at(it.first)
        //           << " " << r_qft.at(it.first)
        //           << " " << r_qlc.at(it.first)
        //           << std::endl;

        ASSERT_DOUBLE_EQ(it.second, r_cdkm.at(it.first));
        ASSERT_DOUBLE_EQ(it.second, r_vbe.at(it.first));
        ASSERT_TRUE((it.second - r_qft.at(it.first) < 1e-9) && (r_qft.at(it.first) - it.second > -1e-9));
        ASSERT_DOUBLE_EQ(it.second, r_qlc.at(it.first));
    }

    for (auto it : r_cdkm_anc)
    {
        double total = it.second + r_qft_anc.at(it.first) + r_vbe_anc.at(it.first) + r_qlc_anc.at(it.first);
        if (total <= 0.01)
            continue;
        // std::cout << it.first << " : " << it.second
        //           << " " << r_vbe_anc.at(it.first)
        //           << " " << r_qft_anc.at(it.first)
        //           << " " << r_qlc_anc.at(it.first)
        //           << std::endl;

        ASSERT_DOUBLE_EQ(it.second, r_vbe_anc.at(it.first));
        ASSERT_TRUE((it.second - r_qft_anc.at(it.first) < 1e-9) && (r_qft_anc.at(it.first) - it.second > -1e-9));
        ASSERT_DOUBLE_EQ(it.second, r_qlc_anc.at(it.first));
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
            for (int b = 1 - max_value; b < max_value; b++)
            {
                checkAdder(bit_len, a, b);
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
    int ancil_capacity = int(2 * bit_len + 1 - std::floor(std::log2(bit_len)));
    QVec ancil = qvm->qAllocMany(ancil_capacity);

    // qcla complement
    QProg qcla_qprog = createEmptyQProg();
    if (0 != value)
    {
        qcla_qprog << bind_data(value, a);
    }

    qcla_qprog << QComplement_V2(a, ancil);

    auto result_qcla = dynamic_cast<CPUQVM *>(qvm)->probRunDict(qcla_qprog, a);
    auto result_qcla_anc = dynamic_cast<CPUQVM *>(qvm)->probRunDict(qcla_qprog, ancil);

    // qft complement
    QProg qft_qprog = createEmptyQProg();
    if (0 != value)
    {
        qft_qprog << bind_data(value, a);
    }

    qft_qprog << QComplement_V2(a, ancil[0]);

    auto result_qft = dynamic_cast<CPUQVM *>(qvm)->probRunDict(qft_qprog, a);
    auto result_qft_anc = dynamic_cast<CPUQVM *>(qvm)->probRunDict(qft_qprog, ancil);

    // original complement
    QProg qprog = createEmptyQProg();
    if (0 != value)
    {
        qprog << bind_data(value, a);
    }
    qprog << QComplement(a, ancil);

    auto result = dynamic_cast<CPUQVM *>(qvm)->probRunDict(qprog, a);
    auto result_anc = dynamic_cast<CPUQVM *>(qvm)->probRunDict(qprog, ancil);

    // ASSERT_EQ(result.size(), result_qcla.size());

    for (auto it : result)
    {
        double total = it.second + result_qcla.at(it.first) + result_qft.at(it.first);
        if (total <= 0.01)
            continue;
        ASSERT_DOUBLE_EQ(result_qcla.at(it.first), it.second);
        ASSERT_TRUE((it.second - result_qft.at(it.first) < 1e-9) && (result_qft.at(it.first) - it.second > -1e-9));
    }

    for (auto it : result_anc)
    {
        double total = it.second + result_qcla_anc.at(it.first) + result_qft_anc.at(it.first);
        if (total <= 0.01)
            continue;
        ASSERT_DOUBLE_EQ(result_qcla_anc.at(it.first), it.second);
        ASSERT_TRUE((it.second - result_qft_anc.at(it.first) < 1e-9) && (result_qft_anc.at(it.first) - it.second > -1e-9));
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
            checkQComplement(bit_len, i);
        }
    }
}