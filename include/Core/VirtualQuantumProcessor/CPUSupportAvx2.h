#ifndef _CPUSupportAvx2_H
#define _CPUSupportAvx2_H

#include "QPandaNamespace.h"

#if defined(__GNUC__) && defined(__x86_64__)
#define GNUC_AVX2
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(GNUC_AVX2)
#include <cpuid.h>
#endif

QPANDA_BEGIN
inline void ccpuid(int cpu_info[4], int function_id)
{
#if defined(_MSC_VER)
    __cpuid(cpu_info, function_id);
#elif defined(GNUC_AVX2)
    __cpuid(function_id,
        cpu_info[0],
        cpu_info[1],
        cpu_info[2],
        cpu_info[3]);
#else 
    cpu_info[0] = cpu_info[1] = cpu_info[2] = cpu_info[3] = 0;
#endif
}

inline void cpuidex(int cpu_info[4], int function_id, int subfunction_id)
{
#if defined(_MSC_VER)
    __cpuidex(cpu_info, function_id, subfunction_id);
#elif defined(GNUC_AVX2)
    __cpuid_count(function_id, subfunction_id, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
#else 
    cpu_info[0] = cpu_info[1] = cpu_info[2] = cpu_info[3] = 0;
#endif
}

inline bool is_supported_avx2()
{ 
#ifdef USE_SIMD 
    static bool cached = false;
    static bool is_supported = false;
    if (cached)
        return is_supported;

    std::array<int, 4> cpui;
    ccpuid(cpui.data(), 0);
    auto num_ids = cpui[0];
    if (num_ids < 7) {
        cached = true;
        is_supported = false;
        return false;
    }

    std::vector<std::array<int, 4>> data;
    for (int i = 0; i <= num_ids; ++i) {
        cpuidex(cpui.data(), i, 0);
        data.push_back(cpui);
    }

    std::bitset<32> f_1_ECX = data[1][2];
    std::bitset<32> f_7_EBX = data[7][1];

    bool is_fma_supported = (f_1_ECX[12] & 1);
    bool is_avx2_supported = (f_7_EBX[5] & 1);

    cached = true;
    is_supported = is_fma_supported && is_avx2_supported;
    return is_supported;
#else
    return false;
#endif
}
template <typename Container =
    std::vector<std::complex<double>>, typename data_t = double >
    class CPUAvx2
{
public:
    QError _three_qubit_gate_simd(Container& m_state, Qnum& qubits, QStat& matrix,
        bool is_dagger, size_t m_qubit_num, const Qnum& controls = {});

    QError _four_qubit_gate_simd(Container& m_state, Qnum& qubits, QStat& matrix,
        bool is_dagger, size_t m_qubit_num, const Qnum& controls = {});

    QError _five_qubit_gate_simd(Container& m_state, Qnum& qubits, QStat& matrix,
        bool is_dagger, size_t m_qubit_num, const Qnum& controls = {});
};


QPANDA_END
#endif