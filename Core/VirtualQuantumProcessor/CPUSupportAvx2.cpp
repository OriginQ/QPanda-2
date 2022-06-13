
#include <vector>
#include <cstdint>
#include <stdio.h>
#include <array>
#include <bitset>
#include <algorithm>
#include "QPandaConfig.h"
#include"Core/Utilities/Tools/QPandaException.h"
#include "Core/VirtualQuantumProcessor/QError.h"
#include "QPandaNamespace.h"
#include "Core/VirtualQuantumProcessor/CPUSupportAvx2.h"
USING_QPANDA
#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_SIMD
#include <immintrin.h>

template <typename data_t>
struct RealStat
{
    RealStat() = delete;
    RealStat(data_t* data) : m_data(data) {}
    inline data_t* operator[](size_t i) {
        return &m_data[i * 2];
    }

    inline const data_t* operator[](size_t i) const {
        return &m_data[i * 2];
    }

    data_t* m_data = nullptr;
};

template <typename data_t>
struct ImagStat : std::false_type {};

template <>
struct ImagStat<double>
{
    ImagStat() = delete;
    ImagStat(double* data) : m_data(data) {}
    inline double* operator[](size_t i) {
        return &m_data[i * 2 + 4];
    }
    inline const double* operator[](size_t i) const {
        return &m_data[i * 2 + 4];
    }
    double* m_data = nullptr;
};

template <>
struct ImagStat<float>
{
    ImagStat() = delete;
    ImagStat(float* data) : m_data(data) {}
    inline float* operator[](size_t i)
    {
        return &m_data[i * 2 + 8];
    }
    inline const float* operator[](size_t i) const
    {
        return &m_data[i * 2 + 8];
    }
    float* m_data = nullptr;
};

template <typename data>
using m256_t = typename std::conditional<std::is_same<data, double>::value, __m256d, __m256>::type;


static auto _mm256_mul(const m256_t<double>& left, const m256_t<double>& right) {
    return _mm256_mul_pd(left, right);
}

static auto _mm256_mul(const m256_t<float>& left, const m256_t<float>& right) {
    return _mm256_mul_ps(left, right);
}

static auto _mm256_fnmadd(const m256_t<double>& left,
    const m256_t<double>& right, const m256_t<double>& ret) {
    return _mm256_fnmadd_pd(left, right, ret);
}

static auto _mm256_fnmadd(const m256_t<float>& left,
    const m256_t<float>& right, const m256_t<float>& ret) {
    return _mm256_fnmadd_ps(left, right, ret);
}

static auto _mm256_fmadd(const m256_t<double>& left,
    const m256_t<double>& right, const m256_t<double>& ret) {
    return _mm256_fmadd_pd(left, right, ret);
}

static auto _mm256_fmadd(const m256_t<float>& left,
    const m256_t<float>& right, const m256_t<float>& ret) {
    return _mm256_fmadd_ps(left, right, ret);
}

static auto _mm256_set1(double d) {
    return _mm256_set1_pd(d);
}

static auto _mm256_set1(float f) {
    return _mm256_set1_ps(f);
}

static auto _mm256_load(double const* d) {
    return _mm256_loadu_pd(d);
}

static auto _mm256_load(float const* f) {
    return _mm256_loadu_ps(f);
}

static void _mm256_store(float* f, const m256_t<float>& c) {
    _mm256_storeu_ps(f, c);
}

static void _mm256_store(double* d, const m256_t<double>& c) {
    _mm256_storeu_pd(d, c);
}

static m256_t<double>_mm256_hsub(m256_t<double>& vec1, m256_t<double>& vec2) {
    return _mm256_hsub_pd(vec1, vec2);
}

static m256_t<float> _mm256_hsub(m256_t<float>& vec1, m256_t<float>& vec2) {
    return _mm256_hsub_ps(vec1, vec2);
}

static m256_t<double> _mm256_swith_real_and_imag(m256_t<double>& vec) {
    return _mm256_permute_pd(vec, 0b0101);
}

static m256_t<float> _mm256_swith_real_and_imag(m256_t<float>& vec) {
    return _mm256_permute_ps(vec, _MM_SHUFFLE(2, 3, 0, 1));
}

static m256_t<double> _mm256_neg(double dummy) {
    return _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
}

static m256_t<float> _mm256_neg(float dummy) {
    return _mm256_setr_ps(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
}

static m256_t<double> _mm256_align(m256_t<double>& vec) {
    return vec;
}

static m256_t<float> _mm256_align(m256_t<float>& vec) {
    return _mm256_permute_ps(vec, _MM_SHUFFLE(3, 1, 2, 0));
}

template <typename FloatType>
static inline void _mm_complex_multiply(m256_t<FloatType>& vec1,
    m256_t<FloatType>& vec2) {
    m256_t<FloatType> vec3 = _mm256_mul(vec1, vec2);
    vec2 = _mm256_swith_real_and_imag(vec2);
    vec2 = _mm256_mul(vec2, _mm256_neg((FloatType)0.0));
    m256_t<FloatType> vec4 = _mm256_mul(vec1, vec2);
    vec1 = _mm256_hsub(vec3, vec4);
    vec1 = _mm256_align(vec1);
}

template <typename FloatType>
static inline void _mm_complex_multiply(m256_t<FloatType>& real_ret,
    m256_t<FloatType>& imag_ret,
    m256_t<FloatType>& real_left,
    m256_t<FloatType>& imag_left,
    const m256_t<FloatType>& real_right,
    const m256_t<FloatType>& imag_right)
{
    real_ret = _mm256_mul(real_left, real_right);
    imag_ret = _mm256_mul(real_left, imag_right);
    real_ret = _mm256_fnmadd(imag_left, imag_right, real_ret);
    imag_ret = _mm256_fmadd(imag_left, real_right, imag_ret);
}

template <typename FloatType>
static inline void _mm_complex_multiply_add(
    m256_t<FloatType>& real_ret,
    m256_t<FloatType>& imag_ret,
    m256_t<FloatType>& real_left,
    m256_t<FloatType>& imag_left,
    const m256_t<FloatType>& real_right,
    const m256_t<FloatType>& imag_right)
{
    real_ret = _mm256_fmadd(real_left, real_right, real_ret);
    imag_ret = _mm256_fmadd(real_left, imag_right, imag_ret);
    real_ret = _mm256_fnmadd(imag_left, imag_right, real_ret);
    imag_ret = _mm256_fmadd(imag_left, real_right, imag_ret);
}

template <typename FloatType>
static inline void _mm_complex_internal_calc(size_t dim,
    m256_t<FloatType> vreals[],
    m256_t<FloatType> vimags[],
    const FloatType* cmplxs,
    m256_t<FloatType>& vret_real,
    m256_t<FloatType>& vret_imag,
    m256_t<FloatType>& vtmp_0,
    m256_t<FloatType>& vtmp_1) {
    vtmp_0 = _mm256_set1(cmplxs[0]);
    vtmp_1 = _mm256_set1(cmplxs[1]);
    _mm_complex_multiply<FloatType>(vret_real, vret_imag, vreals[0], vimags[0],
        vtmp_0, vtmp_1);
    for (size_t i = 1; i < dim; ++i) {
        vtmp_0 = _mm256_set1(cmplxs[i * 2]);
        vtmp_1 = _mm256_set1(cmplxs[i * 2 + 1]);
        _mm_complex_multiply_add<FloatType>(vret_real, vret_imag, vreals[i],
            vimags[i], vtmp_0, vtmp_1);
    }
}

static inline void _mm_load_twoarray_complex_double(const double* real_addr_0,
    const double* imag_addr_1, m256_t<double>& real_ret, m256_t<double>& imag_ret)
{
    real_ret = _mm256_load(real_addr_0);
    imag_ret = _mm256_load(imag_addr_1);
    auto real_tmp = _mm256_permute4x64_pd(real_ret, 2 * 64 + 3 * 16 + 0 * 4 + 1 * 1);
    auto imag_tmp = _mm256_permute4x64_pd(imag_ret, 2 * 64 + 3 * 16 + 0 * 4 + 1 * 1);
    real_ret = _mm256_blend_pd(real_ret, imag_tmp, 0b1010);
    imag_ret = _mm256_blend_pd(real_tmp, imag_ret, 0b1010);
    real_ret = _mm256_permute4x64_pd(real_ret, 3 * 64 + 1 * 16 + 2 * 4 + 0 * 1);
    imag_ret = _mm256_permute4x64_pd(imag_ret, 3 * 64 + 1 * 16 + 2 * 4 + 0 * 1);
}

static inline void _mm_load_twoarray_complex_float(const float* real_addr_0,
    const float* imag_addr_1, m256_t<float>& real_ret, m256_t<float>& imag_ret)
{
    real_ret = _mm256_load(real_addr_0);
    imag_ret = _mm256_load(imag_addr_1);
    auto real_tmp = _mm256_permutevar8x32_ps(real_ret, _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1));
    auto imag_tmp = _mm256_permutevar8x32_ps(imag_ret, _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1));
    real_ret = _mm256_blend_ps(real_ret, imag_tmp, 0b10101010);
    imag_ret = _mm256_blend_ps(real_tmp, imag_ret, 0b10101010);
    real_ret = _mm256_permutevar8x32_ps(real_ret, _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));
    imag_ret = _mm256_permutevar8x32_ps(imag_ret, _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));
}

static inline void _mm_store_twoarray_complex_float(m256_t<float>& real_ret,
    m256_t<float>& imag_ret, float* cmplx_addr_0, float* cmplx_addr_1)
{
    real_ret = _mm256_permutevar8x32_ps(real_ret, _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0));
    imag_ret = _mm256_permutevar8x32_ps(imag_ret, _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0));
    auto real_tmp = _mm256_permutevar8x32_ps(real_ret, _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1));
    auto imag_tmp = _mm256_permutevar8x32_ps(imag_ret, _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1));
    real_ret = _mm256_blend_ps(real_ret, imag_tmp, 0b10101010);
    imag_ret = _mm256_blend_ps(real_tmp, imag_ret, 0b10101010);
    _mm256_store(cmplx_addr_0, real_ret);
    _mm256_store(cmplx_addr_1, imag_ret);
}

static inline void _mm_store_twoarray_complex_double(m256_t<double>& real_ret,
    m256_t<double>& imag_ret, double* cmplx_addr_0, double* cmplx_addr_1)
{
    real_ret = _mm256_permute4x64_pd(real_ret, 3 * 64 + 1 * 16 + 2 * 4 + 0 * 1);
    imag_ret = _mm256_permute4x64_pd(imag_ret, 3 * 64 + 1 * 16 + 2 * 4 + 0 * 1);
    auto real_tmp = _mm256_permute4x64_pd(real_ret, 2 * 64 + 3 * 16 + 0 * 4 + 1 * 1);
    auto imag_tmp = _mm256_permute4x64_pd(imag_ret, 2 * 64 + 3 * 16 + 0 * 4 + 1 * 1);
    real_ret = _mm256_blend_pd(real_ret, imag_tmp, 0b1010);
    imag_ret = _mm256_blend_pd(real_tmp, imag_ret, 0b1010);
    _mm256_store(cmplx_addr_0, real_ret);
    _mm256_store(cmplx_addr_1, imag_ret);
}
template <size_t num_qubits>
void _qoracle_internal_calc1_double(const double* matrix_d, Qnum& qubits, int64_t index, RealStat<double>& reals, ImagStat<double>& imags)
{
    const int double_q0q1_0 = 225;  //3 * 64 + 2 * 16 + 0 * 4 + 1 * 1
    const int double_q0q1_1 = 198;  // 3 * 64 + 0 * 16 + 1 * 4 + 2 * 1
    const int double_q0q1_2 = 39;   //0 * 64 + 2 * 16 + 1 * 4 + 3 * 1

    __m256d real_ret, imag_ret, real_ret1, imag_ret1;
    __m256d vreals[1ull << num_qubits], vimags[1ull << num_qubits];
    __m256d tmp0, tmp1;

    constexpr auto indexes_size = 1ll << num_qubits;
    int64_t indexes[indexes_size];
    load_index(index, qubits.size(), indexes, indexes_size, qubits);

    for (size_t i = 0; i < (1ll << qubits.size()); i += 4)
    {
        auto index = indexes[i];
        _mm_load_twoarray_complex_double(reals[index], imags[index], vreals[i], vimags[i]);
        for (size_t j = 1; j < 4; ++j) {
            switch (j) {
            case 1:
                vreals[i + j] = _mm256_permute4x64_pd(vreals[i], double_q0q1_0);
                vimags[i + j] = _mm256_permute4x64_pd(vimags[i], double_q0q1_0);
                break;
            case 2:
                vreals[i + j] = _mm256_permute4x64_pd(vreals[i], double_q0q1_1);
                vimags[i + j] = _mm256_permute4x64_pd(vimags[i], double_q0q1_1);
                break;
            case 3:
                vreals[i + j] = _mm256_permute4x64_pd(vreals[i], double_q0q1_2);
                vimags[i + j] = _mm256_permute4x64_pd(vimags[i], double_q0q1_2);
                break;
            }
        }
    }

    size_t mindex = 0;
    for (size_t i = 0; i < (1ll << qubits.size()); i += 4)
    {
        auto index = indexes[i];
        _mm_complex_internal_calc<double>((1ll << qubits.size()), vreals, vimags,
            (&matrix_d[mindex]), real_ret, imag_ret, tmp0,
            tmp1);
        mindex += (1ll << (qubits.size() + 1));
        for (size_t j = 1; j < 4; ++j) {
            _mm_complex_internal_calc<double>((1ll << qubits.size()), vreals, vimags,
                (&matrix_d[mindex]), real_ret1, imag_ret1,
                tmp0, tmp1);
            mindex += (1ll << (qubits.size() + 1));
            switch (j) {
            case 1:
                real_ret1 = _mm256_permute4x64_pd(real_ret1, double_q0q1_0);
                imag_ret1 = _mm256_permute4x64_pd(imag_ret1, double_q0q1_0);
                real_ret = _mm256_blend_pd(real_ret, real_ret1, 0b0010);
                imag_ret = _mm256_blend_pd(imag_ret, imag_ret1, 0b0010);
                break;
            case 2:
                real_ret1 = _mm256_permute4x64_pd(real_ret1, double_q0q1_1);
                imag_ret1 = _mm256_permute4x64_pd(imag_ret1, double_q0q1_1);
                real_ret = _mm256_blend_pd(real_ret, real_ret1, 0b0100);
                imag_ret = _mm256_blend_pd(imag_ret, imag_ret1, 0b0100);
                break;
            case 3:
                real_ret1 = _mm256_permute4x64_pd(real_ret1, double_q0q1_2);
                imag_ret1 = _mm256_permute4x64_pd(imag_ret1, double_q0q1_2);
                real_ret = _mm256_blend_pd(real_ret, real_ret1, 0b1000);
                imag_ret = _mm256_blend_pd(imag_ret, imag_ret1, 0b1000);
                break;
            }
        }
        _mm_store_twoarray_complex_double(real_ret, imag_ret, reals[index], imags[index]);
    }
}

template <size_t num_qubits>
void _qoracle_internal_calc2_double(const double* matrix_d, Qnum& qubits, int64_t index, RealStat<double>& reals, ImagStat<double>& imags)
{
    const int PERM_D_Q0 = 2 * 64 + 3 * 16 + 0 * 4 + 1 * 1;
    const int PERM_D_Q1 = 1 * 64 + 0 * 16 + 3 * 4 + 2 * 1;


    constexpr auto indexes_size = 1ull << num_qubits;
    int64_t indexes[indexes_size];
    load_index(index, qubits.size(), indexes, indexes_size, qubits);

    __m256d real_ret, imag_ret, real_ret1, imag_ret1;
    int dim = 1ll << qubits.size();
    __m256d vreals[1ull << num_qubits], vimags[1ull << num_qubits];
    __m256d tmp0, tmp1;

    for (size_t i = 0; i < (1ll << qubits.size()); i += 2)
    {
        auto index = indexes[i];
        _mm_load_twoarray_complex_double(reals[index], imags[index], vreals[i], vimags[i]);
        if (qubits[0] == 0) {
            vreals[i + 1] = _mm256_permute4x64_pd(vreals[i], PERM_D_Q0);
            vimags[i + 1] = _mm256_permute4x64_pd(vimags[i], PERM_D_Q0);
        }
        else {
            vreals[i + 1] = _mm256_permute4x64_pd(vreals[i], PERM_D_Q1);
            vimags[i + 1] = _mm256_permute4x64_pd(vimags[i], PERM_D_Q1);
        }
    }

    size_t mindex = 0;
    for (size_t i = 0; i < (1ll << qubits.size()); i += 2)
    {
        auto index = indexes[i];
        _mm_complex_internal_calc<double>((1ll << qubits.size()), vreals, vimags,
            (&matrix_d[mindex]), real_ret, imag_ret, tmp0,
            tmp1);
        mindex += (1ll << (qubits.size() + 1));

        _mm_complex_internal_calc<double>((1ll << qubits.size()), vreals, vimags,
            (&matrix_d[mindex]), real_ret1, imag_ret1,
            tmp0, tmp1);
        mindex += (1ll << (qubits.size() + 1));

        if (qubits[0] == 0) {
            real_ret1 = _mm256_permute4x64_pd(real_ret1, PERM_D_Q0);
            imag_ret1 = _mm256_permute4x64_pd(imag_ret1, PERM_D_Q0);
            real_ret = _mm256_blend_pd(real_ret, real_ret1, 0b1010);
            imag_ret = _mm256_blend_pd(imag_ret, imag_ret1, 0b1010);
        }
        else {
            real_ret1 = _mm256_permute4x64_pd(real_ret1, PERM_D_Q1);
            imag_ret1 = _mm256_permute4x64_pd(imag_ret1, PERM_D_Q1);
            real_ret = _mm256_blend_pd(real_ret, real_ret1, 0b1100);
            imag_ret = _mm256_blend_pd(imag_ret, imag_ret1, 0b1100);
        }
        _mm_store_twoarray_complex_double(real_ret, imag_ret, reals[index], imags[index]);
    }
}

template <size_t num_qubits>
void _qoracle_internal_calc3_double(const double* matrix_d, Qnum& qubits, int64_t index, RealStat<double>& reals, ImagStat<double>& imags)
{
    constexpr auto indexes_size = 1ull << num_qubits;
    int64_t indexes[indexes_size];
    load_index(index, qubits.size(), indexes, indexes_size, qubits);
    __m256d real_ret, imag_ret;
    int dim = 1ll << qubits.size();
    __m256d vreals[1ull << num_qubits], vimags[1ull << num_qubits];
    __m256d tmp0, tmp1;
    for (int i = 0; i < (1ll << qubits.size()); i++)
    {
        auto index = indexes[i];
        _mm_load_twoarray_complex_double(reals[index], imags[index], vreals[i], vimags[i]);
    }

    size_t mindex = 0;
    for (size_t i = 0; i < (1ll << qubits.size()); ++i)
    {
        auto index = indexes[i];
        _mm_complex_internal_calc<double>((1ll << qubits.size()), vreals, vimags,
            (&matrix_d[mindex]), real_ret, imag_ret, tmp0, tmp1);
        mindex += (1ll << (qubits.size() + 1));
        _mm_store_twoarray_complex_double(real_ret, imag_ret, reals[index], imags[index]);

    }

}

template <size_t num_qubits>
void _qoracle_internal_calc1_float(const float* matrix_d, Qnum& qubits, int64_t index, RealStat<float>& reals, ImagStat<float>& imags)
{
    constexpr auto indexes_size = 1ull << num_qubits;
    int64_t indexes[indexes_size];
    load_index(index, qubits.size(), indexes, indexes_size, qubits);

    __m256 real_ret, imag_ret, real_ret1, imag_ret1;
    __m256 vreals[1ull << num_qubits], vimags[1ull << num_qubits];
    __m256 tmp0, tmp1;

    const __m256i _MASKS[7] = { _mm256_set_epi32(7, 6, 5, 4, 3, 2, 0, 1),
                               _mm256_set_epi32(7, 6, 5, 4, 3, 0, 1, 2),
                               _mm256_set_epi32(7, 6, 5, 4, 0, 2, 1, 3),
                               _mm256_set_epi32(7, 6, 5, 0, 3, 2, 1, 4),
                               _mm256_set_epi32(7, 6, 0, 4, 3, 2, 1, 5),
                               _mm256_set_epi32(7, 0, 5, 4, 3, 2, 1, 6),
                               _mm256_set_epi32(0, 6, 5, 4, 3, 2, 1, 7) };

    for (size_t i = 0; i < indexes_size; i += 8) {
        auto index = indexes[i];
        _mm_load_twoarray_complex_float(reals[index], imags[index], vreals[i], vimags[i]);

        for (size_t j = 1; j < 8; ++j) {
            vreals[i + j] = _mm256_permutevar8x32_ps(vreals[i], _MASKS[j - 1]);
            vimags[i + j] = _mm256_permutevar8x32_ps(vimags[i], _MASKS[j - 1]);
        }
    }

    size_t mindex = 0;
    for (size_t i = 0; i < indexes_size; i += 8) {
        auto index = indexes[i];
        _mm_complex_internal_calc<float>((1ull << num_qubits), vreals, vimags,
            (&matrix_d[mindex]), real_ret, imag_ret, tmp0, tmp1);
        mindex += (1ll << (num_qubits + 1));

        for (size_t j = 1; j < 8; ++j) {
            _mm_complex_internal_calc<float>((1ll << num_qubits), vreals, vimags,
                (&matrix_d[mindex]), real_ret1, imag_ret1, tmp0, tmp1);
            mindex += (1ll << (num_qubits + 1));

            real_ret1 = _mm256_permutevar8x32_ps(real_ret1, _MASKS[j - 1]);
            imag_ret1 = _mm256_permutevar8x32_ps(imag_ret1, _MASKS[j - 1]);

            switch (j) {
            case 1:
                real_ret = _mm256_blend_ps(real_ret, real_ret1, 0b00000010);
                imag_ret = _mm256_blend_ps(imag_ret, imag_ret1, 0b00000010);
                break;
            case 2:
                real_ret = _mm256_blend_ps(real_ret, real_ret1, 0b00000100);
                imag_ret = _mm256_blend_ps(imag_ret, imag_ret1, 0b00000100);
                break;
            case 3:
                real_ret = _mm256_blend_ps(real_ret, real_ret1, 0b00001000);
                imag_ret = _mm256_blend_ps(imag_ret, imag_ret1, 0b00001000);
                break;
            case 4:
                real_ret = _mm256_blend_ps(real_ret, real_ret1, 0b00010000);
                imag_ret = _mm256_blend_ps(imag_ret, imag_ret1, 0b00010000);
                break;
            case 5:
                real_ret = _mm256_blend_ps(real_ret, real_ret1, 0b00100000);
                imag_ret = _mm256_blend_ps(imag_ret, imag_ret1, 0b00100000);
                break;
            case 6:
                real_ret = _mm256_blend_ps(real_ret, real_ret1, 0b01000000);
                imag_ret = _mm256_blend_ps(imag_ret, imag_ret1, 0b01000000);
                break;
            case 7:
                real_ret = _mm256_blend_ps(real_ret, real_ret1, 0b10000000);
                imag_ret = _mm256_blend_ps(imag_ret, imag_ret1, 0b10000000);
                break;
            }
        }
        _mm_store_twoarray_complex_float(real_ret, imag_ret, reals[index], imags[index]);
    }
}

template <size_t num_qubits>
void _qoracle_internal_calc2_float(const float* matrix_d, Qnum& qubits, int64_t index, RealStat<float>& reals, ImagStat<float>& imags)
{
    __m256i masks[3];
    __m256 real_ret, imag_ret, real_ret1, imag_ret1;
    __m256 vreals[1ll << num_qubits], vimags[1ll << num_qubits];
    __m256 tmp0, tmp1;

    constexpr auto indexes_size = 1ull << num_qubits;
    int64_t indexes[indexes_size];
    load_index(index, qubits.size(), indexes, indexes_size, qubits);

    if (qubits[1] == 1) {
        masks[0] = _mm256_set_epi32(7, 6, 4, 5, 3, 2, 0, 1);
        masks[1] = _mm256_set_epi32(7, 4, 5, 6, 3, 0, 1, 2);
        masks[2] = _mm256_set_epi32(4, 6, 5, 7, 0, 2, 1, 3);
    }
    else if (qubits[0] == 0) {
        masks[0] = _mm256_set_epi32(7, 6, 5, 4, 2, 3, 0, 1);
        masks[1] = _mm256_set_epi32(7, 2, 5, 0, 3, 6, 1, 4);
        masks[2] = _mm256_set_epi32(2, 6, 0, 4, 3, 7, 1, 5);
    }
    else {
        masks[0] = _mm256_set_epi32(7, 6, 5, 4, 1, 0, 3, 2);
        masks[1] = _mm256_set_epi32(7, 6, 1, 0, 3, 2, 5, 4);
        masks[2] = _mm256_set_epi32(1, 0, 5, 4, 3, 2, 7, 6);
    }

    for (size_t i = 0; i < (1ll << num_qubits); i += 4) {
        auto index = indexes[i];
        _mm_load_twoarray_complex_float(reals[index], imags[index], vreals[i], vimags[i]);

        for (size_t j = 0; j < 3; ++j) {
            vreals[i + j + 1] = _mm256_permutevar8x32_ps(vreals[i], masks[j]);
            vimags[i + j + 1] = _mm256_permutevar8x32_ps(vimags[i], masks[j]);
        }
    }

    size_t mindex = 0;
    for (size_t i = 0; i < (1ll << num_qubits); i += 4)
    {
        auto index = indexes[i];
        _mm_complex_internal_calc<float>((1ll << num_qubits), vreals, vimags,
            (&matrix_d[mindex]), real_ret, imag_ret, tmp0, tmp1);
        mindex += (1ull << (num_qubits + 1));

        for (size_t j = 0; j < 3; ++j)
        {
            _mm_complex_internal_calc<float>((1ll << num_qubits), vreals, vimags,
                (&matrix_d[mindex]), real_ret1, imag_ret1, tmp0, tmp1);
            mindex += (1ll << (num_qubits + 1));

            real_ret1 = _mm256_permutevar8x32_ps(real_ret1, masks[j]);
            imag_ret1 = _mm256_permutevar8x32_ps(imag_ret1, masks[j]);

            switch (j) {
            case 0:
                real_ret = (qubits[1] == 1)
                    ? _mm256_blend_ps(real_ret, real_ret1, 0b00100010)
                    :  // (0,1)
                    (qubits[0] == 0)
                    ? _mm256_blend_ps(real_ret, real_ret1, 0b00001010)
                    :  // (0,2)
                    _mm256_blend_ps(real_ret, real_ret1,
                        0b00001100);  //  (1,2)
                imag_ret = (qubits[1] == 1)
                    ? _mm256_blend_ps(imag_ret, imag_ret1, 0b00100010)
                    :  // (0,1)
                    (qubits[0] == 0)
                    ? _mm256_blend_ps(imag_ret, imag_ret1, 0b00001010)
                    :  // (0,2)
                    _mm256_blend_ps(imag_ret, imag_ret1,
                        0b00001100);  //  (1,2)
                break;
            case 1:
                real_ret = (qubits[1] == 1)
                    ? _mm256_blend_ps(real_ret, real_ret1, 0b01000100)
                    :  // (0,1)
                    (qubits[0] == 0)
                    ? _mm256_blend_ps(real_ret, real_ret1, 0b01010000)
                    :  // (0,2)
                    _mm256_blend_ps(real_ret, real_ret1,
                        0b00110000);  //   (1,2)
                imag_ret = (qubits[1] == 1)
                    ? _mm256_blend_ps(imag_ret, imag_ret1, 0b01000100)
                    :  // (0,1)
                    (qubits[0] == 0)
                    ? _mm256_blend_ps(imag_ret, imag_ret1, 0b01010000)
                    :  // (0,2)
                    _mm256_blend_ps(imag_ret, imag_ret1,
                        0b00110000);  //   (1,2)
                break;
            case 2:
                real_ret = (qubits[1] == 1)
                    ? _mm256_blend_ps(real_ret, real_ret1, 0b10001000)
                    :  // (0,1)
                    (qubits[0] == 0)
                    ? _mm256_blend_ps(real_ret, real_ret1, 0b10100000)
                    :  // (0,2)
                    _mm256_blend_ps(real_ret, real_ret1,
                        0b11000000);  //  (1,2)
                imag_ret = (qubits[1] == 1)
                    ? _mm256_blend_ps(imag_ret, imag_ret1, 0b10001000)
                    :  // (0,1)
                    (qubits[0] == 0)
                    ? _mm256_blend_ps(imag_ret, imag_ret1, 0b10100000)
                    :  // (0,2)
                    _mm256_blend_ps(imag_ret, imag_ret1,
                        0b11000000);  //  (1,2)
                break;
            }
        }
        _mm_store_twoarray_complex_float(real_ret, imag_ret, reals[index], imags[index]);
    }
}

template <size_t num_qubits>
void _qoracle_internal_calc3_float(const float* matrix_d, Qnum& qubits, int64_t index, RealStat<float>& reals, ImagStat<float>& imags)
{
    constexpr auto indexes_size = 1ull << num_qubits;
    int64_t indexes[indexes_size];
    load_index(index, qubits.size(), indexes, indexes_size, qubits);

    __m256i mask;
    __m256 real_ret, imag_ret, real_ret1, imag_ret1;
    __m256 vreals[1ll << num_qubits], vimags[1ll << num_qubits];
    __m256 tmp0, tmp1;

    if (qubits[0] == 0) {
        mask = _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1);
    }
    else if (qubits[0] == 1) {
        mask = _mm256_set_epi32(5, 4, 7, 6, 1, 0, 3, 2);
    }
    else {  // if (q0 == 2) {
        mask = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
    }

    for (size_t i = 0; i < (1ull << num_qubits); i += 2) {
        auto index = indexes[i];
        _mm_load_twoarray_complex_float(reals[index], imags[index], vreals[i], vimags[i]);

        vreals[i + 1] = _mm256_permutevar8x32_ps(vreals[i], mask);
        vimags[i + 1] = _mm256_permutevar8x32_ps(vimags[i], mask);
    }

    size_t mindex = 0;
    for (size_t i = 0; i < (1ll << num_qubits); i += 2) {
        auto index = indexes[i];
        _mm_complex_internal_calc<float>((1ll << num_qubits), vreals, vimags,
            (&matrix_d[mindex]), real_ret, imag_ret, tmp0, tmp1);
        mindex += (1ll << (num_qubits + 1));

        _mm_complex_internal_calc<float>((1ll << num_qubits), vreals, vimags,
            (&matrix_d[mindex]), real_ret1, imag_ret1, tmp0, tmp1);
        mindex += (1ll << (num_qubits + 1));

        real_ret1 = _mm256_permutevar8x32_ps(real_ret1, mask);
        imag_ret1 = _mm256_permutevar8x32_ps(imag_ret1, mask);

        real_ret =
            (qubits[0] == 0) ? _mm256_blend_ps(real_ret, real_ret1, 0b10101010)
            :  // (0,H,H)
            (qubits[0] == 1) ? _mm256_blend_ps(real_ret, real_ret1, 0b11001100)
            :                                      // (1,H,H)
            _mm256_blend_ps(real_ret, real_ret1, 0b11110000);  //  (2,H,H)
        imag_ret =
            (qubits[0] == 0) ? _mm256_blend_ps(imag_ret, imag_ret1, 0b10101010)
            :  // (0,H,H)
            (qubits[0] == 1) ? _mm256_blend_ps(imag_ret, imag_ret1, 0b11001100)
            :                                      // (1,H,H)
            _mm256_blend_ps(imag_ret, imag_ret1, 0b11110000);  //  (2,H,H)

        _mm_store_twoarray_complex_float(real_ret, imag_ret, reals[index], imags[index]);
    }
}

template <size_t num_qubits>
void _qoracle_internal_calc4_float(const float* matrix_d, Qnum& qubits, int64_t index, RealStat<float>& reals, ImagStat<float>& imags)
{
    constexpr auto indexes_size = 1ull << num_qubits;
    int64_t indexes[indexes_size];
    load_index(index, qubits.size(), indexes, indexes_size, qubits);

    __m256 real_ret, imag_ret, real_ret1, imag_ret1;
    __m256 vreals[1ull << num_qubits], vimags[1ull << num_qubits];
    __m256 tmp0, tmp1;

    for (size_t i = 0; i < (1ull << num_qubits); ++i) {
        auto index = indexes[i];
        _mm_load_twoarray_complex_float(reals[index], imags[index], vreals[i], vimags[i]);
    }

    size_t mindex = 0;
    for (size_t i = 0; i < (1ull << num_qubits); ++i) {
        auto index = indexes[i];
        _mm_complex_internal_calc<float>((1ll << num_qubits), vreals, vimags,
            (&matrix_d[mindex]), real_ret, imag_ret, tmp0, tmp1);
        mindex += (1ll << (num_qubits + 1));
        _mm_store_twoarray_complex_float(real_ret, imag_ret, reals[index], imags[index]);
    }
}

int64_t _insert(Qnum& sorted_qubits, int num_qubits, const int64_t k)
{
    int64_t lowbits, retval = k;
    for (size_t j = 0; j < num_qubits; j++) {
        lowbits = retval & ((1 << sorted_qubits[j]) - 1);
        retval >>= sorted_qubits[j];
        retval <<= sorted_qubits[j] + 1;
        retval |= lowbits;
    }
    return retval;
}

inline void load_index(int64_t index0, int num_qubits, int64_t* indexes,
    const size_t indexes_size, const Qnum& qregs)
{
    for (size_t i = 0; i < indexes_size; ++i) {
        indexes[i] = index0;
    }

    for (size_t n = 0; n < num_qubits; ++n) {
        for (size_t i = 0; i < indexes_size; i += (1ull << (n + 1))) {
            for (size_t j = 0; j < (1ull << n); ++j) {
                indexes[i + j + (1ull << n)] += (1ull << qregs[n]);

            }
        }
    }
}

int _omp_thread_num(size_t size)
{
    if (size > (1ll << 9))
    {
#ifdef USE_OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }
    else
    {
        return 1;
    }
}

template <typename Container, typename data_t >
QError CPUAvx2<Container, data_t>::_three_qubit_gate_simd(Container& m_state, Qnum& qubits, QStat& matrix, bool is_dagger, size_t m_qubit_num, const Qnum& controls)
{
    if (typeid(data_t) == typeid(double))
    {
        int64_t size = 1ll << (m_qubit_num - 3);
        std::sort(qubits.begin(), qubits.end());
        int dim = 1ll << qubits.size();
        RealStat<double> reals = { reinterpret_cast<double*>(&m_state[0]) };
        ImagStat<double> imags = { reinterpret_cast<double*>(&m_state[0]) };
        constexpr int matrix_size = (128);
        double mat1[matrix_size];

        for (int i = 0; i < dim; ++i)
        {
            for (int j = 0; j < dim; ++j)
            {
                mat1[i * dim * 2 + j * 2] = matrix[i * dim + j].real();
                mat1[i * dim * 2 + j * 2 + 1] = matrix[i * dim + j].imag();
            }
        }
        int64_t mask = 0;
        if (controls.size() > 3)
        {
            for_each(controls.begin(), controls.end() - 3, [&](size_t q) {
                mask |= 1ll << q;
            });
        }
        if (qubits.size() > 1 && qubits[1] == 1)
        {
#pragma omp parallel for num_threads( _omp_thread_num(size) )
            for (int64_t i = 0; i < size; i += 1)
            {
                const int64_t index = _insert(qubits, qubits.size(), i);
                if (mask != (mask & index))
                    continue;
                _qoracle_internal_calc1_double<3>(mat1, qubits, index, reals, imags);
            }
        }
        else if (qubits[0] < 2)
        {
#pragma omp parallel for num_threads( _omp_thread_num(size) )
            for (int64_t i = 0; i < size; i += 2)
            {
                const int64_t index = _insert(qubits, qubits.size(), i);
                if (mask != (mask & index))
                    continue;
                _qoracle_internal_calc2_double<3>(mat1, qubits, index, reals, imags);
            }
        }
        else
        {
#pragma omp parallel for num_threads( _omp_thread_num(size) )
            for (int64_t i = 0; i < size; i += 4)
            {
                const int64_t index = _insert(qubits, qubits.size(), i);
                if (mask != (mask & index))
                    continue;
                _qoracle_internal_calc3_double<3>(mat1, qubits, index, reals, imags);
            }
        }
    }
    else if (typeid(data_t) == typeid(float))
    {
        int64_t size = 1ll << (m_qubit_num - 3);
        std::sort(qubits.begin(), qubits.end());
        int dim = 1ll << qubits.size();
        RealStat<float> reals = { reinterpret_cast<float*>(&m_state[0]) };
        ImagStat<float> imags = { reinterpret_cast<float*>(&m_state[0]) };
        constexpr int matrix_size = (128);
        float mat1[matrix_size];

        for (int i = 0; i < dim; ++i)
        {
            for (int j = 0; j < dim; ++j)
            {
                mat1[i * dim * 2 + j * 2] = matrix[i * dim + j].real();
                mat1[i * dim * 2 + j * 2 + 1] = matrix[i * dim + j].imag();
            }
        }
        int64_t mask = 0;
        if (controls.size() > 3)
        {
            for_each(controls.begin(), controls.end() - 3, [&](size_t q) {
                mask |= 1ll << q;
            });
        }
        if (qubits.size() > 2 && qubits[2] == 2)
        {
#pragma omp parallel for num_threads( _omp_thread_num(size) )
            for (int64_t i = 0; i < size; i += 1)
            {
                const int64_t index = _insert(qubits, qubits.size(), i);
                if (mask != (mask & index))
                    continue;
                _qoracle_internal_calc1_float<3>(mat1, qubits, index, reals, imags);
            }

        }
        else if (qubits.size() > 1 && qubits[1] < 3)
        {
#pragma omp parallel for num_threads( _omp_thread_num(size) )
            for (int64_t i = 0; i < size; i += 2)
            {
                const int64_t index = _insert(qubits, qubits.size(), i);
                if (mask != (mask & index))
                    continue;
                _qoracle_internal_calc2_float<3>(mat1, qubits, index, reals, imags);
            }

        }
        else if (qubits[0] < 3)
        {
#pragma omp parallel for num_threads( _omp_thread_num(size) )
            for (int64_t i = 0; i < size; i += 4)
            {
                const int64_t index = _insert(qubits, qubits.size(), i);
                if (mask != (mask & index))
                    continue;
                _qoracle_internal_calc3_float<3>(mat1, qubits, index, reals, imags);
            }
        }
        else
        {
#pragma omp parallel for num_threads( _omp_thread_num(size) )
            for (int64_t i = 0; i < size; i += 8)
            {
                const int64_t index = _insert(qubits, qubits.size(), i);
                if (mask != (mask & index))
                    continue;
                _qoracle_internal_calc4_float<3>(mat1, qubits, index, reals, imags);
            }
        }
    }
    return qErrorNone;
}
template <typename Container, typename data_t>
QError CPUAvx2<Container, data_t>::_four_qubit_gate_simd(Container& m_state, Qnum& qubits, QStat& matrix, bool is_dagger, size_t m_qubit_num, const Qnum& controls)
{

    if (typeid(data_t) == typeid(double))
    {
        int64_t size = 1ll << (m_qubit_num - 4);
        std::sort(qubits.begin(), qubits.end());
        int dim = 1ll << qubits.size();
        RealStat<double> reals = { reinterpret_cast<double*>(&m_state[0]) };
        ImagStat<double> imags = { reinterpret_cast<double*>(&m_state[0]) };
        constexpr int matrix_size = (512);
        double mat1[matrix_size];

        for (int i = 0; i < dim; ++i)
        {
            for (int j = 0; j < dim; ++j)
            {
                mat1[i * dim * 2 + j * 2] = matrix[i * dim + j].real();
                mat1[i * dim * 2 + j * 2 + 1] = matrix[i * dim + j].imag();
            }
        }
        int64_t mask = 0;
        if (controls.size() > 4)
        {
            for_each(controls.begin(), controls.end() - 4, [&](size_t q) {
                mask |= 1ll << q;
            });
        }
        if (qubits.size() > 1 && qubits[1] == 1)
        {
#pragma omp parallel for num_threads( _omp_thread_num(size) )
            for (int64_t i = 0; i < size; i += 1)
            {
                const int64_t index = _insert(qubits, qubits.size(), i);
                if (mask != (mask & index))
                    continue;
                _qoracle_internal_calc1_double<4>(mat1, qubits, index, reals, imags);
            }
        }
        else if (qubits[0] < 2)
        {
#pragma omp parallel for num_threads( _omp_thread_num(size) )
            for (int64_t i = 0; i < size; i += 2)
            {
                const int64_t index = _insert(qubits, qubits.size(), i);
                if (mask != (mask & index))
                    continue;
                _qoracle_internal_calc2_double<4>(mat1, qubits, index, reals, imags);
            }
        }
        else
        {
#pragma omp parallel for num_threads( _omp_thread_num(size) )
            for (int64_t i = 0; i < size; i += 4)
            {
                const int64_t index = _insert(qubits, qubits.size(), i);
                if (mask != (mask & index))
                    continue;
                _qoracle_internal_calc3_double<4>(mat1, qubits, index, reals, imags);
            }
        }
    }
    else if (typeid(data_t) == typeid(float))
    {
        int64_t size = 1ll << (m_qubit_num - 4);
        std::sort(qubits.begin(), qubits.end());
        int dim = 1ll << qubits.size();
        RealStat<float> reals = { reinterpret_cast<float*>(&m_state[0]) };
        ImagStat<float> imags = { reinterpret_cast<float*>(&m_state[0]) };
        constexpr int matrix_size = (512);
        float mat1[matrix_size];

        for (int i = 0; i < dim; ++i)
        {
            for (int j = 0; j < dim; ++j)
            {
                mat1[i * dim * 2 + j * 2] = matrix[i * dim + j].real();
                mat1[i * dim * 2 + j * 2 + 1] = matrix[i * dim + j].imag();
            }
        }
        int64_t mask = 0;
        if (controls.size() > 4)
        {
            for_each(controls.begin(), controls.end() - 4, [&](size_t q) {
                mask |= 1ll << q;
            });
        }
        if (qubits.size() > 2 && qubits[2] == 2)
        {
#pragma omp parallel for num_threads( _omp_thread_num(size) )
            for (int64_t i = 0; i < size; i += 1)
            {
                const int64_t index = _insert(qubits, qubits.size(), i);
                if (mask != (mask & index))
                    continue;
                _qoracle_internal_calc1_float<4>(mat1, qubits, index, reals, imags);
            }

        }
        else if (qubits.size() > 1 && qubits[1] < 3)
        {
#pragma omp parallel for num_threads( _omp_thread_num(size) )
            for (int64_t i = 0; i < size; i += 2)
            {
                const int64_t index = _insert(qubits, qubits.size(), i);
                if (mask != (mask & index))
                    continue;
                _qoracle_internal_calc2_float<4>(mat1, qubits, index, reals, imags);
            }

        }
        else if (qubits[0] < 3)
        {
#pragma omp parallel for num_threads( _omp_thread_num(size) )
            for (int64_t i = 0; i < size; i += 4)
            {
                const int64_t index = _insert(qubits, qubits.size(), i);
                if (mask != (mask & index))
                    continue;
                _qoracle_internal_calc3_float<4>(mat1, qubits, index, reals, imags);
            }
        }
        else
        {
#pragma omp parallel for num_threads( _omp_thread_num(size) )
            for (int64_t i = 0; i < size; i += 8)
            {
                const int64_t index = _insert(qubits, qubits.size(), i);
                if (mask != (mask & index))
                    continue;
                _qoracle_internal_calc4_float<4>(mat1, qubits, index, reals, imags);
            }
        }
    }

    return qErrorNone;
}

template <typename Container, typename data_t >
QError CPUAvx2<Container, data_t>::_five_qubit_gate_simd(Container& m_state, Qnum& qubits, QStat& matrix, bool is_dagger, size_t m_qubit_num, const Qnum& controls)
{
    if (typeid(data_t) == typeid(double))
    {
        int64_t size = 1ll << (m_qubit_num - 5);
        std::sort(qubits.begin(), qubits.end());
        int dim = 1ll << qubits.size();
        RealStat<double> reals = { reinterpret_cast<double*>(&m_state[0]) };
        ImagStat<double> imags = { reinterpret_cast<double*>(&m_state[0]) };
        constexpr int matrix_size = (2048);
        double mat1[matrix_size];

        for (int i = 0; i < dim; ++i)
        {
            for (int j = 0; j < dim; ++j)
            {
                mat1[i * dim * 2 + j * 2] = matrix[i * dim + j].real();
                mat1[i * dim * 2 + j * 2 + 1] = matrix[i * dim + j].imag();
            }
        }
        int64_t mask = 0;
        if (controls.size() > 5)
        {
            for_each(controls.begin(), controls.end() - 5, [&](size_t q) {
                mask |= 1ll << q;
            });
        }
        if (qubits.size() > 1 && qubits[1] == 1)
        {
#pragma omp parallel for num_threads( _omp_thread_num(size) )
            for (int64_t i = 0; i < size; i += 1)
            {
                const int64_t index = _insert(qubits, qubits.size(), i);
                if (mask != (mask & index))
                    continue;
                _qoracle_internal_calc1_double<5>(mat1, qubits, index, reals, imags);
            }
        }
        else if (qubits[0] < 2)
        {
#pragma omp parallel for num_threads( _omp_thread_num(size) )
            for (int64_t i = 0; i < size; i += 2)
            {
                const int64_t index = _insert(qubits, qubits.size(), i);
                if (mask != (mask & index))
                    continue;
                _qoracle_internal_calc2_double<5>(mat1, qubits, index, reals, imags);
            }
        }
        else
        {
#pragma omp parallel for num_threads( _omp_thread_num(size) )
            for (int64_t i = 0; i < size; i += 4)
            {
                const int64_t index = _insert(qubits, qubits.size(), i);
                if (mask != (mask & index))
                    continue;
                _qoracle_internal_calc3_double<5>(mat1, qubits, index, reals, imags);
            }
        }
    }
    else if (typeid(data_t) == typeid(float))
    {
        int64_t size = 1ll << (m_qubit_num - 5);
        std::sort(qubits.begin(), qubits.end());
        int dim = 1ll << qubits.size();
        RealStat<float> reals = { reinterpret_cast<float*>(&m_state[0]) };
        ImagStat<float> imags = { reinterpret_cast<float*>(&m_state[0]) };
        constexpr int matrix_size = (2048);
        float mat1[matrix_size];

        for (int i = 0; i < dim; ++i)
        {
            for (int j = 0; j < dim; ++j)
            {
                mat1[i * dim * 2 + j * 2] = matrix[i * dim + j].real();
                mat1[i * dim * 2 + j * 2 + 1] = matrix[i * dim + j].imag();
            }
        }
        int64_t mask = 0;
        if (controls.size() > 5)
        {
            for_each(controls.begin(), controls.end() - 5, [&](size_t q) {
                mask |= 1ll << q;
            });
        }
        if (qubits.size() > 2 && qubits[2] == 2)
        {
#pragma omp parallel for num_threads( _omp_thread_num(size) )
            for (int64_t i = 0; i < size; i += 1)
            {
                const int64_t index = _insert(qubits, qubits.size(), i);
                if (mask != (mask & index))
                    continue;
                _qoracle_internal_calc1_float<5>(mat1, qubits, index, reals, imags);
            }

        }
        else if (qubits.size() > 1 && qubits[1] < 3)
        {
#pragma omp parallel for num_threads( _omp_thread_num(size) )
            for (int64_t i = 0; i < size; i += 2)
            {
                const int64_t index = _insert(qubits, qubits.size(), i);
                if (mask != (mask & index))
                    continue;
                _qoracle_internal_calc2_float<5>(mat1, qubits, index, reals, imags);
            }

        }
        else if (qubits[0] < 3)
        {
#pragma omp parallel for num_threads( _omp_thread_num(size) )
            for (int64_t i = 0; i < size; i += 4)
            {
                const int64_t index = _insert(qubits, qubits.size(), i);
                if (mask != (mask & index))
                    continue;
                _qoracle_internal_calc3_float<5>(mat1, qubits, index, reals, imags);
            }
        }
        else
        {
#pragma omp parallel for num_threads( _omp_thread_num(size) )
            for (int64_t i = 0; i < size; i += 8)
            {
                const int64_t index = _insert(qubits, qubits.size(), i);
                if (mask != (mask & index))
                    continue;
                _qoracle_internal_calc4_float<5>(mat1, qubits, index, reals, imags);
            }
        }
    }

    return qErrorNone;
}

#else

template <typename Container, typename data_t >
QError CPUAvx2<Container, data_t>::_three_qubit_gate_simd(Container& m_state, Qnum& qubits, QStat& matrix, bool is_dagger, size_t m_qubit_num, const Qnum& controls)
{
    QCERR_AND_THROW(run_fail, "can not use avx2");
    return undefineError;
}

template <typename Container, typename data_t >
QError CPUAvx2<Container, data_t>::_four_qubit_gate_simd(Container& m_state, Qnum& qubits, QStat& matrix, bool is_dagger, size_t m_qubit_num, const Qnum& controls)
{
    QCERR_AND_THROW(run_fail, "can not use avx2");
    return undefineError;
}

template <typename Container, typename data_t>
QError CPUAvx2<Container, data_t>::_five_qubit_gate_simd(Container& m_state, Qnum& qubits, QStat& matrix, bool is_dagger, size_t m_qubit_num, const Qnum& controls)
{
    QCERR_AND_THROW(run_fail, "can not use avx2");
    return undefineError;
}

#endif

template class QPanda::CPUAvx2<std::vector<std::complex<double>>, double>;
template class QPanda::CPUAvx2<std::vector<std::complex<float>>, float>;

