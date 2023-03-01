#pragma once

#include <array>
#include <memory>
#include <vector>
#include <complex>
#include "QPandaConfig.h"
#include "ThirdParty/Eigen/Eigen"
#include "ThirdParty/Eigen/Dense"
#include "ThirdParty/Eigen/Sparse"
#include "Core/Utilities/QPandaNamespace.h"
#ifdef USE_OPENMP
#include <omp.h>
#endif

QPANDA_BEGIN

template <typename T> 
using cvector_t = std::vector<std::complex<T>>;

using cmatrix_t = Eigen::MatrixXcd;

// The vector is formed using column-stacking vectorization
template <typename data_t>
class VectorMatrix
{

public:
    
    cvector_t<data_t> convert_data(const cvector_t<double> &vector) const;

    template<typename Lambda, typename List>
    void apply_lambda(Lambda&& func, const List& qubits);

    template<typename Lambda, typename List, typename Param>
    void apply_lambda(Lambda&& func, const List& qubits, const Param& params);

    virtual void apply_matrix(const Qnum &qubits, const cvector_t<double> &matrix);
    virtual void apply_diagonal_matrix(const Qnum &qubits, const cvector_t<double> &diag_matrix);

protected:

    void apply_matrix_1(const size_t qubit, const cvector_t<double> &matrix);

    void apply_diagonal_matrix_1(const size_t qubit, const cvector_t<double> &matrix);

    void apply_permutation_matrix(const Qnum& qubits, const std::vector<std::pair<size_t, size_t>> &pairs);

    template <size_t qubits_num>
    void apply_matrix_n(const Qnum &qubits, const cvector_t<double> &matrix);


protected:

    cvector_t<data_t> m_data;
    size_t m_data_size;
};

template <typename T>
std::vector<T> vector_kron(const std::vector<T>& lhs, const std::vector<T>& rhs)
{
    std::vector<T> result;

    const auto lhs_size = lhs.size();
    const auto rhs_size = rhs.size();

    result.resize(lhs_size * rhs_size);

    for (size_t i = 0; i < lhs_size; ++i)
        for (size_t j = 0; j < rhs_size; ++j)
        {
            result[rhs_size * i + j] = lhs[i] * rhs[j];
        }

    return result;
}

template <typename T>
std::vector<T> matrix_kron(const std::vector<T>& lhs, const std::vector<T>& rhs)
{
    std::vector<T> result;

    auto lhs_rows = (size_t)std::sqrt(lhs.size());
    auto rhs_cols = (size_t)std::sqrt(rhs.size());

    result.resize(lhs.size() * rhs.size());

    size_t lhs_row = 0, lhs_col = 0;
    size_t rhs_row = 0, rhs_col = 0;

    size_t result_row = 0, result_col = 0;

    for (size_t lhs_idx = 0; lhs_idx < lhs.size(); ++lhs_idx)
    {
        for (size_t rhs_idx = 0; rhs_idx < rhs.size(); ++rhs_idx)
        {
            lhs_row = lhs_idx / lhs_rows;
            lhs_col = lhs_idx % lhs_rows;

            rhs_row = rhs_idx / rhs_cols;
            rhs_col = rhs_idx % rhs_cols;

            result_row = rhs_row + (lhs_row * rhs_cols);
            result_col = rhs_col + (lhs_col * rhs_cols);

            result[(result_row) * (lhs_rows * rhs_cols) + result_col] = (lhs[lhs_idx] * rhs[rhs_idx]);
        }
    }

    return result;
}

template <typename T>
std::vector<std::complex<T>> vector_conj(const std::vector<std::complex<T>> &v)
{
    std::vector<std::complex<T>> result;
    std::transform(v.cbegin(), v.cend(), std::back_inserter(result),
        [](const std::complex<T> &val) -> std::complex<T> { return std::conj(val); });
    return result;
}

template<typename T>
std::vector<T> column_stacking(const std::vector<T>& matrix)
{
    auto dim = std::sqrt(matrix.size());

    if (dim * dim != matrix.size())
        QCERR_AND_THROW(std::runtime_error, "column stacking dims incorrent");

    std::vector<T> result;
    result.resize(matrix.size(), 0.);

    for (size_t i = 0; i < dim; i++)
    {
        for (size_t j = 0; j < dim; j++)
        {
            result[j * dim + i] = matrix[i * dim + j];
        }
    }

    return result;
}

template <class T> 
std::vector<T> unitary_superop(const std::vector<T>& matrix)
{
    return matrix_kron(vector_conj(matrix), matrix);
}

template <class T>
std::vector<T> kraus_superop(const std::vector<std::vector<T>>& matrix_list)
{
    std::vector<T> superop = unitary_superop(matrix_list[0]);
    for (auto i = 1; i < matrix_list.size(); ++i)
    {
        auto matrix = unitary_superop(matrix_list[i]);

        for (auto j = 0; j < superop.size(); ++j)
        {
            superop[j] += matrix[j];
        }
    }


    return superop;
}

// Construct a vectorized superoperator from a vectorized matrix
// This is equivalent to vector(tensor(conj(A), A))
template <typename T>
std::vector<T> to_superop(const std::vector<T>& matrix)
{
    size_t dim = (size_t)std::sqrt(matrix.size());

    std::vector<T> result(dim * dim * dim * dim, 0.);
    for (size_t i = 0; i < dim; i++)
        for (size_t j = 0; j < dim; j++)
            for (size_t k = 0; k < dim; k++)
                for (size_t l = 0; l < dim; l++)
                    result[dim * i + k + (dim * dim) * (dim * j + l)]
                    = std::conj(matrix[i + dim * j]) * matrix[k + dim * l];
    return result;
}

template <typename List>
size_t single_indice(List& sorted_qubits, const size_t k)
{
    const auto num_qubits = sorted_qubits.size();

    size_t lowbits, retval = k;
    for (size_t j = 0; j < num_qubits; j++) 
    {
        lowbits = retval & ((1 << sorted_qubits[j]) - 1);
        retval >>= sorted_qubits[j];
        retval <<= sorted_qubits[j] + 1;
        retval |= lowbits;
    }
    return retval;
}

std::unique_ptr<size_t[]> multi_array_indices(const Qnum& qubits, const Qnum& qubits_sorted, const size_t k);

template <size_t qubits_num>
std::array<size_t, 1 << qubits_num> multi_array_indices(const std::array<size_t, qubits_num> &qubits,
                                                        const std::array<size_t, qubits_num> &qubits_sorted,
                                                        const size_t k) 
{
    std::array<size_t, 1 << qubits_num> result;
    result[0] = single_indice(qubits_sorted, k);

    for (size_t i = 0; i < qubits_num; i++)
    {
        const auto n = (1ull << i);
        const auto bit = (1ull << qubits[i]);

        for (size_t j = 0; j < n; j++)
            result[n + j] = result[j] | bit;
    }
    return result;
}

template<typename Lambda, typename List>
void apply_data_lambda(const size_t start, const size_t stop, Lambda&& func, const List& qubits)
{
    const size_t step = stop >> qubits.size();

    auto qubits_sorted = qubits;
    std::sort(qubits_sorted.begin(), qubits_sorted.end());

#pragma omp parallel for num_threads(omp_get_max_threads())
    for (int64_t k = start; k < step; k++)
    {
        //store entries touched by U
        const auto indices = multi_array_indices(qubits, qubits_sorted, k);
        std::forward<Lambda>(func)(indices);
    }
}

template<typename Lambda, typename List, typename Param>
void apply_data_lambda(const size_t start, const size_t stop, Lambda&& func, const List& qubits, const Param& params)
{
    const size_t step = stop >> qubits.size();

    auto qubits_sorted = qubits;
    std::sort(qubits_sorted.begin(), qubits_sorted.end());

#pragma omp parallel for num_threads(omp_get_max_threads())
    for (int64_t k = start; k < step; k++)
    {
        const auto indices = multi_array_indices(qubits, qubits_sorted, k);
        std::forward<Lambda>(func)(indices, params);
    }
}

template<typename Lambda>
std::complex<double> apply_reduction_lambda(const size_t start, const size_t stop, Lambda&& func) 
{
    // Reduction variables
    double val_real = 0.;
    double val_imag = 0.;

#pragma omp parallel for num_threads(omp_get_max_threads())
    for (int64_t k = start; k < stop; k++)
    {
        std::forward<Lambda>(func)(k, val_real, val_imag);
    }

    return std::complex<double>(val_real, val_imag);
}


template <typename data_t>
template<typename Lambda, typename List>
void VectorMatrix<data_t>::apply_lambda(Lambda&& func, const List& qubits)
{
    return apply_data_lambda(0, m_data_size, func, qubits);
}

template <typename data_t>
template<typename Lambda, typename List, typename Param>
void VectorMatrix<data_t>::apply_lambda(Lambda&& func, const List& qubits, const Param& params)
{
    return apply_data_lambda(0, m_data_size, func, qubits, params);
}

template <typename data_t>
void VectorMatrix<data_t>::apply_permutation_matrix(const Qnum& qv, const std::vector<std::pair<size_t, size_t>> &pairs)
{
    auto qubits_num = qv.size();

    switch (qubits_num)
    {
    case 1:
    {
        auto qubits_array = std::array<size_t, 1>({ qv[0] });
        auto lambda = [&](const std::array<size_t, 1ull << 1>& indices)->void
        {
            for (const auto &p : pairs)
                std::swap(m_data[indices[p.first]], m_data[indices[p.second]]);
        };

        apply_lambda(lambda, qubits_array);
        return;
    }

    case 2:
    {
        auto qubits_array = std::array<size_t, 2>({ qv[0], qv[1] });
        auto lambda = [&](const std::array<size_t, 1ull << 2>& indices)->void
        {
            for (const auto &p : pairs)
                std::swap(m_data[indices[p.first]], m_data[indices[p.second]]);
        };

        apply_lambda(lambda, qubits_array);
        return;
    }

    case 3:
    {
        auto qubits_array = std::array<size_t, 3>({ qv[0], qv[1], qv[2] });
        auto lambda = [&](const std::array<size_t, 1ull << 3>& indices)->void
        {
            for (const auto &p : pairs)
                std::swap(m_data[indices[p.first]], m_data[indices[p.second]]);
        };

        apply_lambda(lambda, qubits_array);
        return;
    }

    case 4:
    {
        auto qubits_array = std::array<size_t, 4>({ qv[0], qv[1], qv[2], qv[3] });
        auto lambda = [&](const std::array<size_t, 1ull << 4>& indices)->void
        {
            for (const auto &p : pairs)
                std::swap(m_data[indices[p.first]], m_data[indices[p.second]]);
        };

        apply_lambda(lambda, qubits_array);
        return;
    }

    case 5:
    {
        auto qubits_array = std::array<size_t, 5>({ qv[0], qv[1], qv[2], qv[3], qv[4] });
        auto lambda = [&](const std::array<size_t, 1ull << 5>& indices)->void
        {
            for (const auto &p : pairs)
                std::swap(m_data[indices[p.first]], m_data[indices[p.second]]);
        };

        apply_lambda(lambda, qubits_array);
        return;
    }

    case 6:
    {
        auto qubits_array = std::array<size_t, 6>({ qv[0], qv[1], qv[2], qv[3], qv[4], qv[5] });
        auto lambda = [&](const std::array<size_t, 1ull << 6>& indices)->void
        {
            for (const auto &p : pairs)
                std::swap(m_data[indices[p.first]], m_data[indices[p.second]]);
        };

        apply_lambda(lambda, qubits_array);
        return;
    }

    default:QCERR_AND_THROW(std::runtime_error, "maximum qv num of apply permutation matrix is 6");
    }

    return;
}

template <typename data_t>
template <size_t qubits_num>
void VectorMatrix<data_t>::apply_matrix_n(const Qnum &qubits, const cvector_t<double>& matrix)
{
    const size_t dim = 1ull << qubits_num;

    auto func = [&](const std::array<size_t, 1ull << qubits_num>& indices,
        const cvector_t<data_t> &lambda_matrix) -> void
    {
        std::array<std::complex<data_t>, 1ull << qubits_num> cache;
        for (size_t i = 0; i < dim; i++)
        {
            const auto current_index = indices[i];
            cache[i] = m_data[current_index];
            m_data[current_index] = 0.;
        }

        // update state vector
        for (size_t i = 0; i < dim; i++)
            for (size_t j = 0; j < dim; j++)
                m_data[indices[i]] += lambda_matrix[i + dim * j] * cache[j];
    };

    std::array<size_t, qubits_num> lambda_qubits;
    std::copy_n(qubits.begin(), qubits_num, lambda_qubits.begin());

    apply_lambda(func, lambda_qubits, convert_data(matrix));
    return;
}

QPANDA_END
