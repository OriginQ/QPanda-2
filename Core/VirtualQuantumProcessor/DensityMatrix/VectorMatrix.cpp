#include <array>
#include "Core/VirtualQuantumProcessor/DensityMatrix/VectorMatrix.h"

USING_QPANDA

std::unique_ptr<size_t[]>  QPanda::multi_array_indices(const Qnum& qubits, const Qnum& qubits_sorted, const size_t k)
{
    const auto qubits_num = qubits_sorted.size();

    std::unique_ptr<size_t[]> result_ptr(new size_t[1ull << qubits_num]);

    result_ptr[0] = single_indice(qubits_sorted, k);
    for (size_t i = 0; i < qubits_num; i++)
    {
        const auto n = (1ull << i);
        const auto bit = (1ull << qubits[i]);
        for (size_t j = 0; j < n; j++)
            result_ptr[n + j] = result_ptr[j] | bit;
    }

    return result_ptr;
}


template <typename data_t>
cvector_t<data_t> VectorMatrix<data_t>::convert_data(const cvector_t<double>& vector) const
{
    cvector_t<data_t> result(vector.size());

    for (size_t i = 0; i < vector.size(); ++i)
        result[i] = vector[i];

    return result;
}

template <typename data_t>
void VectorMatrix<data_t>::apply_matrix(const Qnum &qubits, const cvector_t<double>& matrix)
{
    auto qubits_num = qubits.size();

    switch (qubits_num)
    {
    case 1: return apply_matrix_1(qubits[0], matrix);
    case 2: return apply_matrix_n<2>(qubits, matrix);
    case 3: return apply_matrix_n<3>(qubits, matrix);
    case 4: return apply_matrix_n<4>(qubits, matrix);
    case 5: return apply_matrix_n<5>(qubits, matrix);
    case 6: return apply_matrix_n<6>(qubits, matrix);
    case 7: return apply_matrix_n<7>(qubits, matrix);
    case 8: return apply_matrix_n<8>(qubits, matrix);
    case 9: return apply_matrix_n<9>(qubits, matrix);
    case 10: return apply_matrix_n<10>(qubits, matrix);
    default:
        QCERR_AND_THROW(std::runtime_error, "maximum qubits num of apply matrix is 10");
    }
}

template <typename data_t>
void VectorMatrix<data_t>::apply_matrix_1(const size_t qubit, const cvector_t<double>& matrix)
{
    if (matrix[1] == 0.0 && matrix[2] == 0.0)
    {
        const cvector_t<double> diag_matrix = { {matrix[0], matrix[3]} };
        apply_diagonal_matrix_1(qubit, diag_matrix);
        return;
    }

    // Convert qubit to array register for lambda functions
    std::array<size_t, 1> qubits = { {qubit} };

    // Check if anti-diagonal matrix and if so use optimized lambda
    if (matrix[0] == 0.0 && matrix[3] == 0.0)
    {
        if (matrix[1] == 1.0 && matrix[2] == 1.0)
        {
            // X-matrix
            auto func = [&](const std::array<size_t, 2>& indices) -> void
            {
                std::swap(m_data[indices[0]], m_data[indices[1]]);
            };

            apply_lambda(func, qubits);
            return;
        }
        if (matrix[2] == 0.0)
        {
            // Non-unitary projector
            // possibly used in measure/reset/kraus update
            auto func = [&](const std::array<size_t, 2>& indices,
                const cvector_t<data_t> &lambda_matrix) -> void
            {
                m_data[indices[1]] = lambda_matrix[1] * m_data[indices[0]];
                m_data[indices[0]] = 0.0;
            };

            apply_lambda(func, qubits, convert_data(matrix));
            return;
        }
        if (matrix[1] == 0.0)
        {
            // Non-unitary projector
            // possibly used in measure/reset/kraus update
            auto func = [&](const std::array<size_t, 2>& indices,
                const cvector_t<data_t> &lambda_matrix) -> void
            {
                m_data[indices[0]] = lambda_matrix[2] * m_data[indices[1]];
                m_data[indices[1]] = 0.0;
            };

            apply_lambda(func, qubits, convert_data(matrix));
            return;
        }

        // handle general anti-diagonal matrix
        auto func = [&](const std::array<size_t, 2>& indices,
            const cvector_t<data_t> &lambda_matrix) -> void
        {
            const std::complex<data_t> cache = m_data[indices[0]];
            m_data[indices[0]] = lambda_matrix[2] * m_data[indices[1]];
            m_data[indices[1]] = lambda_matrix[1] * cache;
        };

        apply_lambda(func, qubits, convert_data(matrix));
        return;
    }

    auto func = [&](const std::array<size_t, 2>& indices,
        const cvector_t<data_t> &lambda_matrix) -> void
    {
        const auto cache = m_data[indices[0]];
        m_data[indices[0]] = lambda_matrix[0] * cache + lambda_matrix[2] * m_data[indices[1]];
        m_data[indices[1]] = lambda_matrix[1] * cache + lambda_matrix[3] * m_data[indices[1]];
    };

    apply_lambda(func, qubits, convert_data(matrix));
    return;
}

template <typename data_t>
void VectorMatrix<data_t>::apply_diagonal_matrix(const Qnum &qubits, const cvector_t<double>& diag_matrix)
{
    if (qubits.size() == 1)
        return apply_diagonal_matrix_1(qubits[0], diag_matrix);

    const size_t qubits_num = qubits.size();

    auto func = [&](const std::array<size_t, 2>& indices,
        const cvector_t<data_t> &diag) -> void
    {
        for (size_t i = 0; i < 2; ++i)
        {
            const size_t k = indices[i];

            size_t idx = 0;
            for (size_t j = 0; j < qubits_num; j++)
                if ((k & (1ull << qubits[j])) != 0)
                    idx += (1ull << j);

            if (diag[idx] != (data_t)1.0)
                m_data[k] *= diag[idx];
        }
    };

    const cvector_t<double> diag1 = { 1.0,2.0 };
    auto data11 = convert_data(diag1);
    apply_lambda(func, std::array<size_t, 1>({ {qubits[0]} }), convert_data(diag_matrix));
}

template <typename data_t>
void VectorMatrix<data_t>::apply_diagonal_matrix_1(const size_t qubit, const cvector_t<double>& diag_matrix)
{
    if (diag_matrix[0] == 1.0)
    {
        // [[1, 0], [0, z]] matrix
        if (diag_matrix[1] == 1.0)
            return; // Identity

        if (diag_matrix[1] == std::complex<double>(0., -1.))
        {
            // [[1, 0], [0, -i]]
            auto func = [&](const std::array<size_t, 2>& indices,
                const cvector_t<data_t> &lambda_matrix) -> void
            {
                const auto k = indices[1];
                double cache = m_data[k].imag();
                m_data[k].imag(m_data[k].real() * -1.);
                m_data[k].real(cache);
            };

            apply_lambda(func, std::array<size_t, 1>({ {qubit} }), convert_data(diag_matrix));
            return;
        }

        if (diag_matrix[1] == std::complex<double>(0., 1.))
        {
            // [[1, 0], [0, i]]
            auto func = [&](const std::array<size_t, 2>& indices,
                const cvector_t<data_t> &lambda_matrix) -> void
            {
                const auto k = indices[1];
                double cache = m_data[k].imag();
                m_data[k].imag(m_data[k].real());
                m_data[k].real(cache * -1.);
            };

            apply_lambda(func, std::array<size_t, 1>({ {qubit} }), convert_data(diag_matrix));
            return;
        }
        if (diag_matrix[0] == 0.0)
        {
            // [[1, 0], [0, 0]]
            auto func = [&](const std::array<size_t, 2>& indices,
                const cvector_t<data_t> &lambda_matrix) -> void
            {
                m_data[indices[1]] = 0.0;
            };

            apply_lambda(func, std::array<size_t, 1>({ {qubit} }), convert_data(diag_matrix));
            return;
        }
        // general [[1, 0], [0, z]]
        auto func = [&](const std::array<size_t, 2>& indices,
            const cvector_t<data_t> &lambda_matrix) -> void
        {
            const auto k = indices[1];
            m_data[k] *= lambda_matrix[1];
        };

        apply_lambda(func, std::array<size_t, 1>({ {qubit} }), convert_data(diag_matrix));
        return;
    }
    else if (diag_matrix[1] == 1.0)
    {
        // [[z, 0], [0, 1]]
        if (diag_matrix[0] == std::complex<double>(0., -1.))
        {
            // [[-i, 0], [0, 1]]
            auto func = [&](const std::array<size_t, 2>& indices,
                const cvector_t<data_t> &lambda_matrix) -> void
            {
                const auto k = indices[1];
                double cache = m_data[k].imag();
                m_data[k].imag(m_data[k].real() * -1.);
                m_data[k].real(cache);
            };

            apply_lambda(func, std::array<size_t, 1>({ {qubit} }), convert_data(diag_matrix));
            return;
        }
        if (diag_matrix[0] == std::complex<double>(0., 1.))
        {
            // [[i, 0], [0, 1]]
            auto func = [&](const std::array<size_t, 2>& indices,
                const cvector_t<data_t> &lambda_matrix) -> void
            {
                const auto k = indices[1];
                double cache = m_data[k].imag();
                m_data[k].imag(m_data[k].real());
                m_data[k].real(cache * -1.);
            };

            apply_lambda(func, std::array<size_t, 1>({ {qubit} }),convert_data(diag_matrix));
            return;
        }
        if (diag_matrix[0] == 0.0)
        {
            // [[0, 0], [0, 1]]
            auto func = [&](const std::array<size_t, 2>& indices,
                const cvector_t<data_t> &lambda_matrix) -> void
            {
                m_data[indices[0]] = 0.0;
            };

            apply_lambda(func, std::array<size_t, 1>({ {qubit} }), convert_data(diag_matrix));
            return;
        }

        // handle general matrix [[z, 0], [0, 1]]
        auto func = [&](const std::array<size_t, 2>& indices,
            const cvector_t<data_t> &lambda_matrix) -> void
        {
            const auto k = indices[0];
            m_data[k] *= lambda_matrix[0];
        };

        apply_lambda(func, std::array<size_t, 1>({ {qubit} }), convert_data(diag_matrix));
        return;
    }
    else
    {
        // Lambda function for diagonal matrix multiplication
        auto func = [&](const std::array<size_t, 2>& indices,
            const cvector_t<data_t> &lambda_matrix) -> void
        {
            const auto k0 = indices[0];
            const auto k1 = indices[1];

            m_data[k0] *= lambda_matrix[0];
            m_data[k1] *= lambda_matrix[1];
        };

        apply_lambda(func, std::array<size_t, 1>({ {qubit} }), convert_data(diag_matrix));
        return;
    }
}

template class QPanda::VectorMatrix<double>;
template class QPanda::VectorMatrix<float>;