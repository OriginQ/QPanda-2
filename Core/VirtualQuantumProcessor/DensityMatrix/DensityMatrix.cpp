#include <set>
#include <array>
#include "Core/VirtualQuantumProcessor/DensityMatrix/DensityMatrix.h"

USING_QPANDA

using std::vector;
using std::complex;

template <typename data_t>
DensityMatrix<data_t>::DensityMatrix(size_t qubits_num)
{
    set_num_qubits(qubits_num);
}

template <typename data_t>
void DensityMatrix<data_t>::set_num_qubits(size_t qubits_num)
{
    m_qubits_num = qubits_num;
    m_rows = 1ULL << qubits_num;
    VectorMatrix<data_t>::m_data_size = 1ULL << (qubits_num + qubits_num);;
    VectorMatrix<data_t>::m_data.resize(1ULL << (qubits_num + qubits_num));
}

template <typename data_t>
void DensityMatrix<data_t>::initialize()
{
    std::fill(VectorMatrix<data_t>::m_data.begin(), VectorMatrix<data_t>::m_data.end(), 0);
    VectorMatrix<data_t>::m_data[0] = 1.0;
}

template <typename data_t>
void DensityMatrix<data_t>::init_density_matrix(size_t qubits_num)
{
    set_num_qubits(qubits_num);
    initialize();
}


template <typename data_t>
void DensityMatrix<data_t>::initialize(const std::vector<std::complex<data_t>>& data)
{
    if (VectorMatrix<data_t>::m_data.size() == data.size())
    {
        // Convert a vectorized density matrix into density matrix
        std::move(data.begin(), data.end(), VectorMatrix<data_t>::m_data.data());
    }
    else if (VectorMatrix<data_t>::m_data.size() == data.size() * data.size())
    {
        // Convert a state vector into density matrix
        auto matrix = vector_kron(vector_conj(data), data);
        std::move(matrix.begin(), matrix.end(), VectorMatrix<data_t>::m_data.data());
    }
    else 
    {
        QCERR_AND_THROW(std::runtime_error, "DensityMatrix initialize length incorrect");
    }
}

template <typename data_t>
void DensityMatrix<data_t>::initialize(const cmatrix_t& data)
{
    if (VectorMatrix<data_t>::m_data.size() == data.size())
    {
        // Convert a vectorized density matrix into density matrix
#pragma omp parallel for num_threads(omp_get_max_threads())
        for (int64_t i = 0; i < data.size(); ++i)
            VectorMatrix<data_t>::m_data[i] = (std::complex<data_t>)data(i);
    }
    else
    {
        QCERR_AND_THROW(std::runtime_error, "DensityMatrix initialize length incorrect");
    }
}


template <typename data_t>
Qnum DensityMatrix<data_t>::superop_qubits(const Qnum &qubits) const
{
    Qnum superop_qubits = qubits;
    const auto qubits_num = m_qubits_num;

    for (const auto q : qubits) 
        superop_qubits.push_back(q + qubits_num);

    return superop_qubits;
}

template <typename data_t>
void DensityMatrix<data_t>::apply_karus(const Qnum &qubits, const std::vector<cvector_t<double>>& matrix_list)
{
    apply_superop_matrix(qubits, kraus_superop(matrix_list));
}


template <typename data_t>
void DensityMatrix<data_t>::apply_superop_matrix(const Qnum &qubits, const cvector_t<double>& matrix)
{
    VectorMatrix<data_t>::apply_matrix(superop_qubits(qubits), matrix);
}

template <typename data_t>
void DensityMatrix<data_t>::apply_unitary_matrix(const Qnum &qubits, const cvector_t<double>& matrix)
{
    if (qubits.size() > 4) 
    {
        // Apply as two qubits_num-qubit matrix mults
        Qnum conj_qubits;
        for (const auto q : qubits)
            conj_qubits.push_back(q + m_qubits_num);

        // Apply id \otimes U
        VectorMatrix<data_t>::apply_matrix(qubits, matrix);

        // Apply conj(U) \otimes id
        VectorMatrix<data_t>::apply_matrix(conj_qubits, vector_conj(matrix));
    }
    else 
    {
        // Apply as single 2N-qubit matrix mult.
        apply_superop_matrix(qubits, to_superop(matrix));
    }
}

template <typename data_t>
void DensityMatrix<data_t>::apply_diagonal_superop_matrix(const Qnum &qubits, const cvector_t<double>& diag_matrix) 
{
    VectorMatrix<data_t>::apply_diagonal_matrix(superop_qubits(qubits), diag_matrix);
}

template <typename data_t>
void DensityMatrix<data_t>::apply_diagonal_unitary_matrix(const Qnum &qubits, const cvector_t<double>& diag_matrix) 
{
    // Apply as single 2N-qubit matrix mult.
    apply_diagonal_superop_matrix(qubits, vector_kron(vector_conj(diag_matrix), diag_matrix));
}

template <typename data_t>
void DensityMatrix<data_t>::apply_CNOT(const size_t q0, const size_t q1)
{
    std::vector<std::pair<size_t, size_t>> pairs = 
    {
      {{1, 3}, {4, 12}, {5, 15}, {6, 14}, {7, 13}, {9, 11}}
    };

    const Qnum qubits = { {q0, q1, q0 + m_qubits_num, q1 + m_qubits_num} };
    VectorMatrix<data_t>::apply_permutation_matrix(qubits, pairs);
}

template <typename data_t>
void DensityMatrix<data_t>::apply_CZ(const size_t q0, const size_t q1)
{
    // Lambda function for CZ gate
    auto lambda = [&](const std::array<size_t, 1ULL << 4> &indices)->void 
    {
        VectorMatrix<data_t>::m_data[indices[3]] *= -1.;
        VectorMatrix<data_t>::m_data[indices[7]] *= -1.;
        VectorMatrix<data_t>::m_data[indices[11]] *= -1.;
        VectorMatrix<data_t>::m_data[indices[12]] *= -1.;
        VectorMatrix<data_t>::m_data[indices[13]] *= -1.;
        VectorMatrix<data_t>::m_data[indices[14]] *= -1.;
    };

    const std::array<size_t, 4> qubits = { {q0, q1, q0 + m_qubits_num, q1 + m_qubits_num} };
    VectorMatrix<data_t>::apply_lambda(lambda, qubits);
}

template <typename data_t>
void DensityMatrix<data_t>::apply_SWAP(const size_t q0, const size_t q1) 
{
    std::vector<std::pair<size_t, size_t>> pairs = 
    {
        {{1, 2}, {4, 8}, {5, 10}, {6, 9}, {7, 11}, {13, 14}}
    };

    const Qnum qubits = { {q0, q1, q0 + m_qubits_num, q1 + m_qubits_num} };
    VectorMatrix<data_t>::apply_permutation_matrix(qubits, pairs);
}

template <typename data_t>
void DensityMatrix<data_t>::apply_X(const size_t qubit)
{
    auto lambda = [&](const std::array<size_t, 1ULL << 2> &indices)->void 
    {
        std::swap(VectorMatrix<data_t>::m_data[indices[0]], VectorMatrix<data_t>::m_data[indices[3]]);
        std::swap(VectorMatrix<data_t>::m_data[indices[1]], VectorMatrix<data_t>::m_data[indices[2]]);
    };

    const std::array<size_t, 2> qubits = { {qubit, qubit + m_qubits_num} };
    VectorMatrix<data_t>::apply_lambda(lambda, qubits);
}

template <typename data_t>
void DensityMatrix<data_t>::apply_Y(const size_t qubit)
{
    auto lambda = [&](const std::array<size_t, 1ULL << 2> &indices)->void 
    {
        std::swap(VectorMatrix<data_t>::m_data[indices[0]], VectorMatrix<data_t>::m_data[indices[3]]);
        const std::complex<data_t> cache = std::complex<data_t>(-1) * VectorMatrix<data_t>::m_data[indices[1]];
        VectorMatrix<data_t>::m_data[indices[1]] = std::complex<data_t>(-1) * VectorMatrix<data_t>::m_data[indices[2]];
        VectorMatrix<data_t>::m_data[indices[2]] = cache;
    };

    const std::array<size_t, 2> qubits = { {qubit, qubit + m_qubits_num} };
    VectorMatrix<data_t>::apply_lambda(lambda, qubits);
}

template <typename data_t>
void DensityMatrix<data_t>::apply_Z(const size_t qubit)
{
    auto lambda = [&](const std::array<size_t, 1ULL << 2> &indices)->void 
    {
        VectorMatrix<data_t>::m_data[indices[1]] *= -1;
        VectorMatrix<data_t>::m_data[indices[2]] *= -1;
    };

    const std::array<size_t, 2> qubits = { {qubit, qubit + m_qubits_num} };
    VectorMatrix<data_t>::apply_lambda(lambda, qubits);
}

template <typename data_t>
void DensityMatrix<data_t>::apply_Phase(const size_t q0, const std::complex<double>& phase)
{
    cvector_t<double> diag_matrix(2);
    diag_matrix[0] = 1.0;
    diag_matrix[1] = phase;
    apply_diagonal_unitary_matrix({ q0 }, diag_matrix);
}

template <typename data_t>
void DensityMatrix<data_t>::apply_CPhase(const size_t q0, const size_t q1, const std::complex<double>& phase) 
{
    const std::complex<double> iphase = std::conj(phase);

    auto lambda = [&](const std::array<size_t, 1ULL << 4> &indices)->void
    {
        VectorMatrix<data_t>::m_data[indices[3]] *= phase;
        VectorMatrix<data_t>::m_data[indices[7]] *= phase;
        VectorMatrix<data_t>::m_data[indices[11]] *= phase;
        VectorMatrix<data_t>::m_data[indices[12]] *= iphase;
        VectorMatrix<data_t>::m_data[indices[13]] *= iphase;
        VectorMatrix<data_t>::m_data[indices[14]] *= iphase;
    };

    const std::array<size_t, 4> qubits = { {q0, q1, q0 + m_qubits_num, q1 + m_qubits_num} };
    VectorMatrix<data_t>::apply_lambda(lambda, qubits);
}

template <typename data_t>
void DensityMatrix<data_t>::apply_Toffoli(const size_t q0, const size_t q1, const size_t q2) 
{
    std::vector<std::pair<size_t, size_t>> pairs = 
    {
      {{3, 7}, {11, 15}, {19, 23}, {24, 56}, {25, 57}, {26, 58}, {27, 63},
      {28, 60}, {29, 61}, {30, 62}, {31, 59}, {35, 39}, {43,47}, {51, 55}}
    };

    const Qnum qubits = { {q0, q1, q2, q0 + m_qubits_num, q1 + m_qubits_num, q2 + m_qubits_num} };
    VectorMatrix<data_t>::apply_permutation_matrix(qubits, pairs);
}

template <typename data_t>
void DensityMatrix<data_t>::apply_mcx(const Qnum& qubits)
{
    const size_t qubits_num = qubits.size();
    const size_t index_0 = (1ull << (qubits_num - 1)) - 1;
    const size_t index_1 = (1ull << qubits_num) - 1;

    switch (qubits_num)
    {
        case 1:
        {
            //Lambda function for X gate
            auto lambda = [&](const std::array<size_t, 2> &indices) -> void
            {
                std::swap(VectorMatrix<data_t>::m_data[indices[index_0]], VectorMatrix<data_t>::m_data[indices[index_1]]);
            };

            VectorMatrix<data_t>::apply_lambda(lambda, std::array<size_t, 1>({ {qubits[0]} }));
            return;
        }
        case 2:
        {
            //Lambda function for CX gate
            auto lambda = [&](const std::array<size_t, 4> &indices) -> void
            {
                std::swap(VectorMatrix<data_t>::m_data[indices[index_0]], VectorMatrix<data_t>::m_data[indices[index_1]]);
            };

            VectorMatrix<data_t>::apply_lambda(lambda, std::array<size_t, 2>({ {qubits[0], qubits[1]} }));
            return;
        }
        case 3:
        {
            //Lambda function for Toffli gate
            auto lambda = [&](const std::array<size_t, 8> &indices) -> void
            {
                std::swap(VectorMatrix<data_t>::m_data[indices[index_0]], VectorMatrix<data_t>::m_data[indices[index_1]]);
            };

            VectorMatrix<data_t>::apply_lambda(lambda, std::array<size_t, 3>({ {qubits[0], qubits[1], qubits[2]} }));
            return;
        }
        default:
        {
            //Lambda function for general multi-controlled X gate
            auto lambda = [&](const std::unique_ptr<size_t[]>& indices) -> void
            {
                std::swap(VectorMatrix<data_t>::m_data[indices[index_0]], VectorMatrix<data_t>::m_data[indices[index_1]]);
            };

            VectorMatrix<data_t>::apply_lambda(lambda, qubits);
        }
    } // end switch
}

template <typename data_t>
void DensityMatrix<data_t>::apply_mcy(const Qnum& qubits, bool is_conj)
{
    const size_t qubits_num = qubits.size();

    const size_t index_0 = (1ull << (qubits_num - 1)) - 1;
    const size_t index_1 = (1ull << qubits_num) - 1;

    std::complex<data_t> i(0., 1.);

    if (is_conj)
        i = std::conj(i);

    switch (qubits_num)
    {
        case 1:
        {
            //Lambda function for Y gate
            auto lambda = [&](const std::array<size_t, 2> &indices) -> void
            {
                const std::complex<data_t> cache = VectorMatrix<data_t>::m_data[indices[index_0]];
                VectorMatrix<data_t>::m_data[indices[index_0]] = -i * VectorMatrix<data_t>::m_data[indices[index_1]];
                VectorMatrix<data_t>::m_data[indices[index_1]] = i * cache;
            };

            VectorMatrix<data_t>::apply_lambda(lambda, std::array<size_t, 1>({ {qubits[0]} }));
            return;
        }
        case 2:
        {
            //Lambda function for CY gate
            auto lambda = [&](const std::array<size_t, 4> &indices) -> void
            {
                const std::complex<data_t> cache = VectorMatrix<data_t>::m_data[indices[index_0]];
                VectorMatrix<data_t>::m_data[indices[index_0]] = -i * VectorMatrix<data_t>::m_data[indices[index_1]];
                VectorMatrix<data_t>::m_data[indices[index_1]] = i * cache;
            };

            VectorMatrix<data_t>::apply_lambda(lambda, std::array<size_t, 2>({ {qubits[0], qubits[1]} }));
            return;
        }
        case 3:
        {
            //Lambda function for CCY gate
            auto lambda = [&](const std::array<size_t, 8> &indices) -> void
            {
                const std::complex<data_t> cache = VectorMatrix<data_t>::m_data[indices[index_0]];
                VectorMatrix<data_t>::m_data[indices[index_0]] = -i * VectorMatrix<data_t>::m_data[indices[index_1]];
                VectorMatrix<data_t>::m_data[indices[index_1]] = i * cache;
            };

            VectorMatrix<data_t>::apply_lambda(lambda, std::array<size_t, 3>({ {qubits[0], qubits[1], qubits[2]} }));
            return;
        }
        default:
        {
            //Lambda function for general multi-controlled Y gate
            auto lambda = [&](const std::unique_ptr<size_t[]>& indices) -> void
            {
                const std::complex<data_t> cache = VectorMatrix<data_t>::m_data[indices[index_0]];
                VectorMatrix<data_t>::m_data[indices[index_0]] = -i * VectorMatrix<data_t>::m_data[indices[index_1]];
                VectorMatrix<data_t>::m_data[indices[index_1]] = i * cache;
            };

            VectorMatrix<data_t>::apply_lambda(lambda, qubits);
        }
    } // end switch
}

template <typename data_t>
void DensityMatrix<data_t>::apply_mcswap(const Qnum& qubits)
{
    const size_t qubits_num = qubits.size();

    const size_t index_0 = (1ull << (qubits_num - 1)) - 1;
    const size_t index_1 = index_0 + (1ull << (qubits_num - 2));

    switch (qubits_num)
    {
        case 2:
        {
            //Lambda function for SWAP gate
            auto lambda = [&](const std::array<size_t, 4> &indices) -> void
            {
                std::swap(VectorMatrix<data_t>::m_data[indices[index_0]], VectorMatrix<data_t>::m_data[indices[index_1]]);
            };

            VectorMatrix<data_t>::apply_lambda(lambda, std::array<size_t, 2>({ {qubits[0], qubits[1]} }));
            return;
        }
        case 3:
        {
            //Lambda function for C-SWAP gate
            auto lambda = [&](const std::array<size_t, 8> &indices) -> void
            {
                std::swap(VectorMatrix<data_t>::m_data[indices[index_0]], VectorMatrix<data_t>::m_data[indices[index_1]]);
            };

            VectorMatrix<data_t>::apply_lambda(lambda, std::array<size_t, 3>({ {qubits[0], qubits[1], qubits[2]} }));
            return;
        }
        default:
        {
            //Lambda function for general multi-controlled SWAP gate
            auto lambda = [&](const std::unique_ptr<size_t[]>& indices) -> void
            {
                std::swap(VectorMatrix<data_t>::m_data[indices[index_0]], VectorMatrix<data_t>::m_data[indices[index_1]]);
            };

            VectorMatrix<data_t>::apply_lambda(lambda, qubits);
        }
    } // end switch
}

template <typename data_t>
void DensityMatrix<data_t>::apply_mcphase(const Qnum& qubits, const std::complex<double> phase)
{
    const size_t qubits_num = qubits.size();

    switch (qubits_num)
    {
        case 1:
        {
            //Lambda function for arbitrary Phase gate with diagonal [1, phase]
            auto lambda = [&](const std::array<size_t, 2> &indices) -> void
            {
                VectorMatrix<data_t>::m_data[indices[1]] *= phase;
            };

            VectorMatrix<data_t>::apply_lambda(lambda, std::array<size_t, 1>({ {qubits[0]} }));
            return;
        }
        case 2:
        {
            //Lambda function for CPhase gate with diagonal [1, 1, 1, phase]
            auto lambda = [&](const std::array<size_t, 4> &indices) -> void
            {
                VectorMatrix<data_t>::m_data[indices[3]] *= phase;
            };

            VectorMatrix<data_t>::apply_lambda(lambda, std::array<size_t, 2>({ {qubits[0], qubits[1]} }));
            return;
        }
        case 3:
        {
            auto lambda = [&](const std::array<size_t, 8> &indices) -> void
            {
                VectorMatrix<data_t>::m_data[indices[7]] *= phase;
            };

            VectorMatrix<data_t>::apply_lambda(lambda, std::array<size_t, 3>({ {qubits[0], qubits[1], qubits[2]} }));
            return;
        }
        default:
        {
            // Lambda function for general multi-controlled Phase gate
            // with diagonal [1, ..., 1, phase]
            auto lambda = [&](const std::unique_ptr<size_t[]>& indices) -> void
            {
                VectorMatrix<data_t>::m_data[indices[(1ull << qubits_num) - 1]] *= phase;
            };

            VectorMatrix<data_t>::apply_lambda(lambda, qubits);
        }
    } // end switch
}

template <typename data_t>
void DensityMatrix<data_t>::apply_mcu(const Qnum& qubits, const cvector_t<double> &matrix)
{
    const size_t qubits_num = qubits.size();

    const size_t index_0 = (1ull << (qubits_num - 1)) - 1;
    const size_t index_1 = (1ull << qubits_num) - 1;

    //diagonal matrix lambda function
    if (matrix[1] == 0.0 && matrix[2] == 0.0)
    {
        //apply a phase gate
        if (matrix[0] == 1.0)
        {
            apply_mcphase(qubits, matrix[3]);
            return;
        }

        //apply a general diagonal gate
        const cvector_t<double> diag = { {matrix[0], matrix[3]} };
        switch (qubits_num)
        {
        case 1:
        {
            //apply a single-qubit matrix
            VectorMatrix<data_t>::apply_diagonal_matrix(qubits, diag);
            return;
        }
        case 2:
        {
            //Lambda function for CU gate
            auto lambda = [&](const std::array<size_t, 4> &indices, const cvector_t<data_t> &diag) -> void
            {
                VectorMatrix<data_t>::m_data[indices[index_0]] = diag[0] * VectorMatrix<data_t>::m_data[indices[index_0]];
                VectorMatrix<data_t>::m_data[indices[index_1]] = diag[1] * VectorMatrix<data_t>::m_data[indices[index_1]];
            };

            VectorMatrix<data_t>::apply_lambda(lambda, std::array<size_t, 2>({ {qubits[0], qubits[1]} }), VectorMatrix<data_t>::convert_data(diag));
            return;
        }
        case 3:
        {
            //Lambda function for CCU gate
            auto lambda = [&](const std::array<size_t, 8> &indices, const cvector_t<data_t> &diag) -> void
            {
                VectorMatrix<data_t>::m_data[indices[index_0]] = diag[0] * VectorMatrix<data_t>::m_data[indices[index_0]];
                VectorMatrix<data_t>::m_data[indices[index_1]] = diag[1] * VectorMatrix<data_t>::m_data[indices[index_1]];
            };

            VectorMatrix<data_t>::apply_lambda(lambda, std::array<size_t, 3>({ {qubits[0], qubits[1], qubits[2]} }), VectorMatrix<data_t>::convert_data(diag));
            return;
        }
        default:
        {
            //Lambda function for general multi-controlled U gate
            auto lambda = [&](const std::unique_ptr<size_t[]>& indices, const cvector_t<data_t> &diag) -> void
            {
                VectorMatrix<data_t>::m_data[indices[index_0]] = diag[0] * VectorMatrix<data_t>::m_data[indices[index_0]];
                VectorMatrix<data_t>::m_data[indices[index_1]] = diag[1] * VectorMatrix<data_t>::m_data[indices[index_1]];
            };

            VectorMatrix<data_t>::apply_lambda(lambda, qubits, VectorMatrix<data_t>::convert_data(diag));
            return;
        }
        } // end switch
    }

    //apply a non-diagonal gate
    switch (qubits_num)
    {
        case 1:
        {
            //apply a single-qubit matrix
            VectorMatrix<data_t>::apply_matrix(qubits, matrix);
            return;
        }
        case 2:
        {
            // Lambda function for CU gate
            auto lambda = [&](const std::array<size_t, 4> &indices, const cvector_t<data_t> &lambda_matrix) -> void
            {
                const auto cache = VectorMatrix<data_t>::m_data[indices[index_0]];
                VectorMatrix<data_t>::m_data[indices[index_0]] = lambda_matrix[0] * VectorMatrix<data_t>::m_data[indices[index_0]] + lambda_matrix[2] * VectorMatrix<data_t>::m_data[indices[index_1]];
                VectorMatrix<data_t>::m_data[indices[index_1]] = lambda_matrix[1] * cache + lambda_matrix[3] * VectorMatrix<data_t>::m_data[indices[index_1]];
            };

            VectorMatrix<data_t>::apply_lambda(lambda, std::array<size_t, 2>({ {qubits[0], qubits[1]} }), VectorMatrix<data_t>::convert_data(matrix));
            return;
        }
        case 3:
        {
            //Lambda function for CCU gate
            auto lambda = [&](const std::array<size_t, 8> &indices, const cvector_t<data_t> &lambda_matrix) -> void
            {
                const auto cache = VectorMatrix<data_t>::m_data[indices[index_0]];
                VectorMatrix<data_t>::m_data[indices[index_0]] = lambda_matrix[0] * VectorMatrix<data_t>::m_data[indices[index_0]] + lambda_matrix[2] * VectorMatrix<data_t>::m_data[indices[index_1]];
                VectorMatrix<data_t>::m_data[indices[index_1]] = lambda_matrix[1] * cache + lambda_matrix[3] * VectorMatrix<data_t>::m_data[indices[index_1]];
            };

            VectorMatrix<data_t>::apply_lambda(lambda, std::array<size_t, 3>({ {qubits[0], qubits[1], qubits[2]} }), VectorMatrix<data_t>::convert_data(matrix));
            return;
        }
        default:
        {
            //Lambda function for general multi-controlled U gate
            auto lambda = [&](const std::unique_ptr<size_t[]>& indices, const cvector_t<data_t> &lambda_matrix) -> void
            {
                const auto cache = VectorMatrix<data_t>::m_data[indices[index_0]];
                VectorMatrix<data_t>::m_data[indices[index_0]] = lambda_matrix[0] * VectorMatrix<data_t>::m_data[indices[index_0]] + lambda_matrix[2] * VectorMatrix<data_t>::m_data[indices[index_1]];
                VectorMatrix<data_t>::m_data[indices[index_1]] = lambda_matrix[1] * cache + lambda_matrix[3] * VectorMatrix<data_t>::m_data[indices[index_1]];
            };

            VectorMatrix<data_t>::apply_lambda(lambda, qubits, VectorMatrix<data_t>::convert_data(matrix));
            return;
        }
    } // end switch
}

template <typename data_t>
void DensityMatrix<data_t>::apply_multiplexer(const Qnum& controls, const Qnum& targets, const cvector_t<double>& matrix) 
{
    auto lambda = [&](const std::unique_ptr<size_t[]>& indices, const cvector_t<data_t>& lambda_matrix)->void
    {
        const size_t ctr_nums = controls.size();
        const size_t tar_nums = targets.size();

        const size_t dims = 1ull << (tar_nums + ctr_nums);

        //Lambda function for stacked matrix multiplication
        auto cache = std::make_unique<std::complex<data_t>[]>(dims);
        for (size_t i = 0; i < dims; i++) 
        {
            const auto ii = indices[i];
            cache[i] = VectorMatrix<data_t>::m_data[ii];
            VectorMatrix<data_t>::m_data[ii] = 0.;
        }

        const size_t columns = 1ull << tar_nums;
        const size_t blocks = 1ull << ctr_nums;

        for (size_t b = 0; b < blocks; b++)
            for (size_t i = 0; i < columns; i++)
                for (size_t j = 0; j < columns; j++)
                {
                    VectorMatrix<data_t>::m_data[indices[i + b * columns]] += 
                        lambda_matrix[i + b * columns + dims * j] * cache[b * columns + j];
                }
    };

    //Use the lambda function
    auto qubits = targets;
    for (const auto &qubit : controls) 
        qubits.push_back(qubit);

    VectorMatrix<data_t>::apply_lambda(lambda, qubits, VectorMatrix<data_t>::convert_data(matrix));
}


template <typename data_t>
void DensityMatrix<data_t>::apply_Measure(const Qnum& qubits)
{
    auto probs = probabilities(qubits);

    auto index = std::discrete_distribution<size_t>(probs.begin(), probs.end())(m_mt);

    cvector_t<double> diag_matrix(1ULL << qubits.size(), 0.);
    diag_matrix[index] = 1. / std::sqrt(probs[index]);

    apply_diagonal_unitary_matrix(qubits, diag_matrix);

    return;
}

//-----------------------------------------------------------------------
// Z-measurement outcome probabilities
//-----------------------------------------------------------------------

template <typename data_t>
double DensityMatrix<data_t>::probability(const size_t index)
{
    if (index > (1ull << m_qubits_num) - 1)
        QCERR_AND_THROW(std::runtime_error, "index out of range");

    const auto shift = m_rows + 1;
    return (double)std::real(VectorMatrix<data_t>::m_data[index * shift]);
}

template <typename data_t>
std::vector<double> DensityMatrix<data_t>::probabilities(Qnum qubits)
{
    Qnum check_qubits = qubits;
    std::set<size_t> qubits_set(check_qubits.begin(), check_qubits.end());
    check_qubits.assign(qubits_set.begin(), qubits_set.end());

    if (check_qubits.size() != qubits.size())
        QCERR_AND_THROW(std::runtime_error, "repetitive qubits addr");

    for (auto qubit : qubits)
    {
        if (qubit > m_qubits_num - 1)
            QCERR_AND_THROW(std::runtime_error, "qubit addr out of range");
    }

    auto all_indices_num = 1ull << m_qubits_num;

    // return all qubits probabilities
    auto qubits_num = qubits.size();
    if (!qubits_num || qubits_num == m_qubits_num)
    {
        std::vector<double> result(all_indices_num, 0.);

#pragma omp parallel for num_threads(omp_get_max_threads())
        for (auto i = 0; i < all_indices_num; i++)
            result[i] = probability(i);

        return result;
    }
    else
    {
        // return select qubits probabilities
        auto dims = 1ull << qubits_num;

        std::vector<double> result(dims, 0.);

        auto qubits_sorted = qubits;
        std::sort(qubits_sorted.begin(), qubits_sorted.end());

        auto bit_value = 0;
        for (size_t i = 0; i < all_indices_num; ++i)
        {
            size_t result_index = 0;
            for (auto j = 0; j < qubits_num; ++j)
            {
                bit_value = ((i & (1ull << qubits_sorted[j])) >> qubits_sorted[j]);
                result_index += bit_value ? (1ull << j) : 0;
            }

            result[result_index] += probability(i);
        }

        return result;
    }
}

template <typename data_t>
cmatrix_t DensityMatrix<data_t>::density_matrix()
{
    auto density_matrix = cmatrix_t(m_rows, m_rows);

    // column stacking
    for (auto i = 0; i < m_rows; ++i)
    {
        for (auto j = 0; j < m_rows; ++j)
        {
            density_matrix(j, i) = VectorMatrix<data_t>::m_data[i*m_rows + j];
        }
    }

    return density_matrix;
}

template <typename data_t>
cmatrix_t DensityMatrix<data_t>::reduced_density_matrix(const Qnum& qubits)
{
    Qnum check_qubits = qubits;
    std::set<size_t> qubits_set(check_qubits.begin(), check_qubits.end());
    check_qubits.assign(qubits_set.begin(), qubits_set.end());

    if (check_qubits.size() != qubits.size())
        QCERR_AND_THROW(std::runtime_error, "repetitive qubits addr");

    for (auto qubit : qubits)
    {
        if (qubit > m_qubits_num - 1)
            QCERR_AND_THROW(std::runtime_error, "qubit addr out of range");
    }

    auto qubits_sorted = qubits;
    std::sort(qubits_sorted.begin(), qubits_sorted.end());

    if ((qubits.size() == m_qubits_num) && (qubits == qubits_sorted))
    {
        return density_matrix();
    }
    else
    {
        // Get superoperator qubits
        const Qnum super_qubits = superop_qubits(qubits);
        const Qnum super_qubits_sorted = superop_qubits(qubits_sorted);

        // Get dimensions
        const size_t qubits_num = qubits.size();

        const size_t dim = 1ULL << qubits_num;
        const size_t vector_dim = 1ULL << (2 * qubits_num);

        cmatrix_t reduced_matrix(dim, dim);
        {
            // Fill matrix with first iteration
            const auto indices = multi_array_indices(super_qubits, super_qubits_sorted, 0);

            for (size_t i = 0; i < vector_dim; ++i)
                reduced_matrix(i) = VectorMatrix<data_t>::m_data[indices[i]];
        }

        // Accumulate with remaning blocks
        const size_t step = 1ULL << (m_qubits_num - qubits_num);
        for (size_t k = 1; k < step; k++)
        {
            const auto indices = multi_array_indices(super_qubits, super_qubits_sorted, k * step + k);

            for (size_t i = 0; i < vector_dim; ++i)
                reduced_matrix(i) += VectorMatrix<data_t>::m_data[indices[i]];
        }

        return reduced_matrix;
    }
}



template <typename data_t>
std::complex<double> DensityMatrix<data_t>::trace()
{
    double val_real = 0.;
    double val_imag = 0.;

    for (size_t k = 0; k < m_rows; ++k)
    {
        val_real += std::real(VectorMatrix<data_t>::m_data[k * m_rows + k]);
        val_imag += std::imag(VectorMatrix<data_t>::m_data[k * m_rows + k]);
    }

    return std::complex<double>(val_real, val_imag);
}

template class QPanda::DensityMatrix<double>;
template class QPanda::DensityMatrix<float>;