#pragma once

#include <random>
#include <algorithm>
#include "Core/Module/DataStruct.h"
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/VirtualQuantumProcessor/DensityMatrix/VectorMatrix.h"

QPANDA_BEGIN

class AbstractDensityMatrix
{
public:

    virtual void initialize() = 0;
    virtual void initialize(const cmatrix_t& data) = 0;

    virtual void init_density_matrix(size_t qubits_num) = 0;

    //-----------------------------------------------------------------------
    // Apply Matrices
    //-----------------------------------------------------------------------

    virtual void apply_unitary_matrix(const Qnum &qubits, const cvector_t<double>& matrix) = 0;
    virtual void apply_superop_matrix(const Qnum &qubits, const cvector_t<double>& matrix) = 0;

    virtual void apply_diagonal_unitary_matrix(const Qnum &qubits, const cvector_t<double>& matrix) = 0;
    virtual void apply_diagonal_superop_matrix(const Qnum &qubits, const cvector_t<double>& matrix) = 0;

    virtual void apply_karus(const Qnum &qubits, const std::vector<cvector_t<double>>& matrix_list) = 0;

    //-----------------------------------------------------------------------
    // Apply Specialized Gates
    //-----------------------------------------------------------------------

    virtual void apply_X(const size_t q0) = 0;

    virtual void apply_Y(const size_t q0) = 0;

    virtual void apply_Z(const size_t q0) = 0;

    virtual void apply_CNOT(const size_t q0, const size_t q1) = 0;

    virtual void apply_CZ(const size_t q0, const size_t q1) = 0;

    virtual void apply_SWAP(const size_t q0, const size_t q1) = 0;

    virtual void apply_Phase(const size_t q0, const std::complex<double>& phase) = 0;

    virtual void apply_CPhase(const size_t q0, const size_t q1, const std::complex<double>& phase) = 0;

    // q0, q1: control | q2: target
    virtual void apply_Toffoli(const size_t q0, const size_t q1, const size_t q2) = 0;

    //Apply a general N-qubit multi-controlled X-gate (X,CNOT,Toffoli)
    virtual void apply_mcx(const Qnum& qubits) = 0;

    //Apply a general multi-controlled Y-gate (Y,CY,CCY)
    virtual void apply_mcy(const Qnum& qubits, bool is_conj = false) = 0;

    //Apply a general multi-controlled single-qubit phase gate with diagonal [1, ..., 1, phase]
    //Phase = -1 for Z, CZ, CCZ gate
    virtual void apply_mcphase(const Qnum& qubits, const std::complex<double> phase) = 0;

    //Apply a general multi-controlled single-qubit unitary gate
    virtual void apply_mcu(const Qnum& qubits, const cvector_t<double>& matrix) = 0;

    //Apply a general multi-controlled SWAP gate
    virtual void apply_mcswap(const Qnum& qubits) = 0;

    virtual void apply_multiplexer(const Qnum& controls, const Qnum& targets, const cvector_t<double>& matrix) = 0;

    virtual void apply_Measure(const Qnum& qubits) = 0;

    virtual double probability(const size_t index) = 0;

    virtual std::complex<double> trace() = 0;

    virtual std::vector<double> probabilities(Qnum qubits = {}) = 0;

    virtual cmatrix_t density_matrix() = 0;

    virtual cmatrix_t reduced_density_matrix(const Qnum& qubits) = 0;

private:

};

template <typename data_t = double>
class DensityMatrix : public VectorMatrix<data_t>, public AbstractDensityMatrix
{
public:

    //-----------------------------------------------------------------------
    // Constructors
    //-----------------------------------------------------------------------

    DensityMatrix() : DensityMatrix(0) {};
    DensityMatrix(size_t qubits_num);
    DensityMatrix(const DensityMatrix& obj) = delete;
    DensityMatrix &operator = (const DensityMatrix& obj) = delete;


    void initialize();
    void initialize(const cmatrix_t& data);
    void initialize(const std::vector<std::complex<data_t>>& data);

    void init_density_matrix(size_t qubits_num);

    //-----------------------------------------------------------------------
    // Apply Matrices
    //-----------------------------------------------------------------------

    void apply_unitary_matrix(const Qnum &qubits, const cvector_t<double>& matrix);
    void apply_superop_matrix(const Qnum &qubits, const cvector_t<double>& matrix);

    void apply_diagonal_unitary_matrix(const Qnum &qubits, const cvector_t<double>& matrix);
    void apply_diagonal_superop_matrix(const Qnum &qubits, const cvector_t<double>& matrix);

    void apply_karus(const Qnum &qubits, const std::vector<cvector_t<double>>& matrix_list);

    //-----------------------------------------------------------------------
    // Apply Specialized Gates
    //-----------------------------------------------------------------------

    void apply_X(const size_t q0);

    void apply_Y(const size_t q0);

    void apply_Z(const size_t q0);

    void apply_CNOT(const size_t q0, const size_t q1);

    void apply_CZ(const size_t q0, const size_t q1);

    void apply_SWAP(const size_t q0, const size_t q1);

    void apply_Phase(const size_t q0, const std::complex<double>& phase);

    void apply_CPhase(const size_t q0, const size_t q1, const std::complex<double>& phase);

    // q0, q1: control | q2: target
    void apply_Toffoli(const size_t q0, const size_t q1, const size_t q2);

    //Apply a general N-qubit multi-controlled X-gate (X,CNOT,Toffoli)
    void apply_mcx(const Qnum& qubits);

    //Apply a general multi-controlled Y-gate (Y,CY,CCY)
    void apply_mcy(const Qnum& qubits, bool is_conj = false);

    //Apply a general multi-controlled single-qubit phase gate with diagonal [1, ..., 1, phase]
    //Phase = -1 for Z, CZ, CCZ gate
    void apply_mcphase(const Qnum& qubits, const std::complex<double> phase);

    //Apply a general multi-controlled single-qubit unitary gate
    void apply_mcu(const Qnum& qubits, const cvector_t<double>& matrix);

    //Apply a general multi-controlled SWAP gate
    void apply_mcswap(const Qnum& qubits);

    void apply_multiplexer(const Qnum& controls, const Qnum& targets, const cvector_t<double>& matrix);

    void apply_Measure(const Qnum& qubits);

    //-----------------------------------------------------------------------
    // Z-measurement outcome probabilities
    //-----------------------------------------------------------------------

    // Return the Z-basis measurement outcome probability P(outcome)
    // For N qubits index in [0, 2^N - 1]
    double probability(const size_t index);

    std::vector<double> probabilities(Qnum qubits = {});

    std::complex<double> trace();

    cmatrix_t density_matrix();

    cmatrix_t reduced_density_matrix(const Qnum& qubits);

public:

    size_t get_qubits_num() { return m_qubits_num; }

private:

    size_t m_rows;
    size_t m_qubits_num;

    void set_num_qubits(size_t qubits_num);

    // Convert qubit indicies to vectorized-density matrix qubitvector indices
    Qnum superop_qubits(const Qnum &qubits) const;

    std::mt19937_64 m_mt;
};

QPANDA_END

