#pragma once

#include "QPandaConfig.h"
#include "Core/Module/DataStruct.h"
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/Utilities/Tools/RandomEngine/RandomEngine.h"
#include "Core/VirtualQuantumProcessor/Stabilizer/PauliGroup.h"
#ifdef USE_OPENMP
#include <omp.h>
#endif

QPANDA_BEGIN
class AbstractClifford
{
public:
    virtual void show_tableau() = 0;
    virtual void initialize(uint64_t qubits_num) = 0;

    virtual void append_h(const uint64_t qubit) = 0;
    virtual void append_s(const uint64_t qubit) = 0;

    virtual void append_x(const uint64_t qubit) = 0;
    virtual void append_y(const uint64_t qubit) = 0;
    virtual void append_z(const uint64_t qubit) = 0;

    virtual void append_cx(const uint64_t control, const uint64_t target) = 0;
    virtual void append_cy(const uint64_t control, const uint64_t target) = 0;
    virtual void append_cz(const uint64_t control, const uint64_t target) = 0;

    virtual Qnum measure_and_update(const Qnum qubits) = 0;
    virtual bool measure_and_update(const uint64_t qubit, const uint64_t random_int) = 0;

    virtual prob_vec pmeasure(const Qnum qubits) = 0;
};

//Basic Clifford Simulator Only Support: { H, S, X, Y, Z, CNOT, CY, CZ, SWAP }
class Clifford : public AbstractClifford
{
public:

    void show_tableau();

    void initialize(uint64_t qubits_num);
    void initialize(const Clifford& clifford);

public:

    const PauliGroup& operator[](uint64_t j) const { return m_tableau[j]; }
    const std::vector<PauliGroup>& table() const { return m_tableau; }
    const std::vector<int>& phases() const { return m_phases; }

    const PauliGroup& destabilizer(uint64_t index) const { return m_tableau[index]; }
    const PauliGroup& stabilizer(uint64_t index) const { return m_tableau[m_qubits_num + index]; }

    void append_h(const uint64_t qubit);
    void append_s(const uint64_t qubit);

    void append_x(const uint64_t qubit);
    void append_y(const uint64_t qubit);
    void append_z(const uint64_t qubit);

    void append_cx(const uint64_t control, const uint64_t target);
    void append_cy(const uint64_t control, const uint64_t target);
    void append_cz(const uint64_t control, const uint64_t target);

    std::pair<bool, uint64_t> z_anticommuting(const uint64_t qubit) const;
    std::pair<bool, uint64_t> x_anticommuting(const uint64_t qubit) const;

    Qnum measure_and_update(const Qnum qubits);
    bool measure_and_update(const uint64_t qubit, const uint64_t random_int);

    prob_vec pmeasure(const Qnum qubits);

private:

    bool is_deterministic(const uint64_t& qubit);
    void tableau_row_sum(const PauliGroup& row, const int row_phase, PauliGroup& accum, int& accum_phase);

private:

    uint64_t m_qubits_num = 0;

    RandomEngine19937 m_random;

    std::vector<PauliGroup> m_tableau;
    std::vector<int> m_phases;
};

QPANDA_END