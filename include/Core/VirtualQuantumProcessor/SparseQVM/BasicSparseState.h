#ifndef _BASIC_SPARSE_STATE_H_
#define _BASIC_SPARSE_STATE_H_
#include <list>
#include <map>
#include <string>
#include "Core/Utilities/QPandaNamespace.h"
QPANDA_BEGIN

enum class OP {
    MCPhase,
    Phase,
    T,
    AdjT,
    S,
    AdjS,
    H,
    MCH,
    X,
    MCX,
    Y,
    MCY,
    Z,
    MCZ,
    M,
    Measure,
    Exp,
    MCExp,
    R1,
    MCR1,
    Rx,
    Rz,
    Ry,
    MCRx,
    MCRz,
    MCRy,
    SWAP,
    MCSWAP,
    PermuteSmall,
    PermuteLarge,
    Proj,
    MCProj,
    Allocate,
    Release,
    NUM_OPS // counts gate types; do not add gates after this!
};

// Used in operation queues for phases/permutations
// Different constructors correspond to different 
// kinds of gates, so some data is not initialized 
// for certain types of gates
struct operation {
    OP gate_type;
    size_t target;
    operation(OP gate_type_arg,
        size_t target_arg) :
        gate_type(gate_type_arg),
        target(target_arg) {}

    std::vector<size_t> controls;
    operation(OP gate_type_arg,
        size_t target_arg,
        std::vector<size_t> controls_arg
    ) : gate_type(gate_type_arg),
        target(target_arg),
        controls(controls_arg) {}

    size_t shift;
    size_t target_2;
    //swap
    operation(OP gate_type_arg,
        size_t target1_arg,
        size_t shift_arg,
        size_t target_2_arg
    ) :gate_type(gate_type_arg),
        target(target1_arg),
        shift(shift_arg),
        target_2(target_2_arg) {}
    //mcswap
    operation(OP gate_type_arg,
        size_t target1_arg,
        size_t shift_arg,
        std::vector<size_t> controls_arg,
        size_t target_2_arg
    ) : gate_type(gate_type_arg),
        target(target1_arg),
        shift(shift_arg),
        controls(controls_arg),
        target_2(target_2_arg) {}
    qcomplex_t phase;
    // Phase
    operation(OP gate_type_arg,
        size_t target_arg,
        qcomplex_t phase_arg
    ) :gate_type(gate_type_arg),
        target(target_arg),
        phase(phase_arg) {}
    // MCPhase
    operation(OP gate_type_arg,
        size_t target_arg,
        std::vector<size_t> controls_arg,
        qcomplex_t phase_arg
    ) : gate_type(gate_type_arg),
        target(target_arg),
        controls(controls_arg),
        phase(phase_arg)
    {}
};

// Also represents operations, but uses
// bitsets instead of vectors of qubit ids
// to save time/space
template<size_t num_qubits>
struct condensed_operation {
    OP gate_type;
    size_t target;
    condensed_operation(OP gate_type_arg,
        size_t target_arg) :
        gate_type(gate_type_arg),
        target(target_arg)
    {}

    std::bitset<num_qubits> controls;
    condensed_operation(OP gate_type_arg,
        size_t target_arg,
        std::bitset<num_qubits> const& controls_arg
    ) : gate_type(gate_type_arg),
        target(target_arg),
        controls(controls_arg) {}

    size_t target_2;
    //swap
    condensed_operation(OP gate_type_arg,
        size_t target1_arg,
        size_t target_2_arg
    ) :gate_type(gate_type_arg),
        target(target1_arg),
        target_2(target_2_arg) {}
    //mcswap
    condensed_operation(OP gate_type_arg,
        size_t target1_arg,
        std::bitset<num_qubits> const& controls_arg,
        size_t target_2_arg
    ) : gate_type(gate_type_arg),
        target(target1_arg),
        controls(controls_arg),
        target_2(target_2_arg) {}
    qcomplex_t phase;
    // Phase
    condensed_operation(OP gate_type_arg,
        size_t target_arg,
        qcomplex_t phase_arg
    ) :gate_type(gate_type_arg),
        target(target_arg),
        phase(phase_arg) {}
    // MCPhase
    condensed_operation(OP gate_type_arg,
        size_t target_arg,
        std::bitset<num_qubits> const& controls_arg,
        qcomplex_t phase_arg
    ) : gate_type(gate_type_arg),
        target(target_arg),
        controls(controls_arg),
        phase(phase_arg)
    {}
};

/// a type for runtime basis specification
enum class Basis_Gate
{
    PauliI = 0,
    PauliX = 1,
    PauliY = 3,
    PauliZ = 2
};


class BasicSparseState
{
public:

    virtual size_t get_num_qubits() = 0;
    virtual void init_state(universal_wavefunction& new_qubit_data) = 0;
    virtual void dump_wavefunction(size_t indent = 0) = 0;

    virtual void set_random_seed(std::mt19937::result_type seed = std::mt19937::default_seed) = 0;

    virtual void set_precision(double new_precision) = 0;

    virtual float get_load_factor() = 0;

    virtual void set_load_factor(float new_load_factor) = 0;

    virtual size_t get_wavefunction_size() = 0;

    virtual void pauli_combination(std::vector<Basis_Gate> const&, std::vector<size_t> const&, qcomplex_t, qcomplex_t) = 0;
    virtual void MCPauliCombination(std::vector<size_t> const&, std::vector<Basis_Gate> const&, std::vector<size_t> const&, qcomplex_t, qcomplex_t) = 0;

    virtual unsigned measure_single_qbit(size_t) = 0;

    virtual void Reset(size_t) = 0;

    virtual void Assert(std::vector<Basis_Gate> const&, std::vector<size_t> const&, bool) = 0;

    virtual double MeasurementProbability(std::vector<Basis_Gate> const&, std::vector<size_t> const&) = 0;
    virtual unsigned Measure(std::vector<Basis_Gate> const&, std::vector<size_t> const&) = 0;


    virtual qcomplex_t probe(std::string const& label) = 0;

    virtual bool dump_qubits(std::vector<size_t> const& qubits, std::function<bool(const char*, double, double)>const&) = 0;

    virtual void dump_all(size_t max_qubit_id, std::function<bool(const char*, double, double)>const&) = 0;

    virtual void phase_and_permute(std::list<operation>const &) = 0;

    virtual void R(Basis_Gate b, double phi, size_t index) = 0;
    virtual void MCR(std::vector<size_t> const&, Basis_Gate, double, size_t) = 0;

    virtual void H(size_t index) = 0;
    virtual void MCH(std::vector<size_t> const& controls, size_t index) = 0;

    virtual bool is_qubit_zero(size_t) = 0;
    virtual std::pair<bool, bool> is_qubit_classical(size_t) = 0;

    virtual universal_wavefunction get_universal_wavefunction() = 0;

    virtual std::function<double()> get_rng() = 0;

    virtual std::string Sample() = 0;
};

QPANDA_END
#endif  //!_BASIC_SPARSE_STATE_H_