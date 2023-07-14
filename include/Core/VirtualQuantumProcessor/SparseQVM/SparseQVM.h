#ifndef SPARSEQVM_H 
#define SPARSEQVM_H

#include <string>
#include <random>
#include <cmath>
#include <functional>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <list>
#include <set>
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/VirtualQuantumProcessor/SparseQVM/BasicSparseState.h"
#include "Core/VirtualQuantumProcessor/SparseQVM/SparseState.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Tools/Traversal.h"
#include "Core/Utilities/Tools/QProgFlattening.h"
QPANDA_BEGIN
constexpr size_t MAX_QUBITS = 1024;
constexpr size_t MIN_QUBITS = 64;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// Recrusively compiles sizes of QuantumState types between MIN_QUBITS and MAX_QUBITS
// qubits large, growing by powers of 2


template<size_t max_num_bits>
std::shared_ptr<BasicSparseState> construct_wfn_helper(size_t nqubits) {
    return (nqubits > max_num_bits / 2) ?
        std::shared_ptr<BasicSparseState>(new SparseState<max_num_bits>())
        : (nqubits > MIN_QUBITS ? construct_wfn_helper<max_num_bits / 2>(nqubits) :
            std::shared_ptr<BasicSparseState>(new SparseState<MIN_QUBITS>()));
}

// Constructs a new quantum state, templated to use enough qubits to hold `nqubits`,
// with the same state as `old_sim`
template<size_t max_num_bits>
std::shared_ptr<BasicSparseState> expand_wfn_helper(std::shared_ptr<BasicSparseState> old_sim, size_t nqubits) {
    return (nqubits > max_num_bits / 2) ? std::shared_ptr<BasicSparseState>(new SparseState<max_num_bits>(old_sim)) : expand_wfn_helper<max_num_bits / 2>(old_sim, nqubits);
}

class SparseSimulator : public IdealQVM, public TraversalInterface<>
{
public:

    virtual void _start();

    void finalize();

    void setConfig(const Configuration& config);

    virtual void init();

    virtual QVec allocateQubits(size_t qubitNumber);

    virtual QVec qAllocMany(size_t qubit_count);

    std::set<std::string> operations_done;

    SparseSimulator() {

    }

    ~SparseSimulator() {
        _execute_queued_ops();
        finalize();
    }

    universal_wavefunction get_state();

    void set_state(universal_wavefunction& data);

    prob_dict probRunDict(QProg &prog);

    std::map<std::string, size_t> runWithConfiguration(QProg& prog, std::vector<ClassicalCondition>& cbits, int shots);

    unsigned M(size_t target);
    std::map<std::string, bool> directlyRun(QProg& prog);

protected:
    void dump_ids(void(*callback)(size_t));

    void handle_prog_to_queue(QProg &prog);

    void DumpWavefunction(size_t indent = 0);

    void DumpWavefunctionQuietly(size_t indent = 0);

    void set_random_seed(std::mt19937::result_type seed = std::mt19937::default_seed);

    size_t get_num_qubits();

    void allocate_specific_qubit(size_t qubit);

    bool release(size_t qubit_id);

    void X(size_t index);

    void MCX(std::vector<size_t> const& controls, size_t  target);

    void MCApplyAnd(std::vector<size_t> const& controls, size_t  target);

    void MCApplyAndAdj(std::vector<size_t> const& controls, size_t  target);

    void Y(size_t index);

    void MCY(std::vector<size_t> const& controls, size_t target);

    void Z(size_t index);

    void MCZ(std::vector<size_t> const& controls, size_t target);

    // Any phase gate
    void Phase(qcomplex_t const& phase, size_t index);

    void MCPhase(std::vector<size_t> const& controls, qcomplex_t const& phase, size_t target);

    void T(size_t index);

    void AdjT(size_t index);

    void R1(double const& angle, size_t index);

    void MCR1(std::vector<size_t> const& controls, double const& angle, size_t target);

    void R1Frac(std::int64_t numerator, std::int64_t power, size_t index);

    void MCR1Frac(std::vector<size_t> const& controls, std::int64_t numerator, std::int64_t power, size_t target);

    void S(size_t index);

    void AdjS(size_t index);

    void R(Basis_Gate b, double phi, size_t index);

    void MCR(std::vector<size_t> const& controls, Basis_Gate b, double phi, size_t target);

    void RFrac(Basis_Gate axis, std::int64_t numerator, std::int64_t power, size_t index);

    void MCRFrac(std::vector<size_t> const& controls, Basis_Gate axis, std::int64_t numerator, std::int64_t power, size_t target);

    void Exp(std::vector<Basis_Gate> const& axes, double angle, std::vector<size_t> const& qubits);

    void MCExp(std::vector<size_t> const& controls, std::vector<Basis_Gate> const& axes, double angle, std::vector<size_t> const& qubits);

    void H(size_t index);

    void MCH(std::vector<size_t> const& controls, size_t target);

    void SWAP(size_t index_1, size_t index_2);

    void CSWAP(std::vector<size_t> const& controls, size_t index_1, size_t index_2);

    void Reset(size_t target);

    void Assert(std::vector<Basis_Gate> axes, std::vector<size_t> const& qubits, bool result);

    double MeasurementProbability(std::vector<Basis_Gate> const& axes, std::vector<size_t> const& qubits);

    unsigned Measure(std::vector<Basis_Gate> const& axes, std::vector<size_t> const& qubits);

    qcomplex_t probe(QProg& prog, std::string const& label);




    std::string Sample();

    using callback_t = std::function<bool(const char*, double, double)>;
    using extended_callback_t = std::function<bool(const char*, double, double, void*)>;

    bool dump_qubits(std::vector<size_t> const& qubits, callback_t const& callback) {
        _execute_queued_ops(qubits, OP::Ry);
        return _quantum_state->dump_qubits(qubits, callback);
    }

    bool dump_qubits_ext(std::vector<size_t> const& qubits, extended_callback_t const& callback, void* arg) {
        return dump_qubits(qubits, [arg, &callback](const char* c, double re, double im) -> bool { return callback(c, re, im, arg); });
    }

    void dump_all(callback_t const& callback) {
        _execute_queued_ops();
        size_t max_qubit_id = 0;
        for (std::size_t i = 0; i < _occupied_qubits.size(); ++i) {
            if (_occupied_qubits[i])
                max_qubit_id = i;
        }
        _quantum_state->dump_all(max_qubit_id, callback);
    }

    void dump_all_ext(extended_callback_t const& callback, void* arg) {
        dump_all([arg, &callback](const char* c, double re, double im) -> bool { return callback(c, re, im, arg); });
    }

    void update_state() {
        _execute_queued_ops();
    }

private:

    // These indicate whether there are any H, Rx, or Ry gates
    // that have yet to be applied to the wavefunction.
    // Since HH=I and Rx(theta_1)Rx(theta_2) = Rx(theta_1+theta_2)
    // it only needs a boolean to track them.
    std::vector<bool> _queue_H;
    std::vector<bool> _queue_Rx;
    std::vector<bool> _queue_Ry;

    std::vector<double> _angles_Rx;
    std::vector<double> _angles_Ry;

    // Store which qubits are non-zero as a bitstring
    std::vector<bool> _occupied_qubits;
    size_t _max_num_qubits_used = 0;
    size_t _current_number_qubits_used;

    // In a situation where we know a qubit is zero,
    // this sets the occupied qubit vector and decrements
    // the current number of qubits if necessary
    void _set_qubit_to_zero(size_t index) {
        if (_occupied_qubits[index]) {
            --_current_number_qubits_used;
        }
        _occupied_qubits[index] = false;
    }

    // In a situation where a qubit may be non-zero,
    // we increment which qubits are used, and update the current
    // and maximum number of qubits
    void _set_qubit_to_nonzero(size_t index) {
        if (!_occupied_qubits[index]) {
            ++_current_number_qubits_used;
            _max_num_qubits_used = std::max(_max_num_qubits_used, _current_number_qubits_used);
        }
        _occupied_qubits[index] = true;
    }

    // Normalizer for T gates: 1/sqrt(2)
    const double _normalizer_double = 1.0 / std::sqrt(2.0);


    std::shared_ptr<BasicSparseState> m_simulator = nullptr;
    // Internal quantum state
    std::shared_ptr<BasicSparseState> _quantum_state;

    // Queued phase and permutation operations
    std::list<operation> _queued_operations;

    // The next three functions execute the H, and/or Rx, and/or Ry
    // queues on a single qubit
    void _execute_RyRxH_single_qubit(size_t const &index) {
        if (_queue_H[index]) {
            _quantum_state->H(index);
            _queue_H[index] = false;
        }
        if (_queue_Rx[index]) {
            _quantum_state->R(Basis_Gate::PauliX, _angles_Rx[index], index);
            _angles_Rx[index] = 0.0;
            _queue_Rx[index] = false;
        }
        if (_queue_Ry[index]) {
            _quantum_state->R(Basis_Gate::PauliY, _angles_Ry[index], index);
            _angles_Ry[index] = 0.0;
            _queue_Ry[index] = false;
        }
    }

    void _execute_RxH_single_qubit(size_t const &index) {
        if (_queue_H[index]) {
            _quantum_state->H(index);
            _queue_H[index] = false;
        }
        if (_queue_Rx[index]) {
            _quantum_state->R(Basis_Gate::PauliX, _angles_Rx[index], index);
            _angles_Rx[index] = 0.0;
            _queue_Rx[index] = false;
        }
    }

    void _execute_H_single_qubit(size_t const &index) {
        if (_queue_H[index]) {
            _quantum_state->H(index);
            _queue_H[index] = false;
        }
    }

    // Executes all phase and permutation operations, if any exist
    void _execute_phase_and_permute() {
        if (_queued_operations.size() != 0) {
            _quantum_state->phase_and_permute(_queued_operations);
            _queued_operations.clear();
        }
    }

    // Executes all queued operations (including H and rotations)
    // on all qubits 
    void _execute_queued_ops() {
        _execute_phase_and_permute();
        size_t num_qubits = _quantum_state->get_num_qubits();
        for (size_t index = 0; index < num_qubits; index++) {
            _execute_RyRxH_single_qubit(index);
        }
    }

    // Executes all phase and permutation operations,
    // then any H, Rx, or Ry gates queued on the qubit index,
    // up to the level specified (where H < Rx < Ry)
    void _execute_queued_ops(size_t index, OP level = OP::Ry) {
        _execute_phase_and_permute();
        switch (level) {
        case OP::Ry:
            _execute_RyRxH_single_qubit(index);
            break;
        case OP::Rx:
            _execute_RxH_single_qubit(index);
            break;
        case OP::H:
            _execute_H_single_qubit(index);
            break;
        default:
            break;
        }
    }

    // Executes all phase and permutation operations,
    // then any H, Rx, or Ry gates queued on any of the qubit indices,
    // up to the level specified (where H < Rx < Ry)
    void _execute_queued_ops(std::vector<size_t> const& indices, OP level = OP::Ry) {
        _execute_phase_and_permute();
        switch (level) {
        case OP::Ry:
            for (auto index : indices) {
                _execute_RyRxH_single_qubit(index);
            }
            break;
        case OP::Rx:
            for (auto index : indices) {
                _execute_RxH_single_qubit(index);
            }
            break;
        case OP::H:
            for (auto index : indices) {
                _execute_H_single_qubit(index);
            }
            break;
        default:
            break;
        }
    }


    // Executes if there is anything already queued on the qubit target
    // Used when queuing gates that do not commute well
    void _execute_if(size_t target) {
        if (_queue_Ry[target] || _queue_Rx[target] || _queue_H[target]) {
            _execute_queued_ops(target, OP::Ry);
        }
    }

    // Executes if there is anything already queued on the qubits in controls
    // Used when queuing gates that do not commute well
    void _execute_if(std::vector<size_t> const &controls) {
        for (auto control : controls) {
            if (_queue_Ry[control] || _queue_Rx[control] || _queue_H[control]) {
                _execute_queued_ops(controls, OP::Ry);
                return;
            }
        }
    }



};

QPANDA_END
#endif//!SPARSEQVM_H