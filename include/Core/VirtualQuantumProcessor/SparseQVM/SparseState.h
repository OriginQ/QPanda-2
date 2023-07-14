#ifndef _SPARSE_STATE_H_
#define _SPARSE_STATE_H_

#include <string>
#include <unordered_map>
#include <random>
#include <cmath>
#include <functional>
#include <algorithm>
#include <list>
#include <string>
#include <map>
#include <iostream>
#include <memory>
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/VirtualQuantumProcessor/SparseQVM/BasicSparseState.h"
#include <stdlib.h>



QPANDA_BEGIN

template<size_t num_qubits>
bool get_parity(std::bitset<num_qubits> const& bitstring) {
    return bitstring.count() % 2;
}

template<size_t N>
inline bool operator<(const std::bitset<N>& lhs, const std::bitset<N>& rhs) {
    std::bitset<N> mask = lhs ^ rhs;
    std::bitset<N> const ull_mask = std::bitset<N>((unsigned long long) - 1);
    for (int i = static_cast<int>(N - 8 * sizeof(unsigned long long)); i > 0; i -= static_cast<int>(8 * sizeof(unsigned long long))) {
        if (((mask >> i) & ull_mask).to_ullong() > 0) {
            return ((lhs >> i) & ull_mask).to_ullong() < ((rhs >> i) & ull_mask).to_ullong();
        }
    }
    return ((lhs)& ull_mask).to_ullong() < ((rhs)& ull_mask).to_ullong();
}

template<size_t num_qubits>
std::bitset<num_qubits> get_mask(std::vector<size_t> const& indices) {
    std::bitset<num_qubits> mask;
    for (size_t index : indices) {
        mask.set(index);
    }
    return mask;
}

template<size_t num_qubits>
class SparseState : public BasicSparseState
{
public:
    using qubit_label = qubit_label_type<num_qubits>;

    using wavefunction = abstract_wavefunction<qubit_label>;

    SparseState() {
        _qubit_data = wavefunction();
        _qubit_data.max_load_factor(_load_factor);
        _qubit_data.emplace((size_t)0, 1);
#ifdef USE_RANDOM_DEVICE
        std::random_device rd;
        std::mt19937 gen(rd());
#else   
        std::mt19937 gen(rand());
#endif
        std::uniform_real_distribution<double> dist(0, 1);
        _rng = [gen, dist]() mutable { return dist(gen); };
    }

    SparseState(std::shared_ptr<BasicSparseState> old_state) {
        // Copy any needed data
        _rng = old_state->get_rng();
        // Outputs the previous data with labels as strings
        universal_wavefunction old_qubit_data = old_state->get_universal_wavefunction();
        _qubit_data = wavefunction(old_qubit_data.size());
        _load_factor = old_state->get_load_factor();
        _qubit_data.max_load_factor(_load_factor);
        // Writes this into the current wavefunction as qubit_label types
        for (auto current_state = old_qubit_data.begin(); current_state != old_qubit_data.end(); ++current_state) {
            _qubit_data.emplace(qubit_label(current_state->first), current_state->second);
        }
    }

    wavefunction get_data_state()
    {
        return _qubit_data;
    }

    size_t get_num_qubits() {
        return (size_t)num_qubits;
    }

    // Outputs all states and amplitudes to the console
    void dump_wavefunction(size_t indent = 0) {
        dump_wavefunction(_qubit_data, indent);
    }

    // Outputs all states and amplitudes from an input wavefunction to the console
    void dump_wavefunction(wavefunction &wfn, size_t indent = 0) {
        std::string spacing(indent, ' ');
        std::cout << spacing << "Wavefunction:\n";
        auto line_dump = [spacing](qubit_label label, qcomplex_t val) -> bool {
            std::cout << spacing << "  " << label.to_string() << ": ";
            std::cout << val.real();
            std::cout << (val.imag() < 0 ? " - " : " + ") << std::abs(val.imag()) << "i\n";
            return true;
        };
        _dump_wavefunction_base(wfn, line_dump);
        std::cout << spacing << "--end wavefunction\n";
    }


    void set_random_seed(std::mt19937::result_type seed) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dist(0, 1);
        _rng = [gen, dist]() mutable { return dist(gen); };
    }

    // Used to decide when an amplitude is close enough to 0 to discard
    void set_precision(double new_precision) {
        _precision = new_precision;
        _rotation_precision = new_precision;
    }

    // Load factor of the underlying hash map
    float get_load_factor() {
        return _load_factor;
    }

    void set_load_factor(float new_load_factor) {
        _load_factor = new_load_factor;
    }

    // Returns the number of states in superposition
    size_t get_wavefunction_size() {
        return _qubit_data.size();
    }



    // Applies the operator id_coeff*I + pauli_coeff * P
    // where P is the Pauli operators defined by axes applied to the qubits in qubits.
    void pauli_combination(std::vector<Basis_Gate> const& axes, std::vector<size_t> const& qubits, qcomplex_t id_coeff, qcomplex_t pauli_coeff) {
        // Bit-vectors indexing where gates of each type are applied
        qubit_label XYs = 0;
        qubit_label YZs = 0;
        size_t ycount = 0;
        for (int i = 0; i < axes.size(); i++) {
            switch (axes[i]) {
            case Basis_Gate::PauliY:
                YZs.set(qubits[i]);
                XYs.set(qubits[i]);
                ycount++;
                break;
            case Basis_Gate::PauliX:
                XYs.set(qubits[i]);
                break;
            case Basis_Gate::PauliZ:
                YZs.set(qubits[i]);
                break;
            case Basis_Gate::PauliI:
                break;
            default:
                throw std::runtime_error("Bad Pauli basis");
            }
        }

        // All identity
        if (XYs.none() && YZs.none()) {
            return;
        }

        // This branch handles purely Z Pauli vectors
        // Purely Z has no addition, which would cause
        // problems in the comparison in the next section
        if (XYs.none()) {
            // 0 terms get the sum of the coefficients
            // 1 terms get the difference
            pauli_coeff += id_coeff; // id_coeff + pauli_coeff
            id_coeff *= 2;
            id_coeff -= pauli_coeff; // id_coeff - pauli_coeff

            // To avoid saving states of zero amplitude, these if/else 
            // check for when one of the coefficients is 
            // close enough to zero to regard as zero
            if (std::norm(pauli_coeff) > _rotation_precision) {
                if (std::norm(id_coeff) > _rotation_precision) {
                    // If both coefficients are non-zero, we can just modify the state in-place
                    for (auto current_state = (_qubit_data).begin(); current_state != (_qubit_data).end(); ++current_state) {
                        current_state->second *= (get_parity(current_state->first & YZs) ? id_coeff : pauli_coeff);
                    }
                }
                else {
                    // If id_coeff = 0, then we make a new wavefunction and only add in those that will be multiplied
                    // by the pauli_coeff
                    wavefunction new_qubit_data = make_wavefunction(_qubit_data.size());
                    for (auto current_state = (_qubit_data).begin(); current_state != (_qubit_data).end(); ++current_state) {
                        if (!get_parity(current_state->first & YZs)) {
                            new_qubit_data.emplace(current_state->first, current_state->second * pauli_coeff);
                        }
                    }
                    _qubit_data = std::move(new_qubit_data);
                }
            }
            else {
                // If pauli_coeff=0, don't add states multiplied by the pauli_coeff
                wavefunction new_qubit_data = make_wavefunction(_qubit_data.size());
                for (auto current_state = (_qubit_data).begin(); current_state != (_qubit_data).end(); ++current_state) {
                    if (get_parity(current_state->first & YZs)) {
                        new_qubit_data.emplace(current_state->first, current_state->second * id_coeff);
                    }
                }
                _qubit_data = std::move(new_qubit_data);
            }
        }
        else { // There are some X or Y gates

         // Each Y Pauli adds a global phase of i
            switch (ycount % 4) {
            case 1:
                pauli_coeff *= qcomplex_t(0, 1);
                break;
            case 2:
                pauli_coeff *= -1;
                break;
            case 3:
                pauli_coeff *= qcomplex_t(0, -1);
                break;
            default:
                break;
            }
            // When both the state and flipped state are in superposition, when adding the contribution of
            // the flipped state, we add phase depending on the 1s in the flipped state 
            // This phase would be the parity of (flipped_state->first ^ YZs) 
            // However, we know that flipped_state->first = current_state->first ^ YXs
            // So the parity of the flipped state will be the parity of the current state, plus
            // the parity of YZs & YXs, i.e., the parity of the number of Ys 
            // Since this is constant for all states, we compute it once here and save it
            // Then we only compute the parity of the current state
            qcomplex_t pauli_coeff_alt = ycount % 2 ? -pauli_coeff : pauli_coeff;
            wavefunction new_qubit_data = make_wavefunction(_qubit_data.size() * 2);
            qcomplex_t new_state;
            for (auto current_state = (_qubit_data).begin(); current_state != (_qubit_data).end(); ++current_state) {
                auto alt_state = _qubit_data.find(current_state->first ^ XYs);
                if (alt_state == _qubit_data.end()) { // no matching value
                    new_qubit_data.emplace(current_state->first, current_state->second * id_coeff);
                    new_qubit_data.emplace(current_state->first ^ XYs, current_state->second * (get_parity(current_state->first & YZs) ? -pauli_coeff : pauli_coeff));
                }
                else if (current_state->first < alt_state->first) {
                    // Each Y and Z gate adds a phase (since Y=iXZ)
                    bool parity = get_parity(current_state->first & YZs);
                    new_state = current_state->second * id_coeff + alt_state->second * (parity ? -pauli_coeff_alt : pauli_coeff_alt);
                    if (std::norm(new_state) > _rotation_precision) {
                        new_qubit_data.emplace(current_state->first, new_state);
                    }

                    new_state = alt_state->second * id_coeff + current_state->second * (parity ? -pauli_coeff : pauli_coeff);
                    if (std::norm(new_state) > _rotation_precision) {
                        new_qubit_data.emplace(alt_state->first, new_state);
                    }
                }
            }
            _qubit_data = std::move(new_qubit_data);
        }
    }

    // Applies the operator id_coeff*I + pauli_coeff * P
    // where P is the Pauli operators defined by axes applied to the qubits in qubits.
    // Controlled version
    void MCPauliCombination(std::vector<size_t> const& controls, std::vector<Basis_Gate> const& axes, std::vector<size_t> const& qubits, qcomplex_t id_coeff, qcomplex_t pauli_coeff) {
        // Bit-vectors indexing where gates of each type are applied
        qubit_label cmask = _get_mask(controls);
        qubit_label XYs = 0;
        qubit_label YZs = 0;
        size_t ycount = 0;
        // Used for comparing pairs 
        size_t any_xy = -1;
        for (int i = 0; i < axes.size(); i++) {
            switch (axes[i]) {
            case Basis_Gate::PauliY:
                YZs.set(qubits[i]);
                XYs.set(qubits[i]);
                ycount++;
                any_xy = qubits[i];
                break;
            case Basis_Gate::PauliX:
                XYs.set(qubits[i]);
                any_xy = qubits[i];
                break;
            case Basis_Gate::PauliZ:
                YZs.set(qubits[i]);
                break;
            case Basis_Gate::PauliI:
                break;
            default:
                throw std::runtime_error("Bad Pauli basis");
            }
        }

        // This branch handles purely Z Pauli vectors
        // Purely Z has no addition, which would cause
        // problems in the comparison in the next section
        if (XYs.none()) {
            // 0 terms get the sum of the coefficients
            // 1 terms get the difference
            pauli_coeff += id_coeff; // <- id_coeff + pauli_coeff
            id_coeff *= 2;
            id_coeff -= pauli_coeff; // <- id_coeff - pauli_coeff

            // To avoid saving states of zero amplitude, these if/else 
            // check for when one of the coefficients is 
            // close enough to zero to regard as zero
            if (std::norm(pauli_coeff) > _rotation_precision) {
                if (std::norm(id_coeff) > _rotation_precision) {
                    // If both coefficients are non-zero, we can just modify the state in-place
                    for (auto current_state = (_qubit_data).begin(); current_state != (_qubit_data).end(); ++current_state) {
                        if ((current_state->first & cmask) == cmask) {
                            current_state->second *= (get_parity(current_state->first & YZs) ? id_coeff : pauli_coeff);
                        }
                    }
                }
                else {
                    // If id_coeff = 0, then we make a new wavefunction and only add in those that will be multiplied
                    // by the pauli_coeff
                    wavefunction new_qubit_data = make_wavefunction(_qubit_data.size());
                    for (auto current_state = (_qubit_data).begin(); current_state != (_qubit_data).end(); ++current_state) {
                        if (!get_parity(current_state->first & YZs) && (current_state->first & cmask) == cmask) {
                            new_qubit_data.emplace(current_state->first, current_state->second * pauli_coeff);
                        }
                    }
                    _qubit_data = std::move(new_qubit_data);
                }
            }
            else {
                // If pauli_coeff=0, don't add states multiplied by the pauli_coeff
                wavefunction new_qubit_data = make_wavefunction(_qubit_data.size());
                for (auto current_state = (_qubit_data).begin(); current_state != (_qubit_data).end(); ++current_state) {
                    if (get_parity(current_state->first & YZs) && (current_state->first & cmask) == cmask) {
                        new_qubit_data.emplace(current_state->first, current_state->second * id_coeff);
                    }
                }
                _qubit_data = std::move(new_qubit_data);
            }
        }
        else { // There are some X or Y gates
         // Each Y Pauli adds a global phase of i
            switch (ycount % 4) {
            case 1:
                pauli_coeff *= qcomplex_t(0, 1);
                break;
            case 2:
                pauli_coeff *= -1;
                break;
            case 3:
                pauli_coeff *= qcomplex_t(0, -1);
                break;
            default:
                break;
            }
            // When both the state and flipped state are in superposition, when adding the contribution of
            // the flipped state, we add phase depending on the 1s in the flipped state 
            // This phase would be the parity of (flipped_state->first ^ YZs) 
            // However, we know that flipped_state->first = current_state->first ^ YXs
            // So the parity of the flipped state will be the parity of the current state, plus
            // the parity of YZs & YXs, i.e., the parity of the number of Ys 
            // Since this is constant for all states, we compute it once here and save it
            // Then we only compute the parity of the current state
            qcomplex_t pauli_coeff_alt = ycount % 2 ? -pauli_coeff : pauli_coeff;
            wavefunction new_qubit_data = make_wavefunction(_qubit_data.size() * 2);
            qcomplex_t new_state;
            for (auto current_state = (_qubit_data).begin(); current_state != (_qubit_data).end(); ++current_state) {
                if ((current_state->first & cmask) == cmask) {
                    auto alt_state = _qubit_data.find(current_state->first ^ XYs);
                    if (alt_state == _qubit_data.end()) { // no matching value
                        new_qubit_data.emplace(current_state->first, current_state->second * id_coeff);
                        new_qubit_data.emplace(current_state->first ^ XYs, current_state->second * (get_parity(current_state->first & YZs) ? -pauli_coeff : pauli_coeff));
                    }
                    else if (current_state->first < alt_state->first) { //current_state->first[any_xy]){//
                        // Each Y and Z gate adds a phase (since Y=iXZ)
                        bool parity = get_parity(current_state->first & YZs);
                        new_state = current_state->second * id_coeff + alt_state->second * (parity ? -pauli_coeff_alt : pauli_coeff_alt);
                        if (std::norm(new_state) > _rotation_precision) {
                            new_qubit_data.emplace(current_state->first, new_state);
                        }

                        new_state = alt_state->second * id_coeff + current_state->second * (parity ? -pauli_coeff : pauli_coeff);
                        if (std::norm(new_state) > _rotation_precision) {
                            new_qubit_data.emplace(alt_state->first, new_state);
                        }
                    }
                }
                else {
                    new_qubit_data.emplace(current_state->first, current_state->second);
                }
            }
            _qubit_data = std::move(new_qubit_data);
        }
    }

    unsigned measure_single_qbit(size_t target)
    {
        double zero_probability = 0.0;
        double one_probability = 0.0;

        // Writes data into a ones or zeros wavefunction
        // as it adds up probability

        // Once it's finished, it picks one randomly, normalizes
        // then keeps that one as the new wavefunction
        wavefunction ones = make_wavefunction(_qubit_data.size() / 2);
        wavefunction zeros = make_wavefunction(_qubit_data.size() / 2);
        for (auto current_state = (_qubit_data).begin(); current_state != (_qubit_data).end(); ++current_state) {
            double square_amplitude = std::norm(current_state->second);
            if (current_state->first[target]) {
                one_probability += square_amplitude;
                ones.emplace(current_state->first, current_state->second);
            }
            else {
                zero_probability += square_amplitude;
                zeros.emplace(current_state->first, current_state->second);
            }
        }
        // Randomly select
        auto ran = _rng();
        unsigned result = (ran <= one_probability) ? 1 : 0;

        wavefunction &new_qubit_data = (result == 1) ? ones : zeros;
        // Create a new, normalized state
        double normalizer = 1.0 / std::sqrt((result == 1) ? one_probability : zero_probability);
        for (auto current_state = (new_qubit_data).begin(); current_state != (new_qubit_data).end(); ++current_state) {
            current_state->second *= normalizer;
        }
        _qubit_data = std::move(new_qubit_data);

        return result;
    }

    void Reset(size_t target)
    {
        double zero_probability = 0.0;
        double one_probability = 0.0;

        // Writes data into a ones or zeros wavefunction
        // as it adds up probability
        // Once it's finished, it picks one randomly, normalizes
        // then keeps that one as the new wavefunction

        // Used to set the qubit to 0 in the measured result
        qubit_label new_mask = qubit_label();
        new_mask.set(); // sets all bits to 1
        new_mask.set(target, 0);
        wavefunction ones = make_wavefunction(_qubit_data.size() / 2);
        wavefunction zeros = make_wavefunction(_qubit_data.size() / 2);
        for (auto current_state = (_qubit_data).begin(); current_state != (_qubit_data).end(); ++current_state) {
            double square_amplitude = std::norm(current_state->second);
            if (current_state->first[target]) {
                one_probability += square_amplitude;
                ones.emplace(current_state->first & new_mask, current_state->second);
            }
            else {
                zero_probability += square_amplitude;
                zeros.emplace(current_state->first & new_mask, current_state->second);
            }
        }
        // Randomly select
        bool result = (_rng() <= one_probability);

        wavefunction &new_qubit_data = result ? ones : zeros;
        // Create a new, normalized state
        double normalizer = 1.0 / std::sqrt((result) ? one_probability : zero_probability);
        for (auto current_state = (new_qubit_data).begin(); current_state != (new_qubit_data).end(); ++current_state) {
            current_state->second *= normalizer;
        }
        _qubit_data = std::move(new_qubit_data);
    }


    // Samples a state from the superposition with probably proportion to
    // the amplitude, returning a string of the bits of that state.
    // Unlike measurement, this does not modify the state
    std::string Sample() {
        double probability = _rng();
        for (auto current_state = (_qubit_data).begin(); current_state != (_qubit_data).end(); ++current_state) {
            double square_amplitude = std::norm(current_state->second);
            probability -= square_amplitude;
            if (probability <= 0) {
                return current_state->first.to_string();
            }
        }
        return _qubit_data.begin()->first.to_string();
    }

    void Assert(std::vector<Basis_Gate> const& axes, std::vector<size_t> const& qubits, bool result) {
        // Bit-vectors indexing where gates of each type are applied
        qubit_label XYs = 0;
        qubit_label YZs = 0;
        size_t ycount = 0;
        for (int i = 0; i < axes.size(); i++) {
            switch (axes[i]) {
            case Basis_Gate::PauliY:
                YZs.set(qubits[i]);
                XYs.set(qubits[i]);
                ycount++;
                break;
            case Basis_Gate::PauliX:
                XYs.set(qubits[i]);
                break;
            case Basis_Gate::PauliZ:
                YZs.set(qubits[i]);
                break;
            case Basis_Gate::PauliI:
                break;
            default:
                throw std::runtime_error("Bad Pauli basis");
            }
        }

        qcomplex_t phaseShift = result ? -1 : 1;
        // Each Y Pauli adds a global phase of i
        switch (ycount % 4) {
        case 1:
            phaseShift *= qcomplex_t(0, 1);
            break;
        case 2:
            phaseShift *= -1;
            break;
        case 3:
            phaseShift *= qcomplex_t(0, -1);
            break;
        default:
            break;
        }
        for (auto current_state = (_qubit_data).begin(); current_state != (_qubit_data).end(); ++current_state) {
            // The amplitude of current_state should always be non-zero, if the data structure
            // is properly maintained. Since the flipped state should match the amplitude (up to phase),
            // if the flipped state is not in _qubit_data, it implicitly has an ampltude of 0.0, which
            // is *not* a match, so the assertion should fail. 
            auto flipped_state = _qubit_data.find(current_state->first ^ XYs);
            if (flipped_state == _qubit_data.end() ||
                std::norm(flipped_state->second - current_state->second * (get_parity(current_state->first & YZs) ? -phaseShift : phaseShift)) > _precision) {
                qubit_label label = current_state->first;
                qcomplex_t val = current_state->second;
                std::cout << "Problematic state: " << label << "\n";
                std::cout << "Expected " << val * (get_parity(current_state->first & YZs) ? -phaseShift : phaseShift);
                std::cout << ", got " << (flipped_state == _qubit_data.end() ? 0.0 : flipped_state->second) << "\n";
                std::cout << "Wavefunction size: " << _qubit_data.size() << "\n";
                throw std::runtime_error("Not an eigenstate");
            }
        }
    }

    // Returns the probability of a given measurement in a Pauli basis
    // by decomposing each pair of computational basis states into eigenvectors
    // and adding the coefficients of the respective components
    double MeasurementProbability(std::vector<Basis_Gate> const& axes, std::vector<size_t> const& qubits) {
        // Bit-vectors indexing where gates of each type are applied
        qubit_label XYs = 0;
        qubit_label YZs = 0;
        size_t ycount = 0;
        for (int i = 0; i < axes.size(); i++) {
            switch (axes[i]) {
            case Basis_Gate::PauliY:
                YZs.set(qubits[i]);
                XYs.set(qubits[i]);
                ycount++;
                break;
            case Basis_Gate::PauliX:
                XYs.set(qubits[i]);
                break;
            case Basis_Gate::PauliZ:
                YZs.set(qubits[i]);
                break;
            case Basis_Gate::PauliI:
                break;
            default:
                throw std::runtime_error("Bad Pauli basis");
            }
        }
        qcomplex_t phaseShift = 1;

        // Each Y Pauli adds a global phase of i
        switch (ycount % 4) {
        case 1:
            phaseShift *= qcomplex_t(0, 1);
            break;
        case 2:
            phaseShift *= -1;
            break;
        case 3:
            phaseShift *= qcomplex_t(0, -1);
            break;
        default:
            break;
        }
        // Let P be the pauli operation, |psi> the state
        // projection = <psi|P|psi>

        // _qubit_data represents |psi> as sum_x a_x |x>,
        // where all |x> are orthonormal. Thus, the projection
        // will be the product of a_x and a_P(x), where P|x>=|P(x)>
        // Thus, for each |x>, we compute P(x) and look for that state
        // If there is a match, we add the product of their coefficients
        // to the projection, times a phase dependent on how many Ys and Zs match
        // the 1 bits of x
        qcomplex_t projection = 0.0;
        auto flipped_state = _qubit_data.end();
        for (auto current_state = (_qubit_data).begin(); current_state != (_qubit_data).end(); ++current_state) {
            flipped_state = _qubit_data.find(current_state->first ^ XYs); // no match returns _qubit_data.end()
            projection += current_state->second * (flipped_state == _qubit_data.end() ? 0 : std::conj(flipped_state->second)) * (get_parity(current_state->first & YZs) ? -phaseShift : phaseShift);
        }
        // The projector onto the -1 eigenspace (a result of "One") is 0.5 * (I - P)
        // So <psi| 0.5*(I - P)|psi> = 0.5 - 0.5*<psi|P|psi>
        // <psi|P|psi> should always be real so this only takes the real part
        return 0.5 - 0.5 * projection.real();
    }

    unsigned Measure(std::vector<Basis_Gate> const& axes, std::vector<size_t> const& qubits) {
        // Find a probability to get a specific result
        double probability = MeasurementProbability(axes, qubits);
        bool result = _rng() <= probability;
        if (!result)
            probability = 1 - probability;
        probability = std::sqrt(probability);
        // This step executes immediately so that we reduce the number of states in superposition
        pauli_combination(axes, qubits, 0.5 / probability, (result ? -0.5 : 0.5) / probability);
        return result;
    }


    // Probe the amplitude of a single basis state
    qcomplex_t probe(qubit_label const& label) {
        auto qubit = _qubit_data.find(label);
        // States not in the hash map are assumed to be 0
        if (qubit == _qubit_data.end()) {
            return qcomplex_t(0.0, 0.0);
        }
        else {
            return qubit->second;
        }
    }

    qcomplex_t probe(std::string const& label) {
        qubit_label bit_label = qubit_label(label);
        return probe(bit_label);
    }

    using callback_t = std::function<bool(const char*, double, double)>;
    // Dumps the state of a subspace of particular qubits, if they are not entangled
    // This requires it to detect if the subspace is entangled, construct a new 
    // projected wavefunction, then call the `callback` function on each state.
    bool dump_qubits(std::vector<size_t> const& qubits, callback_t const& callback) {
        // Create two wavefunctions
        // check if they are tensor products
        wavefunction dump_wfn;
        wavefunction leftover_wfn;
        if (!_split_wavefunction(_get_mask(qubits), dump_wfn, leftover_wfn)) {
            return false;
        }
        else {
            _dump_wavefunction_base(dump_wfn, [qubits, callback](qubit_label label, qcomplex_t val) -> bool {
                std::string masked(qubits.size(), '0');
                for (std::size_t i = 0; i < qubits.size(); ++i)
                    masked[i] = label[qubits[i]] ? '1' : '0';
                return callback(masked.c_str(), val.real(), val.imag());
            });
            return true;
        }
    }

    // Dumps all the states in superposition via a callback function
    void dump_all(size_t max_qubit_id, callback_t const& callback) {
        _dump_wavefunction_base(_qubit_data, [max_qubit_id, callback](qubit_label label, qcomplex_t val) -> bool {
            return callback(label.to_string().substr(num_qubits - 1 - max_qubit_id).c_str(), val.real(), val.imag());
        });
    }

    // Execute a queue of phase/permutation gates
    void phase_and_permute(std::list<operation> const &operation_list) {
        if (operation_list.size() == 0) { return; }

        // Condense the list into a memory-efficient vector with qubit labels
        // TODO: Is this still needed after multithreading is removed? Can we work off operation_list?
        std::vector<internal_operation> operation_vector;
        operation_vector.reserve(operation_list.size());

        for (auto op : operation_list) {
            switch (op.gate_type) {
            case OP::X:
            case OP::Y:
            case OP::Z:
                operation_vector.push_back(internal_operation(op.gate_type, op.target));
                break;
            case OP::MCX:
            case OP::MCY:
                operation_vector.push_back(internal_operation(op.gate_type, op.target, _get_mask(op.controls)));
                break;
            case OP::MCZ:
                operation_vector.push_back(internal_operation(op.gate_type, op.target, _get_mask(op.controls).set(op.target)));
                break;
            case OP::Phase:
                operation_vector.push_back(internal_operation(op.gate_type, op.target, op.phase));
                break;
            case OP::MCPhase:
                operation_vector.push_back(internal_operation(op.gate_type, op.target, _get_mask(op.controls).set(op.target), op.phase));
                break;
            case OP::SWAP:
                operation_vector.push_back(internal_operation(op.gate_type, op.target, op.target_2));
                break;
            case OP::MCSWAP:
                operation_vector.push_back(internal_operation(op.gate_type, op.target, _get_mask(op.controls), op.target_2));
                break;
            default:
                throw std::runtime_error("Unsupported operation");
                break;
            }
        }

        wavefunction new_qubit_data = make_wavefunction();

        // Iterates through and applies all operations
        for (auto current_state = _qubit_data.begin(); current_state != _qubit_data.end(); ++current_state) {
            qubit_label label = current_state->first;
            qcomplex_t val = current_state->second;
            // Iterate through vector of operations and apply each gate
            for (int i = 0; i < operation_vector.size(); i++) {
                auto &op = operation_vector[i];
                switch (op.gate_type) {
                case OP::X:
                    label.flip(op.target);
                    break;
                case OP::MCX:
                    if ((op.controls & label) == op.controls) {
                        label.flip(op.target);
                    }
                    break;
                case OP::Y:
                    label.flip(op.target);
                    val *= (label[op.target]) ? qcomplex_t(0, 1) : qcomplex_t(0, -1);
                    break;
                case OP::MCY:
                    if ((op.controls & label) == op.controls) {
                        label.flip(op.target);
                        val *= (label[op.target]) ? qcomplex_t(0, 1) : qcomplex_t(0, -1);
                    }
                    break;
                case OP::Z:
                    val *= (label[op.target] ? -1 : 1);
                    break;
                case OP::MCZ:
                    val *= ((op.controls & label) == op.controls) ? -1 : 1;
                    break;
                case OP::Phase:
                    val *= label[op.target] ? op.phase : 1;
                    break;
                case OP::MCPhase:
                    val *= ((op.controls & label) == op.controls) ? op.phase : 1;
                    break;
                case OP::SWAP:
                    if (label[op.target] != label[op.target_2]) {
                        label.flip(op.target);
                        label.flip(op.target_2);
                    }
                    break;
                case OP::MCSWAP:
                    if (((label & op.controls) == op.controls) && (label[op.target] != label[op.target_2])) {
                        label.flip(op.target);
                        label.flip(op.target_2);
                    }
                    break;
                default:
                    throw std::runtime_error("Unsupported operation");
                    break;
                }
            }
            // Insert the new state into the new wavefunction
            new_qubit_data.emplace(label, val);
        }
        _qubit_data = std::move(new_qubit_data);
        operation_vector.clear();
    }

    void R(Basis_Gate b, double phi, size_t index) {
        // Z rotation can be done in-place
        if (b == Basis_Gate::PauliZ) {
            qcomplex_t exp_0 = std::polar(1.0, -0.5*phi);
            qcomplex_t exp_1 = std::polar(1.0, 0.5*phi);
            for (auto current_state = (_qubit_data).begin(); current_state != (_qubit_data).end(); ++current_state) {
                current_state->second *= current_state->first[index] ? exp_1 : exp_0;
            }
        }
        else if (b == Basis_Gate::PauliX || b == Basis_Gate::PauliY) {
            qcomplex_t M00 = std::cos(phi / 2.0);
            qcomplex_t M01 = qcomplex_t(0, -1)*std::sin(0.5 * phi) * (b == Basis_Gate::PauliY ? qcomplex_t(0, -1) : 1);
            if (std::norm(M00) <= _rotation_precision) {
                // This is just a Y or X gate
                phase_and_permute(std::list<operation>{operation(b == Basis_Gate::PauliY ? OP::Y : OP::X, index)});
                return;
            }
            else if (std::norm(M01) <= _rotation_precision) {
                // just an identity
                return;
            }

            qcomplex_t M10 = M01 * (b == Basis_Gate::PauliY ? -1. : 1.);
            // Holds the amplitude of the new state to make it easier to check if it's non-zero
            qcomplex_t new_state;
            qubit_label flip(0);
            flip.set(index);
            wavefunction new_qubit_data = make_wavefunction();
            for (auto current_state = (_qubit_data).begin(); current_state != (_qubit_data).end(); ++current_state) {
                auto flipped_state = _qubit_data.find(current_state->first ^ flip);
                if (flipped_state == _qubit_data.end()) { // no matching value
                    if (current_state->first[index]) {// 1 on that qubit
                        new_qubit_data.emplace(current_state->first ^ flip, current_state->second * M01);
                        new_qubit_data.emplace(current_state->first, current_state->second * M00);
                    }
                    else {
                        new_qubit_data.emplace(current_state->first, current_state->second * M00);
                        new_qubit_data.emplace(current_state->first ^ flip, current_state->second * M10);
                    }
                }
                // Add up the two values, only when reaching the zero value
                else if (!(current_state->first[index])) {
                    new_state = current_state->second * M00 + flipped_state->second * M01; // zero state
                    if (std::norm(new_state) > _rotation_precision) {
                        new_qubit_data.emplace(current_state->first, new_state);
                    }
                    new_state = current_state->second * M10 + flipped_state->second * M00; // one state
                    if (std::norm(new_state) > _rotation_precision) {
                        new_qubit_data.emplace(flipped_state->first, new_state);
                    }
                }
            }
            _qubit_data = std::move(new_qubit_data);
        }
    }

    // Multi-controlled rotation
    void MCR(std::vector<size_t> const& controls, Basis_Gate b, double phi, size_t target) {
        qubit_label checks = _get_mask(controls);
        // A Z-rotation can be done without recreating the wavefunction
        if (b == Basis_Gate::PauliZ) {
            qcomplex_t exp_0 = std::polar(1.0, -0.5*phi);
            qcomplex_t exp_1 = std::polar(1.0, 0.5*phi);
            for (auto current_state = (_qubit_data).begin(); current_state != (_qubit_data).end(); ++current_state) {
                if ((current_state->first & checks) == checks) {
                    current_state->second *= current_state->first[target] ? exp_1 : exp_0;
                }
            }
        }
        // X or Y requires a new wavefunction
        else if (b == Basis_Gate::PauliX || b == Basis_Gate::PauliY) {
            qcomplex_t M00 = std::cos(0.5 * phi);
            qcomplex_t M01 = qcomplex_t(0, -1)*std::sin(0.5 * phi) * (b == Basis_Gate::PauliY ? qcomplex_t(0, -1) : 1);
            qcomplex_t M10 = (b == Basis_Gate::PauliY ? -1.0 : 1.0) * M01;

            if (std::norm(M00) <= _rotation_precision) {
                // This is just an MCY or MCX gate, but with a phase
                // So we need to preprocess with a multi-controlled phase
                if (b == Basis_Gate::PauliY) {
                    qcomplex_t phase = qcomplex_t(0, -1)*std::sin(0.5 * phi);
                    phase_and_permute(std::list<operation>{
                        operation(OP::MCPhase, controls[0], controls, phase),
                            operation(OP::MCY, target, controls)
                    });
                }
                else {
                    qcomplex_t phase = qcomplex_t(0, -1)*std::sin(0.5 * phi);
                    phase_and_permute(std::list<operation>{
                        operation(OP::MCPhase, controls[0], controls, phase),
                            operation(OP::MCX, target, controls)
                    });
                }
                return;
            }
            else if (std::norm(M01) <= _rotation_precision) {
                phase_and_permute(std::list<operation>{operation(OP::MCPhase, controls[0], controls, M00)});
                return;
            }

            qcomplex_t new_state;
            qubit_label flip(0);
            flip.set(target);
            wavefunction new_qubit_data = make_wavefunction();
            for (auto current_state = (_qubit_data).begin(); current_state != (_qubit_data).end(); ++current_state) {
                if ((current_state->first & checks) == checks) {
                    auto flipped_state = _qubit_data.find(current_state->first ^ flip);
                    if (flipped_state == _qubit_data.end()) { // no matching value
                        if (current_state->first[target]) {// 1 on that qubit
                            new_qubit_data.emplace(current_state->first ^ flip, current_state->second * M01);
                            new_qubit_data.emplace(current_state->first, current_state->second * M00);
                        }
                        else {
                            new_qubit_data.emplace(current_state->first, current_state->second * M00);
                            new_qubit_data.emplace(current_state->first ^ flip, current_state->second * M10);
                        }
                    }
                    // Add up the two values, only when reaching the zero val
                    else if (!(current_state->first[target])) {
                        new_state = current_state->second * M00 + flipped_state->second * M01; // zero state
                        if (std::norm(new_state) > _rotation_precision) {
                            new_qubit_data.emplace(current_state->first, new_state);
                        }
                        new_state = current_state->second * M10 + flipped_state->second * M00; // one state
                        if (std::norm(new_state) > _rotation_precision) {
                            new_qubit_data.emplace(flipped_state->first, new_state);
                        }
                    }
                }
                else {
                    new_qubit_data.emplace(current_state->first, current_state->second);
                }
            }
            _qubit_data = std::move(new_qubit_data);
        }
    }

    void H(size_t index) {
        // Initialize a new wavefunction, which will store the modified state
        // We initialize with twice as much space as the current one,
        // as this is the worst case result of an H gate
        wavefunction new_qubit_data = make_wavefunction(_qubit_data.size() * 2);
        // This label makes it easier to find associated labels (where the index is flipped)
        qubit_label flip(0);
        flip.set(index);
        // The amplitude for the new state
        qcomplex_t new_state;
        // Loops over all states in the wavefunction _qubit_data
        for (auto current_state = (_qubit_data).begin(); current_state != (_qubit_data).end(); ++current_state) {
            // An iterator pointing to the state labelled by the flip
            auto flipped_state = _qubit_data.find(current_state->first ^ flip);
            // Checks for whether it needs to add amplitudes from matching states
            // or create two new states
            if (flipped_state == _qubit_data.end()) { // no matching value
                new_qubit_data.emplace(current_state->first & (~flip), current_state->second * _normalizer);
                // Flip the value if the second bit, depending on whether the original had 1 or 0
                new_qubit_data.emplace(current_state->first | flip, current_state->second * (current_state->first[index] ? -_normalizer : _normalizer));
            }
            else if (!(current_state->first[index])) {
                new_state = current_state->second + flipped_state->second; // zero state
                if (std::norm(new_state) > _rotation_precision) {
                    new_qubit_data.emplace(current_state->first, new_state * _normalizer);
                }

                new_state = current_state->second - flipped_state->second; // one state
                if (std::norm(new_state) > _rotation_precision) {
                    new_qubit_data.emplace(current_state->first | flip, new_state * _normalizer);
                }
            }
        }
        // Moves the new data back into the old one (thus destroying
        // the old data)
        _qubit_data = std::move(new_qubit_data);
    }

    void MCH(std::vector<size_t> const& controls, size_t index) {
        wavefunction new_qubit_data = make_wavefunction(_qubit_data.size() * 2);
        qubit_label flip(0);
        flip.set(index);
        qcomplex_t new_state;
        qubit_label checks = _get_mask(controls);
        for (auto current_state = (_qubit_data).begin(); current_state != (_qubit_data).end(); ++current_state) {
            if ((checks & current_state->first) == checks) {
                auto flipped_state = _qubit_data.find(current_state->first ^ flip);
                if (flipped_state == _qubit_data.end()) { // no matching value
                    new_qubit_data.emplace(current_state->first & (~flip), current_state->second * _normalizer);
                    // Flip the value if the second bit, depending on whether the original had 1 or 0
                    new_qubit_data.emplace(current_state->first | flip, current_state->second * (current_state->first[index] ? -_normalizer : _normalizer));
                }
                else if (!(current_state->first[index])) {
                    new_state = current_state->second + flipped_state->second; // zero state
                    if (std::norm(new_state) > _rotation_precision) {
                        new_qubit_data.emplace(current_state->first, new_state * _normalizer);
                    }

                    new_state = current_state->second - flipped_state->second; // one state
                    if (std::norm(new_state) > _rotation_precision) {
                        new_qubit_data.emplace(current_state->first | flip, new_state * _normalizer);
                    }
                }
            }
            else {
                new_qubit_data.emplace(current_state->first, current_state->second);
            }
        }
        _qubit_data = std::move(new_qubit_data);
    }

    // Checks whether a qubit is 0 in all states in the superposition
    bool is_qubit_zero(size_t target) {
        for (auto current_state = _qubit_data.begin(); current_state != _qubit_data.end(); ++current_state) {
            if (current_state->first[target] && std::norm(current_state->second) > _precision) {
                return false;
            }
        }
        return true;
    }

    // Checks whether a qubit is classical
    // result.first is true iff it is classical
    // result.second holds its classical value if result.first == true
    std::pair<bool, bool> is_qubit_classical(size_t target) {
        bool value_found = false;
        bool value = false;
        for (auto current_state = _qubit_data.begin(); current_state != _qubit_data.end(); ++current_state) {
            if (std::norm(current_state->second) > _precision) {
                if (!value_found) {
                    value_found = true;
                    value = current_state->first[target];
                }
                else if (value != current_state->first[target])
                    return std::make_pair(false, false);
            }
        }
        return std::make_pair(true, value);
    }

    // Creates a new wavefunction hash map indexed by strings
    // Not intended for computations but as a way to transfer between
    // simulators templated with different numbers of qubits
    universal_wavefunction get_universal_wavefunction() {
        universal_wavefunction universal_qubit_data = universal_wavefunction(_qubit_data.bucket_count());
        for (auto current_state = _qubit_data.begin(); current_state != _qubit_data.end(); ++current_state) {
            universal_qubit_data.emplace(current_state->first.to_string(), current_state->second);
        }
        return universal_qubit_data;
    }


    void init_state(universal_wavefunction& new_qubit_data)
    {
        _qubit_data.clear();
        _qubit_data = wavefunction(new_qubit_data.size());
        for (auto current_state = new_qubit_data.begin(); current_state != new_qubit_data.end(); ++current_state) {
            _qubit_data.emplace(qubit_label(current_state->first), current_state->second);
        }

        /*for (auto &r : new_qubit_data)
        {
            auto bits = r.first;
            for (auto &q : _qubit_data)
            {
                if (q.first.to_string() == bits)
                {
                    q.second = r.second;
                }
            }

        }*/
        //_qubit_data(new_qubit_data);
        //_qubit_data = std::move(new_qubit_data);
    }

    // Returns the rng from this simulator
    std::function<double()> get_rng() { return _rng; }

private:
    // Internal type used to store operations with bitsets 
    // instead of vectors of qubit ids
    using internal_operation = condensed_operation<num_qubits>;

    // Hash table of the wavefunction
    wavefunction _qubit_data;

    // Internal random numbers
    std::function<double()> _rng;

    // Threshold to assert that something is zero when asserting it is 0
    double _precision = 1e-11;
    // Threshold at which something is zero when
    // deciding whether to add it into the superposition
    double _rotation_precision = 1e-11;

    // Normalizer for H and T gates (1/sqrt(2) as an amplitude)
    const qcomplex_t _normalizer = qcomplex_t(1.0, 0.0) / std::sqrt(2.0);

    // Used when allocating new wavefunctions
    float _load_factor = 0.9375;

    // Makes a wavefunction that is preallocated to the right size
    // and has the correct load factor
    wavefunction make_wavefunction() {
        wavefunction data((size_t)(_qubit_data.size() / _load_factor));
        data.max_load_factor(_load_factor);
        return data;
    }
    wavefunction make_wavefunction(uint64_t	 n_states) {
        wavefunction data((size_t)(n_states / _load_factor));
        data.max_load_factor(_load_factor);
        return data;
    }

    // Creates a qubit_label as a bit mask from a set of indices
    qubit_label _get_mask(std::vector<size_t> const& indices) {
        return get_mask<num_qubits>(indices);
    }

    // Split the wavefunction if separable, otherwise return false
    // Idea is that if we have a_bb|b1>|b2> as the first state, then for
    // any other state a_xx|x1>|x2>, we must also have a_xb|x1>|b2> and a_bx|b1>|x2>
    // in superposition.
    // Also, the coefficients must separate as a_bb=c_b*d_b and a_xx = c_x*d_x, implying
    // that a_xb = c_x*d_b and a_bx = c_b * d_x, and thus we can check this holds if
    // a_bb*a_xx = a_bx * a_xb. 
    // If this holds: we write (a_xx/a_bx)|x1> into the first wavefunction and (a_xx/a_xb)|x2>
    // into the second. 
    bool _split_wavefunction(qubit_label const& first_mask, wavefunction &wfn1, wavefunction &wfn2) {
        qubit_label second_mask = ~first_mask;
        // Guesses size
        wfn1 = wavefunction((int)std::sqrt(_qubit_data.size()));
        wfn2 = wavefunction((int)std::sqrt(_qubit_data.size()));
        // base_label_1 = b1 and base_label_2 = b2 in the notation above
        auto base_state = _qubit_data.begin();
        for (; base_state != _qubit_data.end() && std::norm(base_state->second) <= _precision; ++base_state);
        if (base_state == _qubit_data.end())
            throw std::runtime_error("Invalid state: All amplitudes are ~ zero.");
        qubit_label base_label_1 = base_state->first & first_mask;
        qubit_label base_label_2 = base_state->first & second_mask;
        // base_val = a_bb
        qcomplex_t base_val = base_state->second;
        double norm1 = 1., norm2 = 1.;
        wfn1[base_label_1] = 1.;
        wfn2[base_label_2] = 1.;
        std::size_t num_nonzero_states = 1;
        // From here on, base_state is |x1>|x2>
        ++base_state;
        for (; base_state != _qubit_data.end(); ++base_state) {
            qubit_label label_1 = base_state->first & first_mask;
            qubit_label label_2 = base_state->first & second_mask;
            // first_state is |x1>|b2>, second_state is |b1>|x2>
            auto first_state = _qubit_data.find(label_1 | base_label_2);
            auto second_state = _qubit_data.find(base_label_1 | label_2);
            // Ensures that both |x1>|b2> and |b1>|x2> are in the superposition
            if (first_state == _qubit_data.end() || second_state == _qubit_data.end()) {
                // state does not exist
                // therefore states are entangled
                return false;
            }
            else { // label with base label exists
             // Checks that a_bba_xx = a_xb*a_bx
                if (std::norm(first_state->second * second_state->second - base_val * base_state->second) > _precision) {
                    return false;
                }
                else {
                    if (std::norm(base_state->second) <= _precision)
                        continue;
                    num_nonzero_states++;
                    // Not entangled so far, save the two states, with amplitudes a_xx/a_bx and a_xx/a_xb, respectively
                    if (wfn1.find(label_1) == wfn1.end()) {
                        auto amp1 = base_state->second / second_state->second;
                        auto nrm = std::norm(amp1);
                        if (nrm > _precision)
                            wfn1[label_1] = amp1;
                        norm1 += nrm;
                    }
                    if (wfn2.find(label_2) == wfn2.end()) {
                        auto amp2 = base_state->second / first_state->second;
                        auto nrm = std::norm(amp2);
                        if (nrm > _precision)
                            wfn2[label_2] = amp2;
                        norm2 += nrm;
                    }
                }
            }
        }
        if (num_nonzero_states != wfn1.size()*wfn2.size())
            return false;
        // Normalize
        for (auto current_state = wfn1.begin(); current_state != wfn1.end(); ++current_state) {
            current_state->second *= 1. / std::sqrt(norm1);
        }
        for (auto current_state = wfn2.begin(); current_state != wfn2.end(); ++current_state) {
            current_state->second *= 1. / std::sqrt(norm2);
        }
        return true;
    }

    // Iterates through a wavefunction and calls the output function on each value
    // It first sorts the labels before outputting
    void _dump_wavefunction_base(wavefunction &wfn, std::function<bool(qubit_label, qcomplex_t)> output) {
        if (wfn.size() == 0) { return; }
        using pair_t = std::pair<qubit_label, qcomplex_t>;
        std::vector<pair_t> sortedByLabels;
        sortedByLabels.reserve(wfn.size());
        for (auto current_state = (wfn).begin(); current_state != (wfn).end(); ++current_state) {
            sortedByLabels.push_back(*current_state);
        }
        std::sort(
            sortedByLabels.begin(),
            sortedByLabels.end(),
            [](const pair_t& lhs, const pair_t& rhs) {return lhs.first < rhs.first; });
        qcomplex_t val;
        for (pair_t entry : sortedByLabels) {
            if (!output(entry.first, entry.second))
                break;
        }
    }

};


QPANDA_END

#endif  //!_SPARSE_STATE_H_