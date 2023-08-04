#include "Core/VirtualQuantumProcessor/SparseQVM/SparseQVM.h"

USING_QPANDA

static void swap_bool(std::vector<bool>::reference a, std::vector<bool>::reference b) 
{
    bool temp = a;
    a = b;
    b = temp;
}

void SparseSimulator::_start()
{
    _Qubit_Pool =
        QubitPoolFactory::GetFactoryInstance().
        GetPoolWithoutTopology(_Config.maxQubit);
    _ptrIsNull(_Qubit_Pool, "_Qubit_Pool");

    _CMem =
        CMemFactory::GetFactoryInstance().
        GetInstanceFromSize(_Config.maxCMem);

    _ptrIsNull(_CMem, "_CMem");

    _QResult =
        QResultFactory::GetFactoryInstance().
        GetEmptyQResult();

    _ptrIsNull(_QResult, "_QResult");

    _QMachineStatus =
        QMachineStatusFactory::
        GetQMachineStatus();

    _ptrIsNull(_QMachineStatus, "_QMachineStatus");
}

void SparseSimulator::finalize()
{
    if (nullptr != _AsyncTask)
    {
        _AsyncTask->wait();
        delete _AsyncTask;
    }

    if (nullptr != _Qubit_Pool)
    {
        delete _Qubit_Pool;
    }

    if (nullptr != _CMem)
    {
        delete _CMem;
    }

    if (nullptr != _QResult)
    {
        delete _QResult;
    }

    if (nullptr != _QMachineStatus)
    {
        delete _QMachineStatus;
    }

    if (nullptr != _pGates)
    {
        delete _pGates;
    }

    _Qubit_Pool = nullptr;
    _CMem = nullptr;
    _QResult = nullptr;
    _QMachineStatus = nullptr;
    _pGates = nullptr;
    _AsyncTask = nullptr;
    _ExecId = 0;
}

void SparseSimulator::setConfig(const Configuration& config)
{
    finalize();
    _Config.maxQubit = config.maxQubit;
    _Config.maxCMem = config.maxCMem;
    init();
}

void SparseSimulator::init()
{
    _start();
}

QVec SparseSimulator::allocateQubits(size_t qubitNumber)
{
    if (_Qubit_Pool == nullptr)
    {
        // check if the pointer is nullptr
        // Before init
        // After finalize
        QCERR("Must initialize the system first");
        throw(qvm_attributes_error("Must initialize the system first"));
    }

    if (qubitNumber + getAllocateQubitNum() > _Config.maxQubit)
    {
        QCERR("qubitNumber > maxQubit");
        throw(qalloc_fail("qubitNumber > maxQubit"));
    }

    try
    {
        QVec vQubit;

        for (size_t i = 0; i < qubitNumber; i++)
        {
            vQubit.push_back(_Qubit_Pool->allocateQubit());
        }
        return vQubit;
    }
    catch (const std::exception& e)
    {
        QCERR(e.what());
        throw(qalloc_fail(e.what()));
    }
}

QVec SparseSimulator::qAllocMany(size_t qubit_count)
{
    // Constructs a quantum state templated to the right number of qubits
    // and returns a pointer to it as a basic_quantum_state
    _quantum_state = construct_wfn_helper<MAX_QUBITS>(qubit_count);
    // Return the number of qubits this actually produces
    auto new_qubit_count = _quantum_state->get_num_qubits();
    // Initialize with no qubits occupied
    _occupied_qubits = std::vector<bool>(new_qubit_count, 0);
    _max_num_qubits_used = 0;
    _current_number_qubits_used = 0;

    _queue_Ry = std::vector<bool>(new_qubit_count, 0);
    _queue_Rx = std::vector<bool>(new_qubit_count, 0);
    _queue_H = std::vector<bool>(new_qubit_count, 0);
    _angles_Rx = std::vector<double>(new_qubit_count, 0.0);
    _angles_Ry = std::vector<double>(new_qubit_count, 0.0);
    return allocateQubits(qubit_count);
}

void SparseSimulator::dump_ids(void(*callback)(size_t))
{
    for (size_t qid = 0; qid < _occupied_qubits.size(); ++qid)
    {
        if (_occupied_qubits[qid])
        {
            callback((size_t)qid);
        }
    }
}

universal_wavefunction SparseSimulator::get_state()
{
    auto res = _quantum_state->get_universal_wavefunction();
    return res;
}

void SparseSimulator::set_state(universal_wavefunction& data)
{
    _quantum_state->init_state(data);
}

void SparseSimulator::handle_prog_to_queue(QProg &prog)
{
    flatten(prog);
    auto prog_node = prog.getImplementationPtr();
    for (auto itr = prog_node->getFirstNodeIter(); itr != prog_node->getEndNodeIter(); ++itr)
    {
        auto gate_tmp = std::dynamic_pointer_cast<QNode>(*itr);

        if ((*gate_tmp).getNodeType() == NodeType::MEASURE_GATE) {

            /*auto measure_node = std::dynamic_pointer_cast<AbstractQuantumMeasure>(gate_tmp);
            auto qbit = measure_node->getQuBit();
            auto phy_qbit = qbit->get_phy_addr();
            this->M(phy_qbit);*/
            continue;
        }
        auto gate_node = std::dynamic_pointer_cast<AbstractQGateNode>(gate_tmp);

        auto gate_type = gate_node->getQGate()->getGateType();

        QVec used_qv;
        QVec control_qv;
        std::vector<size_t> controls;
        gate_node->getControlVector(control_qv);
        auto qvtophy = [](QVec &control_qv, std::vector<size_t>& controls)
        {
            for (int i = 0; i < control_qv.size(); i++)
            {
                controls.push_back(control_qv[i]->get_phy_addr());
            }
        };
        switch (gate_type)
        {
        case GateType::HADAMARD_GATE:
        {

            gate_node->getQuBitVector(used_qv);
            qvtophy(control_qv, controls);
            this->MCH(controls, used_qv[0]->get_phy_addr());

            /* else
             {
                 gate_node->getQuBitVector(used_qv);
                 this->H(used_qv[0]->get_phy_addr());
             }*/



        }
        break;
        case GateType::PAULI_X_GATE:
        {
            qvtophy(control_qv, controls);
            gate_node->getQuBitVector(used_qv);
            this->MCX(controls, used_qv[0]->get_phy_addr());

        }
        break;
        case GateType::PAULI_Y_GATE:
        {
            gate_node->getQuBitVector(used_qv);
            qvtophy(control_qv, controls);
            this->MCY(controls, used_qv[0]->get_phy_addr());

        }
        break;
        case GateType::PAULI_Z_GATE:
        {
            gate_node->getQuBitVector(used_qv);
            qvtophy(control_qv, controls);
            this->MCZ(controls, used_qv[0]->get_phy_addr());

        }
        break;
        case GateType::S_GATE:
        {
            gate_node->getQuBitVector(used_qv);

            this->S(used_qv[0]->get_phy_addr());

        }
        break;
        case GateType::T_GATE:
        {
            gate_node->getQuBitVector(used_qv);

            this->T(used_qv[0]->get_phy_addr());

        }
        break;
        case GateType::RX_GATE:
        {
            gate_node->getQuBitVector(used_qv);
            auto gate_parameter = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter*>(gate_node->getQGate());
            double angle = gate_parameter->getParameter();
            if (gate_node->isDagger())
            {
                angle = -angle;
            }
            qvtophy(control_qv, controls);
            this->MCR(controls, Basis_Gate::PauliX, angle, used_qv[0]->get_phy_addr());

        }
        break;
        case GateType::P_GATE:
        {
            gate_node->getQuBitVector(used_qv);
            auto gate_parameter = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter*>(gate_node->getQGate());
            double angle = gate_parameter->getParameter();
            if (gate_node->isDagger())
            {
                angle = -angle;
            }
            qvtophy(control_qv, controls);
            qcomplex_t phase_angle;
            phase_angle.real(std::cos(angle));
            phase_angle.imag(1 * std::sin(angle));
            this->MCPhase(controls, phase_angle, used_qv[0]->get_phy_addr());

        }
        break;
        case GateType::RY_GATE:
        {
            gate_node->getQuBitVector(used_qv);
            auto gate_parameter = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter*>(gate_node->getQGate());
            double angle = gate_parameter->getParameter();
            if (gate_node->isDagger())
            {
                angle = -angle;
            }
            qvtophy(control_qv, controls);
            this->MCR(controls, Basis_Gate::PauliY, angle, used_qv[0]->get_phy_addr());

        }
        break;
        case GateType::RZ_GATE:
        {
            gate_node->getQuBitVector(used_qv);
            auto gate_parameter = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter*>(gate_node->getQGate());
            double angle = gate_parameter->getParameter();
            if (gate_node->isDagger())
            {
                angle = -angle;
            }
            qvtophy(control_qv, controls);
            this->MCR(controls, Basis_Gate::PauliZ, angle, used_qv[0]->get_phy_addr());

        }
        break;
        case GateType::CNOT_GATE:
        {
            gate_node->getQuBitVector(used_qv);

            qvtophy(control_qv, controls);
            controls.push_back(used_qv[0]->get_phy_addr());
            this->MCX(controls, used_qv[1]->get_phy_addr());

        }
        break;

        case GateType::TOFFOLI_GATE:
        {
            gate_node->getQuBitVector(used_qv);
            qvtophy(control_qv, controls);
            controls.push_back(used_qv[0]->get_phy_addr());
            this->MCX(controls, used_qv[1]->get_phy_addr());

        }
        break;

        case GateType::SWAP_GATE:
        {
            gate_node->getQuBitVector(used_qv);
            qvtophy(control_qv, controls);
            this->CSWAP(controls, used_qv[0]->get_phy_addr(), used_qv[1]->get_phy_addr());
        }
        break;

        case GateType::CPHASE_GATE:
        {
            gate_node->getQuBitVector(used_qv);
            auto gate_parameter = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter*>(gate_node->getQGate());
            double angle = gate_parameter->getParameter();
            if (gate_node->isDagger())
            {
                angle = -angle;
            }
            qvtophy(control_qv, controls);
            controls.push_back(used_qv[0]->get_phy_addr());
            qcomplex_t phase_angle;
            phase_angle.real(std::cos(angle));
            phase_angle.imag(1 * std::sin(angle));
            this->MCPhase(controls, phase_angle, used_qv[1]->get_phy_addr());
        }
        break;

        case GateType::U1_GATE:
        {
            gate_node->getQuBitVector(used_qv);
            auto gate_parameter = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter*>(gate_node->getQGate());
            double angle = gate_parameter->getParameter();
            if (gate_node->isDagger())
            {
                angle = -angle;
            }
            qvtophy(control_qv, controls);
            qcomplex_t phase_angle;
            phase_angle.real(std::cos(angle));
            phase_angle.imag(1 * std::sin(angle));
            this->MCPhase(controls, phase_angle, used_qv[0]->get_phy_addr());
        }
        break;

        }


    }
}

void SparseSimulator::DumpWavefunction(size_t indent)
{
    _execute_queued_ops();
    _quantum_state->dump_wavefunction(indent);
}

void SparseSimulator::DumpWavefunctionQuietly(size_t indent) {
    _quantum_state->dump_wavefunction(indent);
}

void SparseSimulator::set_random_seed(std::mt19937::result_type seed)
{
    _quantum_state->set_random_seed(seed);
}

size_t SparseSimulator::get_num_qubits() {
    return _quantum_state->get_num_qubits();
}

void SparseSimulator::allocate_specific_qubit(size_t qubit)
{
    size_t num_qubits = _quantum_state->get_num_qubits();
    // Checks that there are enough qubits
    if (qubit >= num_qubits) {
        // We create a new wavefunction and reallocate
        std::shared_ptr<BasicSparseState> old_state = _quantum_state;
        _quantum_state = expand_wfn_helper<MAX_QUBITS>(old_state, qubit + 1);

        num_qubits = _quantum_state->get_num_qubits();
        _occupied_qubits.resize(num_qubits, 0);
        _queue_Ry.resize(num_qubits, 0);
        _queue_Rx.resize(num_qubits, 0);
        _queue_H.resize(num_qubits, 0);
        _angles_Rx.resize(num_qubits, 0.0);
        _angles_Ry.resize(num_qubits, 0.0);
    }
    // The external qubit manager should prevent this, but this checks anyway
    if (_occupied_qubits[qubit]) {
        throw std::runtime_error("Qubit " + std::to_string(qubit) + " is already occupied");
    }
    // There is actually nothing to do to "allocate" a qubit, as every qubit
    // is already available for use with this data structure
}

bool SparseSimulator::release(size_t qubit_id)
{
    // Quick check if it's zero
    if (_occupied_qubits[qubit_id]) {
        // If not zero here, we must execute any remaining operations
        // Then check if the result is all zero
        _execute_queued_ops(qubit_id);
        auto is_classical = _quantum_state->is_qubit_classical(qubit_id);
        if (!is_classical.first) { // qubit isn't classical
            _quantum_state->Reset(qubit_id);
            _set_qubit_to_zero(qubit_id);
            return false;
        }
        else if (is_classical.second) {// qubit is in |1>
            X(qubit_id); // reset to |0> and release
            _execute_queued_ops(qubit_id);
        }
    }
    _set_qubit_to_zero(qubit_id);
    return true;
}

void SparseSimulator::X(size_t index)
{
    if (_queue_Ry[index]) {
        _angles_Ry[index] *= -1.0;
    }
    // Rx trivially commutes
    if (_queue_H[index]) {
        _queued_operations.push_back(operation(OP::Z, index));
        return;
    }
    _queued_operations.push_back(operation(OP::X, index));
    _set_qubit_to_nonzero(index);
}

void SparseSimulator::MCX(std::vector<size_t> const& controls, size_t  target)
{
    if (controls.size() == 0) {
        X(target);
        return;
    }
    // Check for anything on the controls
    if (controls.size() > 1) {
        _execute_if(controls);
    }
    else {
        // An H on the control but not the target forces execution
        if (_queue_Ry[controls[0]] || _queue_Rx[controls[0]] || (_queue_H[controls[0]] && !_queue_H[target])) {
            _execute_queued_ops(controls, OP::Ry);
        }
    }
    // Ry on the target causes issues
    if (_queue_Ry[target]) {
        _execute_queued_ops(target, OP::Ry);
    }
    // Rx on the target trivially commutes

    // An H on the target flips the operation
    if (_queue_H[target]) {
        // If it is a CNOT and there is also an H on the control, we swap control and target
        if (controls.size() == 1 && _queue_H[controls[0]]) {
            _queued_operations.push_back(operation(OP::MCX, controls[0], std::vector<size_t>{target}));
            _set_qubit_to_nonzero(controls[0]);
        }
        else {
            _queued_operations.push_back(operation(OP::MCZ, target, controls));
        }
        return;
    }
    // Queue the operation at this point
    _queued_operations.push_back(operation(OP::MCX, target, controls));
    _set_qubit_to_nonzero(target);
}

void SparseSimulator::MCApplyAnd(std::vector<size_t> const& controls, size_t  target)
{
    Assert(std::vector<Basis_Gate>{Basis_Gate::PauliZ}, std::vector<size_t>{target}, 0);
    MCX(controls, target);
}

void SparseSimulator::MCApplyAndAdj(std::vector<size_t> const& controls, size_t  target) {
    MCX(controls, target);
    Assert(std::vector<Basis_Gate>{Basis_Gate::PauliZ}, std::vector<size_t>{target}, 0);
    _set_qubit_to_zero(target);
}

void SparseSimulator::Y(size_t index)
{
    // XY = -YX
    if (_queue_Rx[index]) {
        _angles_Rx[index] *= -1.0;
    }
    // commutes with H up to phase, so we ignore the H queue
    _queued_operations.push_back(operation(OP::Y, index));
    _set_qubit_to_nonzero(index);
}

void SparseSimulator::MCY(std::vector<size_t> const& controls, size_t target) {
    if (controls.size() == 0) {
        Y(target);
        return;
    }
    _execute_if(controls);
    // Commutes with Ry on the target, not Rx
    if (_queue_Rx[target]) {
        _execute_queued_ops(target, OP::Rx);
    }
    // HY = -YH, so we add a phase to track this
    if (_queue_H[target]) {
        // The phase added does not depend on the target
        // Thus we use one of the controls as a target
        if (controls.size() == 1)
            _queued_operations.push_back(operation(OP::Z, controls[0]));
        else if (controls.size() > 1)
            _queued_operations.push_back(operation(OP::MCZ, controls[0], controls));
    }
    _queued_operations.push_back(operation(OP::MCY, target, controls));
    _set_qubit_to_nonzero(target);
}

void SparseSimulator::Z(size_t index)
{
    // ZY = -YZ
    if (_queue_Ry[index]) {
        _angles_Ry[index] *= -1;
    }
    // XZ = -ZX
    if (_queue_Rx[index]) {
        _angles_Rx[index] *= -1;
    }
    // HZ = XH
    if (_queue_H[index]) {
        _queued_operations.push_back(operation(OP::X, index));
        _set_qubit_to_nonzero(index);
        return;
    }
    // No need to modified _occupied_qubits, since if a qubit is 0
    // a Z will not change that
    _queued_operations.push_back(operation(OP::Z, index));
}

void SparseSimulator::MCZ(std::vector<size_t> const& controls, size_t target)
{
    if (controls.size() == 0) {
        Z(target);
        return;
    }
    // If the only thing on the controls is one H, we can switch
    // this to an MCX. Any Rx or Ry, or more than 1 H, means we
    // must execute.
    size_t count = 0;
    for (auto control : controls) {
        if (_queue_Ry[control] || _queue_Rx[control]) {
            count += 2;
        }
        if (_queue_H[control]) {
            count++;
        }
    }
    if (_queue_Ry[target] || _queue_Rx[target]) {
        count += 2;
    }
    if (_queue_H[target]) { count++; }
    if (count > 1) {
        _execute_queued_ops(controls, OP::Ry);
        _execute_queued_ops(target, OP::Ry);
    }
    else if (count == 1) {
        // Transform to an MCX, but we need to swap one of the controls
        // with the target if the Hadamard is on one of the control qubits
        std::vector<size_t> new_controls(controls);
        for (std::size_t i = 0; i < new_controls.size(); ++i) {
            if (_queue_H[new_controls[i]]) {
                std::swap(new_controls[i], target);
                break;
            }
        }
        _queued_operations.push_back(operation(OP::MCX, target, new_controls));
        _set_qubit_to_nonzero(target);
        return;
    }
    _queued_operations.push_back(operation(OP::MCZ, target, controls));
}

void SparseSimulator::Phase(qcomplex_t const& phase, size_t index)
{
    // Rx, Ry, and H do not commute well with arbitrary phase gates
    if (_queue_Ry[index] || _queue_Rx[index] || _queue_H[index]) {
        _execute_queued_ops(index, OP::Ry);
    }
    _queued_operations.push_back(operation(OP::Phase, index, phase));
}

void SparseSimulator::MCPhase(std::vector<size_t> const& controls, qcomplex_t const& phase, size_t target) {
    if (controls.size() == 0) {
        Phase(phase, target);
        return;
    }
    _execute_if(controls);
    _execute_if(target);
    _queued_operations.push_back(operation(OP::MCPhase, target, controls, phase));
}

void SparseSimulator::T(size_t index)
{
    Phase(qcomplex_t(_normalizer_double, _normalizer_double), index);
}

void SparseSimulator::AdjT(size_t index)
{
    Phase(qcomplex_t(_normalizer_double, -_normalizer_double), index);
}

void SparseSimulator::R1(double const& angle, size_t index)
{
    Phase(std::polar(1.0, angle), index);
}

void SparseSimulator::MCR1(std::vector<size_t> const& controls, double const& angle, size_t target)
{
    if (controls.size() > 0)
        MCPhase(controls, std::polar(1.0, angle), target);
    else
        R1(angle, target);
}

void SparseSimulator::R1Frac(std::int64_t numerator, std::int64_t power, size_t index)
{
    R1((double)numerator * pow(0.5, power)*M_PI, index);
}

void SparseSimulator::MCR1Frac(std::vector<size_t> const& controls, std::int64_t numerator, std::int64_t power, size_t target)
{
    if (controls.size() > 0)
        MCR1(controls, (double)numerator * pow(0.5, power) * M_PI, target);
    else
        R1Frac(numerator, power, target);
}

void SparseSimulator::S(size_t index)
{
    Phase(qcomplex_t(0, 1), index);
}

void SparseSimulator::AdjS(size_t index)
{
    Phase(qcomplex_t(0, -1), index);
}

void SparseSimulator::R(Basis_Gate b, double phi, size_t index)
{
    if (b == Basis_Gate::PauliI) {
        return;
    }

    // Tries to absorb the rotation into the existing queue,
    // if it hits a different kind of rotation, the queue executes
    if (b == Basis_Gate::PauliY) {
        _queue_Ry[index] = true;
        _angles_Ry[index] += phi;
        _set_qubit_to_nonzero(index);
        return;
    }
    else if (_queue_Ry[index]) {
        _execute_queued_ops(index, OP::Ry);
    }

    if (b == Basis_Gate::PauliX) {
        _queue_Rx[index] = true;
        _angles_Rx[index] += phi;
        _set_qubit_to_nonzero(index);
        return;
    }
    else if (_queue_Rx[index]) {
        _execute_queued_ops(index, OP::Rz);
    }

    // An Rz is just a phase
    if (b == Basis_Gate::PauliZ) {
        // HRz = RxH, but that's the wrong order for this structure
        // Thus we must execute the H queue
        if (_queue_H[index]) {
            _execute_queued_ops(index, OP::H);
        }
        // Rz(phi) = RI(phi)*R1(-2*phi)
        // Global phase from RI is ignored
        R1(phi, index);
    }
}

void SparseSimulator::MCR(std::vector<size_t> const& controls, Basis_Gate b, double phi, size_t target) {
    if (controls.size() == 0) {
        R(b, phi, target);
        return;
    }
    if (b == Basis_Gate::PauliI) {
        // Controlled I rotations are equivalent to controlled phase gates
        if (controls.size() > 1) {
            MCPhase(controls, std::polar(1.0, -0.5*phi), controls[0]);
        }
        else {
            Phase(std::polar(1.0, -0.5*phi), controls[0]);
        }
        return;
    }

    _execute_if(controls);
    // The target can commute with rotations of the same type
    if (_queue_Ry[target] && b != Basis_Gate::PauliY) {
        _execute_queued_ops(target, OP::Ry);
    }
    if (_queue_Rx[target] && b != Basis_Gate::PauliX) {
        _execute_queued_ops(target, OP::Rx);
    }
    if (_queue_H[target]) {
        _execute_queued_ops(target, OP::H);
    }
    // Execute any phase and permutation gates
    // These are not indexed by qubit so it does
    // not matter what the qubit argument is
    _execute_queued_ops(0, OP::PermuteLarge);
    _quantum_state->MCR(controls, b, phi, target);
    _set_qubit_to_nonzero(target);
}

void SparseSimulator::RFrac(Basis_Gate axis, std::int64_t numerator, std::int64_t power, size_t index)
{
    // Opposite sign convention
    R(axis, -(double)numerator * std::pow(0.5, power - 1)*M_PI, index);
}

void SparseSimulator::MCRFrac(std::vector<size_t> const& controls, Basis_Gate axis, std::int64_t numerator, std::int64_t power, size_t target) {
    // Opposite sign convention
    MCR(controls, axis, -(double)numerator * std::pow(0.5, power - 1) * M_PI, target);
}

void SparseSimulator::Exp(std::vector<Basis_Gate> const& axes, double angle, std::vector<size_t> const& qubits) {
    qcomplex_t cosAngle = std::cos(angle);
    qcomplex_t sinAngle = qcomplex_t(0, 1)*std::sin(angle);
    // This does not commute nicely with anything, so we execute everything
    _execute_queued_ops(qubits);
    _quantum_state->pauli_combination(axes, qubits, cosAngle, sinAngle);
    for (auto qubit : qubits) {
        _set_qubit_to_nonzero(qubit);
    }
}

void SparseSimulator::MCExp(std::vector<size_t> const& controls, std::vector<Basis_Gate> const& axes, double angle, std::vector<size_t> const& qubits) {
    if (controls.size() == 0) {
        Exp(axes, angle, qubits);
        return;
    }
    qcomplex_t cosAngle = std::cos(angle);
    qcomplex_t sinAngle = qcomplex_t(0, 1)*std::sin(angle);
    // This does not commute nicely with anything, so we execute everything
    _execute_queued_ops(qubits);
    _execute_queued_ops(controls);
    _quantum_state->MCPauliCombination(controls, axes, qubits, cosAngle, sinAngle);
    for (auto qubit : qubits) {
        _set_qubit_to_nonzero(qubit);
    }
}

void SparseSimulator::H(size_t index)
{
    // YH = -HY
    _angles_Ry[index] *= (_queue_Ry[index] ? -1.0 : 1.0);
    // Commuting with Rx creates a phase, but on the wrong side
    // So we execute any Rx immediately
    if (_queue_Rx[index]) {
        _execute_queued_ops(index, OP::Rx);
    }
    _queue_H[index] = !_queue_H[index];
    _set_qubit_to_nonzero(index);
}

void SparseSimulator::MCH(std::vector<size_t> const& controls, size_t target)
{
    if (controls.size() == 0) {
        H(target);
        return;
    }
    // No commutation on controls
    _execute_if(controls);
    // No Ry or Rx commutation on target
    if (_queue_Ry[target] || _queue_Rx[target]) {
        _execute_queued_ops(target, OP::Ry);
    }
    // Commutes through H gates on the target, so it does not check
    _execute_phase_and_permute();
    _quantum_state->MCH(controls, target);
    _set_qubit_to_nonzero(target);
}

void SparseSimulator::SWAP(size_t index_1, size_t index_2)
{
    // This is necessary for the "shift" to make sense
    if (index_1 > index_2) {
        std::swap(index_2, index_1);
    }
    // Everything commutes nicely with a swap
    swap_bool(_queue_Ry[index_1], _queue_Ry[index_2]);
    std::swap(_angles_Ry[index_1], _angles_Ry[index_2]);
    swap_bool(_queue_Rx[index_1], _queue_Rx[index_2]);
    std::swap(_angles_Rx[index_1], _angles_Rx[index_2]);
    swap_bool(_queue_H[index_1], _queue_H[index_2]);
    swap_bool(_occupied_qubits[index_1], _occupied_qubits[index_2]);
    size_t shift = index_2 - index_1;
    _queued_operations.push_back(operation(OP::SWAP, index_1, shift, index_2));
}

void SparseSimulator::CSWAP(std::vector<size_t> const& controls, size_t index_1, size_t index_2)
{
    if (controls.size() == 0) {
        SWAP(index_1, index_2);
        return;
    }
    if (index_1 > index_2) {
        std::swap(index_2, index_1);
    }
    // Nothing commutes nicely with a controlled swap
    _execute_if(controls);
    _execute_if(index_1);
    _execute_if(index_2);

    size_t shift = index_2 - index_1;
    _queued_operations.push_back(operation(OP::MCSWAP, index_1, shift, controls, index_2));
    // If either qubit is occupied, then set them both to occupied
    if (_occupied_qubits[index_1] || _occupied_qubits[index_2]) {
        _set_qubit_to_nonzero(index_1);
        _set_qubit_to_nonzero(index_2);
    }
}

prob_dict SparseSimulator::probRunDict(QProg &prog)
{
    handle_prog_to_queue(prog);
    _execute_queued_ops();

    auto m_state = _quantum_state->get_universal_wavefunction();
    QVec used_qv;
    auto size = prog.get_max_qubit_addr() + 1;
    prob_dict mResult;

    prob_vec probs;
    std::vector<std::string> key_data;
    for (auto current_state = (m_state).begin(); current_state != (m_state).end(); ++current_state) {
        double square_amplitude = std::norm(current_state->second);
        probs.push_back(square_amplitude);
        std::string str = current_state->first;
        std::reverse(str.begin(), str.end());
        key_data.push_back(str);

    }

    std::vector<int> key_count;
    for (int i = 0; i < key_data.size(); i++)
    {
        auto str = key_data[i].substr(0, size);
        std::reverse(str.begin(), str.end());
        mResult.insert(std::pair<std::string, double>(str, probs[i]));
    }


    return mResult;
}

std::map<std::string, size_t> SparseSimulator::runWithConfiguration(QProg& prog, std::vector<ClassicalCondition>& cbits, int shots)
{

    handle_prog_to_queue(prog);
    _execute_queued_ops();

    auto m_state = _quantum_state->get_universal_wavefunction();

    TraversalConfig traver_param;
    QProgCheck prog_check;
    prog_check.execute(prog.getImplementationPtr(), nullptr, traver_param);

    if (0 == traver_param.m_measure_qubits.size())
    {
        return std::map<std::string, size_t>();
    }

    if (shots < 1)
        QCERR_AND_THROW(run_fail, "shots data error");
    std::map<std::string, size_t> result_map;
    std::vector<double> random_nums(shots, 0);
    for (size_t i = 0; i < shots; i++)
    {
        random_nums[i] = random_generator19937();
    }
    std::sort(random_nums.begin(), random_nums.end(), [](double& a, double b) { return a > b; });
    prob_vec probs;
    Qnum qubits_nums = traver_param.m_measure_qubits;

    std::unordered_multimap<size_t, CBit*> qubit_cbit_map;
    for (size_t i = 0; i < traver_param.m_measure_cc.size(); i++)
    {
        qubit_cbit_map.insert({ traver_param.m_measure_qubits[i], traver_param.m_measure_cc[i] });
    }

    prob_vec probs_tmp;
    std::map<std::string, double> prob_map;
    std::vector<std::string> key_data;
    for (auto current_state = (m_state).begin(); current_state != (m_state).end(); ++current_state) {
        double square_amplitude = std::norm(current_state->second);
        probs.push_back(square_amplitude);
        std::string str = current_state->first;
        std::reverse(str.begin(), str.end());
        key_data.push_back(str);

    }

    std::vector<int> key_count;
    for (int i = 0; i < key_data.size(); i++)
    {
        key_data[i] = key_data[i].substr(0, cbits.size());
        prob_map.insert(std::pair<std::string, double>(key_data[i], probs[i]));
    }

    double p_sum = 0;

    for (auto current_prob = prob_map.begin(); current_prob != prob_map.end(); ++current_prob)
    {
        p_sum += current_prob->second;
        auto iter = random_nums.rbegin();
        while (iter != random_nums.rend() && *iter < p_sum)
        {
            random_nums.pop_back();
            iter = random_nums.rbegin();
            std::string result_bin_str = current_prob->first;
            std::reverse(result_bin_str.begin(), result_bin_str.end());
            if (result_map.find(result_bin_str) == result_map.end())
            {
                result_map[result_bin_str] = 1;
            }
            else
            {
                result_map[result_bin_str] += 1;
            }
        }

        if (0 == random_nums.size())
        {
            break;
        }
    }

    return result_map;
}

std::map<std::string, bool>  SparseSimulator::directlyRun(QProg& prog)
{
    std::map<std::string, bool> m_result;
    handle_prog_to_queue(prog);
    flatten(prog);
    auto prog_node = prog.getImplementationPtr();
    for (auto itr = prog_node->getFirstNodeIter(); itr != prog_node->getEndNodeIter(); ++itr)
    {
        auto gate_tmp = std::dynamic_pointer_cast<QNode>(*itr);

        if ((*gate_tmp).getNodeType() == NodeType::MEASURE_GATE)
        {
            auto measure_node = std::dynamic_pointer_cast<AbstractQuantumMeasure>(gate_tmp);
            auto qbit = measure_node->getQuBit();
            auto phy_qbit = qbit->get_phy_addr();
            auto iResult = this->M(phy_qbit);
            if (iResult < 0)
            {
                QCERR("result error");
            }
            CBit * cexpr = measure_node->getCBit();
            if (nullptr == cexpr)
            {
                QCERR("unknow error");
            }

            cexpr->set_val(iResult);
            std::string name = cexpr->getName();
            auto aiter = m_result.find(name);
            if (aiter != m_result.end())
            {
                aiter->second = (bool)iResult;
            }
            else
            {
                m_result.insert(std::pair<std::string, bool>(name, (bool)iResult));
            }
        }

    }

    return m_result;

}

unsigned SparseSimulator::M(size_t target)
{
    // Do nothing if the qubit is known to be 0
    if (!_occupied_qubits[target]) {
        return 0;
    }
    // If we get a measurement, we take it as soon as we can
    _execute_queued_ops(target, OP::Ry);
    // If we measure 0, then this resets the occupied qubit register
    unsigned res = _quantum_state->measure_single_qbit(target);
    if (res == 0)
        _set_qubit_to_zero(target);
    return res;
}

void  SparseSimulator::Reset(size_t target)
{
    if (!_occupied_qubits[target]) { return; }
    _execute_queued_ops(target, OP::Ry);
    _quantum_state->Reset(target);
    _set_qubit_to_zero(target);
}

void SparseSimulator::Assert(std::vector<Basis_Gate> axes, std::vector<size_t> const& qubits, bool result) {
    // Assertions will not commute well with Rx or Ry
    for (auto qubit : qubits) {
        if (_queue_Rx[qubit] || _queue_Ry[qubit])
            _execute_queued_ops(qubits, OP::Ry);
    }
    bool isEmpty = true;
    // Process each assertion by H commutation
    for (int i = 0; i < qubits.size(); i++) {
        switch (axes[i]) {
        case Basis_Gate::PauliY:
            // HY=-YH, so we switch the eigenvalue
            if (_queue_H[qubits[i]])
                result ^= 1;
            isEmpty = false;
            break;
        case Basis_Gate::PauliX:
            // HX = ZH
            if (_queue_H[qubits[i]])
                axes[i] = Basis_Gate::PauliZ;
            isEmpty = false;
            break;
        case Basis_Gate::PauliZ:
            // HZ = XH
            if (_queue_H[qubits[i]])
                axes[i] = Basis_Gate::PauliX;
            isEmpty = false;
            break;
        default:
            break;
        }
    }
    if (isEmpty) {
        return;
    }
    _execute_queued_ops(qubits, OP::PermuteLarge);
    _quantum_state->Assert(axes, qubits, result);
}

double SparseSimulator::MeasurementProbability(std::vector<Basis_Gate> const& axes, std::vector<size_t> const& qubits)
{
    _execute_queued_ops(qubits, OP::Ry);
    return _quantum_state->MeasurementProbability(axes, qubits);
}

unsigned SparseSimulator::Measure(std::vector<Basis_Gate> const& axes, std::vector<size_t> const& qubits)
{
    _execute_queued_ops(qubits, OP::Ry);
    unsigned result = _quantum_state->Measure(axes, qubits);
    // Switch basis to save space
    // Idea being that, e.g., HH = I, but if we know
    // that the qubit is in the X-basis, we can apply H
    // and execute, and this will send that qubit to all ones
    // or all zeros; then we leave the second H in the queue
    // Ideally we would also do that with Y, but HS would force execution,
    // rendering it pointless
    std::vector<size_t> measurements;
    for (int i = 0; i < axes.size(); i++) {
        if (axes[i] == Basis_Gate::PauliX) {
            H(qubits[i]);
            measurements.push_back(qubits[i]);
        }
    }
    _execute_queued_ops(measurements, OP::H);
    // These operations undo the previous operations, but they will be
    // queued
    for (int i = 0; i < axes.size(); i++) {
        if (axes[i] == Basis_Gate::PauliX) {
            H(qubits[i]);
        }
    }
    return result;
}

qcomplex_t SparseSimulator::probe(QProg& prog, std::string const& label)
{
    handle_prog_to_queue(prog);
    _execute_queued_ops();
    return _quantum_state->probe(label);
}

std::string SparseSimulator::Sample()
{
    _execute_queued_ops();
    return _quantum_state->Sample();
}