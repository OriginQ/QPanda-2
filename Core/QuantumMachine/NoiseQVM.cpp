#include "OriginQuantumMachine.h"
#include "NoiseQPU/NoiseCPUImplQPU.h"
#include "Core/Utilities/QProgInfo/MetadataValidity.h"
#include "Core/Utilities/Tools/TranformQGateTypeStringAndEnum.h"
#include "Core/Utilities/QProgTransform/TransformDecomposition.h"
#include "Core/Utilities/Tools/QPandaException.h"
#include "Core/QuantumMachine/QProgExecution.h"
#include "QPandaConfig.h"
#ifdef USE_MPI
#include "mpi.h"
#include "Core/Utilities/Tools/Uinteger.h"
#endif
USING_QPANDA
using namespace std;


NoiseQVM::NoiseQVM()
{

}

void NoiseQVM::init()
{
    try
    {
        _start();
        _pGates = new NoisyCPUImplQPU(m_quantum_noise);
        _ptrIsNull(_pGates, "NoisyCPUImplQPU");
    }
    catch (const std::exception&e)
    {
        finalize();
        QCERR(e.what());
        throw init_fail(e.what());
    }
}


void NoiseQVM::init(rapidjson::Document &)
{
    return ;
}

std::map<string, size_t> NoiseQVM::runWithConfiguration(QProg &prog, std::vector<ClassicalCondition> &cbits,
                                                          rapidjson::Document &doc, const NoiseModel&)
{
    QPANDA_ASSERT(doc.HasParseError() || !doc.HasMember("shots") || !doc["shots"].IsUint64(),
                  "runWithConfiguration param don't  have shots");
    size_t shots = doc["shots"].GetUint64();
    return runWithConfiguration(prog, cbits, shots);
}

double NoiseQVM::get_expectation(QProg prog, const QHamiltonian& hamiltonian, const QVec& qv, int shots)
{
    double total_expectation = 0;

    for (size_t i = 0; i < hamiltonian.size(); i++)
    {
        auto component = hamiltonian[i];
        if (component.first.empty())
        {
            total_expectation += component.second;
            continue;
        }

        QProg qprog;
        qprog << prog;

        QVec vqubit;
        vector<ClassicalCondition> vcbit;

        for (auto iter : component.first)
        {
            vqubit.push_back(qv[iter.first]);
            vcbit.push_back(cAlloc(iter.first));
            if (iter.second == 'X')
                qprog << H(qv[iter.first]);
            else if (iter.second == 'Y')
                qprog << RX(qv[iter.first], PI / 2);

        }
        for (auto i = 0; i < vqubit.size(); i++)
            qprog << Measure(vqubit[i], vcbit[i]);

        double expectation = 0;
        auto outcome = runWithConfiguration(qprog, vcbit, shots);
        size_t label = 0;
        for (auto iter : outcome)
        {
            label = 0;
            for (auto iter1 : iter.first)
            {
                if (iter1 == '1')
                    label++;
            }

            if (label % 2 == 0)
                expectation += iter.second * 1.0 / shots;
            else
                expectation -= iter.second * 1.0 / shots;
        }
        total_expectation += component.second * expectation;
    }

    return total_expectation;
}

std::map<std::string, size_t> NoiseQVM::runWithConfiguration(QProg& prog, std::vector<int>& cibts_addr, int shots, const NoiseModel&)
{
    std::vector<ClassicalCondition> cbits_vect;
	auto cmem = OriginCMem::get_instance();
	for (auto addr : cibts_addr)
		cbits_vect.push_back(cmem->get_cbit_by_addr(addr));

	return runWithConfiguration(prog, cbits_vect, shots);
}

std::map<std::string, size_t> NoiseQVM::runWithConfiguration(QProg& prog, int shots, const NoiseModel&)
{
    if (shots < 1)
        QCERR_AND_THROW(run_fail, "shots data error");

    TraversalConfig traver_param;
    QProgCheck prog_check;
    prog_check.execute(prog.getImplementationPtr(), nullptr, traver_param);

    auto measure_cbits_vector = traver_param.m_measure_cc;
    std::sort(measure_cbits_vector.begin(), measure_cbits_vector.end(), [&](CBit* a, CBit* b)
    {
        auto current_cbit_a_name = a->getName();
        auto current_cbit_b_name = b->getName();

        string current_cbit_a_number_str = current_cbit_a_name.substr(1);
        string current_cbit_b_number_str = current_cbit_b_name.substr(1);

        size_t current_a_cbit_addr = stoul(current_cbit_a_number_str);
        size_t current_b_cbit_addr = stoul(current_cbit_b_number_str);

        return current_a_cbit_addr < current_b_cbit_addr;
    });

    vector<ClassicalCondition> cbits_vector;
    for (auto cbit : measure_cbits_vector)
        cbits_vector.push_back(ClassicalCondition(cbit));

    return runWithConfiguration(prog, cbits_vector, shots);
}


std::map<string, size_t> NoiseQVM::runWithConfiguration(QProg &prog, std::vector<ClassicalCondition> &cbits, int shots, const NoiseModel&)
{
    auto qpu = dynamic_cast<NoisyCPUImplQPU *>(_pGates);
    QPANDA_ASSERT(nullptr == qpu, "Error: NoisyCPUImplQPU.");
    qpu->set_quantum_noise(m_quantum_noise);

    map<string, size_t> mResult;
    for (size_t i = 0; i < shots; i++)
    {
        run(prog);
        string sResult = _ResultToBinaryString(cbits);

        std::reverse(sResult.begin(), sResult.end());
        if (mResult.find(sResult) == mResult.end())
        {
            mResult[sResult] = 1;
        }
        else
        {
            mResult[sResult] += 1;
        }
    }

    return mResult;
}

std::map<std::string, bool> NoiseQVM::directlyRun(QProg &prog, const NoiseModel&)
{
    auto qpu = dynamic_cast<NoisyCPUImplQPU *>(_pGates);
    QPANDA_ASSERT(nullptr == qpu, "Error: NoisyCPUImplQPU.");
    qpu->set_quantum_noise(m_quantum_noise);

    run(prog);
    return _QResult->getResultMap();
}

void NoiseQVM::run(QProg & prog, const NoiseModel&)
{
    try
    {
        TraversalConfig config(m_rotation_angle_error);
        config.m_can_optimize_measure = false;

		//_pGates->initState(0, 1, _Qubit_Pool->get_max_usedqubit_addr() + 1);
        _pGates->initState(0, 1, prog.get_max_qubit_addr() + 1);
        QProgExecution prog_exec;
        prog_exec.execute(prog.getImplementationPtr(), nullptr, config, _pGates);

        std::map<string, bool>result;
        prog_exec.get_return_value(result);

        /* aiter has been used in line 120 */
        for (auto aiter : result)
        {
            _QResult->append(aiter);
        }
    }
    catch (const std::exception&e)
    {
        QCERR(e.what());
        throw run_fail(e.what());
    }

    return;
}

void NoiseQVM::set_noise_model(const NOISE_MODEL &model, const GateType &type, double prob)
{
    set_noise_model(model, type, prob, vector<QVec>());
    return ;
}

void NoiseQVM::set_noise_model(const NOISE_MODEL& model, const std::vector<GateType> &types, double prob)
{
	for (auto &type : types)
	{
		set_noise_model(model, type, prob, vector<QVec>());
	}
	return;
}

void NoiseQVM::set_noise_model(const NOISE_MODEL &model, const GateType &type, double prob, const std::vector<QVec> &qubits)
{
    size_t type_qubit_num = 0;
    if ((type >= GateType::P0_GATE && type <= U4_GATE)
        || GateType::I_GATE == type
        || GATE_TYPE_MEASURE == type
        || GATE_TYPE_RESET == type)

    {
        type_qubit_num = 1;
    }
    else if (type >= CU_GATE && type <= P11_GATE)
    {
        type_qubit_num = 2;
    }
    else
    {
        throw std::runtime_error("Error: noise qubit");
    }

    QuantumError quantum_error;
    quantum_error.set_noise(model, prob, type_qubit_num);

    vector<vector<size_t>> noise_qubits(qubits.size());
    for (size_t i = 0; i < qubits.size(); i++)
    {
        vector<size_t> addrs(qubits[i].size());
        for (size_t j = 0; j < qubits[i].size(); j++)
        {
            addrs[j] = qubits[i].at(j)->get_phy_addr();
        }
        noise_qubits[i] = addrs;
    }

    m_quantum_noise.add_quamtum_error(type, quantum_error, noise_qubits);
    return ;
}

void NoiseQVM::set_noise_model(const NOISE_MODEL &model, const GateType &type, double prob, const QVec &qubits)
{
    vector<QVec> noise_qubits;
    noise_qubits.reserve(qubits.size());
    for (auto &val : qubits)
    {
        noise_qubits.push_back({val});
    }
    set_noise_model(model, type, prob, noise_qubits);
}

void NoiseQVM::set_noise_model(const NOISE_MODEL &model, const std::vector<GateType> &types, double prob, const QVec &qubits)
{
	vector<QVec> noise_qubits;
	noise_qubits.reserve(qubits.size());
	for (auto &val : qubits)
	{
		noise_qubits.push_back({ val });
	}
	
	for (auto &type : types)
	{
		set_noise_model(model, type, prob, noise_qubits);
	}	
}

void NoiseQVM::set_noise_model(const NOISE_MODEL &model, const GateType &type,
                                 double T1, double T2, double t_gate)
{
    set_noise_model(model, type, T1, T2, t_gate, vector<QVec>());
    return ;
}

void NoiseQVM::set_noise_model(const NOISE_MODEL &model, const std::vector<GateType> &types,
	double T1, double T2, double t_gate)
{
	for (auto &type : types)
	{
		set_noise_model(model, type, T1, T2, t_gate, vector<QVec>());
	}

	return;
}

void NoiseQVM::set_noise_model(const NOISE_MODEL &model, const GateType &type, double T1, double T2, 
	double t_gate, const QVec &qubits)
{
    vector<QVec> noise_qubits;
    noise_qubits.reserve(qubits.size());
    for (auto &val : qubits)
    {
        noise_qubits.push_back({val});
    }

    set_noise_model(model, type, T1, T2, t_gate, noise_qubits);
    return ;
}

void NoiseQVM::set_noise_model(const NOISE_MODEL &model, const std::vector<GateType> &types, double T1, double T2,
	double t_gate, const QVec &qubits)
{
	vector<QVec> noise_qubits;
	noise_qubits.reserve(qubits.size());
	for (auto &val : qubits)
	{
		noise_qubits.push_back({ val });
	}

	for (auto &type : types)
	{
		set_noise_model(model, type, T1, T2, t_gate, noise_qubits);
	}

	return;
}

void NoiseQVM::set_noise_model(const NOISE_MODEL &model, const GateType &type, double T1, double T2, double t_gate,
                                 const std::vector<QVec> &qubits)
{
    size_t type_qubit_num = 0;
    if ((type >= GateType::P0_GATE && type <= U4_GATE)
        || GateType::I_GATE == type
        || GATE_TYPE_MEASURE == type
        || GATE_TYPE_RESET == type)
    {
        type_qubit_num = 1;
    }
    else if (type >= CU_GATE && type <= P11_GATE)
    {
        type_qubit_num = 2;
    }
    else
    {
        throw std::runtime_error("Error: noise qubit");
    }

    QuantumError quantum_error;
    quantum_error.set_noise(model, T1, T2, t_gate, type_qubit_num);

    vector<vector<size_t>> noise_qubits(qubits.size());
    for (size_t i = 0; i < qubits.size(); i++)
    {
        vector<size_t> addrs(qubits[i].size());
        for (size_t j = 0; j < qubits[i].size(); j++)
        {
            addrs[j] = qubits[i].at(j)->get_phy_addr();
        }
        noise_qubits[i] = addrs;
    }

    m_quantum_noise.add_quamtum_error(type, quantum_error, noise_qubits);
    return ;
}


void NoiseQVM::set_measure_error(const NOISE_MODEL &model, double prob, const QVec &qubits)
{
    set_noise_model(model, GATE_TYPE_MEASURE, prob, qubits);
    return ;
}

void NoiseQVM::set_measure_error(const NOISE_MODEL &model, double T1, double T2, double t_gate, const QVec &qubits)
{
    set_noise_model(model, GATE_TYPE_MEASURE, T1, T2, t_gate, qubits);
    return ;
}

void NoiseQVM::set_mixed_unitary_error(const GateType &type, const std::vector<QStat> &unitary_matrices, const std::vector<double> &probs)
{
    set_mixed_unitary_error(type, unitary_matrices, probs, vector<QVec>());
    return ;
}

void NoiseQVM::set_mixed_unitary_error(const GateType &type, const std::vector<QStat> &unitary_matrices, const std::vector<double> &probs, const QVec &qubits)
{
    vector<QVec> noise_qubits;
    noise_qubits.reserve(qubits.size());
    for (auto &val : qubits)
    {
        noise_qubits.push_back({val});
    }

    set_mixed_unitary_error(type, unitary_matrices, probs, noise_qubits);
    return ;
}

void NoiseQVM::set_mixed_unitary_error(const GateType &type, const std::vector<QStat> &unitary_matrices,
                                         const std::vector<double> &probs, const std::vector<QVec> &qubits)
{
    size_t type_qubit_num = 0;
    if ((type >= GateType::P0_GATE && type <= U4_GATE)
        || GateType::I_GATE == type
        || GATE_TYPE_MEASURE == type
        || GATE_TYPE_RESET == type)
    {
        type_qubit_num = 1;
    }
    else if (type >= CU_GATE && type <= P11_GATE)
    {
        type_qubit_num = 2;
    }
    else
    {
        throw std::runtime_error("Error: noise qubit");
    }

    QuantumError quantum_error;
    quantum_error.set_noise(MIXED_UNITARY_OPRATOR, unitary_matrices, probs, type_qubit_num);

    vector<vector<size_t>> noise_qubits(qubits.size());
    for (size_t i = 0; i < qubits.size(); i++)
    {
        vector<size_t> addrs(qubits[i].size());
        for (size_t j = 0; j < qubits[i].size(); j++)
        {
            addrs[j] = qubits[i].at(j)->get_phy_addr();
        }
        noise_qubits[i] = addrs;
    }

    m_quantum_noise.add_quamtum_error(type, quantum_error, noise_qubits);
    return ;
}


void NoiseQVM::set_reset_error(double p0, double p1, const QVec &qubits)
{
    QuantumError quantum_error;
    quantum_error.set_reset_error(p0, p1);

    vector<vector<size_t>> noise_qubits(qubits.size());
    for (size_t i = 0; i < qubits.size(); i++)
    {
        auto addr = qubits.at(i)->get_phy_addr();
        noise_qubits[i] = {addr};
    }

    m_quantum_noise.add_quamtum_error(GATE_TYPE_RESET, quantum_error, noise_qubits);
    return ;
}

void NoiseQVM::set_readout_error(const std::vector<std::vector<double> > &probs_list, const QVec &qubits)
{
    QPANDA_ASSERT(0 == qubits.size() && 2 != probs_list.size(), "Error: readout paramters.");
    if (0 == qubits.size())
    {
        QuantumError quantum_error;
        quantum_error.set_readout_error(probs_list, 1);
        m_quantum_noise.add_quamtum_error(GATE_TYPE_READOUT, quantum_error, vector<vector<size_t>>());
        return ;
    }

    for (size_t i = 0; i < qubits.size(); i++)
    {
        auto addr = qubits.at(i)->get_phy_addr();
        QuantumError quantum_error;
        //auto iter = probs_list.begin() + 2*i;
        auto iter = probs_list.begin();
        quantum_error.set_readout_error({iter, iter + 2}, 1);
        m_quantum_noise.add_quamtum_error(GATE_TYPE_READOUT, quantum_error, {{addr}});
    }

    return ;
}

void NoiseQVM::set_parallel_threads(size_t size) {

    if (size > 0)
    {
        _pGates->set_parallel_threads_size(size);
    }
    else
    {
        QCERR("_Set max thread is zero");
        throw qvm_attributes_error("_Set max thread is zero");
    }
}

void NoiseQVM::set_rotation_error(double error)
{
    m_rotation_angle_error = error;
}


REGISTER_QUANTUM_MACHINE(NoiseQVM);
