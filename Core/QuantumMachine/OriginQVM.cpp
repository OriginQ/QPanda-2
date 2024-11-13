/*
Copyright (c) 2017-2023 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "OriginQuantumMachine.h"
#include "Factory.h"
#include "Core/Utilities/QProgInfo/ConfigMap.h"
#include "QPandaConfig.h"
#include "VirtualQuantumProcessor/GPUImplQPU.h"
#include "VirtualQuantumProcessor/CPUImplQPU.h"
#include "VirtualQuantumProcessor/CPUImplQPUSingleThread.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include "Core/Utilities/Tools/QPandaException.h"
#include "Core/Utilities/Tools/Utils.h"
#include "Core/Utilities/QProgInfo/QuantumMetadata.h"
#include "Core/QuantumMachine/QProgExecution.h"
#include "Core/Utilities/Tools/Uinteger.h"
#include "Core/QuantumMachine/QProgCheck.h"
#include "Core/Utilities/QProgInfo/QProgProgress.h"
#include "Core/Utilities/Tools/QStatMatrix.h"
#include "Core/Utilities/Tools/QProgFlattening.h"
#include <set>
#include <thread>
#ifdef USE_OPENMP
#include <omp.h>
#endif


USING_QPANDA
using namespace std;

QuantumMachine* CPUQVM_Constructor()
{
	return new CPUQVM();
}

volatile QuantumMachineFactoryHelper _Quantum_Machine_Factory_Helper_CPUQVM(
	"CPUQVM",
	CPUQVM_Constructor
);

//REGISTER_QUANTUM_MACHINE(CPUQVM);
REGISTER_QUANTUM_MACHINE(CPUSingleThreadQVM);
REGISTER_QUANTUM_MACHINE(GPUQVM);

void QVM::setConfig(const Configuration& config)
{
	finalize();
	_Config.maxQubit = config.maxQubit;
	_Config.maxCMem = config.maxCMem;
	init();
}

void QVM::set_parallel_threads(size_t size){

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

void QVM::init() 
{
}

void QVM::initSparseState(const std::map<std::string, qcomplex_t> &sparse_state, const QVec &qlist)
{
    auto state = sparse_state_to_full_amplitude(sparse_state);

    return initState(state,qlist);
}


void QVM::initState(const QStat& state, const QVec& qlist)
{
	if (0 == qlist.size())
	{
        _pGates->initState(getAllocateQubitNum(), state);
	}
	else
	{
		auto qubit_alloc_size = getAllocateQubitNum();
		QPANDA_ASSERT(qlist.size() > qubit_alloc_size || (1ull << qlist.size()) != state.size(),
			"Error: initState state and qlist size.");
		set<size_t> qubit_set;
		for_each(qlist.begin(), qlist.end(), [&](Qubit* q) {
			qubit_set.insert(q->get_phy_addr());
		});
		QPANDA_ASSERT(qlist.size() != qubit_set.size(), "Error: initState state qlist.");

		QStat init_state(1ull << qubit_alloc_size, 0);
		for (int64_t i = 0; i < state.size(); i++)
		{
			int64_t index = i;
			int64_t init_state_index = 0;
			int64_t j = 0;

			do
			{
				auto base = index % 2;
				init_state_index += base * (1ll << qlist[j]->get_phy_addr());
				index >>= 1;
				j++;
			} while (index != 0);

			init_state[init_state_index] = state[i];
		}
		_pGates->initState(qubit_alloc_size, init_state);
	}

	return;
}

std::map<string, size_t> QVM::run_with_optimizing(QProg& prog, std::vector<ClassicalCondition>& cbits, int shots, TraversalConfig& traver_param)
{
	if (0 == traver_param.m_measure_qubits.size())
	{
		return map<string, size_t>();
	}

	map<string, size_t> result_map;
	//_pGates->initState(0, 1, _Qubit_Pool->get_max_usedqubit_addr() + 1);
	_pGates->initState(0, 1, prog.get_max_qubit_addr() +1);

	QProgExecution prog_exec;
	prog_exec.execute(prog.getImplementationPtr(), nullptr, traver_param, _pGates);
	map<size_t, size_t> result;

	vector<double> random_nums(shots, 0);
	for (size_t i = 0; i < shots; i++)
	{
		random_nums[i] = random_generator19937();
	}

	std::sort(random_nums.begin(), random_nums.end(), [](double& a, double b) { return a > b; });
	prob_vec probs;

	Qnum qubits_nums = traver_param.m_measure_qubits;
	_pGates->pMeasure(qubits_nums, probs);
	std::unordered_multimap<size_t, CBit*> qubit_cbit_map;
	for (size_t i = 0; i < traver_param.m_measure_cc.size(); i++)
	{
		qubit_cbit_map.insert({ traver_param.m_measure_qubits[i], traver_param.m_measure_cc[i] });
	}

	double p_sum = 0;
	for (size_t i = 0; i < probs.size(); i++)
	{
		if (probs[i] < DBL_EPSILON && probs[i] > -DBL_EPSILON)
		{
			continue;
		}

		p_sum += probs[i];
		auto iter = random_nums.rbegin();
		while (iter != random_nums.rend() && *iter < p_sum)
		{
			size_t measure_num = traver_param.m_measure_cc.size();
			auto bin_str_index = integerToBinary(i, measure_num);
			for (size_t j = 0; j < measure_num; j++)
			{
				auto qubit_idx = qubits_nums[j];
				auto mulit_iter = qubit_cbit_map.equal_range(qubit_idx);
				while (mulit_iter.first != mulit_iter.second)
				{
					auto cbit = mulit_iter.first->second;
					cbit->set_val(bin_str_index[measure_num - j - 1] - 0x30);
					_QResult->append({ cbit->getName(), cbit->getValue() });
					++mulit_iter.first;
				}
			}

			random_nums.pop_back();
			iter = random_nums.rbegin();

			string result_bin_str = _ResultToBinaryString(cbits);
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

std::map<string, size_t> QVM::run_with_normal(QProg& prog, std::vector<ClassicalCondition>& cbits, int shots, const NoiseModel& noise_model)
{
	map<string, size_t> mResult;
	for (size_t i = 0; i < shots; i++)
	{
		run(prog, noise_model);
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

Qubit* QVM::allocateQubit()
{
	if (_Qubit_Pool == nullptr)
	{
		// check if the pointer is nullptr
		// Before init
		// After finalize
		QCERR("Must initialize the system first");
		throw(qvm_attributes_error("Must initialize the system first"));
	}
	else
	{
		try
		{
			auto qubit = _Qubit_Pool->allocateQubit();
			if (nullptr == qubit)
			{
				throw qalloc_fail("allocateQubit error");
			}

			return qubit;
		}
		catch (const std::exception& e)
		{
			QCERR(e.what());
			throw(qalloc_fail(e.what()));
		}
	}
}

QVec QVM::allocateQubits(size_t qubitNumber)
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

ClassicalCondition QVM::allocateCBit()
{
	if (_CMem == nullptr)
	{
		// check if the pointer is nullptr
		// Before init
		// After finalize
		QCERR("Must initialize the system first");
		throw(qvm_attributes_error("Must initialize the system first"));
	}
	else
	{
		try
		{
			auto cbit = _CMem->Allocate_CBit();
			if (nullptr == cbit)
			{
				throw calloc_fail("cbitNumber > maxCMem");
			}

			ClassicalCondition temp(cbit);
			return temp;
		}
		catch (const std::exception& e)
		{
			QCERR(e.what());
			throw(calloc_fail(e.what()));
		}
	}
}


vector<ClassicalCondition> QVM::allocateCBits(size_t cbitNumber)
{
	if (_CMem == nullptr)
	{
		// check if the pointer is nullptr
		// Before init
		// After finalize
		QCERR("Must initialize the system first");
		throw(qvm_attributes_error("Must initialize the system first"));
	}
	else
	{
		if (cbitNumber + getAllocateCMemNum() > _Config.maxCMem)
		{
			QCERR("cbitNumber > maxCMem");
			throw(calloc_fail("cbitNumber > maxCMem"));
		}
		try
		{
			vector<ClassicalCondition> cbit_vector;
			for (size_t i = 0; i < cbitNumber; i++)
			{
				auto cbit = _CMem->Allocate_CBit();
				cbit_vector.push_back(cbit);
			}
			return cbit_vector;
		}
		catch (const std::exception& e)
		{
			QCERR(e.what());
			throw(calloc_fail(e.what()));
		}
	}
}

ClassicalCondition QVM::allocateCBit(size_t stCBitaddr)
{
	if (_CMem == nullptr)
	{
		// check if the pointer is nullptr
		// Before init
		// After finalize
		QCERR("Must initialize the system first");
		throw(qvm_attributes_error("Must initialize the system first"));
	}
	else
	{
		try
		{
			auto cbit = _CMem->Allocate_CBit(stCBitaddr);
			if (nullptr == cbit)
			{
				QCERR("stCBitaddr > maxCMem");
				throw calloc_fail("stCBitaddr > maxCMem");
			}
			ClassicalCondition temp(cbit);
			return temp;
		}
		catch (const std::exception& e)
		{
			QCERR(e.what());
			throw(calloc_fail(e.what()));
		}
	}
}

Qubit* QVM::allocateQubitThroughPhyAddress(size_t stQubitNum)
{
	if (_Qubit_Pool == nullptr)
	{
		// check if the pointer is nullptr
		// Before init
		// After finalize
		QCERR("Must initialize the system first");
		throw(qvm_attributes_error("Must initialize the system first"));
	}
	else
	{
		try
		{
			auto qubit = _Qubit_Pool->allocateQubitThroughPhyAddress(stQubitNum);
			if (nullptr == qubit)
			{
				throw qalloc_fail("qubits addr > _Config.maxQubit");
			}

			return qubit;
		}
		catch (const std::exception& e)
		{
			QCERR(e.what());
			throw(qalloc_fail(e.what()));
		}
	}
}

Qubit* QVM::allocateQubitThroughVirAddress(size_t qubit_num)
{
	if (nullptr == _Qubit_Pool)
	{
		QCERR("_Qubit_Pool is nullptr ,you must init global_quantum_machine at first");
		throw qvm_attributes_error("_Qubit_Pool is nullptr ,you must init global_quantum_machine at first");
	}
	return _Qubit_Pool->allocateQubitThroughVirAddress(qubit_num);
}

void QVM::Free_Qubit(Qubit* qubit)
{
	if (qubit == nullptr)
	{
		return;
	}
	this->_Qubit_Pool->Free_Qubit(qubit);
}

void QVM::Free_Qubits(QVec& vQubit)
{
	for (auto iter : vQubit)
	{
		if (iter == nullptr)
		{
			break;
		}
		this->_Qubit_Pool->Free_Qubit(iter);
		//delete iter;
		//iter = nullptr;
	}
}

void QVM::Free_CBit(ClassicalCondition& class_cond)
{
	auto cbit = class_cond.getExprPtr()->getCBit();
	if (nullptr == cbit)
	{
		QCERR("cbit is null");
		throw invalid_argument("cbit is null");
	}
	_CMem->Free_CBit(cbit);
}

void QVM::Free_CBits(vector<ClassicalCondition>& vCBit)
{
	for (auto iter : vCBit)
	{
		auto cbit = iter.getExprPtr()->getCBit();
		if (nullptr == cbit)
		{
			QCERR("cbit is null");
			throw invalid_argument("cbit is null");
		}
		this->_CMem->Free_CBit(cbit);
	}
}

void QVM::run(QProg& qprog, const NoiseModel& noise_model)
{
	try
	{
		TraversalConfig config(noise_model.rotation_error());
		config.m_can_optimize_measure = false;

		std::shared_ptr<AbstractQuantumProgram> qp = nullptr;
		if (noise_model.enabled())
		{
			/* generate simulate prog contains virtual noise gate */
			auto noise_qprog = NoiseProgGenerator().generate_noise_prog(noise_model, qprog.getImplementationPtr());
			qp = noise_qprog.getImplementationPtr();
		}
		else {
            QVec used_qv;
            this->get_allocate_qubits(used_qv);
            if (used_qv.size() > 18)
            {
                QProg prog = deepCopy(qprog);
                merge_QGate(prog);
                qp = prog.getImplementationPtr();
            }
            else
            {
                qp = qprog.getImplementationPtr();
            }
		}
		QPANDA_ASSERT(qp == nullptr, "Error: not valid quantum program");

		//_pGates->initState(0, 1, _Qubit_Pool->get_max_usedqubit_addr() + 1);
		_pGates->initState(0, 1, qp->get_max_qubit_addr() + 1);

		QProgExecution prog_exec;
		/* use QProgExecution object address(uniqe in process) as qprog process id _ExecId for recording execute progress */
		_ExecId = uint64_t(&prog_exec);
		QProgProgress::getInstance().prog_start(_ExecId);
		prog_exec.execute(qp, nullptr, config, _pGates);
		QProgProgress::getInstance().prog_end(_ExecId);
		std::map<string, bool>result;
		prog_exec.get_return_value(result);

		/* add readout error */
		if (noise_model.readout_error_enabled())
		{
			NoiseReadOutGenerator::append_noise_readout(noise_model, result);
		}

		/* aiter has been used in line 120 */
		for (auto aiter : result)
		{
			_QResult->append(aiter);
		}
	}
	catch (const std::exception& e)
	{
		QCERR(e.what());
		throw run_fail(e.what());
	}
}

void CPUQVM::run(QProg& qprog, const NoiseModel& noise_model)
{
    try
    {
        TraversalConfig config(noise_model.rotation_error());
        config.m_can_optimize_measure = false;

        std::shared_ptr<AbstractQuantumProgram> qp = nullptr;
        if (noise_model.enabled())
        {
            /* generate simulate prog contains virtual noise gate */
            auto noise_qprog = NoiseProgGenerator().generate_noise_prog(noise_model, qprog.getImplementationPtr());
            qp = noise_qprog.getImplementationPtr();
        }
        else {
            
            QVec used_qv;
            this->get_allocate_qubits(used_qv);
            if (used_qv.size() > 18)
            {
                QProg prog = deepCopy(qprog);
                merge_QGate(prog);
                qp = prog.getImplementationPtr();
            }
            else
            {
                 qp = qprog.getImplementationPtr();
            }
        }
        QPANDA_ASSERT(qp == nullptr, "Error: not valid quantum program");

        //_pGates->initState(0, 1, _Qubit_Pool->get_max_usedqubit_addr() + 1);
        _pGates->initState(0, 1, qp->get_max_qubit_addr() + 1);

        QProgExecution prog_exec;
        /* use QProgExecution object address(uniqe in process) as qprog process id _ExecId for recording execute progress */
        _ExecId = uint64_t(&prog_exec);
        QProgProgress::getInstance().prog_start(_ExecId);
        prog_exec.execute(qp, nullptr, config, _pGates);
        QProgProgress::getInstance().prog_end(_ExecId);
        std::map<string, bool>result;
        prog_exec.get_return_value(result);

        /* add readout error */
        if (noise_model.readout_error_enabled())
        {
            NoiseReadOutGenerator::append_noise_readout(noise_model, result);
        }

        /* aiter has been used in line 120 */
        for (auto aiter : result)
        {
            _QResult->append(aiter);
        }
    }
    catch (const std::exception& e)
    {
        QCERR(e.what());
        throw run_fail(e.what());
    }
}

QMachineStatus* QVM::getStatus() const
{
	if (nullptr == _QMachineStatus)
	{
		QCERR("_QMachineStatus is null");
		throw qvm_attributes_error("_QMachineStatus is null");
	}
	return _QMachineStatus;
}

QStat QVM::getQState() const
{
	if (nullptr == _pGates)
	{
		QCERR("pgates is nullptr");
		throw qvm_attributes_error("pgates is nullptr");
	}
	return _pGates->getQState();
}

size_t QVM::getVirtualQubitAddress(Qubit* qubit)const
{
	if (nullptr == qubit)
	{
		QCERR("qubit is nullptr");
		throw invalid_argument("qubit is nullptr");
	}

	if (nullptr == _Qubit_Pool)
	{
		QCERR("_Qubit_Pool is nullptr,you must init global_quantum_machine");
		throw qvm_attributes_error("_Qubit_Pool is nullptr,you must init global_quantum_machine");
	}

	return _Qubit_Pool->getVirtualQubitAddress(qubit);
}

bool QVM::swapQubitPhysicalAddress(Qubit* first_qubit, Qubit* second_qubit)
{
	if ((nullptr == first_qubit) || (nullptr == second_qubit))
	{
		return false;
	}

	auto first_addr = first_qubit->getPhysicalQubitPtr()->getQubitAddr();
	auto second_addr = second_qubit->getPhysicalQubitPtr()->getQubitAddr();

	first_qubit->getPhysicalQubitPtr()->setQubitAddr(second_addr);
	second_qubit->getPhysicalQubitPtr()->setQubitAddr(first_addr);
}

void QVM::set_random_engine(RandomEngine* rng)
{
	random_engine = rng;
}

QResult* QVM::getResult()
{
	if (nullptr == _QResult)
	{
		QCERR("_QResult is nullptr");
		throw qvm_attributes_error("_QResult is nullptr");
	}
	return _QResult;
}

void QVM::finalize()
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

size_t QVM::getAllocateQubit()
{
	if (nullptr == _Qubit_Pool)
	{
		QCERR("_QResult is nullptr");
		throw qvm_attributes_error("_QResult is nullptr");
	}
	return _Qubit_Pool->getMaxQubit() - _Qubit_Pool->getIdleQubit();
}

size_t QVM::getAllocateCMem()
{
	if (nullptr == _CMem)
	{
		QCERR("_CMem is nullptr");
		throw qvm_attributes_error("_CMem is nullptr");
	}
	return _CMem->getMaxMem() - _CMem->getIdleMem();
}

map<string, bool> QVM::getResultMap()
{
	if (nullptr == _QResult)
	{
		QCERR("QResult is null");
		throw qvm_attributes_error("QResult is null");
	}
	return _QResult->getResultMap();
}

prob_tuple IdealQVM::PMeasure(QVec qubit_vector, int select_max)
{
	if (0 == qubit_vector.size())
	{
		QCERR("the size of qubit_vector is zero");
		throw invalid_argument("the size of qubit_vector is zero");
	}

	if (nullptr == _pGates)
	{
		QCERR("_pGates is null");
		throw qvm_attributes_error("_pGates is null");
	}
	try
	{
		Qnum vqubit;
		for (auto aiter = qubit_vector.begin(); aiter != qubit_vector.end(); ++aiter)
		{
			vqubit.push_back((*aiter)->getPhysicalQubitPtr()->getQubitAddr());
		}

		prob_tuple result_vec;

		prob_vec pmeasure_vector;
		_pGates->pMeasure(vqubit, pmeasure_vector);

		for (auto i = 0; i < pmeasure_vector.size(); ++i)
		{
			result_vec.emplace_back(make_pair(i, pmeasure_vector[i]));
		}
		sort(result_vec.begin(), result_vec.end(),
			[=](std::pair<size_t, double> a, std::pair<size_t, double> b) {return a.second > b.second; });

		if ((select_max == -1) || (pmeasure_vector.size() <= select_max))
		{
			return result_vec;
		}
		else
		{
			result_vec.erase(result_vec.begin() + select_max, result_vec.end());
			return result_vec;
		}
	}
	catch (const std::exception& e)
	{
		QCERR(e.what());
		throw result_get_fail(e.what());
	}
}

prob_vec IdealQVM::PMeasure_no_index(QVec qubit_vector)
{
	if (0 == qubit_vector.size())
	{
		QCERR("the size of qubit_vector is zero");
		throw invalid_argument("the size of qubit_vector is zero");
	}

	if (nullptr == _pGates)
	{
		QCERR("_pGates is null");
		throw qvm_attributes_error("_pGates is null");
	}
	try
	{
		Qnum vqubit;
		for (auto aiter = qubit_vector.begin(); aiter != qubit_vector.end(); ++aiter)
		{
			vqubit.push_back((*aiter)->getPhysicalQubitPtr()->getQubitAddr());
		}

		prob_vec pmeasure_vector;
		_pGates->pMeasure(vqubit, pmeasure_vector);

		return pmeasure_vector;
	}
	catch (const std::exception& e)
	{
		QCERR(e.what());
		throw result_get_fail(e.what());
	}
}

void QVM::merge_QGate(QProg& src_prog)
{
    flatten(src_prog, true);
    /* for (auto gate_tmp = prog_node->getFirstNodeIter(); gate_tmp != prog_node->getEndNodeIter();gate_tmp++)
     {
         auto gate_itr = std::dynamic_pointer_cast<QNode>(*gate_tmp);
         copy_prog.insertQNode(copy_prog.getLastNodeIter(), std::dynamic_pointer_cast<QNode>(gate_itr));
     }*/
    _fusion_gate(src_prog, 1);
    _fusion_gate(src_prog, 2);
}

void  QVM::_fusion_gate(QProg& src_prog, const int fusion_bit)
{
    auto prog_node = src_prog.getImplementationPtr();
    for (auto itr = prog_node->getLastNodeIter(); itr != prog_node->getHeadNodeIter(); --itr)
    {
        if (itr == nullptr) {
            break;
        }

        auto gate_tmp = std::dynamic_pointer_cast<QNode>(*itr);
        if ((*gate_tmp).getNodeType() != NodeType::GATE_NODE) {
            continue;
        }

        auto gate_node = std::dynamic_pointer_cast<AbstractQGateNode>(gate_tmp);
        if (gate_node->getControlQubitNum() > 0) {
            continue;
        }
        if (gate_node->getQGate()->getGateType() == GateType::RXX_GATE ||
            gate_node->getQGate()->getGateType() == GateType::RYY_GATE ||
            gate_node->getQGate()->getGateType() == GateType::RZZ_GATE ||
            gate_node->getQGate()->getGateType() == GateType::RZX_GATE)
        {
            continue;
        }

        QVec qubit_vec;
        gate_node->getQuBitVector(qubit_vec);
        if (qubit_vec.size() != fusion_bit) {
            continue;
        }

        std::vector<NodeIter> fusing_gate_idxs = { itr };
        std::vector<int> fusing_qubits;
        for (const auto qbit : qubit_vec) {
            fusing_qubits.insert(fusing_qubits.end(), qbit->get_phy_addr());
        }

        /*2.Fuse gate with backwarding*/
        if (itr != prog_node->getLastNodeIter())
        {
            auto fusion_gate_itr = itr;
            ++fusion_gate_itr;
            for (; fusion_gate_itr != prog_node->getEndNodeIter(); ++fusion_gate_itr)
            {
                auto q_gate = std::dynamic_pointer_cast<QNode>(*fusion_gate_itr);
                if (q_gate->getNodeType() != NodeType::GATE_NODE) {
                    continue;
                }

                auto gate_tmp = std::dynamic_pointer_cast<AbstractQGateNode>(q_gate);
                if (gate_tmp->getControlQubitNum() > 0) {
                    break;
                }
                if (gate_tmp->getQGate()->getGateType() == GateType::RXX_GATE ||
                    gate_tmp->getQGate()->getGateType() == GateType::RYY_GATE ||
                    gate_tmp->getQGate()->getGateType() == GateType::RZZ_GATE ||
                    gate_tmp->getQGate()->getGateType() == GateType::RZX_GATE)
                {
                    continue;
                }

                auto &t_gate = gate_tmp;
                if (!_exclude_escaped_qubits(fusing_qubits, t_gate)) {
                    fusing_gate_idxs.push_back(fusion_gate_itr); /*All the qubits of tgt_op are in fusing_qubits*/
                }

                else if (fusing_qubits.empty()) {
                    break;
                }
            }
        }

        std::reverse(fusing_gate_idxs.begin(), fusing_gate_idxs.end());
        fusing_qubits.clear();
        for (auto &qbit : qubit_vec) {
            fusing_qubits.insert(fusing_qubits.end(), qbit->get_phy_addr());
        }

        /*3.fuse gate with forwarding */
        if (itr != prog_node->getFirstNodeIter())
        {
            auto fusion_gate_itr = itr;
            --fusion_gate_itr;
            for (; fusion_gate_itr != prog_node->getHeadNodeIter(); --fusion_gate_itr)
            {
                auto q_gate = std::dynamic_pointer_cast<QNode>(*fusion_gate_itr);
                if (q_gate->getNodeType() != NodeType::GATE_NODE) {
                    continue;
                }
                auto gate_tmp = std::dynamic_pointer_cast<AbstractQGateNode>(q_gate);
                if (gate_tmp->getControlQubitNum() > 0) {
                    break;
                }

                if (gate_tmp->getQGate()->getGateType() == GateType::RXX_GATE ||
                    gate_tmp->getQGate()->getGateType() == GateType::RYY_GATE ||
                    gate_tmp->getQGate()->getGateType() == GateType::RZZ_GATE ||
                    gate_tmp->getQGate()->getGateType() == GateType::RZX_GATE)
                {
                    continue;
                }

                auto &t_gate = gate_tmp;
                if (!_exclude_escaped_qubits(fusing_qubits, t_gate)) {
                    fusing_gate_idxs.push_back(fusion_gate_itr); /*All the qubits of tgt_op are in fusing_qubits*/
                }
                else if (fusing_qubits.empty()) {
                    break;
                }
            }
        }

        if (fusing_gate_idxs.size() <= 1) {
            continue;
        }

        /*4.generate a fused gate*/
        _allocate_new_operation(src_prog, itr, fusing_gate_idxs);
    }

}

bool QVM::_exclude_escaped_qubits(std::vector<int>& fusion_qubits,
    const QGate& tgt_op)  const
{

    bool included = true;
    QVec used_qv;
    tgt_op.getQuBitVector(used_qv);
    if (tgt_op.getControlQubitNum() > 0)
        return true;
    for (const auto qubit : used_qv) {
        included &= (std::find(fusion_qubits.begin(), fusion_qubits.end(), qubit->get_phy_addr()) != fusion_qubits.end());

    }

    if (included) {
        return false;
    }

    for (const auto op_qubit : used_qv) {
        auto found = std::find(fusion_qubits.begin(), fusion_qubits.end(), op_qubit->get_phy_addr());
        if (found != fusion_qubits.end())
            fusion_qubits.erase(found);
    }
    return true;
}

void QVM::_allocate_new_operation(QProg& prog, NodeIter& index_itr,
    std::vector<NodeIter>& fusing_gate_itrs)
{
    std::vector<QGate> fusion_gates;
    for (auto& itr : fusing_gate_itrs) {
        auto p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(*itr);
        fusion_gates.push_back(p_gate);
    }

    auto q_gate = _generate_operation(fusion_gates);
    prog.insertQNode(index_itr, std::dynamic_pointer_cast<QNode>(q_gate.getImplementationPtr()));
    index_itr++;
    for (auto &itr : fusing_gate_itrs) {
        prog.deleteQNode(itr);
    }
}

QGate QVM::_generate_operation(std::vector<QGate>& fusion_gates)
{
    std::set<int> fusioned_qubits;
    std::vector<QVec> tmp;
    for (auto &t_gate : fusion_gates)
    {
        QVec t_vec;
        t_gate.getQuBitVector(t_vec);
		for (int i = 0; i < t_vec.size(); i++) 
		{
			for (int j = i + 1; j < t_vec.size(); j++)
			{
				if (t_vec[i]->get_phy_addr() ==  t_vec[j]->get_phy_addr())
				{
					QCERR("have the same qubit ");
					throw invalid_argument("have the same qubit ");
				}
			}

			fusioned_qubits.insert(t_vec[i]->get_phy_addr());
		}
    }

    std::vector<int> remapped2orig(fusioned_qubits.begin(), fusioned_qubits.end());
    std::unordered_map<int, int> orig2remapped;
    std::vector<int> arg_qubits;

    arg_qubits.resize(fusioned_qubits.size(), 0);

    for (int i = 0; i < remapped2orig.size(); i++)
    {
        orig2remapped[remapped2orig[i]] = i;
        arg_qubits[i] = i;
    }
    std::map<int, Qubit*> tmp_map;
    QVec used_qv;
    this->get_allocate_qubits(used_qv);
    //qvm->get_allocate_qubits(used_qv);

    for (auto &it : used_qv)
    {
        tmp_map[it->get_phy_addr()] = it;
    }
    for (auto &op : fusion_gates)
    {
        QVec tmp_qv;
        op.getQuBitVector(tmp_qv);
        for (int i = 0; i < tmp_qv.size(); i++)
        {
            tmp_qv[i] = tmp_map[orig2remapped[tmp_qv[i]->get_phy_addr()]];

        }
        op.remap(tmp_qv);
    }

    if (arg_qubits.size() > 2)
    {
        auto fusioned_op = _generate_oracle_gate(fusion_gates, arg_qubits);
        QVec gate_qv;
        fusioned_op.getQuBitVector(gate_qv);

        for (auto &it : used_qv)
        {
            tmp_map[it->get_phy_addr()] = it;
        }
        for (size_t i = 0; i < gate_qv.size(); i++)
        {

            gate_qv[i] = tmp_map[remapped2orig[i]];
        }

        fusioned_op.remap(gate_qv);
        return fusioned_op;
    }
    else
    {
        auto fusioned_op = _generate_operation_internal(fusion_gates, arg_qubits);
        QVec gate_qv;
        fusioned_op.getQuBitVector(gate_qv);

        for (auto &it : used_qv)
        {
            tmp_map[it->get_phy_addr()] = it;
        }
        for (size_t i = 0; i < gate_qv.size(); i++)
        {

            gate_qv[i] = tmp_map[remapped2orig[i]];
        }

        fusioned_op.remap(gate_qv);
        return fusioned_op;
    }
}

QGate QVM::_generate_operation_internal(const std::vector<QGate> &fusion_gates,
    const std::vector<int> &qubits)
{
    CPUImplQPU<double> cpu;
    QStat state;
    cpu.initMatrixState(qubits.size() * 2, state);
    for (int i = 0; i < fusion_gates.size(); i++)
    {
        QStat matrix;
        fusion_gates[i].getQGate()->getMatrix(matrix);
        if (fusion_gates[i].isDagger()) {
            dagger(matrix);
        }

        QStat tmp_matrix;
        tmp_matrix.resize(16);
        QStat temp_init = { qcomplex_t(1,0), qcomplex_t(0,0),qcomplex_t(0,0),
        qcomplex_t(1,0) };

        QVec qubit_vector;
        fusion_gates[i].getQuBitVector(qubit_vector);
        Qubit* qubit = *(qubit_vector.begin());
        size_t bit = qubit->getPhysicalQubitPtr()->getQubitAddr();
        auto gate_type = fusion_gates[i].getQGate()->getGateType();

        std::vector<size_t> phy_qv;
        for (auto &i : qubit_vector) {
            phy_qv.push_back(i->get_phy_addr());
        }

        if (gate_type == GateType::ORACLE_GATE) {
            cpu.OracleGate(phy_qv, matrix, false);
        }
        else
        {
            if (qubit_vector.size() > 1) {
                cpu.double_qubit_gate_fusion(phy_qv[0], phy_qv[1], matrix);
            }
            else
            {
                if (qubit_vector.size() > 1) {
                    cpu.double_qubit_gate_fusion(qubits[0], qubits[1], matrix);
                }
                else
                {
                    if (qubits.size() > 1)
                    {
                        if (bit == qubits[0])
                        {
                            tmp_matrix = tensor(matrix, temp_init);
                        }
                        else
                        {
                            tmp_matrix = tensor(temp_init, matrix);
                        }
                        cpu.double_qubit_gate_fusion(qubits[0], qubits[1], tmp_matrix);
                    }
                    else
                    {
                        cpu.single_qubit_gate_fusion(bit, matrix);
                    }
                }
            }
        }

    }

    QStat data = cpu.getQState();
    QVec gate_qv;
    gate_qv.resize(qubits.size(), 0);
    std::map<int, Qubit*> tmp_map;
    QVec used_qv;
    this->get_allocate_qubits(used_qv);

    for (auto &qv : used_qv) {
        tmp_map[qv->get_phy_addr()] = qv;
    }

    for (int i = 0; i < qubits.size(); i++) {
        gate_qv[i] = tmp_map[qubits[i]];
    }

    if (gate_qv.size() > 1)
    {
        for (int j = 0; j < 4; j++)
        {
            qcomplex_t tmp = data[j * 4 + 1];
            data[j * 4 + 1] = data[j * 4 + 2];
            data[j * 4 + 2] = tmp;
        }
        for (int j = 0; j < 4; j++)
        {
            qcomplex_t tmp = data[4 + j];
            data[4 + j] = data[8 + j];
            data[8 + j] = tmp;
        }
        return QDouble(gate_qv[0], gate_qv[1], data);
    }
    else
    {
        return U4(gate_qv[0], data);
    }

}

QGate QVM::_generate_oracle_gate(const std::vector<QGate>& fusion_gates,
    const std::vector<int>& qubits)
{
    CPUImplQPU<double> cpu;
    QStat state;
    cpu.initMatrixState(qubits.size() * 2, state);
    for (int i = fusion_gates.size() - 1; i >= 0; i--)
    {
        QStat matrix;
        fusion_gates[i].getQGate()->getMatrix(matrix);
        if (fusion_gates[i].isDagger()) {
            dagger(matrix);
        }

        QVec qubit_vector;
        fusion_gates[i].getQuBitVector(qubit_vector);
        Qubit* qubit = *(qubit_vector.begin());
        size_t bit = qubit->getPhysicalQubitPtr()->getQubitAddr();
        auto gate_type = fusion_gates[i].getQGate()->getGateType();

        std::vector<size_t> phy_qv;
        for (auto &i : qubit_vector) {
            phy_qv.push_back(i->get_phy_addr());
        }

        if (gate_type == GateType::ORACLE_GATE) {
            cpu.OracleGate(phy_qv, matrix, false);
        }
        else
        {
            if (qubit_vector.size() == 1)
            {
                cpu.unitarySingleQubitGate(phy_qv[0], matrix, false, (GateType)gate_type);
            }
            else
            {
                if (gate_type == GateType::CNOT_GATE)
                {
                    cpu.unitaryDoubleQubitGate(phy_qv[0], phy_qv[1], matrix, false, (GateType)gate_type);
                }
                else
                {
                    cpu.three_qubit_gate_fusion(phy_qv[0], phy_qv[1], matrix);
                }
            }
        }
    }

    QStat data = cpu.getQState();
    QVec gate_qv;
    gate_qv.resize(qubits.size(), 0);
    std::map<int, Qubit*> tmp_map;
    QVec used_qv;

    this->get_allocate_qubits(used_qv);
    for (auto &qv : used_qv) {
        tmp_map[qv->get_phy_addr()] = qv;
    }
    for (int i = 0; i < qubits.size(); i++) {
        gate_qv[i] = tmp_map[qubits[i]];
    }

    return QOracle(gate_qv, data);
}

map<string, bool> QVM::directlyRun(QProg& qProg, const NoiseModel& noise_model)
{
	run(qProg, noise_model);
	return _QResult->getResultMap();
}

prob_tuple IdealQVM::getProbTupleList(QVec vQubit, int select_max)
{
	if (0 == vQubit.size())
	{
		QCERR("the size of qubit_vector is zero");
		throw invalid_argument("the size of qubit_vector is zero");
	}

	if (nullptr == _pGates)
	{
		QCERR("_pGates is null");
		throw qvm_attributes_error("_pGates is null");
	}

	try
	{
		return PMeasure(vQubit, select_max);
	}
	catch (const std::exception& e)
	{
		QCERR(e.what());
		throw result_get_fail(e.what());
	}
}

prob_vec IdealQVM::getProbList(QVec vQubit, int selectMax)
{
	if (0 == vQubit.size())
	{
		QCERR("the size of qubit_vector is zero");
		throw invalid_argument("the size of qubit_vector is zero");
	}

	if (nullptr == _pGates)
	{
		QCERR("_pGates is null");
		throw qvm_attributes_error("_pGates is null");
	}

	try
	{
		prob_vec vResult;
		Qnum vqubitAddr;
		for (auto aiter = vQubit.begin(); aiter != vQubit.end(); ++aiter)
		{
			vqubitAddr.push_back((*aiter)->getPhysicalQubitPtr()->getQubitAddr());
		}

		_pGates->pMeasure(vqubitAddr, vResult);

        if ((selectMax == -1) || (vResult.size() <= selectMax))
        {
            return vResult;
        }
        else
        {
            vResult.erase(vResult.begin() + selectMax, vResult.end());
            return vResult;
        }
	}
	catch (const std::exception& e)
	{
		QCERR(e.what());
		throw result_get_fail(e.what());
	}


}

string QVM::_ResultToBinaryString(vector<ClassicalCondition>& vCBit)
{
	string sTemp;
	if (nullptr == _QResult)
	{
		QCERR("_QResult is null");
		throw qvm_attributes_error("_QResult is null");
	}
	auto resmap = _QResult->getResultMap();
	for (auto c : vCBit)
	{
		auto cbit = c.getExprPtr()->getCBit();
		if (nullptr == cbit)
		{
			QCERR("vcbit is error");
			throw runtime_error("vcbit is error");
		}
		if (resmap[cbit->getName()])
		{
			sTemp.push_back('1');
		}
		else
		{
			sTemp.push_back('0');
		}
	}
	return sTemp;
}

void QVM::_ptrIsNull(void* ptr, std::string name)
{
	if (nullptr == ptr)
	{
		stringstream error;
		error << "alloc " << name << " fail";
		QCERR(error.str());
		throw bad_alloc();
	}
}

void QVM::_start()
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

prob_dict IdealQVM::getProbDict(QVec vQubit, int selectMax)
{
	if (0 == vQubit.size())
	{
		QCERR("the size of qubit_vector is zero");
		throw invalid_argument("the size of qubit_vector is zero");
	}

	prob_dict mResult;

	size_t stLength = vQubit.size();

    for (auto qv_idx = vQubit.begin(); qv_idx != vQubit.end(); qv_idx++)
    {
        auto result= count(vQubit.begin(), vQubit.end(), *qv_idx);
        
        if (result > 1)
        {
            QCERR("the getProbDict qubit_vector has duplicate members");
            throw invalid_argument("the getProbDict squbit_vector has duplicate members");
        }
   
    }
   
	auto vTemp = PMeasure(vQubit, selectMax);
	for (auto iter : vTemp)
	{
		mResult.insert(make_pair(dec2bin(iter.first, stLength), iter.second));
	}
	return mResult;
}

prob_tuple IdealQVM::
probRunTupleList(QProg& qProg, QVec vQubit, int selectMax)
{
	run(qProg);
	return getProbTupleList(vQubit, selectMax);
}

prob_vec IdealQVM::
probRunList(QProg& qProg, QVec vQubit, int selectMax)
{
	run(qProg);
	return getProbList(vQubit, selectMax);
}

//vector<prob_vec> IdealQVM::
//probRunList(vector<QProg>& QProgs, vector<QVec> &vQubits, vector<int> &selectMaxs,int threads)
//{
//	size_t size = QProgs.size();
//	vector<prob_vec> results(size);
//	size_t thread = omp_get_max_threads();
//	if (thread > threads)
//	{
//		run_fail("Select threads overflow");
//	}
//#pragma omp parallel for num_threads(threads)
//	for (int i = 0; i < QProgs.size(); ++i) 
//	{
//		run(QProgs[i]);
//		results[i] = getProbList(vQubits[i], selectMaxs[i]);
//	}
//	
//	return results;
//}

prob_dict IdealQVM::
probRunDict(QProg& qProg, QVec vQubit, int selectMax)
{
	run(qProg);
	return getProbDict(vQubit, selectMax);
}

//vector<prob_dict> IdealQVM::
//probRunDict(vector<QProg>& QProgs, vector<QVec>& vQubits, vector<int>& selectMaxs, int threads)
//{
//	size_t size = QProgs.size();
//	vector<prob_dict> results(size);
//	size_t thread = omp_get_max_threads();
//	if (thread > threads) 
//	{
//		run_fail("Select threads overflow");
//	}
//#pragma omp parallel for num_threads(threads)
//	for (int i = 0; i < QProgs.size(); ++i)
//	{
//		run(QProgs[i]);
//		results[i] = getProbDict(vQubits[i], selectMaxs[i]);
//	}
//
//	return results;
//}

prob_tuple IdealQVM::probRunTupleList(QProg& qProg, const std::vector<int>& qubits_addr, int selectMax)
{
	return probRunTupleList(qProg, get_qubits_by_phyaddrs(qubits_addr), selectMax);
}

prob_vec IdealQVM::probRunList(QProg& qProg, const std::vector<int>& qubits_addr, int selectMax)
{
	return probRunList(qProg, get_qubits_by_phyaddrs(qubits_addr), selectMax);
}

prob_dict IdealQVM::probRunDict(QProg& qProg, const std::vector<int>& qubits_addr, int selectMax)
{
	return probRunDict(qProg, get_qubits_by_phyaddrs(qubits_addr), selectMax);
}


map<string, size_t> QVM::
runWithConfiguration(QProg& qProg, vector<ClassicalCondition>& vCBit, int shots, const NoiseModel& noise_model)
{
	rapidjson::Document doc;
	doc.Parse("{}");
	auto& alloc = doc.GetAllocator();
	doc.AddMember("shots", shots, alloc);
	return runWithConfiguration(qProg, vCBit, doc, noise_model);
}

map<string, size_t> QVM::
runWithConfiguration(QProg& qProg, vector<int>& cibts_addr, int shots, const NoiseModel& noise_model)
{
	vector<ClassicalCondition> cbits_vect;
	auto cmem = OriginCMem::get_instance();
	for (auto addr : cibts_addr)
		cbits_vect.push_back(cmem->get_cbit_by_addr(addr));

	return runWithConfiguration(qProg, cbits_vect, shots, noise_model);
}

map<string, size_t> QVM::
runWithConfiguration(QProg& prog, int shots, const NoiseModel& noise_model)
{
    if (shots < 1)
        QCERR_AND_THROW(run_fail,"shots data error");

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

    if (traver_param.m_can_optimize_measure && shots > 1 && !noise_model.enabled() && !noise_model.readout_error_enabled())
    {
        return run_with_optimizing(prog, cbits_vector, shots, traver_param);
    }
    else
    {
        return run_with_normal(prog, cbits_vector, shots, noise_model);
    }
}



map<string, size_t> QVM::
runWithConfiguration(QProg& qProg, vector<ClassicalCondition>& vCBit, rapidjson::Document& param, const NoiseModel& noise_model)
{
	if (!param.HasMember("shots"))
	{
		QCERR("OriginCollection don't  have shots");
		throw run_fail("runWithConfiguration param don't  have shots");
	}
	int shots = 0;

	if (param["shots"].IsInt())
	{
		shots = param["shots"].GetInt();
	}
	else
	{
		QCERR("shots data type error");
		throw run_fail("shots data type error");
	}

	if (shots < 1)
	{
		QCERR("shots data error");
		throw run_fail("shots data error");
	}

	TraversalConfig traver_param;
	QProgCheck prog_check;
	prog_check.execute(qProg.getImplementationPtr(), nullptr, traver_param);

	if (traver_param.m_can_optimize_measure && shots > 1 && !noise_model.enabled() && !noise_model.readout_error_enabled())
	{
		return run_with_optimizing(qProg, vCBit, shots, traver_param);
	}
	else
	{
		return run_with_normal(qProg, vCBit, shots, noise_model);
	}
}


double QVM::get_expectation(QProg prog, const QHamiltonian& hamiltonian, const QVec& qv)
{
	double total_expectation = 0;
	directlyRun(prog);
	auto qstate = getQState();
	initState(qstate);

	auto _parity_check = [](size_t number)
	{
		bool label = true;
		size_t i = 0;
		while ((number >> i) != 0)
		{
			if ((number >> i) % 2 == 1)
				label = !label;

			++i;
		}
		return label;
	};

	for (size_t i = 0; i < hamiltonian.size(); i++)
	{
		auto component = hamiltonian[i];
		if (component.first.empty())
		{
			total_expectation += component.second;
			continue;
		}

		QProg qprog;
		Qnum vqubit;
		for (auto iter : component.first)
		{
			vqubit.push_back(qv[iter.first]->get_phy_addr());
			if (iter.second == 'X')
				qprog << H(qv[iter.first]);
			else if (iter.second == 'Y')
				qprog << RX(qv[iter.first], PI / 2);
		}

		directlyRun(qprog);
		prob_vec pmeasure_vector;
		_pGates->pMeasure(vqubit, pmeasure_vector);

		double expectation = 0;
#pragma omp parallel for reduction(+:expectation)
		for (auto i = 0; i < pmeasure_vector.size(); i++)
		{
			if (_parity_check(i))
				expectation += pmeasure_vector[i];
			else
				expectation -= pmeasure_vector[i];
		}

		total_expectation += component.second * expectation;
	}

	initState();
	return total_expectation;
}

void QVM::get_expectation_vector(QProg prog, const QHamiltonian& hamiltonian, const QVec& qv, vector_d& measure_probably)
{
	directlyRun(prog);
	auto qstate = getQState();
	initState(qstate);

	auto _parity_check = [](size_t number)
	{
		bool label = true;
		size_t i = 0;
		while ((number >> i) != 0)
		{
			if ((number >> i) % 2 == 1)
				label = !label;

			++i;
		}
		return label;
	};


	for (size_t i = 0; i < hamiltonian.size(); i++)
	{
		auto component = hamiltonian[i];
		if (component.first.empty())
		{
			measure_probably[i] = 1.0;
			continue;
		}

		QProg qprog;
		Qnum vqubit;
		for (auto iter : component.first)
		{
			vqubit.push_back(qv[iter.first]->get_phy_addr());
			if (iter.second == 'X')
				qprog << H(qv[iter.first]);
			else if (iter.second == 'Y')
				qprog << RX(qv[iter.first], PI / 2);
		}

		directlyRun(qprog);
		prob_vec pmeasure_vector;
		_pGates->pMeasure(vqubit, pmeasure_vector);

		double expectation = 0;
#pragma omp parallel for reduction(+:expectation)
		for (auto i = 0; i < pmeasure_vector.size(); i++)
		{
			if (_parity_check(i))
			{
				expectation += pmeasure_vector[i];
			}
			else
			{
				expectation -= pmeasure_vector[i];
			}
		}
		measure_probably[i] = expectation;
	}
	initState();
}

double QVM::get_expectation(QProg prog, const QHamiltonian& hamiltonian, const QVec& qv, int shots)
{
	double total_expectation = 0;
	directlyRun(prog);
	auto qstate = getQState();
	initState(qstate);

	for (size_t i = 0; i < hamiltonian.size(); i++)
	{
		auto component = hamiltonian[i];
		if (component.first.empty())
		{
			total_expectation += component.second;
			continue;
		}
		QProg qprog;
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
	initState();
	return total_expectation;
}

void QVM::get_expectation_vector(QProg prog, const QHamiltonian& hamiltonian, const QVec& qv, vector_d& measure_probably, int shots)
{
	directlyRun(prog);
	auto qstate = getQState();
	initState(qstate);

	for (size_t i = 0; i < hamiltonian.size(); i++)
	{
		auto component = hamiltonian[i];
		if (component.first.empty())
		{
			measure_probably[i] = 1.0;
			continue;
		}
		QProg qprog;
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
		auto comp_tmp = component.first;

		size_t label = 0;
		for (auto iter : outcome)
		{
			double tmp1 = 0;
			label = 0;
			for (auto iter1 : iter.first)
			{
				if (iter1 == '1')
					label++;
			}

			tmp1 = iter.second * 1.0 / shots;
			if (label % 2 == 0)
			{
				expectation += tmp1;
			}
			else
			{
				expectation -= tmp1;
			}
		}
		measure_probably[i] = expectation;
	}
	initState();
}

size_t QVM::get_processed_qgate_num()
{
	return _AsyncTask->get_process(QProgProgress::getInstance(), _ExecId);
}

void QVM::async_run(QProg& qProg, const NoiseModel& noise_model)
{
	_AsyncTask->run(this, qProg, noise_model);
}

bool QVM::is_async_finished()
{
	return _AsyncTask->is_finished();
}

std::map<std::string, bool> QVM::get_async_result()
{
	_AsyncTask->result();
	return _QResult->getResultMap();
}

static void accumulateProbability(prob_vec& probList, prob_vec& accumulateProb)
{
	accumulateProb.clear();
	accumulateProb.push_back(probList[0]);
	for (int i = 1; i < probList.size(); ++i)
	{
		accumulateProb.push_back(accumulateProb[i - 1] + probList[i]);
	}
}

map<string, size_t> IdealQVM::quickMeasure(QVec vQubit, size_t shots)
{
	map<string, size_t>  meas_result;
	prob_vec probList = getProbList(vQubit, -1);
	prob_vec accumulate_probabilites;
	accumulateProbability(probList, accumulate_probabilites);
	for (int i = 0; i < shots; ++i)
	{
		double rng = RandomNumberGenerator();
		if (rng < accumulate_probabilites[0])
			add_up_a_map(meas_result, dec2bin(0, vQubit.size()));
		for (int i = 1; i < accumulate_probabilites.size(); ++i)
		{
			if (rng < accumulate_probabilites[i] &&
				rng >= accumulate_probabilites[i - 1]
				)
			{
				add_up_a_map(meas_result,
					dec2bin(i, vQubit.size())
				);
				break;
			}
		}
	}
	return meas_result;
}


map<GateType, size_t> QVM::getGateTimeMap() const
{
	QuantumMetadata metadata;
	map<GateType, size_t> gate_time;
	metadata.getGateTime(gate_time);

	return gate_time;
}

QStat IdealQVM::getQStat()
{
	if (nullptr == _pGates)
	{
		QCERR("_pGates is null");
		throw qvm_attributes_error("_pGates is null");
	}
    
    return _pGates->getQState();
    
}

QStat IdealQVM::getQState()
{
	if (nullptr == _pGates)
	{
		QCERR("_pGates is null");
		throw qvm_attributes_error("_pGates is null");
	}
	return _pGates->getQState();
}

CPUQVM::~CPUQVM()
{
	if(_CMem) {
		delete _CMem;
		_CMem = nullptr;
	}
}

void CPUQVM::set_parallel_threads(size_t size) {

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

void CPUQVM::init(bool is_double_precision)
{
    QVM::finalize();
	try
	{
		_start();
        if (is_double_precision == true)
        {
            _pGates = new CPUImplQPU<double>();
            _ptrIsNull(_pGates, "CPUImplQPU");
        }
        else 
        {
            _pGates = new CPUImplQPU<float>();
            _ptrIsNull(_pGates, "CPUImplQPU");
        }
       
	}
	catch (const std::exception& e)
	{
		QCERR(e.what());
		throw init_fail(e.what());
	}

}

void CPUQVM::init()
{
    QVM::finalize();
    try
    {
        _start();
        _pGates = new CPUImplQPU<double>();
        _ptrIsNull(_pGates, "CPUImplQPU");
    }
    catch (const std::exception& e)
    {
        QCERR(e.what());
        throw init_fail(e.what());
    }

}


void GPUQVM::init()
{
    QVM::finalize();
	try
	{
		_start();
#ifdef USE_CUDA
		_pGates = new GPUImplQPU();
#else
		_pGates = nullptr;
#endif // USE_CUDA
		_ptrIsNull(_pGates, "GPUImplQPU");
	}
	catch (const std::exception& e)
	{
		QCERR(e.what());
		throw init_fail(e.what());
	}

}

void CPUSingleThreadQVM::init()
{
    QVM::finalize();
	try
	{
		_start();
		_pGates = new CPUImplQPUSingleThread();
		_ptrIsNull(_pGates, "CPUImplQPUSingleThread");
		if (!random_engine)
			_pGates->set_random_engine(random_engine);
	}
	catch (const std::exception& e)
	{
		QCERR(e.what());
		throw init_fail(e.what());
	}

}


void QVM::setConfigure(const Configuration& config)
{
    setConfig(config);
    return ;
}
Qubit* QVM::qAlloc()
{
	return allocateQubit();
}

QVec QVM::qAllocMany(size_t qubit_count)
{
	return allocateQubits(qubit_count);
}

ClassicalCondition QVM::cAlloc()
{
	return allocateCBit();
}

ClassicalCondition QVM::cAlloc(size_t cbitNum)
{
	return allocateCBit(cbitNum);
}

std::vector<ClassicalCondition> QVM::cAllocMany(size_t count)
{
	return allocateCBits(count);
}

void QVM::qFree(Qubit* qubit)
{
    if (qubit == nullptr)
    {
        return;
    }
    this->_Qubit_Pool->Free_Qubit(qubit);
}

void QVM::qFreeAll(QVec& qubit_vec)
{
    for (auto iter : qubit_vec)
    {
        if (iter == nullptr)
        {
            break;
        }
        this->_Qubit_Pool->Free_Qubit(iter);
        //delete iter;
        //iter = nullptr;
    }
}

void QVM::qFreeAll()
{
    QVec qubit_vec;
    get_allocate_qubits(qubit_vec);
    this->qFreeAll(qubit_vec);

    return ;
}

void QVM::cFree(ClassicalCondition& class_cond)
{
    auto cbit = class_cond.getExprPtr()->getCBit();
    if (nullptr == cbit)
    {
        QCERR("cbit is null");
        throw invalid_argument("cbit is null");
    }
    _CMem->Free_CBit(cbit);
}
void QVM::cFreeAll(std::vector<ClassicalCondition >& cbit_vec)
{
    for (auto &iter : cbit_vec)
    {
        auto cbit = iter.getExprPtr()->getCBit();
        if (nullptr == cbit)
        {
            QCERR("cbit is null");
            throw invalid_argument("cbit is null");
        }
        this->_CMem->Free_CBit(cbit);
    }
}

void QVM::cFreeAll()
{
    std::vector<ClassicalCondition> cc_vec;
    get_allocate_cbits(cc_vec);
    cFreeAll(cc_vec);
    return ;
}

size_t QVM::getAllocateQubitNum()
{
	return getAllocateQubit();
}

size_t QVM::getAllocateCMemNum()
{
	return getAllocateCMem();
}

size_t QVM::get_allocate_qubits(QVec& qubit_vect)
{
	if (nullptr == _Qubit_Pool)
	{
		QCERR("_QResult is nullptr");
		throw qvm_attributes_error("_QResult is nullptr");
	}
	return _Qubit_Pool->get_allocate_qubits(qubit_vect);
}


size_t QVM::get_allocate_cbits(std::vector<ClassicalCondition>& cc_vect)
{
	if (nullptr == _CMem)
	{
		QCERR("_CMem is nullptr");
		throw qvm_attributes_error("_CMem is nullptr");
	}
	std::vector<CBit*> cbit_vect;
	size_t allocate_size = _CMem->get_allocate_cbits(cbit_vect);
	for (auto iter : cbit_vect)
	{
		cc_vect.push_back(ClassicalCondition(iter));
	}
	return allocate_size;
}

QVM::~QVM()
{
	finalize();
}

prob_tuple IdealQVM::pMeasure(QVec qubit_vector, int select_max)
{
	return PMeasure(qubit_vector, select_max);
}

prob_vec IdealQVM::pMeasureNoIndex(QVec qubit_vector)
{
	return PMeasure_no_index(qubit_vector);
}
