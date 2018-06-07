/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.

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
#include "../QuantumCircuit/QProgram.h"
#include "../QuantumInstructionHandle/X86QuantumGates.h"

OriginQVM::OriginQVM()
{
	return;
}

void OriginQVM::init()
{
	_Qubit_Pool = 
		Factory::
		QubitPoolFactory::GetFactoryInstance().
		GetPoolWithoutTopology(_Config.maxQubit);
	_CMem = 
		Factory::
		CMemFactory::GetFactoryInstance().
		GetInstanceFromSize(_Config.maxCMem);
    QuantumProgram & temp = CreateEmptyQProg();
    _QProgram = temp.getPosition();
	_QResult =
		Factory::
		QResultFactory::GetFactoryInstance().
		GetEmptyQResult();
	_QMachineStatus =
		Factory::
		QMachineStatusFactory::
		GetQMachineStatus();
    _pGates = new X86QuantumGates();
}

Qubit * OriginQVM::Allocate_Qubit()
{
	if (_Qubit_Pool == nullptr)
	{
		// check if the pointer is nullptr
		// Before init
		// After finalize
		throw(exception());
	}
	else
	{
		return _Qubit_Pool->Allocate_Qubit();
	}
		
}

CBit * OriginQVM::Allocate_CBit()
{
	if (_CMem == nullptr)
	{
		// check if the pointer is nullptr
		// Before init
		// After finalize
		throw(exception());
	}
	else
	{
		return _CMem->Allocate_CBit();
	}
}

void OriginQVM::Free_Qubit(Qubit *qubit)
{
    this->_Qubit_Pool->Free_Qubit(qubit);
}

void OriginQVM::Free_CBit(CBit *cbit)
{
    this->_CMem->Free_CBit(cbit);
}

void OriginQVM::load(QuantumProgram &loadProgram)
{
    QNodeAgency temp(&loadProgram, nullptr, nullptr);
    if (!temp.verify())
    {
        throw exception();
    }
    _QProgram = loadProgram.getPosition();
}

void OriginQVM::append(QuantumProgram & prog)
{
    QNodeAgency tempAgency(&prog, nullptr, nullptr);
    if (!tempAgency.verify())
    {
        throw exception();
    }
    auto aiter = _G_QNodeVector.getNode(_QProgram);
    QuantumProgram * temp = dynamic_cast<QuantumProgram *>(*aiter);
    temp->operator<<(prog);
}

void OriginQVM::run()
{
    if (_QProgram < 0)
    {
        return;
    }
    
    auto aiter =_G_QNodeVector.getNode(_QProgram);
    QuantumProgram * pNode = (QuantumProgram *)*aiter;
    _pParam = new QuantumGateParam();

	/* error may occur if user free working qubit before run() */
    _pParam->mQuantumBitNumber = _Qubit_Pool->getMaxQubit() - _Qubit_Pool->getIdleQubit();

    _pGates->initState(_pParam);
    QNodeAgency temp(pNode, _pParam, _pGates);
    if (temp.executeAction())
    {
		/* aiter has been used in line 120 */
        for (auto aiter : _pParam->mReturnValue)
        {
            _QResult->append(aiter);
        }
    }
    delete _pParam;
    _pParam = nullptr;
    return ;

}

QMachineStatus * OriginQVM::getStatus() const
{
	return _QMachineStatus;
}

QResult * OriginQVM::getResult()
{
	return _QResult;
}

void OriginQVM::finalize()
{
	delete _Qubit_Pool;
	delete _CMem;
	delete _QResult;
	delete _QMachineStatus;
    delete _pGates;
}

//REGISTER_QUANTUM_MACHINE(OriginQVM)

