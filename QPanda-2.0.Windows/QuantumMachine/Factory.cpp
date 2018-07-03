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

#include "Factory.h"
#include "OriginQuantumMachine.h"
#include "../QPanda/QPandaException.h"
#include "QPanda/ConfigMap.h"
#include "OriginClassicalExpression.h"
#include "OriginQMachine.h"
/* 1. static variable initialization start */

#if false
Factory::QuantumMachineFactory::constructor_map_t
Factory::QuantumMachineFactory::
_Quantum_Machine_Constructor;
	// empty
	// constructor map of quantum machine
	// realize the Reflection

Factory::PhysicalQubitFactory::constructor_stack_t
Factory::PhysicalQubitFactory::
_Physical_Qubit_Constructor;

Factory::QubitFactory::constructor_stack_t
Factory::QubitFactory::
_Qubit_Constructor;

Factory::QubitPoolFactory::size_constructor_stack_t
Factory::QubitPoolFactory::_Qubit_Pool_Constructor;

Factory::CBitFactory::name_constructor_stack_t
Factory::CBitFactory::
_CBit_Constructor;

Factory::CMemFactory::size_constructor_stack_t
Factory::CMemFactory::
_CMem_Constructor;

Factory::QResultFactory::constructor_stack_t
Factory::QResultFactory::
_QResult_Constructor;

Factory::CExprFactory::cbit_constructor_stack_t
Factory::CExprFactory::_CExpr_CBit_Constructor;

Factory::CExprFactory::operator_constructor_stack_t
Factory::CExprFactory::_CExpr_Operator_Constructor;

#endif

/* 1. static variable initialization end */

/* 2. Quantum Machine Status Factory */

QMachineStatus * Factory::QMachineStatusFactory::GetQMachineStatus()
{
	return new OriginQMachineStatus();
}

/* 2. Quantum Machine Status Factory End*/

/* 3. Quantum Machine Factory*/

/* Factory Helper class */

Factory::QuantumMachineFactory::QuantumMachineFactory()
{
	// constructor
    /*
	if (_Quantum_Machine_Constructor.find("OriginQVM")
		!= _Quantum_Machine_Constructor.end())
	{
		throw(factory_init_error("QuantumMachineFactory"));
	}
	_Quantum_Machine_Constructor.insert(
		make_pair(
			"OriginQVM", 
			[]() {return new OriginQVM(); }
		)
	);
    */
    
}

FactoryHelper::QuantumMachineFactoryHelper::
QuantumMachineFactoryHelper(
	string _Classname,
	constructor_t _Constructor
)
{
	// create a static variable of the helper class
	// can execute the constructor code
	
	// Here we insert the classname and constructor pair
	// into the map of _Quantum_Machine_Constructor
	auto &fac=
		Factory::QuantumMachineFactory::GetFactoryInstance();
	fac.registerclass(_Classname, _Constructor);
}

Factory::QuantumMachineFactory & Factory::QuantumMachineFactory::GetFactoryInstance()
{
	static QuantumMachineFactory fac;
	return fac;
}

/* Factory Method */
QuantumMachine * Factory::QuantumMachineFactory::
	CreateByName(string _Name)
{
	// create a Quantum Machine by its name string
	if (_Quantum_Machine_Constructor.find(_Name)
		== _Quantum_Machine_Constructor.end())
	{
		// that means the quantum machine is not registered
		return nullptr;
	}
	return Factory::QuantumMachineFactory::
		_Quantum_Machine_Constructor[_Name]();
}

void Factory::QuantumMachineFactory::registerclass(string name, constructor_t constructor)
{
	if (_Quantum_Machine_Constructor.find(name) != _Quantum_Machine_Constructor.end())
	{
		cerr << "QuantumMachineFactory Warning:" << endl
		     << "re-register the class " << name << endl;
	}
	_Quantum_Machine_Constructor.insert(make_pair(name, constructor));
}

REGISTER_QUANTUM_MACHINE(OriginQVM);
REGISTER_QUANTUM_MACHINE(OriginQMachine);

/* 3. Quantum Machine Factory End*/

/* 4. Physical Qubit Factory*/
Factory::PhysicalQubitFactory::PhysicalQubitFactory()
{

}

Factory::PhysicalQubitFactory& 
Factory::PhysicalQubitFactory::GetFactoryInstance()
{
	static PhysicalQubitFactory fac;
	return fac;
}

/* Factory Get Instance */
PhysicalQubit * Factory::PhysicalQubitFactory::GetInstance()
{
	// I think the last Constructor that user push
	// to the top of the stack is the user defined
	// class. So only Get the top is ok

    auto sClassName = _G_configMap["PhysicalQubit"];
    if (sClassName.size()<=0)
    {
        return nullptr;
    }
	auto aiter = _Physical_Qubit_Constructor.find(sClassName); 

    if (aiter == _Physical_Qubit_Constructor.end())
    {
        return nullptr;
    }

    return aiter->second();
}

void Factory::PhysicalQubitFactory::registerclass(string & sClassName, constructor_t constructor)
{
	// Only push the Constructor to the top
	_Physical_Qubit_Constructor.insert(make_pair(sClassName, constructor));
}

/* Factory Helper Class */
FactoryHelper::PhysicalQubitFactoryHelper::PhysicalQubitFactoryHelper(string  sClassName,constructor_t _Constructor)
{
	auto &fac = Factory::PhysicalQubitFactory::GetFactoryInstance();
	fac.registerclass(sClassName,_Constructor);
}


REGISTER_PHYSICAL_QUBIT(OriginPhysicalQubit);

/* 4. Physical Qubit Factory End */

/* 5. Qubit Factory */
Factory::QubitFactory::
QubitFactory()
{
}

/* Factory Helper Class */
FactoryHelper::QubitFactoryHelper::QubitFactoryHelper(string sClassName ,constructor_t _Constructor)
{
	auto &fac=Factory::QubitFactory::GetFactoryInstance();
	fac.registerclass(sClassName,_Constructor);
}

Factory::QubitFactory & Factory::QubitFactory::GetFactoryInstance()
{
	static QubitFactory fac;
	return fac;
}

/* Factory Get Instance */
Qubit * Factory::QubitFactory::GetInstance(PhysicalQubit *_physQ)
{
	// I think the last Constructor that user push
	// to the top of the stack is the user defined
	// class. So only Get the top is ok

    auto sClassName = _G_configMap["Qubit"];
    if (sClassName.size() <= 0)
    {
        return nullptr;
    }
    auto aiter = _Qubit_Constructor.find(sClassName);

    if (aiter == _Qubit_Constructor.end())
    {
        return nullptr;
    }

	return aiter->second(_physQ);
}

void Factory::QubitFactory::registerclass(string &sClassName,constructor_t constructor)
{
	// Only push the Constructor to the top
	_Qubit_Constructor.insert(make_pair(sClassName, constructor));
}

REGISTER_QUBIT(OriginQubit);

/* 6. Qubit Pool Factory */

Factory::QubitPoolFactory::QubitPoolFactory()
{
    /*
	_Qubit_Pool_Constructor.push(
		[](size_t _Size)
	{
		return new OriginQubitPool(_Size);
	}
	);
    */
}
Factory::QubitPoolFactory &
Factory::QubitPoolFactory::GetFactoryInstance()
{
	static QubitPoolFactory fac;
	return fac;
}

/* Factory Get Instance */
QubitPool * Factory::QubitPoolFactory::GetPoolWithoutTopology(size_t size)
{
	// I think the last Constructor that user push
	// to the top of the stack is the user defined
	// class. So only Get the top is ok

	if (_Qubit_Pool_Constructor.empty())
	{
		return nullptr;
	}

    auto sClassName = _G_configMap["QubitPool"];
    if (sClassName.size() <= 0)
    {
        return nullptr;
    }
    auto aiter = _Qubit_Pool_Constructor.find(sClassName);

    if (aiter == _Qubit_Pool_Constructor.end())
    {
        return nullptr;
    }

    return aiter->second(size);

}

void Factory::QubitPoolFactory::registerclass_size_(string & sClassName,
	size_constructor_t constructor)
{
	_Qubit_Pool_Constructor.insert(make_pair(sClassName, constructor));
}

/* Factory Helper Class */
FactoryHelper::QubitPoolFactoryHelper::QubitPoolFactoryHelper(string  sClassName,
	size_constructor_t constructor)
{
	// Only push the Constructor to the top
	auto &fac = Factory::QubitPoolFactory::GetFactoryInstance();
	fac.registerclass_size_(sClassName,constructor);
}

REGISTER_QUBIT_POOL_SIZE_(OriginQubitPool);
/* 6. Qubit Pool Factory End*/

/* 7. CBit Factory */

Factory::CBitFactory::CBitFactory()
{
    /*
	if (_CBit_Constructor.empty() != true)
	{
		throw(factory_init_error("CBitFactory"));
	}
	_CBit_Constructor.push([](string name)
	{
		return new OriginCBit(name);
	});
    */
}

Factory::CBitFactory&
Factory::CBitFactory::GetFactoryInstance()
{
	static CBitFactory fac;
	return fac;
}

void Factory::CBitFactory::registerclass_name_
(string &sClassName, name_constructor_t constructor)
{
	_CBit_Constructor.insert(make_pair(sClassName, constructor));
}

CBit * Factory::CBitFactory::CreateCBitFromName(string name)
{
    if (_CBit_Constructor.empty())
    {
        return nullptr;
    }

    auto sClassName = _G_configMap["CBit"];
    if (sClassName.size() <= 0)
    {
        return nullptr;
    }
    auto aiter = _CBit_Constructor.find(sClassName);

    if (aiter == _CBit_Constructor.end())
    {
        return nullptr;
    }
    return aiter->second(name);
	//return _CBit_Constructor.top()(name);
}

FactoryHelper::CBitFactoryHelper::CBitFactoryHelper(string sClassName,name_constructor_t _Constructor)
{
	auto &fac = Factory::CBitFactory::GetFactoryInstance();
	fac.registerclass_name_(sClassName,_Constructor);
}

REGISTER_CBIT_NAME_(OriginCBit);
/* 7. CBit Factory End*/

/* 8. CMem Factory*/
Factory::CMemFactory::CMemFactory()
{
    /*
	if (_CMem_Constructor.empty() != true)
	{
		throw(factory_init_error("CMemFactory"));
	}
	_CMem_Constructor.push(
		[](size_t _Size)
	{
		return new OriginCMem(_Size);
	}
	);

    */
}

CMem * Factory::CMemFactory::GetInstanceFromSize(size_t _Size)
{
    if (_CMem_Constructor.empty())
    {
        return nullptr;
    }

    auto sClassName = _G_configMap["CMem"];
    if (sClassName.size() <= 0)
    {
        return nullptr;
    }
    auto aiter = _CMem_Constructor.find(sClassName);

    if (aiter == _CMem_Constructor.end())
    {
        return nullptr;
    }
    return aiter->second(_Size);
}

FactoryHelper::CMemFactoryHelper::CMemFactoryHelper
(string sClassName,size_constructor_t _Constructor)
{
	auto &fac = Factory::CMemFactory::GetFactoryInstance();
	fac.registerclass_size_(sClassName,_Constructor);
}

void Factory::CMemFactory::
registerclass_size_(string & sClassName,size_constructor_t constructor)
{
	_CMem_Constructor.insert(make_pair(sClassName, constructor));
}

Factory::CMemFactory&
Factory::CMemFactory::GetFactoryInstance()
{
	static CMemFactory fac;
	return fac;
}
REGISTER_CMEM_SIZE_(OriginCMem);
/* 8. CMem Factory End*/

/* 10. QResult Factory */
Factory::QResultFactory::
QResultFactory()
{
    /*
	if (_QResult_Constructor.empty() != true)
	{
		throw(factory_init_error("QResult"));
	}
	_QResult_Constructor.push([]()
	{
		return new OriginQResult();
	});
    */
}

QResult * Factory::QResultFactory::GetEmptyQResult()
{
    if (_QResult_Constructor.empty())
    {
        return nullptr;
    }

    auto sClassName = _G_configMap["QResult"];
    if (sClassName.size() <= 0)
    {
        return nullptr;
    }
    auto aiter = _QResult_Constructor.find(sClassName);

    if (aiter == _QResult_Constructor.end())
    {
        return nullptr;
    }
    return aiter->second();

}

void Factory::QResultFactory::registerclass(string& sClassName,constructor_t
_Constructor)
{
	_QResult_Constructor.insert(make_pair(sClassName, _Constructor));
}

Factory::QResultFactory&
Factory::QResultFactory::GetFactoryInstance()
{ 
	static QResultFactory fac;
	return fac;
}

FactoryHelper::QResultFactoryHelper::
QResultFactoryHelper(string sClassName,constructor_t _Constructor)
{
	auto &fac = Factory::QResultFactory::GetFactoryInstance();
	fac.registerclass(sClassName,_Constructor);
}
REGISTER_QRES_NAME(OriginQResult);
/* 10. QResult Factory End */

/* 11. CExpr Factory */

Factory::CExprFactory::CExprFactory()
{
    /*
	if (!_CExpr_CBit_Constructor.empty())
	{
		throw(factory_init_error("CExpr(CBit)"));
	}
	_CExpr_CBit_Constructor.push(
		[](CBit* m)
	{
		return new OriginCExpr(m);
	});
	if (!_CExpr_Operator_Constructor.empty())
	{
		throw(factory_init_error("CExpr(Operator)"));
	}
	_CExpr_Operator_Constructor.push(
		[](CExpr* left, CExpr* right, int op)
	{
		return new OriginCExpr(left, right, op);
	}
	);
    */
}

Factory::CExprFactory&
Factory::CExprFactory::GetFactoryInstance()
{
	static CExprFactory fac;
	return fac;
}


CExpr * Factory::CExprFactory::GetCExprByCBit(CBit *bit)
{
    if (_CExpr_CBit_Constructor.empty())
    {
        return nullptr;
    }

    auto sClassName = _G_configMap["CExpr"];
    if (sClassName.size() <= 0)
    {
        return nullptr;
    }
    auto aiter = _CExpr_CBit_Constructor.find(sClassName);

    if (aiter == _CExpr_CBit_Constructor.end())
    {
        return nullptr;
    }
    return aiter->second(bit);

}

CExpr * Factory::CExprFactory::GetCExprByOperation(
	CExpr *leftexpr, 
	CExpr *rightexpr, 
	int op)
{
    if (_CExpr_Operator_Constructor.empty())
    {
        return nullptr;
    }

    auto sClassName = _G_configMap["CExpr"];
    if (sClassName.size() <= 0)
    {
        return nullptr;
    }
    auto aiter = _CExpr_Operator_Constructor.find(sClassName);

    if (aiter == _CExpr_Operator_Constructor.end())
    {
        return nullptr;
    }
    return aiter->second(leftexpr, rightexpr, op);

}

void Factory::CExprFactory::registerclass_CBit_(
	string &sClassName,cbit_constructor_t _Constructor
)
{
	_CExpr_CBit_Constructor.insert(make_pair(sClassName, _Constructor));
}

FactoryHelper::CExprFactoryHelper::CExprFactoryHelper(
    string sClassName,
	cbit_constructor_t _Constructor
)
{
	auto &fac =
		Factory::CExprFactory::GetFactoryInstance();
	fac.registerclass_CBit_(sClassName,_Constructor);
}

void Factory::CExprFactory::registerclass_operator_
(string &sClassName,operator_constructor_t _Constructor)
{
	_CExpr_Operator_Constructor.insert(make_pair(sClassName, _Constructor));
}

FactoryHelper::CExprFactoryHelper::CExprFactoryHelper(
    string sClassName,
	operator_constructor_t _Constructor
)
{
	auto &fac =
		Factory::CExprFactory::GetFactoryInstance();
	fac.registerclass_operator_(sClassName,_Constructor);
}

REGISTER_CEXPR(OriginCExpr);




