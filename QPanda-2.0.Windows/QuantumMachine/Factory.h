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

#ifndef FACTORY_H
#define FACTORY_H

#include "QuantumMachineInterface.h"
#include "ClassicalConditionInterface.h"
#include <functional>
#include <stack>
using namespace std;

#define REGISTER_QUANTUM_MACHINE(classname) \
QuantumMachine* classname##_Constructor()\
{\
    return new classname();\
}\
static FactoryHelper::QuantumMachineFactoryHelper _Quantum_Machine_Factory_Helper(\
	#classname,\
	classname##_Constructor\
);

#define REGISTER_PHYSICAL_QUBIT(classname) \
PhysicalQubit* classname##_Constructor()\
{\
    return new classname();\
}\
static FactoryHelper::PhysicalQubitFactoryHelper _Physical_Qubit_Factory_Helper(\
	classname##_Constructor\
);

#define REGISTER_QUBIT(classname) \
Qubit* classname##_Constructor(PhysicalQubit* physQ)\
{\
    return new classname(physQ);\
}\
static FactoryHelper::QubitFactoryHelper _Qubit_Factory_Helper(\
	classname##_Constructor\
);

#define REGISTER_QUBIT_POOL_SIZE_(classname) \
QubitPool* classname##_Constructor(size_t size)\
{\
    return new classname(size);\
}\
static FactoryHelper::QubitPoolFactoryHelper _Qubit_Pool_Factory_Helper(\
	classname##_Constructor\
);

#define REGISTER_CBIT_NAME_(classname) \
CBit* classname##_Constructor(string name)\
{\
    return new classname(name);\
}\
static FactoryHelper::CBitFactoryHelper _CBit_Factory_Helper(\
	classname##_Constructor\
);

#define REGISTER_CMEM_SIZE_(classname) \
CMem* classname##_Constructor(size_t size)\
{\
    return new classname(size);\
}\
static FactoryHelper::CMemFactoryHelper _CMem_Factory_Helper(\
	classname##_Constructor\
);

#define REGISTER_QRES_NAME(classname) \
QResult* classname##_Constructor()\
{\
    return new classname();\
}\
static FactoryHelper::QResultFactoryHelper _QRes_Factory_Helper(\
	classname##_Constructor\
);

#define REGISTER_CEXPR(classname)\
CExpr* classname##_CBit_Constructor(CBit* cbit)\
{\
	return new classname(cbit);\
}\
CExpr* classname##_Operator_Constructor(\
	CExpr* leftexpr,\
	CExpr* rightexpr,\
	int op\
)\
{\
	return new classname(leftexpr, rightexpr, op);\
}\
static FactoryHelper::CExprFactoryHelper \
_classname##_CBit_Constructor_Helper\
(classname##_CBit_Constructor);\
static FactoryHelper::CExprFactoryHelper \
_classname##_Operator_Constructor_Helper\
(classname##_Operator_Constructor);

class Factory
{
public:	

	/* 2. Quantum Machine Status Factory */
	class QMachineStatusFactory
	{
	public:
		static QMachineStatus* GetQMachineStatus();
	};

	/* 3. Quantum Machine Factory */
	class QuantumMachineFactory
	{
		// Factory for class QuantumMachine
		QuantumMachineFactory();
	public:
		static QuantumMachineFactory & GetFactoryInstance();
		typedef function<QuantumMachine*()> constructor_t;
		typedef map<string, constructor_t> constructor_map_t;

		// create the quantum machine by its class name
		QuantumMachine* CreateByName(string);

		// a constructor map
		constructor_map_t _Quantum_Machine_Constructor;	

		// class register function
		void registerclass(string name, constructor_t constructor);
	};

	/* 4. Physical Qubit Factory*/
	class PhysicalQubitFactory
	{
		// Factory for class PhysicalQubit
		PhysicalQubitFactory();
	public:

		static PhysicalQubitFactory & GetFactoryInstance();

		PhysicalQubit* GetInstance();
		typedef function<PhysicalQubit*()> constructor_t;
		typedef stack<constructor_t> constructor_stack_t;
		void registerclass(constructor_t constructor);
		// the constructor stack
		constructor_stack_t _Physical_Qubit_Constructor;
	};

	/* 5. Qubit Factory */
	class QubitFactory
	{
		QubitFactory();
	public:
		static QubitFactory & GetFactoryInstance();
		Qubit* GetInstance(PhysicalQubit*);
		typedef function<Qubit*(PhysicalQubit*)> constructor_t;
		typedef stack<constructor_t> constructor_stack_t;
		void registerclass(constructor_t constructor);
		// the constructor stack
		constructor_stack_t _Qubit_Constructor;
	};

	/* 6. Qubit Pool Factory */

	class QubitPoolFactory
	{
		QubitPoolFactory();
	public:
		static QubitPoolFactory& GetFactoryInstance();
		typedef function<QubitPool*(size_t)> size_constructor_t;
		typedef stack<size_constructor_t> size_constructor_stack_t;
        size_constructor_stack_t _Qubit_Pool_Constructor;
		QubitPool* GetPoolWithoutTopology(size_t);
		void registerclass_size_(size_constructor_t constructor);
	};

	/* 7. CBit Factory */
	class CBitFactory
	{
		CBitFactory();
	public:
		static CBitFactory & GetFactoryInstance();
		typedef function<CBit*(string)> name_constructor_t;
		typedef stack<name_constructor_t> name_constructor_stack_t;
		name_constructor_stack_t _CBit_Constructor;
		void registerclass_name_(name_constructor_t constructor);
		CBit* CreateCBitFromName(string);
	};

	/* 8. CMem Factory */
	class CMemFactory
	{
		CMemFactory();
	public:
		typedef function<CMem*(size_t)> size_constructor_t;
		typedef stack<size_constructor_t> size_constructor_stack_t;		
		size_constructor_stack_t _CMem_Constructor;
		CMem* GetInstanceFromSize(size_t);
		void registerclass_size_(size_constructor_t);
		static CMemFactory& GetFactoryInstance();
	};

	/* 10. QResult Factory */
	class QResultFactory
	{
		QResultFactory();
	public:
		typedef function<QResult*()> constructor_t;
		typedef stack<constructor_t> constructor_stack_t;
        constructor_stack_t _QResult_Constructor ;
		QResult* GetEmptyQResult();
		void registerclass(constructor_t);
		static QResultFactory& GetFactoryInstance();
	};

	/* 11. CExpr Factory */
	class CExprFactory
	{
		CExprFactory();
	public:

		static CExprFactory & GetFactoryInstance();

		typedef function<CExpr*(CBit*)> cbit_constructor_t;
		typedef stack<cbit_constructor_t> cbit_constructor_stack_t;
		cbit_constructor_stack_t _CExpr_CBit_Constructor;
		CExpr* GetCExprByCBit(CBit*);
		void registerclass_CBit_(cbit_constructor_t);
		
		typedef function<CExpr*(CExpr*, CExpr*, int)> operator_constructor_t;
		typedef stack<operator_constructor_t> operator_constructor_stack_t;
		operator_constructor_stack_t _CExpr_Operator_Constructor;
		CExpr* GetCExprByOperation(CExpr*, CExpr*, int);
		void registerclass_operator_(operator_constructor_t);
	};

};

// Factory Helper Class
class FactoryHelper
{
public:

	class QuantumMachineFactoryHelper
	{
		typedef Factory::QuantumMachineFactory::constructor_t constructor_t;			
	public:
		QuantumMachineFactoryHelper(string, constructor_t);
	};

	class PhysicalQubitFactoryHelper
	{
		typedef Factory::PhysicalQubitFactory::constructor_t constructor_t;
	public:
		PhysicalQubitFactoryHelper(constructor_t);
	};

	class QubitFactoryHelper
	{
		typedef Factory::QubitFactory::constructor_t constructor_t;
	public:
		QubitFactoryHelper(constructor_t);
	};

	class QubitPoolFactoryHelper
	{
		typedef Factory::QubitPoolFactory::
			size_constructor_t size_constructor_t;
	public:
		QubitPoolFactoryHelper(size_constructor_t);
	};

	class CBitFactoryHelper
	{
		typedef Factory::CBitFactory::name_constructor_t
			name_constructor_t;
	public:
		CBitFactoryHelper(name_constructor_t);
	};

	class CMemFactoryHelper
	{
		typedef Factory::CMemFactory::size_constructor_t
			size_constructor_t;
	public:
		CMemFactoryHelper(size_constructor_t _Constructor);
	};
	
	class QResultFactoryHelper
	{
		typedef Factory::QResultFactory::constructor_t
			constructor_t;
	public:
		QResultFactoryHelper(constructor_t _Constructor);
	};

	class CExprFactoryHelper
	{
		typedef Factory::CExprFactory::cbit_constructor_t
			cbit_constructor_t;
		typedef Factory::CExprFactory::operator_constructor_t
			operator_constructor_t;
	public:
		CExprFactoryHelper(cbit_constructor_t _Constructor);
		CExprFactoryHelper(operator_constructor_t _Constructor);
	};
};

#endif