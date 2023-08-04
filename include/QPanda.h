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
/*! \file QPanda.h */

#ifndef _QPANDA_H
#define _QPANDA_H

#include "Core/Core.h"
#include "QPandaConfig.h"
#include "Components/Components.h"

#if defined(USE_EXTENSION)

#include "Extensions/QAlg/QAlg.h"

#endif

/**
* @defgroup Components
* @brief QPanda2  Components  Group
*
* @defgroup HamiltonianSimulation
* @ingroup Components
*
* @defgroup MaxCutProblemGenerator
* @ingroup Components
*
* @defgroup Operator
* @ingroup Components
*
* @defgroup Optimizer
* @ingroup Components
*
* @defgroup Utils
* @ingroup Components
*/


/**
* @defgroup Core
* @brief QPanda2 Core Group
*
* @defgroup Module
* @ingroup Core
*
* @defgroup QuantumCircuit
* @brief QPanda2  quantum circuit and quantum program
* @ingroup Core
*
* @defgroup QuantumMachine
* @brief  QPanda2 quantum virtual machine
* @ingroup Core
*
* @defgroup Utilities
* @brief QPanda2  base  Utilities  classes and  interface
* @ingroup Core
*
* @defgroup Variational
* @brief QPanda2   variate
* @ingroup Core
*
* @defgroup VirtualQuantumProcessor
* @brief QPanda2  virtual quantum processor
* @ingroup Core
*/


/**
* @defgroup QAlg
* @brief QPanda2 Algorithm Group
*
* @defgroup B_V_Algorithm
* @brief  Bernstein-Vazirani algorithm
* @ingroup QAlg
*
* @defgroup DJ_Algorithm
* @brief Deutsch Jozsa algorithm
* @ingroup QAlg
*
* @defgroup Grover_Algorithm
* @brief Grover Algorithm
* @ingroup QAlg
*
*
*/


#endif // !_QPANDA_H
































































