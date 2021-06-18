/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.

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
/*! \file Components.h */
#ifndef _COMPONENTS_H
#define _COMPONENTS_H

#include "Components/ChemiQ/ChemiqUtil.h"
#include "Components/ChemiQ/Psi4Wrapper.h"
#include "Components/HamiltonianSimulation/HamiltonianSimulation.h"
#include "Components/MaxCutProblemGenerator/MaxCutProblemGenerator.h"
#include "Components/NodeSortProblemGenerator/NodeSortProblemGenerator.h"
#include "Components/Operator/FermionOperator.h"
#include "Components/Operator/PauliOperator.h"
#include "Components/Optimizer/AbstractOptimizer.h"
#include "Components/Optimizer/OptimizerFactory.h"
#include "Components/Optimizer/OriginNelderMead.h"
#include "Components/Optimizer/OriginPowell.h"
#include "Components/Optimizer/OriginCOBYLA.h"
#include "Components/Optimizer/OriginLBFGSB.h"
#include "Components/Optimizer/OriginSLSQP.h"
#include "Components/Utils/RJson/RJson.h"
#include "Components/macro.h"


#endif // !_COMPONENTS_H