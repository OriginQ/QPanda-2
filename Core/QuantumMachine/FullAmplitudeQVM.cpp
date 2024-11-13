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
#include "Core/QuantumMachine/QProgExecution.h"
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


void FullAmplitudeQVM::init(BackendType type)
{
    QVM::finalize();
    try
    {
        _start();

        if (BackendType::CPU == type)
        {
            _pGates = new CPUImplQPU<double>();
            _ptrIsNull(_pGates, "CPUImplQPU");
        }
        else if (BackendType::GPU == type)
        {
#ifdef USE_CUDA
            try
            {
                _pGates = new GPUImplQPU();
            }
            catch (...)
            {
                std::cout << "WARNING : cuda not found ,use cpu backend." << std::endl;
                _pGates = new CPUImplQPU<double>();
            }
#else 
            std::cout << "cuda not found ,use cpu backend." << std::endl;
            _pGates = new CPUImplQPU<double>();

#endif // USE_CUDA
        }
        else
        {
            QCERR_AND_THROW(run_fail, "PartialAmplitudeQVM::init BackendType error.");
        }

    }
    catch (const std::exception& e)
    {
        QCERR(e.what());
        throw init_fail(e.what());
    }
}

void FullAmplitudeQVM::init()
{
    init(BackendType::CPU);
}
