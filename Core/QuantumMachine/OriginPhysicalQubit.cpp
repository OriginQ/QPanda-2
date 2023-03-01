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
using namespace std;
USING_QPANDA
OriginPhysicalQubit::OriginPhysicalQubit()
{
    bIsOccupied = false;
}

bool OriginPhysicalQubit::getOccupancy() const
{
    return bIsOccupied;
}

void OriginPhysicalQubit::setOccupancy(bool bStat)
{
    bIsOccupied = bStat;
}





