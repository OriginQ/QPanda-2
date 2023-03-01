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
USING_QPANDA
using namespace std;
OriginQResult::OriginQResult()
{ }

void OriginQResult::append(pair<string, bool> pairResult)
{
    // first search in map to find if CBit is existed
    
    auto iter=_Result_Map.find(pairResult.first);

    if (iter == _Result_Map.end())
    {    // if not existed
        _Result_Map.insert(pairResult);
    }
    else
    {    // if already existed
        _Result_Map[pairResult.first] = pairResult.second;
    }
}