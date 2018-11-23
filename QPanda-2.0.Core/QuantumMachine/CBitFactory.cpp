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
#include "CBitFactory.h"
#include "QPanda/ConfigMap.h"
#include "OriginQuantumMachine.h"

/* CBit Factory */
CBitFactory::CBitFactory()
{
}

CBitFactory&
CBitFactory::GetFactoryInstance()
{
	static CBitFactory fac;
	return fac;
}

void CBitFactory::registerclass_name_
(string &sClassName, name_constructor_t constructor)
{
	_CBit_Constructor.insert(make_pair(sClassName, constructor));
}

CBit * CBitFactory::CreateCBitFromName(string name)
{
	if (_CBit_Constructor.empty())
	{
		return nullptr;
	}

	auto sClassName = ConfigMap::getInstance()["CBit"];
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

CBitFactoryHelper::CBitFactoryHelper(string sClassName, name_constructor_t _Constructor)
{
	auto &fac = CBitFactory::GetFactoryInstance();
	fac.registerclass_name_(sClassName, _Constructor);
}

REGISTER_CBIT_NAME_(OriginCBit);