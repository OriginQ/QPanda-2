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

#ifndef CMEM_FACTORY_H
#define CMEM_FACTORY_H

#include "CBitFactory.h"
#include "QPandaNamespace.h"
#include <functional>
#include <stack>

QPANDA_BEGIN
class CMem
{
    // this class is considered as the container of the CBit
public:
    virtual CBit* Allocate_CBit() = 0;
    virtual CBit* Allocate_CBit(size_t) = 0;
    virtual size_t getMaxMem() const = 0;
    virtual size_t getIdleMem() const = 0;
    virtual void Free_CBit(CBit*) = 0;
    virtual void clearAll() = 0;
    virtual ~CMem() {}
};
#define REGISTER_CMEM_SIZE_(classname) \
CMem* classname##_Constructor(size_t size)\
{\
    return new classname(size);\
}\
static CMemFactoryHelper _CMem_Factory_Helper_##classname(\
    #classname,\
    classname##_Constructor\
)

/* CMem Factory */
class CMemFactory
{
    CMemFactory();
public:
    typedef std::function<CMem*(size_t)> size_constructor_t;
    typedef std::map<std::string, size_constructor_t> size_constructor_stack_t;
    size_constructor_stack_t _CMem_Constructor;
    CMem* GetInstanceFromSize(size_t);
    void registerclass_size_(std::string &, size_constructor_t);
    static CMemFactory& GetFactoryInstance();
};

class CMemFactoryHelper
{
    typedef CMemFactory::size_constructor_t
        size_constructor_t;
public:
    CMemFactoryHelper(std::string, size_constructor_t _Constructor);
};
QPANDA_END
#endif