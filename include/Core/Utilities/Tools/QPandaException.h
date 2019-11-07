/*! \file QPandaException.h */
#ifndef QPANDA_EXCEPTION_H
#define QPANDA_EXCEPTION_H
#include "Core/Utilities/QPandaNamespace.h"
#include <exception>
#include <string>
QPANDA_BEGIN
/**
* @namespace QPanda
*/
/*
|-QPandaException
|---qalloc_fail
|---calloc_fail
<<<<<<< HEAD
|---factory_init_error
|---duplicate_free

=======
|---duplicate_free

|---factory_exception
|------factory_init_error
|------factory_get_instance_fail

>>>>>>> c081d7d6d4a9f182e66b5a2e1e764f4bf35f0c19
|---load_exception
|------qubit_not_allocated
|------cbit_not_allocated

|---classical_system_exception
|------operator_specifier_error
|------content_specifier_error
|------eval_error
|------invalid_cbit_ptr
|------invalid_cmem

|---qubit_pool_exception
|------invalid_qubit_ptr
|------invalid_pool
*/

/**
* @class QPandaException
* @brief QPanda2 exception basic class
* @ingroup Core
*/
class QPandaException : public std::exception
{
    std::string errmsg;
    bool isFree;
    public:
    QPandaException()
    : exception()
    {
        errmsg = "Unknown error";
        isFree = false;
    }
    QPandaException(const char* str)
    : exception() {
        errmsg = str;
    }

    QPandaException(std::string str)
    : exception() {
        errmsg = str;
    }

    virtual const char* what()
    {
        return errmsg.c_str();
    }	
};

/**
* @class qalloc_fail
* @brief QPanda2 alloc qubit failed exception
*/
class qalloc_fail : public QPandaException
{
    public:
    qalloc_fail() : QPandaException(
        "qalloc fail"
        ) {}
    qalloc_fail(std::string errmsg)
        : QPandaException(
            errmsg
        )
    {}
};

/**
* @class qalloc_fail
* @brief QPanda2 alloc cbit failed exception
*/
class calloc_fail : public QPandaException
{
    public:
    calloc_fail() : QPandaException(
        "calloc fail"
        ) {}
    calloc_fail(std::string errmsg)
        : QPandaException(
            errmsg
        )
    {}
};

/**
* @class init_fail
* @brief QPanda2 init failed exception
*/
class init_fail : public QPandaException
{
    public:
        init_fail() : QPandaException(
        "init_fail"
        ){}
        init_fail(std::string errmsg)
    : QPandaException(
        errmsg
        )
    {}
};

/**
* @class run_fail
* @brief QPanda2 running time error exception
*/
class run_fail : public QPandaException
{
    public:
        run_fail(std::string cls) : QPandaException(
        cls+" initialization error"
        ) {}
};

/**
* @class result_get_fail
* @brief QPanda2 get result failed  exception
*/
class result_get_fail :public QPandaException
{
    public:
        result_get_fail(std::string cls) :QPandaException(
        cls+ " get result fail"
        )
    {}
};

/**
* @class result_get_fail
* @brief QPanda2 alloc quantum gate failed exception
*/
class gate_alloc_fail : public QPandaException
{
    public:
        gate_alloc_fail()
    :QPandaException(
        "gate alloc fail")
    {}
        gate_alloc_fail(std::string err)
    : QPandaException(err)
    {};
};

/**
* @class qcircuit_construction_fail
* @brief QPanda2 qcircuit construction failed exception
*/
class qcircuit_construction_fail : public QPandaException
{
public:
    qcircuit_construction_fail()
        : QPandaException(
            "quantum circuit construction failed")
    {}
    qcircuit_construction_fail(std::string err)
        : QPandaException(err)
    {};
};

/**
* @class qprog_construction_fail
* @brief QPanda2 quantum program construction failed exception
*/
class qprog_construction_fail : public QPandaException
{
    public:
        qprog_construction_fail()
    : QPandaException(
        "quantumprogramm construction failed")
    {}
        qprog_construction_fail(std::string err)
        : QPandaException(err)
    {};
};

/**
* @class qprog_syntax_error
* @brief QPanda2 quantum program syntax error exception
*/
class qprog_syntax_error : public QPandaException
{
public:
    qprog_syntax_error()
        : QPandaException(
            "syntax_error")
    {}
    qprog_syntax_error(std::string err)
        : QPandaException(err +" syntax_error")
    {};
};

/**
* @class undefine_error
* @brief QPanda2  undefined error exception
*/
class undefine_error : public QPandaException
{
public:
    undefine_error()
        : QPandaException(
            "undefine error")
    {}
    undefine_error(std::string err)
        : QPandaException("undefine " + err + " error")
    {};
};

/**
* @class qvm_attributes_error
* @brief QPanda2  quantum machine attributes error exception
*/
class qvm_attributes_error : public QPandaException
{
public:
    qvm_attributes_error()
        : QPandaException(
            "global_quantum_machine attributes is nullptr")
    {}
    qvm_attributes_error(std::string err)
        : QPandaException(err)
    {};
};
QPANDA_END
#endif


