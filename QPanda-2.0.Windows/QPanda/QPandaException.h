#ifndef QPANDA_EXCEPTION_H
#define QPANDA_EXCEPTION_H

#include <exception>
#include <string>
using namespace std;

class QPandaException : public exception
{
public:

	QPandaException(const char* str, bool isfree)
#if defined (_WIN32)
		: exception(str, isfree) {}
#else
		: exception(string(str), isfree) {}
#endif

	QPandaException(string str,bool isfree)
#if defined (_WIN32)
		: exception(str.c_str(), isfree) {}
#else
		: exception(str, isfree) {}
#endif
	
};

class qalloc_fail : public QPandaException
{
public:
	qalloc_fail() : QPandaException(
		"qalloc fail",
		false
	) {}
};

class calloc_fail : public QPandaException
{
public:
	calloc_fail() : QPandaException(
		"calloc fail",
		false
	) {}
};

class factory_init_error : public QPandaException
{
public:
	factory_init_error(string cls) : QPandaException(
		cls+" initialization error",
		false
	) {}
};

class classical_system_exception : public QPandaException
{
public:
	classical_system_exception(string err, bool isfree) 
		: QPandaException(err,isfree)
	{};
};

class operator_specifier_error : public classical_system_exception
{
public:
	operator_specifier_error()
		: classical_system_exception(
			"Operator not existed",
			false)
	{}
};

class content_specifier_error : public classical_system_exception
{
public:
	content_specifier_error()
		: classical_system_exception(
			"contentSpecifier invalid",
			false)
	{}
};

class eval_error : public classical_system_exception
{
public:
	eval_error()
		: classical_system_exception(
			"Evaluation of expression has failed",
			false)
	{}
};

#endif