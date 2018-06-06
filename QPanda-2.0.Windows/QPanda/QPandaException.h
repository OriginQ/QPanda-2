#ifndef _QPANDA_EXCEPTION_H
#define _QPANDA_EXCEPTION_H
#include <exception>
#include <iostream>
using std::exception;
using std::string;

class QPandaException : public exception
{
public:
    QPandaException(const char* str, bool isfree)
        :exception(str, isfree) {}
    QPandaException(string str, bool isfree)
        :exception(str.c_str(), isfree) {}

};

#endif // !_QPANDA_EXCEPTION_H

