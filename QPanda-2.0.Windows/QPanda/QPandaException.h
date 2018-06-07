#ifndef _QPANDA_EXCEPTION_H
#define _QPANDA_EXCEPTION_H
#include <exception>
#include <iostream>
using std::exception;
using std::string;

class QPandaException : public exception
{
    bool isFree;
    string errmsg;
public:
    QPandaException(const char* str, bool isfree)
        :exception() {
        errmsg = str;
        isFree = isfree;
    }
    QPandaException(string str, bool isfree)
        :exception() {
        errmsg = str;
        isFree = isfree;
    }

};

#endif // !_QPANDA_EXCEPTION_H

