/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

AbstractTest.h

Author: LiYe
Created in 2018-11-08


*/

#ifndef ABSTRACTTEST_H
#define ABSTRACTTEST_H

#include <string>
#include "QAlg/DataStruct.h"
#include "RJson/RJson.h"
#include "QAlg/macro.h"
#include "tag_macro.h"

namespace QPanda
{

class AbstractOptimizer;
class AbstractTest
{
public:
    AbstractTest(const std::string &tag);
    virtual ~AbstractTest();
    bool canHandle(const std::string &tag)
    {
        return tag == m_tag;
    }

    virtual bool exec(rapidjson::Document &doc) = 0;
protected:
    void setOptimizerPara(
            AbstractOptimizer *optimizer,
            rapidjson::Value &value);
    std::string getOutputFile(rapidjson::Value &value,
                              std::string default_name_prefix = "");
protected:
    std::string m_tag;
};

}

#endif // ABSTRACTTEST_H
