/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

TestManager.h
Author: LiYe
Created in 2018-11-08



*/

#ifndef TESTMANAGER_H
#define TESTMANAGER_H

#include <string>
#include <vector>
#include "QAlg/marco.h"

namespace QPanda
{

class AbstractTest;
class TestManager
{
    SINGLETON_DECLARE(TestManager)
public:
    TestManager();
    void registerTest(AbstractTest *test)
    {
        if (nullptr != test)
        {
            m_test_vec.push_back(test);
        }
    }

    void setConfigFile(const std::string &file)
    {
        m_file = file;
    }

    void setConfigSchema(const std::string &schema)
    {
        m_config_schema = schema;
    }

    bool exec();

    TestManager(const TestManager &) = delete;
    TestManager& operator = (const TestManager &) = delete;
private:
    std::string m_file;
    std::string m_config_schema;
    std::vector<AbstractTest*> m_test_vec;
};

}


#endif // TESTMANAGER_H
