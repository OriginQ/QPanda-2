/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QAOATest.h

Author: LiYe
Created in 2018-11-08


*/

#ifndef QAOATEST_H
#define QAOATEST_H

#include "TestInterface/AbstractTest.h"

namespace QPanda
{

    class QAOA;
    class QAOATest : public AbstractTest
    {
        SINGLETON_DECLARE(QAOATest)
    public:
        virtual bool exec(rapidjson::Document &doc);
    private:
        void setQAOAPara(QAOA &qaoa, rapidjson::Value &value);
        QPauliMap getProblem(rapidjson::Value &value);
        bool doTest(QAOA &qaoa, rapidjson::Value &value);
        QAOATest();
    };

}

#endif // QAOATEST_H
