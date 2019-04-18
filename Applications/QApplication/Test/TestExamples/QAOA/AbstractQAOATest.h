/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

AbstractQAOATest.h

Author: LiYe
Created in 2018-11-12


*/

#ifndef ABSTRACTQAOATEST_H
#define ABSTRACTQAOATEST_H

#include "QAlg/DataStruct.h"
#include "rapidjson/document.h"

namespace QPanda
{

    class QAOA;
    class AbstractQAOATest
    {
    public:
        AbstractQAOATest();
        virtual ~AbstractQAOATest();

        void setUseMPI(bool use_mpi)
        {
            m_use_mpi = use_mpi;
        }

        void setOutputFile(const std::string &file)
        {
            m_output_file = file;
        }

        void setPara(rapidjson::Value &para)
        {
            m_para = para;
        }

        virtual bool exec(QAOA &qaoa) = 0;
    protected:
        bool m_use_mpi{false};
        std::string m_output_file;
        rapidjson::Value m_para;
    };

}

#endif // ABSTRACTQAOATEST_H
