/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QAOAParaScan.h

Author: LiYe
Created in 2018-11-12


*/

#ifndef QAOAPARASCAN_H
#define QAOAPARASCAN_H

#include "AbstractQAOATest.h"

namespace QPanda
{

    class QAOAParaScan : public AbstractQAOATest
    {
    public:
        QAOAParaScan();
        virtual bool exec(QAOA &qaoa);
    private:
        QTwoPara  get2P(rapidjson::Value &value);
        vector_i get2Pos(rapidjson::Value &value);
        vector_i getKeys(rapidjson::Value &value);
    };

}

#endif // QAOAPARASCAN_H
