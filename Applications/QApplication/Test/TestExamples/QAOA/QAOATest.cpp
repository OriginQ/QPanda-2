#include <time.h>
#include "QAOATest.h"
#include "QAOA/QAOA.h"
#include "Operator/PauliOperator.h"
#include "QAOATestFactory.h"
#include "AbstractQAOATest.h"
#include "TestManager/TestManager.h"

namespace QPanda
{

    SINGLETON_IMPLEMENT_EAGER(QAOATest)

    double myFunc(const std::string &key, const QPanda::QPauliMap &pauli_map)
    {
        double sum = 0;

        PauliOperator pauli_op(pauli_map);
        QHamiltonian hamiltonian = pauli_op.toHamiltonian();
        //    std::map<size_t, size_t> index_map = pauli_op.getIndexMap();

        for_each(hamiltonian.begin(),
            hamiltonian.end(),
            [&](const QHamiltonianItem &item)
        {
            std::vector<size_t> index_vec;
            for (auto iter = item.first.begin();
                iter != item.first.end();
                iter++)
            {
                index_vec.push_back(iter->first);
            }

            //        double value = item.second;
            size_t i = index_vec.front();
            size_t j = index_vec.back();
            if (key[i] != key[j])
            {
                sum += item.second;
            }
        });

        return sum;
    }

    bool QAOATest::exec(rapidjson::Document &doc)
    {
        auto &optimizer_para = doc[STR_OPTIMIZER];
        std::string optimizer_name;
        RJson::GetStr(optimizer_name, STR_NAME, &optimizer_para);

        QPauliMap pauli_map = getProblem(doc[STR_PROBLEM]);
        QAOA qaoa(optimizer_name);
        qaoa.setHamiltonian(pauli_map);
        qaoa.regiestUserDefinedFunc(std::bind(&myFunc,
                    std::placeholders::_1,
                    pauli_map));

        auto optimizer = qaoa.getOptimizer();
        setOptimizerPara(optimizer, optimizer_para);
        setQAOAPara(qaoa, doc[STR_PARAMETERS]);

        if (doc[STR_PARAMETERS].HasMember(STR_TEST))
        {
            rapidjson::Value &test_value = doc[STR_PARAMETERS][STR_TEST];
            return doTest(qaoa, test_value);
        }

        return qaoa.exec();
    }

    void QAOATest::setQAOAPara(QAOA &qaoa, rapidjson::Value &value)
    {
        if (value.HasMember(STR_STEP))
        {
            qaoa.setStep(static_cast<size_t>(value[STR_STEP].GetInt()));
        }

        if (value.HasMember(STR_SHOTS))
        {
            qaoa.setShots(static_cast<size_t>(value[STR_SHOTS].GetInt()));
        }

        if (value.HasMember(STR_DELTA_T))
        {
            qaoa.setDeltaT(value[STR_DELTA_T].GetDouble());
        }

        if (value.HasMember(STR_INITIAL))
        {
            rapidjson::Value &initial = value[STR_INITIAL];
            bool random = initial[STR_RANDOM].GetBool();
            vector_d para_ves;

            rapidjson::Value &value = initial[STR_VALUE];
            if (!random)
            {
                for (rapidjson::SizeType i = 0; i < value.Size(); i++)
                {
                    para_ves.push_back(value[i].GetDouble());
                }
            }
            else
            {
                double min = value[STR_MIN].GetDouble();
                double max = value[STR_MAX].GetDouble();
                if (min > max)
                {
                    std::swap(min, max);
                }

                srand(static_cast<unsigned>(time(nullptr)));
                size_t step = qaoa.step();
                for (size_t i = 0; i < step*2; i++)
                {
                    para_ves.push_back(rand()*1.0 / RAND_MAX * (max - min) + min);
                }
            }

            qaoa.setDefaultOptimizePara(para_ves);
        }
    }

    QPauliMap QAOATest::getProblem(rapidjson::Value &value)
    {
        QPauliMap pauli_map;
        for (rapidjson::SizeType i = 0; i < value.Size(); i++)
        {
            std::string str1;
            RJson::GetStr(str1, 0, &value[i]);

            std::string str2;
            RJson::GetStr(str2, 1, &value[i]);

            double node_value;
            RJson::GetDouble(node_value, 2, &value[i]);

            auto node = str1 + " " + str2;
            pauli_map.insert(std::make_pair(node, node_value));
        }

        return  pauli_map;
    }

    bool QAOATest::doTest(QAOA &qaoa, rapidjson::Value &value)
    {
        std::string test_name;
        RJson::GetStr(test_name, STR_NAME, &value);
        auto test = QAOATestFactory::makeQAOATest(test_name);
        if (test.get())
        {
            if (value.HasMember(STR_USE_MPI))
            {
                test->setUseMPI(value[STR_USE_MPI].GetBool());
            }

            std::string filename = getOutputFile(value, "QAOATest_");
            test->setOutputFile(filename);
            test->setPara(value[STR_PARA]);

            return test->exec(qaoa);
        }

        std::cout << "No test can be found." << std::endl;
        return true;
    }

    QAOATest::QAOATest():
        AbstractTest("QAOA")
    {
        TestManager::getInstance()->registerTest(this);
    }

}
