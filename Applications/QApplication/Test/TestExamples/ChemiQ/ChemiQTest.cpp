#include <fstream>
#include <algorithm>
#include <cctype>
#include "ChemiQTest.h"
#include "ChemiQ.h"
#include "TestManager/TestManager.h"

namespace QPanda
{
    SINGLETON_IMPLEMENT_EAGER(ChemiQTest)

    bool ChemiQTest::exec(rapidjson::Document &doc)
    {
        ChemiQ chemiq;
        setChemiQPara(chemiq, doc);

        bool ret = chemiq.exec();
        chemiq.finalize();

        return ret;
    }

    void ChemiQTest::setOptimizerPara(
        ChemiQ& chemiq, 
        const rapidjson::Value& value) const
    {
        if (value.HasMember(STR_NAME))
        {
            std::string s;
            RJson::GetStr(s, STR_NAME, &value);
            transform(s.begin(), s.end(), s.begin(), ::toupper);
            if (s == "NELDER-MEAD")
            {
                chemiq.setOptimizerType(OptimizerType::NELDER_MEAD);
            }
            else if (s == "POWELL")
            {
                chemiq.setOptimizerType(OptimizerType::POWELL);
            }
            else if (s == "GRADIENT")
            {
                chemiq.setOptimizerType(OptimizerType::GRADIENT);
                if (value.HasMember(STR_LEARNING_RATE))
                {
                    chemiq.setLearningRate(value[STR_LEARNING_RATE].GetDouble());
                }
            }
            else
            {
                chemiq.setOptimizerType(OptimizerType::NELDER_MEAD);
            }

        }
        if (value.HasMember(STR_MAX_ITER))
        {
            chemiq.setOptimizerIterNum(
                static_cast<size_t>(value[STR_MAX_ITER].GetInt()));
        }

        if (value.HasMember(STR_MAX_FCALLS))
        {
            chemiq.setOptimizerFuncCallNum(
                static_cast<size_t>(value[STR_MAX_FCALLS].GetInt()));
        }

        if (value.HasMember(STR_XATOL))
        {
            chemiq.setOptimizerXatol(value[STR_XATOL].GetDouble());
        }

        if (value.HasMember(STR_FATOL))
        {
            chemiq.setOptimizerFatol(value[STR_FATOL].GetDouble());
        }
    }

    void ChemiQTest::setChemiQPara(
            ChemiQ &chemiq,
            const rapidjson::Value &doc) const
    {
        auto& optimizer_para = doc[STR_OPTIMIZER];
        setOptimizerPara(chemiq, optimizer_para);

        auto &value = doc[STR_PARAMETERS];

        std::string psi4_path;
        RJson::GetStr(psi4_path, STR_PSI4_PATH, &value);
        chemiq.initialize(psi4_path);

        if (value.HasMember(STR_DATA_SAVE_PATH))
        {
            std::string data_save_path;
            RJson::GetStr(data_save_path, STR_DATA_SAVE_PATH, &value);
            chemiq.setSaveDataDir(data_save_path);
        }

        if (value.HasMember(STR_QUANTUM_MARCHINE_TYPE))
        {
            int cpu_type = value[STR_QUANTUM_MARCHINE_TYPE].GetInt();
            if (cpu_type <= NOISE)
            {
                chemiq.setQuantumMachineType(QMachineType(cpu_type));
            }
        }

        if (value.HasMember(STR_UCC_TYPE))
        {
            std::string ucc_type;
            RJson::GetStr(ucc_type, STR_UCC_TYPE, &value);
            transform(ucc_type.begin(), ucc_type.end(),
                ucc_type.begin(), ::toupper);
            if (ucc_type == "UCCS")
            {
                chemiq.setUccType(UccType::UCCS);
            }
            else if (ucc_type == "UCCSD")
            {
                std::cout << "set ucc type with UCCSD" << std::endl;
                chemiq.setUccType(UccType::UCCSD);
            }
        }

        if (value.HasMember(STR_TRANSFORM_TYPE))
        {
            std::string transform_type;
            RJson::GetStr(transform_type, STR_TRANSFORM_TYPE, &value);
            transform(transform_type.begin(), transform_type.end(), 
                transform_type.begin(), ::toupper);
            if (transform_type == "JW")
            {
                chemiq.setTransformType(TransFormType::Jordan_Wigner);
            }
            else if (transform_type == "PARITY")
            {
                chemiq.setTransformType(TransFormType::Parity);
            }
            else if (transform_type == "BK")
            {
                chemiq.setTransformType(TransFormType::Bravyi_Ktaev);
            }
            else
            {
                chemiq.setTransformType(TransFormType::Jordan_Wigner);
            }
        }

        if (value.HasMember(STR_BASIS))
        {
            std::string str_basis;
            RJson::GetStr(str_basis, STR_BASIS, &value);
            chemiq.setBasis(str_basis);
        }
        
        if (value.HasMember(STR_MULTIPLICITY))
        {
            chemiq.setMultiplicity(value[STR_MULTIPLICITY].GetInt());
        }

        if (value.HasMember(STR_CHARGE))
        {
            chemiq.setCharge(value[STR_CHARGE].GetInt());
        }
        
        if (value.HasMember(STR_EVOLUTION_TIME))
        {
            chemiq.setEvolutionTime(value[STR_EVOLUTION_TIME].GetDouble());
        }

        if (value.HasMember(STR_HAMILTONIAN_SIMULATION_SLICES))
        {
            chemiq.setEvolutionTime(
                value[STR_HAMILTONIAN_SIMULATION_SLICES].GetInt()
            );
        }

        if (value.HasMember(STR_GET_HAMILTONINAN_FROM_FILE))
        {
            chemiq.setToGetHamiltonianFromFile(
                value[STR_GET_HAMILTONINAN_FROM_FILE].GetBool()
            );
        }

        if (value.HasMember(STR_RANDOM))
        {
            chemiq.setRandomPara(value[STR_RANDOM].GetBool());
        }

        if (value.HasMember(STR_HAMILTONIAN_GENERATION_ONLY))
        {
            chemiq.setHamiltonianGenerationOnly(
                value[STR_HAMILTONIAN_GENERATION_ONLY].GetBool()
            );
        }

        if (value.HasMember(STR_INITIAL))
        {
            auto &intial_para = value[STR_INITIAL];
            vector_d para;
            for (rapidjson::SizeType i = 0; i < intial_para.Size(); i++)
            {
                para.push_back(intial_para[i].GetDouble());
            }
            chemiq.setDefaultOptimizedPara(para);
        }

        QMoleculeGeometry geometry = getProblem(doc[STR_PROBLEM]);
        size_t atom_num = geometry.size();
        QAtomsPosGroup atom_pos_group;
        if (2 == atom_num)
        {
            atom_pos_group = get2AtomPosGroup(value[STR_ATOMS_POS]);
        }
        else
        {
            atom_pos_group = getNormalAtomPosGroup(value[STR_ATOMS_POS]);
        }
        vector_s molecules;
        for (auto i = 0u; i < atom_pos_group.size(); i++)
        {
            std::string melecular;
            for (auto j = 0u; j < atom_pos_group[i].size(); j++)
            {
                auto& atom = geometry[j].first;
                auto& pos = atom_pos_group[i][j];

                melecular += atom + " " + std::to_string(pos.x) +
                    " " + std::to_string(pos.y) + " "
                        + std::to_string(pos.z) + "\n";
            }
            molecules.push_back(melecular);
        }

        chemiq.setMolecules(molecules);
    }

    QMoleculeGeometry
    ChemiQTest::getProblem(const rapidjson::Value &value) const
    {
        QMoleculeGeometry geometry;
        for (rapidjson::SizeType i = 0; i < value.Size(); i++)
        {
            std::string atom;
            RJson::GetStr(atom, 0, &value[i]);

            QPosition atom_pos = getPos(value[i][1]);
            geometry.push_back(std::make_pair(atom, atom_pos));
        }

        return geometry;
    }

    QAtomsPosGroup
    ChemiQTest::get2AtomPosGroup(const rapidjson::Value &value) const
    {
        QAtomsPosGroup atoms_pos_group;
        QPosition first(0, 0, 0);

        if (value.IsArray())
        {
            if (value[0].IsArray())
            {
                atoms_pos_group = getNormalAtomPosGroup(value);
            }
            else
            {
                for (rapidjson::SizeType i = 0; i < value.Size(); i++)
                {
                    std::vector<QPosition> vec;
                    QPosition second(0, 0, 0);
                    RJson::GetDouble(second.z, i, &value);

                    vec.push_back(first);
                    vec.push_back(second);
                    atoms_pos_group.push_back(vec);
                }
            }
        }
        else
        {
            double begin = 0.0;
            RJson::GetDouble(begin, STR_BEGIN, &value);

            double end = 0.0;
            RJson::GetDouble(end, STR_END, &value);

            int size = 0;
            RJson::GetInt(size, STR_SIZE, &value);

            if (begin > end)
            {
                std::swap(begin, end);
            }

            double delta = (end - begin)/size;
            for (int i = 0; i < size; i++)
            {
                std::vector<QPosition> vec;
                QPosition second(0, 0, 0);
                second.z = begin + i*delta;

                vec.push_back(first);
                vec.push_back(second);
                atoms_pos_group.push_back(vec);
            }
        }

        return atoms_pos_group;
    }

    QAtomsPosGroup
    ChemiQTest::getNormalAtomPosGroup(const rapidjson::Value &value) const
    {
        QAtomsPosGroup atom_pos_group;
        size_t check_size = 0;
        size_t size = value.Size();
        for (rapidjson::SizeType i = 0; i < size; i++)
        {
            auto &item = value[i];
            if (i == 0)
            {
                check_size = item.Size();
            }
            else if (item.Size() != check_size)
            {
                atom_pos_group.clear();
                break;
            }

            std::vector<QPosition> vec;
            for (rapidjson::SizeType j = 0; j < item.Size(); j++)
            {
                QPosition pos = getPos(item[j]);
                vec.push_back(pos);
            }

            atom_pos_group.push_back(vec);
        }

        return atom_pos_group;
    }

    vector_d ChemiQTest::get2AtomDistances(const rapidjson::Value &value) const
    {
        vector_d distances;
        if (value.IsArray())
        {
            for (rapidjson::SizeType i = 0; i < value.Size(); i++)
            {
                double distance = 0.0;
                RJson::GetDouble(distance, i, &value);
                distances.push_back(distance);
            }
        }
        else
        {
            double begin = 0.0;
            RJson::GetDouble(begin, STR_BEGIN, &value);

            double end = 0.0;
            RJson::GetDouble(end, STR_END, &value);

            int size = 0;
            RJson::GetInt(size, STR_SIZE, &value);

            if (begin > end)
            {
                std::swap(begin, end);
            }

            double delta = (end - begin)/size;
            for (int i = 0; i < size; i++)
            {
                distances.push_back(begin + i*delta);
            }
        }

        return distances;
    }

    QPosition ChemiQTest::getPos(const rapidjson::Value &value) const
    {
        QPosition atom_pos;
        RJson::GetDouble(atom_pos.x, 0, &value);
        RJson::GetDouble(atom_pos.y, 1, &value);
        RJson::GetDouble(atom_pos.z, 2, &value);

        return atom_pos;
    }

    ChemiQTest::ChemiQTest():
        AbstractTest("ChemiQ")
    {
        TestManager::getInstance()->registerTest(this);
    }

}
