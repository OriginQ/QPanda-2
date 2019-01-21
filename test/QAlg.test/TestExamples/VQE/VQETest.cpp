#include <fstream>
#include "VQETest.h"
#include "VQE/VQE.h"
#include "TestManager/TestManager.h"
#include "mpi.h"

namespace QPanda
{
    SINGLETON_IMPLEMENT_EAGER(VQETest)

    bool VQETest::exec(rapidjson::Document &doc)
    {
        auto &optimizer_para = doc[STR_OPTIMIZER];
        std::string optimizer_name;
        RJson::GetStr(optimizer_name, STR_NAME, &optimizer_para);

        QMoleculeGeometry geometry = getProblem(doc[STR_PROBLEM]);
        VQE vqe(optimizer_name);
        vqe.setMoleculeGeometry(geometry);

        auto optimizer = vqe.getOptimizer();
        setOptimizerPara(optimizer, optimizer_para);

        std::string filename = getOutputFile(doc[STR_PARAMETERS], "VQETest_");

        bool use_mpi = false;
        if (doc[STR_PARAMETERS].HasMember(STR_USE_MPI))
        {
            use_mpi = doc[STR_PARAMETERS][STR_USE_MPI].GetBool();
        }

        if (use_mpi)
        {
            return mpiTest(vqe, filename, doc[STR_PARAMETERS]);
        }
        else
        {
            setVQEPara(vqe, doc[STR_PARAMETERS]);
            if (!vqe.exec())
            {
                return false;
            }

            return writeResultToFile(filename, vqe, doc[STR_PARAMETERS]);
        }
    }

    bool VQETest::mpiTest(
            VQE &vqe,
            const std::string &filename,
            const rapidjson::Value &value) const
    {
        MPI_Init(nullptr, nullptr);
        int s = 0;
        int r = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &s);
        MPI_Comm_rank(MPI_COMM_WORLD, &r);

        size_t size = static_cast<size_t>(s);
        size_t rank = static_cast<size_t>(r);

        setVQEPara(vqe, value, rank, size);
        bool ret = vqe.exec();
        if (ret)
        {
            if (vqe.getMoleculeGeometry().size() == 2)
            {
                std::string filename_new = std::to_string(rank) + filename;
                std::fstream out(filename_new, std::ios::out|std::ios::trunc);
                if (out.fail())
                {
                    std::cout << "Open file failed. filename: "
                              << filename_new << std::endl;
                }
                else
                {
                    vector_d distances =
                            get2AtomDistances(value[STR_ATOMS_POS]);
                    vector_d energies = vqe.getEnergies();

                    auto pair = getIndexAndCount(distances.size(), rank, size);
                    for (size_t i = 0; i < energies.size(); i++)
                    {
                        out << distances[i + pair.first]
                                << "\t"
                                << energies[i] << std::endl;
                    }

                    out.close();
                }
            }
        }

        MPI_Finalize();

        return ret;
    }

    void VQETest::setVQEPara(
            VQE &vqe,
            const rapidjson::Value &value,
            size_t rank,
            size_t size) const
    {
        std::string psi4_path;
        RJson::GetStr(psi4_path, STR_PSI4_PATH, &value);
        vqe.setPsi4Path(psi4_path);

        if (value.HasMember(STR_SHOTS))
        {
            vqe.setShots(static_cast<size_t>(value[STR_SHOTS].GetInt()));
        }

        std::string str_basis;
        RJson::GetStr(str_basis, STR_BASIS, &value);
        vqe.setBasis(str_basis);

        vqe.setMultiplicity(value[STR_MULTIPLICITY].GetInt());
        vqe.setCharge(value[STR_CHARGE].GetInt());

        size_t atom_num = vqe.getMoleculeGeometry().size();
        QAtomsPosGroup atom_pos_group;
        if (2 == atom_num)
        {
            atom_pos_group = get2AtomPosGroup(value[STR_ATOMS_POS]);
        }
        else
        {
            size_t check_size = 0;
            size_t size = value[STR_ATOMS_POS].Size();
            for (rapidjson::SizeType i = 0; i < size; i++)
            {
                auto &item = value[STR_ATOMS_POS][i];
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
        }

        if (size != 0)
        {
            auto pair = getIndexAndCount(atom_pos_group.size(), rank, size);

            QAtomsPosGroup tmp;
            for (size_t i = pair.first; i < pair.first + pair.second; i++)
            {
                tmp.push_back(atom_pos_group[i]);
            }
            atom_pos_group = tmp;
        }

        vqe.setAtomsPosGroup(atom_pos_group);
    }

    QMoleculeGeometry
    VQETest::getProblem(const rapidjson::Value &value) const
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
    VQETest::get2AtomPosGroup(const rapidjson::Value &value) const
    {
        QAtomsPosGroup atoms_pos_group;
        QPosition first(0, 0, 0);

        if (value.IsArray())
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

    vector_d VQETest::get2AtomDistances(const rapidjson::Value &value) const
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

    QPosition VQETest::getPos(const rapidjson::Value &value) const
    {
        QPosition atom_pos;
        RJson::GetDouble(atom_pos.x, 0, &value);
        RJson::GetDouble(atom_pos.y, 1, &value);
        RJson::GetDouble(atom_pos.z, 2, &value);

        return atom_pos;
    }

    bool VQETest::writeResultToFile(const std::string &filename,
                                    const VQE &vqe,
                                    const rapidjson::Value &value) const
    {
        if (vqe.getMoleculeGeometry().size() == 2)
        {
            std::fstream out(filename, std::ios::out|std::ios::trunc);
            if (out.fail())
            {
                std::cout << "Open file failed. filename: "
                          << filename << std::endl;
                return false;
            }

            vector_d distances = get2AtomDistances(value[STR_ATOMS_POS]);
            vector_d energies = vqe.getEnergies();

            for (size_t i = 0; i < distances.size(); i++)
            {
                out << distances[i] << "\t" << energies[i] << std::endl;
            }

            out.close();
        }

        return true;
    }

    std::pair<size_t, size_t> VQETest::getIndexAndCount(
            size_t length,
            size_t rank,
            size_t size) const
    {
        if (size == 0)
        {
            return std::make_pair(0, length);
        }
        size_t delta = length / size;
        size_t last = length % size;

        size_t index = (rank < last ? rank : last) + delta * rank;
        size_t count = delta + (rank < last ? 1 : 0);

        return std::make_pair(index, count);
    }

    VQETest::VQETest():
        AbstractTest("VQE")
    {
        TestManager::getInstance()->registerTest(this);
    }

}
