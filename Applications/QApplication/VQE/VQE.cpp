#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <cstdio>
#include "VQE.h"
#include "Optimizer/OptimizerFactory.h"
#include "Optimizer/AbstractOptimizer.h"
#include "HamiltonianSimulation/HamiltonianSimulation.h"
#include "QString.h"
#include "psi4_input_template.h"
#include "OriginCollection.h"
#include "Operator/PauliOperator.h"

#ifdef _WIN32
#include <Windows.h>
#endif

using namespace std;
namespace QPanda
{

#define STR_PSI4_INPUT_FILE                         ("psi4.input.inp")
#define STR_PSI4_OUPUT_FILE                         ("psi4.input.inp.dat")
#define STR_PSI4_USER_DEFINED_DATA_FILE             ("psi4.data.tmp")
#define STR_PSI4_LOG_FILE                           ("psi4.log")

    std::string dec2Bin(size_t n, size_t size)
    {
        std::string binstr = "";
        for (size_t i = 0; i < size; i++)
        {
            std::string t = n & 1 ? "1":"0";
            binstr = t + binstr;
            n >>= 1;
        }
        return binstr;
    }

    VQE::VQE(OptimizerType optimizer)
    {
        m_optimizer = OptimizerFactory::makeOptimizer(optimizer);
        if (nullptr == m_optimizer.get())
        {
            QCERR("No optimizer.");
            throw runtime_error("No optimizer.");
        }
    }

    VQE::VQE(const string &optimizer)
    {
        m_optimizer = OptimizerFactory::makeOptimizer(optimizer);
        if (nullptr == m_optimizer.get())
        {
            QCERR("No optimizer.");
            throw runtime_error("No optimizer.");
        }
    }

    VQE::~VQE()
    {

    }

    bool VQE::exec()
    {
        bool exec_flag = true;
        do
        {
#ifdef _WIN32
        ::ShowWindow(::GetConsoleWindow(), SW_HIDE);
#endif
            if (!initVQE())
            {
                exec_flag = false;
                break;
            }

            QInit();

            m_energies.clear();
            for (size_t i = 0; i < m_atoms_pos_group.size(); i++)
            {
                PauliOperator pauli;
                auto geometry = genMoleculeGeometry(i);
                if (!getDataFromPsi4(geometry, pauli))
                {
                    exec_flag = false;
                    break;
                }

                m_optimizer->registerFunc(std::bind(&VQE::callVQE,
                    this,
                    std::placeholders::_1,
                    pauli.toHamiltonian()),
                    m_optimized_para);

                if (m_enable_optimizer_data)
                {
                    std::string filename = m_data_save_path + "optimizer_" +
                            std::to_string(i) + ".dat";
                    m_optimizer_data_db = OriginCollection(filename, false);
                    m_optimizer_data_db = {"index", "fun_val", "para"};
                    m_func_calls = 0;
                }

                m_optimizer->exec();

                if (m_enable_optimizer_data)
                {
                    m_optimizer_data_db.write();
                }

                auto result = m_optimizer->getResult();
                m_energies.push_back(result.fun_val);

                saveResult(result, i);
            }

            QFinalize();
        } while (0);

        writeExecLog(exec_flag);

#ifdef _WIN32
        ::ShowWindow(::GetConsoleWindow(), SW_SHOW);
#endif

        return exec_flag;
    }

    void VQE::QInit()
    {
        init();

        m_qubit_vec.clear();
        m_cbit_vec.clear();
        for (size_t i = 0; i < m_qn; i++)
        {
            m_qubit_vec.emplace_back(qAlloc());
            m_cbit_vec.emplace_back(cAlloc());
        }
    }

    void VQE::QFinalize()
    {
        std::for_each(m_cbit_vec.begin(), m_cbit_vec.end(),
            [](ClassicalCondition cbit) { cFree(cbit); });

        finalize();
    }

    bool VQE::initVQE()
    {
        do {
            if (!checkPara(m_last_err))
            {
                break;
            }

            PauliOperator pauli;
            if (!getDataFromPsi4(m_geometry, pauli))
            {
                break;
            }

            m_qn = pauli.getMaxIndex();
            m_electron_num = getElectronNum(m_geometry);

            auto n_param = getCCS_N_Trem(m_qn, m_electron_num);
            m_optimized_para.resize(n_param, 0.5);

            if (!m_data_save_path.empty())
            {
                auto pos = m_data_save_path.find_last_of('/');
                if (pos != (m_data_save_path.length() -1))
                {
                    m_data_save_path += '/';
                }
            }

            return true;
        } while (0);

        return false;
    }

    bool VQE::checkPara(std::string &err_msg)
    {
        do
        {
            if (m_psi4_path.empty())
            {
                err_msg = "Psi4 file path is empty.";
                break;
            }

            if (m_geometry.empty())
            {
                err_msg = "Molecule geometry is empty.";
                break;
            }

            if (m_atoms_pos_group.empty()||
                    m_atoms_pos_group[0].empty())
            {
                err_msg = "Distance range is empty.";
                break;
            }

            if (m_geometry.size() != m_atoms_pos_group[0].size())
            {
                err_msg = "Atoms and atom positions are not equal. ";
                break;
            }

            if (0 == m_shots)
            {
                err_msg = "Shots num is zero.";
                break;
            }

            return true;
        } while (0);

        return false;
    }

    QMoleculeGeometry VQE::genMoleculeGeometry(const size_t &index)
    {
        QMoleculeGeometry geometry = m_geometry;

        auto &group = m_atoms_pos_group[index];
        for (size_t i = 0; i < group.size(); i++)
        {
            auto &atom = geometry[i];
            atom.second = group[i];
        }

        return geometry;
    }

    PauliOperator VQE::genPauliOpComplex(const std::string &s, bool *ok)
    {
        QString str(s);
        auto vec = str.split("\n", QString::SkipEmptyParts);

        QPauliMap map;
        bool flag = true;
        for (size_t i = 0; i < vec.size(); i++)
        {
            auto item = vec[i];
            auto item_vec = item.split("j)", QString::SkipEmptyParts);

            if (item_vec.size() < 2)
            {
                flag = false;
                break;
            }

            auto tmp_key =
                   item_vec[1].split("+", QString::SkipEmptyParts)[0].trimmed();
            std::string key = tmp_key.mid(1, tmp_key.size() - 2).data();
            auto value = genComplex(item_vec[0].trimmed().data());
            map.insert(std::make_pair(key, value));
        }

        if (ok)
        {
            *ok = flag;
        }

        PauliOperator pauli(map);
        return pauli;
    }

    complex_d VQE::genComplex(const std::string &s)
    {
        QString str(s);
        str = str.mid(1);

        bool real_negative_sign = false;
        if ('-' == str.at(0))
        {
            real_negative_sign = true;
            str = str.right(str.size() - 2);
        }

        bool imag_negative_sign = false;
        auto vec = str.split("+", QString::SkipEmptyParts);
        if (vec.empty())
        {
            imag_negative_sign = true;
            vec = str.split("-", QString::SkipEmptyParts);
        }

        if (2 != vec.size())
        {

            std::string err = std::string("Bad complex string: ") + str.data();
            std::cout << err << std::endl;
            QCERR(err);
            throw runtime_error(err);
        }

        auto real = vec[0].toDouble();
        auto imag = vec[1].toDouble();

        if (real_negative_sign)
        {
            real = -real;
        }

        if (imag_negative_sign)
        {
            imag = -imag;
        }

        return complex_d(real, imag);
    }

    size_t VQE::getElectronNum(const QMoleculeGeometry &geometry)
    {
        size_t electron_num = 0;
        for (size_t i = 0; i < geometry.size(); i++)
        {
            electron_num += g_kAtomElectrons.at(geometry[i].first);
        }

        return electron_num;
    }

    size_t VQE::getCCS_N_Trem(const size_t & qn, const size_t & en)
    {
        if (qn < en)
        {
            std::string err = "Qubit num is less than electron num.";
            std::cout << err << std::endl;
            QCERR(err);
            throw runtime_error(err);
        }

        return (qn - en) * en;
    }

    size_t VQE::getCCSD_N_Trem(const size_t & qn, const size_t & en)
    {
        if (qn < en)
        {
            std::string err = "Qubit num is less than electron num.";
            std::cout << err << std::endl;
            QCERR(err);
            throw runtime_error(err);
        }

        return (qn - en) * en + (qn - en)* (qn -en - 1) * en * (en - 1) / 4;
    }

    PauliOperator VQE::getCCS(
        const size_t & qn, 
        const size_t & en, 
        const vector_d & para_vec)
    {
        if (qn < en)
        {
            std::string err = "Qubit num is less than electron num.";
            std::cout << err << std::endl;
            QCERR(err);
            throw runtime_error(err);
        }

        if (qn == en)
        {
            return PauliOperator();
        }

        size_t cnt = 0;
        PauliOperator result_op;
        for (size_t i = 0; i < en; i++)
        {
            for (auto ex = en; ex < qn; ex++)
            {
                auto t1 = getFermionJordanWigner('c', ex);
                auto t2 = getFermionJordanWigner('a', i);
                auto t3 = t1*t2*para_vec[cnt];

                auto tmp_op = result_op;
                result_op = tmp_op + t3;

                cnt++;
            }
        }

        return result_op;
    }

    PauliOperator VQE::getCCSD(
        const size_t &qn, 
        const size_t &en, 
        const vector_d &para_vec)
    {
        if (qn < en)
        {
            std::string err = "Qubit num is less than electron num.";
            std::cout << err << std::endl;
            QCERR(err);
            throw runtime_error(err);
        }

        if (qn == en)
        {
            return PauliOperator();
        }

        size_t cnt = 0;
        PauliOperator result_op;
        for (size_t i = 0; i < en; i++)
        {
            for (size_t j = i + 1; j < en; j++)
            {
                for (size_t ex1 = en; ex1 < qn; ex1++)
                {
                    for (size_t ex2 = ex1 + 1; ex2 < qn; ex2++)
                    {
                        result_op +=
                            getFermionJordanWigner('c', ex2)*
                            getFermionJordanWigner('c', ex1)*
                            getFermionJordanWigner('a', j)*
                            getFermionJordanWigner('a', i)*
                            para_vec[cnt];
                        cnt++;
                    }
                }
            }
        }

        return result_op;
    }

    PauliOperator VQE::getFermionJordanWigner(
        const char &fermion_type, 
        const size_t &op_qubit)
    {
        std::string op_str;
        for (size_t i = 0; i < op_qubit; i++)
        {
            op_str += "Z" + std::to_string(i) + " ";
        }

        std::string op_str1 = op_str + "X" + std::to_string(op_qubit);
        std::string op_str2 = op_str + "Y" + std::to_string(op_qubit);

        QPauliMap map;
        map.insert(std::make_pair(op_str1, 1));
        if ('a' == fermion_type)
        {
            map.insert(std::make_pair(op_str2, complex_d(0,1)));
        }
        else if ('c' == fermion_type)
        {
            map.insert(std::make_pair(op_str2, complex_d(0, -1)));
        }
        else
        {
            std::string err = "Bad fermion type.";
            std::cout << err << std::endl;
            QCERR(err);
            throw runtime_error(err);
        }

        return PauliOperator(map);
    }

    PauliOperator VQE::transCC2UCC(const PauliOperator &cc)
    {
        return complex_d(0, 1)*(cc - cc.dagger());
    }

    QCircuit VQE::transformBase(const QTerm &base)
    {
        size_t cnt = 0;
        auto instance = QGateNodeFactory::getInstance();

        QCircuit circuit;
        for (auto iter = base.begin(); iter != base.end(); iter++, cnt++)
        {
            if (cnt % 3 == 0)
            {
                char ch = iter->second;
                if (ch == 'X')
                {
                    circuit << instance->getGateNode(
                        "H", 
                        m_qubit_vec[iter->first]);
                }
                else if (ch == 'Y')
                {
                    circuit << instance->getGateNode(
                        "RX", 
                        m_qubit_vec[iter->first],
                        Q_PI_2);
                }
                else if (ch == 'Z')
                {
                    continue;
                }
                else
                {
                    std::string err =
                            std::string("Error char not in [XYZ]. char: ") + ch;
                    std::cout << err << std::endl;
                    QCERR(err);
                    throw runtime_error(err);
                }
            }
        }

        return circuit;
    }

    double VQE::getExpectation(
            const QHamiltonian & unitary_cc,
            const QHamiltonianItem & component)
    {
        m_prog.clear();

        auto instance = QGateNodeFactory::getInstance();
        m_prog << instance->getGateNode("X", m_qubit_vec[0])
               << instance->getGateNode("X", m_qubit_vec[2])
               << simulateHamiltonian(m_qubit_vec, unitary_cc, 1, 3);

        if (!component.first.empty())
        {
            m_prog << transformBase(component.first);
        }

        directlyRun(m_prog);

        double expectation = 0.0;
        auto result = getProbabilites();
        for (auto iter = result.begin(); iter != result.end(); iter++)
        {
            if (PairtyCheck(iter->first, component.first))
            {
                expectation -= iter->second;
            }
            else
            {
                expectation += iter->second;
            }
        }

        return expectation * component.second;
    }

    QProbMap VQE::getProbabilites(int select_max)
    {
        const auto prob = PMeasure(m_qubit_vec, select_max);

        QProbMap map;
        for (size_t i = 0; i < prob.size(); i++)
        {
            const auto &item = prob[i];
            map.insert(std::make_pair(
                dec2Bin(item.first, m_qubit_vec.size()),
                item.second));
        }

        return map;
    }

    bool VQE::PairtyCheck(const std::string &str, const QTerm &paulis)
    {
        size_t check = 0;

        for (auto iter = paulis.begin(); iter != paulis.end(); iter++)
        {
            if ('1' == str[iter->first])
            {
                check++;
            }
        }

        return 1 == check % 2;
    }

    bool VQE::getDataFromPsi4(
            const QMoleculeGeometry &geometry,
            PauliOperator &pauli)
    {
        std::fstream f(STR_PSI4_INPUT_FILE, std::ios::out);
        if (f.fail())
        {
            std::cout << "Open file failed. filename: "
                      << STR_PSI4_INPUT_FILE << std::endl;
            return false;
        }

        std::string file_content =
                std::string(kPsi4_s) +
                std::string(kMolecular_s) +
                QMoleculeGeometry2String(geometry) +
                std::string(kMolecular_e) +
                std::string(kMultiplicit_s) +
                std::to_string(m_multiplicity) +
                std::string(kMultiplicit_e) +
                std::string(kCharge_s) +
                std::to_string(m_charge) +
                std::string(kCharge_e) +
                std::string(kGlobals_s) +
                std::string(kBasis) + m_basis +
                std::string(kGlobals_e) +
                std::string(kPsi4_e);

        f << file_content << std::endl;
        f.close();

        std::string cmd = std::string("python ")
                + m_psi4_path + " " + STR_PSI4_INPUT_FILE;
        int ret = 0;

#ifdef _WIN32
//        ::ShowWindow(::GetConsoleWindow(), SW_HIDE);
        ret = system(cmd.c_str());
//        ::ShowWindow(::GetConsoleWindow(), SW_SHOW);
#else
        ret = system(cmd.c_str());
#endif

        if (0 != ret)
        {
            std::fstream out(STR_PSI4_LOG_FILE, std::ios::in);
            if (out.fail())
            {
                m_last_err = "Unknow error!";
            }
            else
            {
                std::string line;
                while(std::getline(out, line))
                {
                    m_last_err += line + "\n";
                }

                out.close();
                std::remove(STR_PSI4_LOG_FILE);
            }
            std::cout << "Run cmd falid. cmd: " << cmd << std::endl;
            return false;
        }

        std::remove(STR_PSI4_OUPUT_FILE);
        std::remove("timer.dat");

        if (!psi4DataToPauli(STR_PSI4_USER_DEFINED_DATA_FILE, pauli))
        {
            return false;
        }

        std::remove(STR_PSI4_USER_DEFINED_DATA_FILE);

        std::cout << pauli << std::endl;

        return true;
    }

    string VQE::QMoleculeGeometry2String(const QMoleculeGeometry &geometry)
    {
        std::string str;
        for (size_t i = 0; i < geometry.size(); i++)
        {
            const auto &item = geometry[i];
            str += item.first + " " + std::to_string(item.second.x)
                    + " " + std::to_string(item.second.y) + " "
                    + std::to_string(item.second.z) + "\n";
        }

        return str;
    }

    bool VQE::psi4DataToPauli(
            const string &filename,
            PauliOperator &pauli)
    {
        std::ifstream f(filename);
        if (f.fail())
        {
            std::cout << "Open file failed! " << filename
                << std::endl;
            return false;
        }
        std::stringstream buffer;
        buffer << f.rdbuf();
        f.close();

        QString contents(buffer.str());
        auto contents_vec = contents.split("\r\n", QString::SkipEmptyParts);
        for (size_t i = 0; i < contents_vec.size(); i++)
        {
            const auto &item  = contents_vec[i];
            auto item_vec = item.split(":",  QString::SkipEmptyParts);
            if (2 != item_vec.size())
            {
                std::cout << "file format error! filename: "
                          << filename << std::endl;
                return false;
            }

            auto real_value = item_vec[1].toDouble();
            complex_d value(real_value);
            auto transformed_term =
                    PauliOperator({{"", value}});
            auto len = item_vec[0].size();
            auto inner_str = item_vec[0].mid(1, len-2);
            auto inner_vec =
                    inner_str.splitByStr("), (", QString::SkipEmptyParts);
            if (!inner_vec.empty())
            {
                inner_vec[0] = inner_vec[0].mid(1);
                auto &last = inner_vec[inner_vec.size()-1];
                last = last.left(last.size() - 1);

                for (size_t j = 0; j < inner_vec.size(); j++)
                {
                    auto tmp_vec =
                            inner_vec[j].split(",", QString::SkipEmptyParts);

                    if (2 != tmp_vec.size())
                    {
                        std::cout << "file content format error! filename: "
                                  << filename << std::endl;
                        return false;
                    }

                    int first = tmp_vec[0].toInt();
                    int second = tmp_vec[1].toInt();
                    std::string str_item;
                    for (int k = 0; k < first; k++)
                    {
                        str_item += "Z" + std::to_string(k) + " ";
                    }

                    auto x_str = str_item + "X" + std::to_string(first);
                    auto y_str = str_item + "Y" + std::to_string(first);

                    PauliOperator pauli_x_component({{x_str, 0.5}});
                    complex_d value =
                            second ? complex_d(0, -0.5) : complex_d(0, 0.5);
                    PauliOperator pauli_y_component({{y_str, value}});

                    transformed_term *= pauli_x_component + pauli_y_component;
                }
            }

            pauli += transformed_term;
        }

        return true;
    }

    QResultPair VQE::callVQE(
            const vector_d &para,
            const QHamiltonian &hamiltonian)
    {
        PauliOperator cc = getCCS(m_qn, m_electron_num, para);
        PauliOperator ucc = transCC2UCC(cc);
        QHamiltonian ucc_hamiltonian = ucc.toHamiltonian();

        double expectation = 0.0;
        for (size_t i = 0; i < hamiltonian.size(); i++)
        {
            expectation += getExpectation(ucc_hamiltonian, hamiltonian[i]);
        }

        if (m_enable_optimizer_data)
        {
            m_optimizer_data_db.insertValue(m_func_calls, expectation, para);
            m_func_calls++;
        }

        return std::make_pair("", expectation);
    }

    bool VQE::saveResult(const QOptimizationResult &result, size_t index)
    {
        if (m_data_save_path.empty())
        {
            return true;
        }

        std::string filename = m_data_save_path + "result_" +
                std::to_string(index) + ".dat";

        OriginCollection collection(filename, false);
        collection = {"index", "fun_val", "key",
                      "iters", "fcalls", "message", "para"};

        collection.insertValue(index, result.fun_val, result.key, result.iters,
                               result.fcalls, result.message, result.para);

        return collection.write();
    }

    void VQE::writeExecLog(bool exec_flag)
    {
        OriginCollection collection("VQE.log", false);
        collection = {"status", "message"};

        collection.insertValue(exec_flag ? 0: -1, m_last_err);
    }

}
