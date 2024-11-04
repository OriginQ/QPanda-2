/*
Copyright (c) 2017-2023 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

Author: LiYe
Created in 2019-01-22

*/

#ifndef _PAULIROPERATOR_H_
#define _PAULIROPERATOR_H_

#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include "Core/Module/DataStruct.h"
#include "Core/Utilities/Tools/QString.h"
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/Tools/MatrixDecomposition.h"
#include "Components/HamiltonianSimulation/HamiltonianSimulation.h"

QPANDA_BEGIN

/**
* @brief Pauli operator class
* @ingroup Operator
*/
template<class T>
class PauliOp
{
public:
    using PauliItem = std::pair<QPauliPair, T>; // <[3X, 4Y], "3X 4Y">, T>
    using PauliData = std::vector<PauliItem>;  // [<[3X, 4Y], "3X 4Y">, T>, <<>,>]
    using PauliMap = std::map<std::string, T>;
    using PauliMatrix = std::vector<T>;

    /**
    * @note
        The QTerm_reversed value char only will be 'X','Y','Z', 'I'.
    */
    using QTerm_reversed = std::map<size_t, char, std::greater<size_t>>;
    using QPauliPair_reversed = std::pair<QTerm_reversed, std::string>;
    using PauliItem_reversed = std::pair<QPauliPair_reversed, T>; // <<[4X, 3Y], "4X 3Y">, T>
    using PauliData_reversed = std::vector<PauliItem_reversed>;  // [<<[4X, 3Y], "4X 3Y">, T>, <<>,>]

public:
	/**
	* @brief  Constructor of PauliOp class
	*/
    PauliOp(){}

    PauliOp(const T &value)
    {
        insertData("", value);
    }

    PauliOp(double value)
    {
        insertData("", T(value));
    }

    PauliOp(const std::string &key, const T &value, bool is_reduce_duplicates = false)
    {
        insertData(key, value);

        if (is_reduce_duplicates)
            reduceDuplicates();
    }

    PauliOp(const PauliMap &map, bool is_reduce_duplicates = false)
    {
        for (auto iter = map.begin(); iter != map.end(); iter++)
        {
            insertData(iter->first, iter->second);
        }

        if (is_reduce_duplicates)
            reduceDuplicates();
    }

    PauliOp(PauliOp &&op):
        m_data(std::move(op.m_data))
    {
    }

    PauliOp(const PauliOp &op):
        m_data(op.m_data)
    {
    }

    PauliOp(PauliData &&pauli, bool is_reduce_duplicates = false):
        m_data(std::move(pauli))
    {
        if (is_reduce_duplicates)
            reduceDuplicates();
    }

    PauliOp(const PauliData &pauli, bool is_reduce_duplicates = false):
        m_data(pauli)
    {
        if (is_reduce_duplicates)
            reduceDuplicates();
    }

    PauliOp(EigenMatrixX &matrix, bool is_reduce_duplicates = false)
    {
        if (std::is_same<T, complex_d>::value)
        {
            CPUQVM machine; 
            machine.init();
            PualiOperatorLinearCombination linear_result;
            matrix_decompose_paulis(&machine, matrix, linear_result);

            std::map<std::string, complex_d> pauli_map;
            for (auto item : linear_result)
            {
                double val = item.first;
                QCircuit cir = item.second;

                QCircuitToPauliOperator cir_to_opt(val);
                auto pauli_value = cir_to_opt.traversal(cir);

                pauli_map.insert(pauli_value);
            }

            for (auto iter = pauli_map.begin(); iter != pauli_map.end(); iter++)
            {
                insertData(iter->first, iter->second);
            }

            machine.finalize();
        }
        else
        {
            QCERR_AND_THROW(run_fail, "matrix data type error")
        }
       
        if (is_reduce_duplicates)
            reduceDuplicates();
    }

    PauliOp &operator = (const PauliOp &op)
    {
        m_data = op.m_data;
        return *this;
    }

    PauliOp &operator = (PauliOp &&op)
    {
        m_data = std::move(op.m_data);
        return *this;
    }


	/**
	* @brief  get the Transposed conjugate matrix
	* @return PauliOp return the Transposed conjugate matrix
	*/
    PauliOp dagger() const
    {
        auto tmp_data = m_data;
        for (size_t i = 0; i < tmp_data.size(); i++)
        {
            auto &item = tmp_data[i];
            item.second = T(item.second.real(), -item.second.imag());
        }

        return PauliOp(tmp_data);
    }

	/**
	* @brief get the max index of pauli string
	* @return size_t
	*/
    size_t getMaxIndex()
    {
        int max_index = -1;
        for (size_t i = 0; i < m_data.size(); i++)
        {
            auto index_map = m_data[i].first.first;
            auto iter = index_map.rbegin();
            if (iter != index_map.rend())
            {
                if (int(iter->first) > max_index)
                {
                    max_index = iter->first;
                }
            }
        }
        if (m_data.size() == 0) {
            return SIZE_MAX;
        }

        return max_index;
    }

      /**
    * @brief  remap qubit index
    * @param[in] std::map<size_t, size_t>& qubit index map
    * @return PauliOp return remapped qubit index map
    */
    PauliOp remapQubitIndex(std::map<size_t, size_t> &index_map) const
    {
        index_map.clear();
        for (size_t i = 0; i < m_data.size(); i++)
        {
            auto pair = m_data[i].first;
            auto term = pair.first;
            for (auto iter = term.begin(); iter != term.end(); iter++)
            {
                index_map.insert(std::make_pair(iter->first, 1));
            }
        }

        size_t cnt = 0;
        for (auto iter = index_map.begin(); iter != index_map.end(); iter++)
        {
            index_map[iter->first] = cnt;
            cnt++;
        }

        PauliData pauli_data;
        for (size_t i = 0; i < m_data.size(); i++)
        {
            QTerm tmp_term;

            auto pair = m_data[i].first;
            QTerm term = pair.first;
            for (auto iter = term.begin(); iter != term.end(); iter++)
            {
                tmp_term.insert(std::make_pair(index_map[iter->first],
                    iter->second));
            }

            pauli_data.emplace_back(
                std::make_pair(QPauliPair(tmp_term, QTerm2StdString(tmp_term)),
                    m_data[i].second));
        }

        return PauliOp(pauli_data);
    }


	/**
	* @brief Judge whether it is empty
	* @return bool if data is empty, return true, or else return false
	*/
    bool isEmpty() const { return m_data.empty(); }

	/**
	* @brief Judge whether all of data is "Z"
	* @return bool if all data is "Z", return true, or else return false
	*/
    bool isAllPauliZorI() const
    {
        for (size_t i = 0; i < m_data.size(); i++)
        {
            auto item = m_data[i];
            auto map = item.first.first;

            for (auto iter = map.begin(); iter != map.end(); iter++)
            {
                if ('Z' != iter->second)
                {
                    return false;
                }
            }
        }

        return true;
    }

	/**
	* @brief set error threshold
	* @param[in] double threshold val
	*/
    void setErrorThreshold(double threshold)
    {
        m_error_threshold = threshold;
    }

	/**
	* @brief get error threshold
	* @return double return the error threshold val
	*/
    double error_threshold() const { return m_error_threshold; }

	/**
	* @brief data to string
	* @return std::string convert data val to string
	*/
    std::string  toString() const
    {
        std::string str = "{";
        for (size_t i = 0; i < m_data.size(); i++)
        {
            str += "\n";

            auto item = m_data[i];
            auto pair = item.first;
            auto value = item.second;

            //str += pair.second + " : ";
            str += "\"" + pair.second + "\"" + " : ";

            if (fabs(value.real()) < m_error_threshold)
            {
                str += std::to_string(value.imag()) + "i";
            }
            else if (fabs(value.imag()) < m_error_threshold)
            {
                str += std::to_string(value.real());
            }
            else
            {
                if (value.imag() < 0)
                {
                    str += "(" + std::to_string(value.real()) +
                        std::to_string(value.imag()) + "i)";
                }
                else
                {
                    str += "(" + std::to_string(value.real()) + "+" +
                        std::to_string(value.imag()) + "i)";
                }
            }

            if (i != m_data.size() - 1)
            {
                str += ",";
            }
        }

        if (!m_data.empty())
        {
            str += "\n";
        }

        str += "}";
        return str;
    }

    /**
* @brief number of gate
* @return size_t 
*/
    size_t ngate() const
    {
        size_t n = 0;

        for (size_t i = 0; i < m_data.size(); i++)
        {
            auto item = m_data[i];
            auto pair = item.first;

            n += count(pair.second.begin(), pair.second.end(), ' ');

        }

        return n;
    }

    /**
    * @brief transform Pauli Hamiltonian to matrix by getCircuitMatrix
    * @param[in] void
    * @return QMatrixXcd
    */
    QMatrixXcd to_matrix()
    {
        auto max_idx = getMaxIndex();
        if (std::is_same<T, complex_d>::value)
        {
            auto qubit_pool = OriginQubitPool::get_instance();
            qubit_pool->set_capacity(30);

            auto full_matrix = qstat_zero_matrix(1ull << (max_idx + 1));

            for (const auto& val : m_data)
            {
                QCircuit circuit;
                auto pauli_pair = val.first;
                auto item = val.second;

                auto pauli_map = pauli_pair.first;
                for (const auto& pauli: pauli_map)
                {
                    auto qubit = qubit_pool->allocateQubitThroughPhyAddress(pauli.first);

                    if (pauli.second == 'X')
                    {
                        circuit << X(qubit);
                    }
                    else if (pauli.second == 'Y')
                    {
                        circuit << Y(qubit);
                    }
                    else if (pauli.second == 'Z')
                    {
                        circuit << Z(qubit);
                    }
                    else
                    {
                        QCERR_AND_THROW(run_fail, "pauli type error");
                    }
                }

                for (auto i = 0; i < max_idx + 1; ++i)
                {
                    auto qubit = qubit_pool->allocateQubitThroughPhyAddress(i);
                    circuit << BARRIER(qubit);
                }
                
                auto matrix = item * QPanda::getCircuitMatrix(circuit);
                full_matrix = full_matrix + matrix;

            }

            return QStat_to_Eigen(full_matrix);
        }
        else
        {
            QCERR_AND_THROW(run_fail, "matrix data type error")
        }
    }

    /**
    * @brief transform Pauli Hamiltonian to matrix by kroneckerProduct
    * @param[in] void
    * @return QMatrixXcd
    */
    QMatrixXcd matrix()
    {
        // transform "X0 X3" to "X3 I2 I1 X0"
        PauliData_reversed pauli_data;
        size_t nbit = getMaxIndex() + 1;
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            auto item = m_data[i]; // [3X, 4Y], "3X 4Y">, T
            auto pair = item.first; // [3X, 4Y], "3X 4Y"
            auto value_T = item.second; // T
            QTerm term = pair.first; // [3X, 4Y], 

            QPauliPair_reversed pair_reversed;
            QTerm_reversed term_reversed;
            for (size_t j = 0; j < nbit; ++j)
            {
                bool find = false;
                if (!term.empty())
                {
                    for (auto& t : term)
                    {
                        if (t.first == j)
                        {
                            term_reversed.insert(std::make_pair(t.first, t.second));
                            find = true;
                            break;
                        }
                    }
                }
                if (!find)
                {
                    term_reversed.insert(std::make_pair(j, 'I'));
                }
            }

            std::string sn = "";
            for (auto iter = term_reversed.begin(); iter != term_reversed.end(); iter++)
            {
                if (std::next(iter) == term_reversed.end())
                {
                    sn += (iter->second + std::to_string(iter->first));
                }
                else
                {
                    sn += (iter->second + std::to_string(iter->first) + " ");
                }
            }
            pair_reversed.first = term_reversed;
            pair_reversed.second = sn;

            pauli_data.emplace_back(std::make_pair(pair_reversed, value_T));
        }

        // transform "X3 I2 I1 X0" to QMatrixXcd
        QMatrixXcd I_matrix = QMatrixXcd::Identity(2, 2);
        QMatrixXcd X_matrix(2, 2);  X_matrix << 0, 1, 1, 0;
        QMatrixXcd Y_matrix(2, 2); Y_matrix << 0, complex_d(0, -1), complex_d(0, 1), 0;
        QMatrixXcd Z_matrix(2, 2); Z_matrix << 1, 0, 0, -1;
        QMatrixXcd full_matrix; bool full_empty = true;
        if (std::is_same<T, complex_d>::value)
        {
            for (const auto& val : pauli_data)
            {
                auto t1 = std::chrono::steady_clock::now();
                QMatrixXcd subitem_matrix; bool sub_empty = true;
                auto pauli_pair = val.first;
                auto item = val.second;

                auto pauli_map = pauli_pair.first;
                for (const auto& pauli : pauli_map)
                {
                    QMatrixXcd gate_mat;
                    if (pauli.second == 'X')
                    {
                        gate_mat = X_matrix;
                    }
                    else if (pauli.second == 'Y')
                    {
                        gate_mat = Y_matrix;
                    }
                    else if (pauli.second == 'Z')
                    {
                        gate_mat = Z_matrix;
                    }
                    else if (pauli.second == 'I')
                    {
                        gate_mat = I_matrix;
                    }
                    else
                    {
                        QCERR_AND_THROW(run_fail, "pauli type error");
                    }

                    if (sub_empty)
                    {
                        subitem_matrix = gate_mat;  sub_empty = false;
                    }
                    else
                    {
                        QMatrixXcd temp = kroneckerProduct(subitem_matrix, gate_mat);
                        subitem_matrix = temp;
                    }
                }
                if (full_empty)
                {
                    full_matrix = item * subitem_matrix; full_empty = false;
                }
                else
                {
                    full_matrix = full_matrix + item * subitem_matrix;
                }
            }
            return full_matrix;
        }
        else
        {
            QCERR_AND_THROW(run_fail, "matrix data type error")
        }
    }

	/**
	* @brief get data
	* @return PauliData return Pauli data
	*/
    PauliData data() const { return m_data; }

	/**
	* @brief convert data to Hamiltonian 
	* @param[out] bool* save the convert result, default is nullptr
	* @return QHamiltonian the convert result
	*/
    QHamiltonian toHamiltonian(bool *ok = nullptr) const
    {
        QHamiltonian hamiltonian;

        for (size_t i = 0; i < m_data.size(); i++)
        {
            auto item = m_data[i];
            auto pair = item.first;
            auto value = item.second;

            if (fabs(value.imag()) > fabs(m_error_threshold))
            {
                std::cout << "PauliOperator data cannot convert to Hamiltonian."
                          << std::endl;
                if (ok)
                {
                    *ok = false;
                }

                return QHamiltonian();
            }

            hamiltonian.emplace_back(std::make_pair(pair.first, value.real()));
        }

        if (ok)
        {
            *ok = true;
        }

        return hamiltonian;
    }

	/**
	* @brief overload +
	* @return PauliOp return (PauliOp_left + PauliOp_right)
	*/
    PauliOp  operator + (const PauliOp &rhs) const
    {
        PauliData pauli_data = m_data;
        pauli_data.insert(pauli_data.end(),
                          rhs.m_data.begin(),
                          rhs.m_data.end());

        PauliOp pauli_op(std::move(pauli_data));

        return pauli_op;
    }

	/**
	* @brief overload -
	* @return PauliOp return (PauliOp_left - PauliOp_right)
	*/
    PauliOp  operator - (const PauliOp &rhs) const
    {
        PauliData tmp_data = rhs.m_data;
        for (auto i = 0u; i < tmp_data.size(); i++)
        {
            tmp_data[i].second = tmp_data[i].second * T(-1.0,0);
        }

        PauliData pauli_data = m_data;
        pauli_data.insert(pauli_data.end(), tmp_data.begin(), tmp_data.end());

        PauliOp pauli_op(std::move(pauli_data));

        return pauli_op;
    }

	/**
	* @brief overload *
	* @return PauliOp return (PauliOp_left * PauliOp_right)
	*/
    PauliOp  operator * (const PauliOp &rhs) const
    {
        PauliData pauli_data;
        PauliData rhs_data = rhs.m_data;

        for (size_t i = 0; i < m_data.size(); i++)
        {
            auto &item_i = m_data[i];
            for (size_t j = 0; j < rhs_data.size(); j++)
            {
                auto &item_j = rhs_data[j];
                auto &map_i = item_i.first.first;
                auto &map_j = item_j.first.first;

                auto item = genPauliItem(map_i,
                                            map_j,
                                            item_i.second * item_j.second);
                pauli_data.emplace_back(item);
            }
        }

        PauliOp pauli(std::move(pauli_data));

        return pauli;
    }

	/**
	* @brief overload +=
	* @return PauliOp return (PauliOp_left += PauliOp_right)
	*/
    PauliOp &operator +=(const PauliOp &rhs)
    {
        m_data.insert(m_data.end(), rhs.m_data.begin(), rhs.m_data.end());
        //reduceDuplicates();

        return *this;
    }

	/**
	* @brief overload -=
	* @return PauliOp return (PauliOp_left -= PauliOp_right)
	*/
    PauliOp &operator -=(const PauliOp &rhs)
    {
        PauliData tmp_data = rhs.m_data;
        for (auto i = 0u; i < tmp_data.size(); i++)
        {
            tmp_data[i].second = tmp_data[i].second * T(-1.0,0);
        }

        m_data.insert(m_data.end(), tmp_data.begin(), tmp_data.end());
        //reduceDuplicates();

        return *this;
    }

	/**
	* @brief overload *=
	* @return PauliOp return (PauliOp_left *= PauliOp_right)
	*/
    PauliOp &operator *=(const PauliOp &rhs)
    {
        PauliData tmp_data;
        PauliData rhs_data = rhs.m_data;

        for (size_t i = 0; i < m_data.size(); i++)
        {
            auto &item_i = m_data[i];
            for (size_t j = 0; j < rhs_data.size(); j++)
            {
                auto &item_j = rhs_data[j];
                auto &map_i = item_i.first.first;
                auto &map_j = item_j.first.first;

                auto item = genPauliItem(map_i,
                    map_j,
                    item_i.second * item_j.second);
                tmp_data.emplace_back(item);
            }
        }

        m_data = std::move(tmp_data);
        //reduceDuplicates();

        return *this;
    }

	/**
	* @brief overload +
	* @return PauliOp return (PauliOp_left + PauliOp_right)
	*/
    friend PauliOp operator + (const T &lhs,
                                        const PauliOp &rhs)
    {
        return rhs + lhs;
    }

	/**
	* @brief overload -
	* @return PauliOp return (PauliOp_left - PauliOp_right)
	*/
    friend PauliOp operator - (const T &lhs,
                                        const PauliOp &rhs)
    {
        PauliData tmp_data = rhs.m_data;
        for (auto i = 0u; i < tmp_data.size(); i++)
        {
            tmp_data[i].second = tmp_data[i].second * T(-1.0,0);
        }

        PauliOp tmp(std::move(tmp_data));

        return tmp + lhs;
    }

	/**
	* @brief overload *
	* @return PauliOp return (PauliOp_left * PauliOp_right)
	*/
    friend PauliOp operator * (const T &lhs,
                                        const PauliOp &rhs)
    {
        return rhs * lhs;
    }

	/**
	* @brief overload +
	* @return PauliOp return (double + PauliOp_right)
	*/
    friend PauliOp operator + (const double &lhs,
                                        const PauliOp &rhs)
    {
        return rhs + lhs;
    }

	/**
	* @brief overload -
	* @return PauliOp return (double - PauliOp_right)
	*/
    friend PauliOp operator - (const double &lhs,
                                        const PauliOp &rhs)
    {
        PauliData tmp_data = rhs.m_data;
        for (auto i = 0u; i < tmp_data.size(); i++)
        {
            tmp_data[i].second = tmp_data[i].second * T(-1.0,0);
        }

        PauliOp tmp(std::move(tmp_data));

        return tmp + lhs;
    }

	/**
	* @brief overload *
	* @return PauliOp return (PauliOp_left * PauliOp_right)
	*/
    friend PauliOp operator * (const double &lhs,
                                        const PauliOp &rhs)
    {
        return rhs * lhs;
    }

	/**
	* @brief overload std::cout, convert PauliOp to string and output to std::cout
	* @return std::ostream&
	*/
    friend std::ostream  &operator <<(std::ostream &out,
                                        const PauliOp &rhs)
    {
        out << rhs.toString();
        return out;
    }

    void reduceDuplicates()
    {
        std::map<std::string, T> data_map;
        std::map<std::string, QTerm> term_map;

        for (size_t i = 0; i < m_data.size(); i++)
        {
            auto item = m_data[i];
            auto pair = item.first;
            auto value = item.second;
            QTerm term = pair.first;
            std::string str = pair.second;

            if (data_map.find(str) != data_map.end())
            {
                data_map[str] = data_map[str] + value;
            }
            else
            {
                data_map.insert(std::make_pair(str, value));
                term_map.insert(std::make_pair(str, term));
            }
        }


        PauliData pauli_data;
        for (auto iter = data_map.begin(); iter != data_map.end(); iter++)
        {
            auto err = std::abs(iter->second);
            if (err > fabs(m_error_threshold))
            {
                QPauliPair pair;
                pair.first = term_map[iter->first];
                pair.second = iter->first;

                pauli_data.emplace_back(std::make_pair(pair, iter->second));
            }
        }

        m_data = std::move(pauli_data);
    }

    void delSimilar()
    {
        std::map<std::string, T> value_map;
        std::map<std::string, QPauliPair> pair_map;

        for (size_t i = 0; i < m_data.size(); ++i)
        {
            auto item = m_data[i]; // [3X, 4Y], "3X 4Y">, T
            auto pair = item.first; // [3X, 4Y], "3X 4Y"
            auto value = item.second; // T
            QTerm term = pair.first; // [3X, 4Y], 

            std::string XY = "", sn = ""; char c;
            for (auto& t : term)
            {
                sn += std::to_string(t.first);
                if (t.second!='X')
                {
                    XY += 'X';
                }
                else
                {
                    XY += t.second;
                }
            }
            std::sort(sn.begin(), sn.end());
            std::string snXY = sn + XY;
            auto result = value_map.find(snXY);
            if (result == value_map.end())
            {
                value_map.insert(std::make_pair(snXY, value));
                pair_map.insert(std::make_pair(snXY, pair));
            }
        }

        PauliData pauli_data;
        for (auto iter = value_map.begin(); iter != value_map.end(); iter++)
        {
            auto r = iter->second.real();
            auto i = iter->second.imag();
            if (fabs(r) < m_error_threshold && fabs(i) < m_error_threshold)
            {
                continue;
            }

            QPauliPair pair = pair_map[iter->first];
            pauli_data.emplace_back(std::make_pair(pair, iter->second));
        }

        m_data = std::move(pauli_data);
    }

    void reorderReduce(const size_t ne)
    {
        if (ne % 2)  return;
        using pairPTerm = std::pair<size_t, char>;

        //printf("NOTE: The number of qubits has been reduced by two\n");

        size_t nbit = getMaxIndex(), nb2 = nbit / 2; bool fg = (ne / 2) % 2;
        std::string snd; QTerm fst;
        size_t o; char v;

        for (PauliItem& pi : m_data) // <[3X, 4Y], "3X 4Y">, T>,
        {
            QPauliPair pif = pi.first; // <[3X, 4Y], "3X 4Y">
            if (!pif.second.size()) continue;

            fst.clear(); snd = ""; // "3X 4Y"
            for (auto& p : pif.first) // [3X, 4Y]
            {
                o = p.first; v = p.second;

                if (o < nb2)
                {
                    fst.insert(p);
                    snd += v + std::to_string(o) + " ";
                }
                else if (o == nb2)
                {
                    if (fg) pi.second = pi.second * (-1.0);
                }
                else if (o < nbit)
                {
                    fst.insert(pairPTerm(o - 1, v));
                    snd += v + std::to_string(o - 1) + " ";
                }
            }
            pi.first = QPauliPair(fst, snd);
        }

        //reduceDuplicates();
    }

    /**
    * @brief reduce two qubits when using parity mapping (even electorn)
    * @param[in] size_t
    * @return void
    */
    void twoQubitReduce(size_t nelec)
    {
        if (nelec % 2 == 1 )
        {
            QCERR_AND_THROW_ERRSTR(std::runtime_error, "electron number is odd, can't use qubit reducing!");
        }
        size_t nbit = getMaxIndex() + 1;
        std::vector<size_t> reduced_qubits = {nbit/2 - 1, nbit - 1};
        std::vector<size_t> reduced_qubit_state = {1, 0};
        if ((nelec / 2) % 2 == 0)
        {
            reduced_qubit_state = {0, 0};
        }
        PauliData pauli_data;
        for (size_t i = 0; i < m_data.size(); ++i)
        {
            auto item = m_data[i]; // [3X, 4Y], "3X 4Y">, T
            auto pair = item.first; // [3X, 4Y], "3X 4Y"
            auto value = item.second; // T
            QTerm term = pair.first; // [3X, 4Y], 

            std::vector<size_t> term_key;
            for (auto& t : term)
            {
                size_t tmp = t.first;
                for (size_t j = 0; j < reduced_qubits.size(); ++j)
                {
                    // find pauli char that will be deleted by qubit id
                    if (t.first == reduced_qubits[j])
                    {
                        if (t.second != 'Z')
                        {
                            throw std::runtime_error("error in twoQubitReduce!\n");
                        }
                        // if reduced_qubit_state = 1, T * -1; else T invariability
                        if (reduced_qubit_state[j] == 1)
                        {
                            auto re = -1 * value.real();
                            auto im = -1 * value.imag();
                            value = complex_d(re, im);
                        }
                        term_key.emplace_back(reduced_qubits[j]);
                    }
                }
            }

            // delete pauli char
            if (term_key.size() > 0)
            {
                for (auto& key : term_key)
                {
                    term.erase(key);
                }
            }

            QTerm term_map;
            for (auto& t : term)
            {
                size_t tmp = t.first;
                if (t.first > reduced_qubits[0] && t.first < reduced_qubits[1])
                {
                    tmp = t.first - 1;
                }
                term_map.insert(std::make_pair(tmp, t.second));
            }

            std::string sn = "";
            for (auto iter = term_map.begin(); iter != term_map.end(); iter++)
            {
                if (std::next(iter) == term_map.end())
                {
                    sn += (iter->second + std::to_string(iter->first));
                }
                else
                {
                    sn += (iter->second + std::to_string(iter->first) + " ");
                }
            }
            pair.first = term_map;
            pair.second = sn;
            pauli_data.emplace_back(std::make_pair(pair, value));
        }
        m_data = std::move(pauli_data);
        reduceDuplicates();
    }

private:

    std::string QTerm2StdString(const QTerm &map) const
    {
        std::string str;
        bool next = false;

        for (auto iter = map.begin(); iter != map.end(); iter++)
        {
            if (!next)
            {
                next = true;
            }
            else
            {
                str += " ";
            }

            char ch = static_cast<char>(toupper(iter->second));
            str += ch + std::to_string(iter->first);
        }

        return str;
    }

    QTermPair genQTermPair(const QString &str) const
    {
        if (str.size() < 2)
        {
            std::string err = "size < 2.";
            QCERR_AND_THROW(std::invalid_argument, "pauli size not complete");
        }

        char ch = static_cast<char>(toupper(str.at(0)));
        std::string check_str = "XYZ";
        if (check_str.find(ch) == std::string::npos)
        {
            std::string err = std::string("Param not in [XYZ]. str: ")
                    + str.data();
            std::cout << err << std::endl;
            throw err;
        }

        bool ok = false;
        auto index = str.mid(1).toInt(&ok);
        if (!ok)
        {
            std::string err = "Convert index to int failed.";
            std::cout << err << std::endl;
            throw err;
        }

        return QTermPair(index, ch);
    }

    void insertData(const std::string &str, const T &value)
    {
        QString key(str);

        auto str_vec = key.split(" ", QString::SkipEmptyParts);
        if (str_vec.empty())
        {
            QPauliPair pair;
            pair.first = QTerm();
            pair.second = "";
            m_data.emplace_back(std::make_pair(pair, value));
            return;
        }

        QTerm one_term;
        for (size_t i = 0; i < str_vec.size(); i++)
        {
            QTermPair one_pair = genQTermPair(str_vec[i]);

            if (one_term.find(one_pair.first) != one_term.end())
                QCERR_AND_THROW(std::invalid_argument, "Bad param in QPuliMap: Index repeat.");
            one_term.insert(one_pair);
        }

        QPauliPair pair;
        pair.first = one_term;
        pair.second = QTerm2StdString(one_term);

        m_data.emplace_back(std::make_pair(pair, value));
    }

    PauliItem  genPauliItem(const QTerm &map_i,
                                const QTerm &map_j,
                                const T &value) const
    {
        auto tmp_map = map_i;
        auto result = value;
        auto iter_j = map_j.begin();
        for (; iter_j != map_j.end(); iter_j++)
        {
            auto iter_t = tmp_map.find(iter_j->first);
            if (iter_t == tmp_map.end())
            {
                tmp_map.insert(*iter_j);
            }
            else
            {
                std::string tmp = iter_t->second + std::string() +
                    iter_j->second;
                if (("XX" == tmp) ||
                    ("YY" == tmp) ||
                    ("ZZ" == tmp))
                {
                    tmp_map.erase(iter_t->first);
                }
                else if ("XY" == tmp)
                {
                    result *= complex_d{0, 1};
                    iter_t->second = 'Z';
                }
                else if ("XZ" == tmp)
                {
                    result *= complex_d{0, -1};
                    iter_t->second = 'Y';
                }
                else if ("YX" == tmp)
                {
                    result *= complex_d{0, -1};
                    iter_t->second = 'Z';
                }
                else if ("YZ" == tmp)
                {
                    result *= complex_d{0, 1};
                    iter_t->second = 'X';
                }
                else if ("ZX" == tmp)
                {
                    result *= complex_d{0, 1};
                    iter_t->second = 'Y';
                }
                else if ("ZY" == tmp)
                {
                    result *= complex_d{0, -1};
                    iter_t->second = 'X';
                }
                else
                {
                    std::string err = "Bad para in QPauli.";
                    std::cout << err << std::endl;
                    throw err;
                }
            }
        }

        QPauliPair pair;
        pair.first = tmp_map;
        pair.second = QTerm2StdString(tmp_map);

        PauliItem item;
        item.first = pair;
        item.second = result;

        return item;
    }

private:
    PauliData m_data;
    double m_error_threshold{1.0e-6};
};

using PauliOperator = PauliOp<complex_d>;

PauliOperator x(int index);
PauliOperator y(int index);
PauliOperator z(int index);
PauliOperator i(int index);

/*
* @brief Transfrom vector to pauli operator
* @param[in]  data_vec a vector
* @return  PauliOperator
*/
template<class T>
PauliOperator transVecToPauliOperator(const std::vector<T>& data_vec)
{
    size_t size = data_vec.size();
    size_t qnum = std::ceil(std::log2(size));
    if (qnum == 0)
    {
        return PauliOperator();
    }

    using PauliPair = std::vector<PauliOperator>;
    std::vector<PauliPair> q_pauli_vec;
    for (size_t i = 0; i < qnum; i++)
    {
        PauliPair pauli_pair;
        std::string pauli_z = "Z" + std::to_string(i);
        PauliOperator p0 = 0.5 + PauliOperator(pauli_z, 0.5);
        PauliOperator p1 = 0.5 + PauliOperator(pauli_z, -0.5);

        pauli_pair.push_back(p0);
        pauli_pair.push_back(p1);
        q_pauli_vec.push_back(pauli_pair);
    }

    PauliOperator result;
    for (size_t i = 0; i < size; i++)
    {
        PauliOperator item(1);
        for (int j = qnum - 1; j >= 0; j--)
        {
            int bit_j = i & (1 << j);
            bit_j = bit_j >> j;
            item *= q_pauli_vec[j][bit_j];
        }

        item *= data_vec[i];
        result += item;
    }

    return result;
}

std::vector<double> kron(const std::vector<double>& vec1, const std::vector<double>& vec2);
std::vector<double> dot(const std::vector<double>& vec1, const std::vector<double>& vec2);
std::vector<double> operator +(const std::vector<double>& vec1, const std::vector<double>& vec2);
std::vector<double> operator *(const std::vector<double>& vec, double value);

/*
* @brief Transfrom Pauli operator to vector
* @param[in]  pauli Pauli operator
* @return  a vector with double value
* @note The subterms of the Pauli operator must be I and Z
*/
std::vector<double> transPauliOperatorToVec(PauliOperator pauli);
PauliOperator matrix_decompose_hamiltonian(EigenMatrixX& matrix);
std::vector<complex_d> transPauliOperatorToMatrix(const PauliOperator& opt);
QPANDA_END

#endif // _PAULIROPERATOR_H_
