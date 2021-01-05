/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

Author: LiYe
Created in 2019-01-22

*/

#ifndef PAULIROPERATOR_H
#define PAULIROPERATOR_H

#include <map>
#include <vector>
#include "Core/Utilities/Tools/QString.h"
#include "Core/Utilities/QPandaNamespace.h"
#include "Components/DataStruct.h"

QPANDA_BEGIN

/**
* @brief Pauli operator class
* @ingroup Operator
*/
template<class T>
class PauliOp
{
public:
    using PauliItem = std::pair<QPauliPair, T>;
    using PauliData = std::vector<PauliItem>;
    using PauliMap = std::map<std::string, T>;

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

    PauliOp(const std::string &key, const T &value)
    {
        insertData(key, value);
    }

    PauliOp(const PauliMap &map)
    {
        for (auto iter = map.begin(); iter != map.end(); iter++)
        {
            insertData(iter->first, iter->second);
        }

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

    PauliOp(PauliData &&pauli):
        m_data(std::move(pauli))
    {
        reduceDuplicates();
    }

    PauliOp(const PauliData &pauli):
        m_data(pauli)
    {
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
	* @brief  remap qubit index
	* @param[in] std::map<size_t, size_t>& qubit index map
	* @return PauliOp return remapped qubit index map
	*/
    PauliOp remapQubitIndex(std::map<size_t, size_t> &index_map)
    {
        index_map.clear();
        for (size_t i = 0; i < m_data.size(); i++)
        {
            auto pair = m_data[i].first;
            QTerm term = pair.first;
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
	* @brief get the max index
	* @return size_t the max index
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

        max_index++;

        return max_index;
    }

	/**
	* @brief Judge whether it is empty
	* @return bool if data is empty, return true, or else return false
	*/
    bool isEmpty() { return m_data.empty(); }

	/**
	* @brief Judge whether all of data is "Z"
	* @return bool if all data is "Z", return true, or else return false
	*/
    bool isAllPauliZorI()
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

            str += pair.second + " : ";

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
        }

        if (!m_data.empty())
        {
            str += "\n";
        }

        str += "}";
        return str;
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
        reduceDuplicates();

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
        reduceDuplicates();

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
        reduceDuplicates();

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

private:
    QTermPair genQTermPair(const QString &str) const
    {
        if (str.size() < 2)
        {
            std::string err = "size < 2.";
            std::cout << err << std::endl;
            throw err;
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
            {
                std::string err = "Bad param in QPuliMap: Index repeat.";
                std::cout << err << std::endl;
                QCERR(err);
                throw std::invalid_argument(err);
            }
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

private:
    PauliData m_data;
    double m_error_threshold{1e-6};
};

using PauliOperator = PauliOp<complex_d>;

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

/*
* @brief Transfrom Pauli operator to vector
* @param[in]  pauli Pauli operator
* @return  a vector with double value
* @note The subterms of the Pauli operator must be I and Z
*/
std::vector<double> transPauliOperatorToVec(PauliOperator pauli);

QPANDA_END
#endif // PAULIROPERATOR_H
