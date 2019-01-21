#include <iostream>
#include "QPauliOperator.h"

namespace QPanda
{
    QPauliOperator::QPauliOperator(const QPauliMap &map)
    {
        for (auto iter = map.begin(); iter != map.end(); iter++)
        {
            QString key(iter->first);
            auto value = iter->second;

            auto str_vec = key.split(" ", QString::SkipEmptyParts);
            if (str_vec.empty())
            {
                QPauliPair pair;
                pair.first = QTerm();
                pair.second = "";
                m_data.emplace_back(std::make_pair(pair, value));
                continue;
            }

            QTerm one_term;
            for (auto i = 0; i < str_vec.size(); i++)
            {
                QTermPair one_pair = genQTermPair(str_vec[i]);
                
                if (one_term.find(one_pair.first) != one_term.end())
                {
                    throw std::string("Bad param in QPuliMap: "
                        "Index repeat.");
                }
                one_term.insert(one_pair);
            }

            QPauliPair pair;
            pair.first = one_term;
            pair.second = QTerm2StdString(one_term);

            m_data.emplace_back(std::make_pair(pair, value));
        }

        reduceDuplicates();
        remapQubitIndex();
    }

    QPauliOperator::QPauliOperator(const QHamiltonian &hamiltonian)
    {
        for (auto i = 0; i < hamiltonian.size(); i++)
        {
            auto item = hamiltonian[i];

            QPauliPair pair;
            pair.first = item.first;
            pair.second = QTerm2StdString(item.first);

            m_data.emplace_back(std::make_pair(pair, item.second));
        }

        reduceDuplicates();
        remapQubitIndex();
    }

    QPauliOperator::QPauliOperator(QPauliOperator &&op):
        m_data(op.m_data)
    {
    }

    QPauliOperator::QPauliOperator(const QPauliOperator &op):
        m_data(op.m_data)
    {
    }

    QPauliOperator::QPauliOperator(QPauli &&pauli):
        m_data(pauli)
    {
        reduceDuplicates();
        remapQubitIndex();
    }

    QPauliOperator::QPauliOperator(const QPauli &pauli):
        m_data(pauli)
    {
        reduceDuplicates();
        remapQubitIndex();
    }

    QPauliOperator & QPauliOperator::operator=(const QPauliOperator &op)
    {
        m_data = op.m_data;
        return *this;
    }

    QPauliOperator & QPauliOperator::operator=(QPauliOperator &&op)
    {
        m_data = op.m_data;
        return *this;
    }

    bool QPauliOperator::isAllPauliZorI()
    {
        for (auto i = 0; i < m_data.size(); i++)
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

    QHamiltonian QPauliOperator::toHamiltonian(bool *ok) const
    {
        QHamiltonian hamiltonian;

        for (auto i = 0; i < m_data.size(); i++)
        {
            auto item = m_data[i];
            auto pair = item.first;
            auto value = item.second;

            if (value.imag() > m_error_threshold)
            {
                std::cerr << "QPauli data cannot convert to QHamiltonian." << std::endl;
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

    std::string QPauliOperator::toString() const
    {
        std::string str = "{";
        bool next = false;

        for (auto i = 0; i < m_data.size(); i++)
        {
            if (!next)
            {
                next = true;
            }
            else
            {
                str += ", ";
            }

            auto item = m_data[i];
            auto pair = item.first;
            auto value = item.second;

            str += "\"" + pair.second + "\": ";
            
            if (value.real() < m_error_threshold)
            {
                str += std::to_string(value.imag()) + "i";
            }
            else if (value.imag() < m_error_threshold)
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

        return str;
    }

    QPauliOperator QPauliOperator::operator+(const complex_d &rhs) const
    {
        QPauli pauli_data = m_data;
        for (auto i = 0; i < pauli_data.size(); i++)
        {
            auto &item = pauli_data[i];
            item.second += rhs;
        }

        QPauliOperator pauli(std::move(pauli_data));
        pauli.reduceDuplicates();
        pauli.remapQubitIndex();

        return pauli;
    }

    QPauliOperator QPauliOperator::operator-(const complex_d &rhs) const
    {
        QPauli pauli_data = m_data;
        for (auto i = 0; i < pauli_data.size(); i++)
        {
            auto &item = pauli_data[i];
            item.second -= rhs;
        }

        QPauliOperator pauli(std::move(pauli_data));
        pauli.reduceDuplicates();
        pauli.remapQubitIndex();

        return pauli;
    }

    QPauliOperator QPauliOperator::operator*(const complex_d &rhs) const
    {
        QPauli pauli_data = m_data;
        for (auto i = 0; i < pauli_data.size(); i++)
        {
            auto &item = pauli_data[i];
            item.second *= rhs;
        }

        QPauliOperator pauli(std::move(pauli_data));
        pauli.reduceDuplicates();
        pauli.remapQubitIndex();

        return pauli;
    }

    QPauliOperator &QPauliOperator::operator+=(const complex_d &rhs)
    {
        for (auto i = 0; i < m_data.size(); i++)
        {
            auto &item = m_data[i];
            item.second += rhs;
        }

        reduceDuplicates();
        remapQubitIndex();

        return *this;
    }

    QPauliOperator &QPauliOperator::operator-=(const complex_d &rhs)
    {
        for (auto i = 0; i < m_data.size(); i++)
        {
            auto &item = m_data[i];
            item.second -= rhs;
        }

        reduceDuplicates();
        remapQubitIndex();

        return *this;
    }

    QPauliOperator &QPauliOperator::operator*=(const complex_d &rhs)
    {
        for (auto i = 0; i < m_data.size(); i++)
        {
            auto &item = m_data[i];
            item.second *= rhs;
        }

        reduceDuplicates();
        remapQubitIndex();

        return *this;
    }

    QPauliOperator QPauliOperator::operator+(const QPauliOperator &rhs) const
    {
        const QPauli &cdata = rhs.data();
        QPauli pauli_data = m_data;
        pauli_data.insert(pauli_data.end(), cdata.begin(), cdata.end());
        
        QPauliOperator pauli_op(std::move(pauli_data));
        pauli_op.reduceDuplicates();
        pauli_op.remapQubitIndex();

        return pauli_op;
    }

    QPauliOperator QPauliOperator::operator-(const QPauliOperator &rhs) const
    {
        QPauliOperator tmp_pauli(rhs);
        tmp_pauli *= -1;

        const QPauli &cdata = tmp_pauli.data();
        QPauli pauli_data = m_data;
        pauli_data.insert(pauli_data.end(), cdata.begin(), cdata.end());

        QPauliOperator pauli_op(std::move(pauli_data));
        pauli_op.reduceDuplicates();
        pauli_op.remapQubitIndex();

        return pauli_op;
    }

    QPauliOperator QPauliOperator::operator*(const QPauliOperator &rhs) const
    {
        QPauli pauli_data;
        const QPauli &rhs_data = rhs.data();

        for (auto i = 0; i < m_data.size(); i++)
        {
            auto &item_i = m_data[i];
            for (auto j = 0; j < rhs_data.size(); j++)
            {
                auto &item_j = rhs_data[j];
                auto &map_i = item_i.first.first;
                auto &map_j = item_j.first.first;

                auto item = genQPauliItem(map_i, 
                                          map_j, 
                                          item_i.second * item_j.second);
                pauli_data.emplace_back(item);
            }
        }

        QPauliOperator pauli(std::move(pauli_data));
        pauli.reduceDuplicates();
        pauli.remapQubitIndex();

        return pauli;
    }

    QPauliOperator &QPauliOperator::operator+=(const QPauliOperator &rhs)
    {
        const QPauli &cdata = rhs.data();

        m_data.insert(m_data.end(), cdata.begin(), cdata.end());
        reduceDuplicates();
        remapQubitIndex();

        return *this;
    }

    QPauliOperator &QPauliOperator::operator-=(const QPauliOperator &rhs)
    {
        QPauliOperator tmp_pauli(rhs);
        tmp_pauli *= -1;

        const QPauli &cdata = tmp_pauli.data();
        
        m_data.insert(m_data.end(), cdata.begin(), cdata.end());
        reduceDuplicates();
        remapQubitIndex();

        return *this;
    }

    QPauliOperator &QPauliOperator::operator*=(const QPauliOperator &rhs)
    {
        QPauli tmp_data;
        const QPauli &rhs_data = rhs.data();

        for (auto i = 0; i < m_data.size(); i++)
        {
            auto &item_i = m_data[i];
            for (auto j = 0; j < rhs_data.size(); j++)
            {
                auto &item_j = rhs_data[j];
                auto &map_i = item_i.first.first;
                auto &map_j = item_j.first.first;

                auto item = genQPauliItem(map_i,
                    map_j,
                    item_i.second * item_j.second);
                tmp_data.emplace_back(item);
            }
        }

        m_data = std::move(tmp_data);
        reduceDuplicates();
        remapQubitIndex();

        return *this;
    }

    QPauliOperator QPauliOperator::operator+(const QHamiltonian &rhs) const
    {
        QPauliOperator rhs_pauli(rhs);

        return *this + rhs_pauli;
    }

    QPauliOperator QPauliOperator::operator-(const QHamiltonian &rhs) const
    {
        QPauliOperator rhs_pauli(rhs);

        return *this - rhs_pauli;
    }

    QPauliOperator QPauliOperator::operator*(const QHamiltonian &rhs) const
    {
        QPauliOperator rhs_pauli(rhs);

        return *this * rhs_pauli;
    }

    QPauliOperator &QPauliOperator::operator+=(const QHamiltonian &rhs)
    {
        QPauliOperator rhs_pauli(rhs);

        return *this += rhs_pauli;
    }

    QPauliOperator &QPauliOperator::operator-=(const QHamiltonian &rhs)
    {
        QPauliOperator rhs_pauli(rhs);

        return *this -= rhs_pauli;
    }

    QPauliOperator &QPauliOperator::operator*=(const QHamiltonian &rhs)
    {
        QPauliOperator rhs_pauli(rhs);

        return *this *= rhs_pauli;
    }

    QPauliOperator operator+(const complex_d &lhs, const QPauliOperator &rhs)
    {
        QPauli pauli_data = rhs.data();
        for (auto i = 0; i < pauli_data.size(); i++)
        {
            auto &item = pauli_data[i];
            item.second += lhs;
        }

        QPauliOperator pauli(std::move(pauli_data));
        pauli.reduceDuplicates();
        pauli.remapQubitIndex();

        return pauli;
    }

    QPauliOperator operator-(const complex_d &lhs, const QPauliOperator &rhs)
    {
        QPauli pauli_data = rhs.data();
        for (auto i = 0; i < pauli_data.size(); i++)
        {
            auto &item = pauli_data[i];
            item.second -= lhs;
        }

        QPauliOperator pauli(std::move(pauli_data));
        pauli.reduceDuplicates();
        pauli.remapQubitIndex();

        return pauli;
    }

    QPauliOperator operator*(const complex_d &lhs, const QPauliOperator &rhs)
    {
        QPauli pauli_data = rhs.data();
        for (auto i = 0; i < pauli_data.size(); i++)
        {
            auto &item = pauli_data[i];
            item.second *= lhs;
        }

        QPauliOperator pauli(std::move(pauli_data));
        pauli.reduceDuplicates();
        pauli.remapQubitIndex();

        return pauli;
    }

    QPauliOperator operator+(const QHamiltonian &lhs, const QPauliOperator &rhs)
    {
        QPauliOperator rhs_pauli(lhs);

        return rhs_pauli + rhs;
    }

    QPauliOperator operator-(const QHamiltonian &lhs, const QPauliOperator &rhs)
    {
        QPauliOperator rhs_pauli(lhs);

        return rhs_pauli - rhs;
    }

    QPauliOperator operator*(const QHamiltonian &lhs, const QPauliOperator &rhs)
    {
        QPauliOperator rhs_pauli(lhs);

        return rhs_pauli * rhs;
    }

    std::ostream & operator<<(std::ostream &out, const QPauliOperator &rhs)
    {
        out << rhs.toString();
        return out;
    }

    std::ostream & operator<<(std::ostream &out, const QHamiltonian &rhs)
    {
        out << QPauliOperator(rhs).toString();
        return out;
    }

    std::ostream & operator<<(std::ostream &out, const QPauliMap &rhs)
    {
        out << QPauliOperator(rhs).toString();
        return out;
    }

    std::ostream & operator<<(std::ostream &out, const QPauli &rhs)
    {
        out << QPauliOperator(rhs).toString();
        return out;
    }

    QTermPair QPauliOperator::genQTermPair(const QString &str) const
    {
        if (str.size() < 2)
        {
            throw std::string("Bad param in QPuliMap: "
                "size < 2.");
        }

        auto ch = toupper(str.at(0));
        std::string check_str = "XYZ";
        if (check_str.find(ch) == std::string::npos)
        {
            throw std::string("Bad param in QPuliMap: "
                "Param not in [XYZ].");
        }

        bool ok = false;
        auto index = str.mid(1).toInt(&ok);
        if (!ok)
        {
            throw std::string("Bad param in QPuliMap: "
                "Convert index to int failed.");
        }

        return QTermPair(index, ch);
    }

    QPauliItem QPauliOperator::genQPauliItem(const QTerm &map_i, 
                                             const QTerm &map_j, 
                                             const complex_d &value) const
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
                    throw std::string("Bad para in QPauli.");
                }
            }
        }
        
        QPauliPair pair;
        pair.first = tmp_map;
        pair.second = QTerm2StdString(tmp_map);

        QPauliItem item;
        item.first = pair;
        item.second = result;

        return item;
    }

    std::string QPauliOperator::QTerm2StdString(const QTerm &map) const
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

            char ch = toupper(iter->second);
            str += ch + std::to_string(iter->first);
        }

        return str;
    }

    void QPauliOperator::reduceDuplicates()
    {
        std::map<std::string, complex_d> data_map;
        std::map<std::string, QTerm> term_map;

        for (auto i = 0; i < m_data.size(); i++)
        {
            auto item = m_data[i];
            auto pair = item.first;
            complex_d value = item.second;
            QTerm term = pair.first;
            std::string str = pair.second;

            if (data_map.find(str) != data_map.end())
            {
                data_map[str] += value;
            }
            else
            {
                data_map.insert(std::make_pair(str, value));
                term_map.insert(std::make_pair(str, term));
            }
        }

        QPauli pauli_data;
        for (auto iter = data_map.begin(); iter != data_map.end(); iter++)
        {
            auto err = std::abs(iter->second);
            if (err > m_error_threshold)
            {
                QPauliPair pair;
                pair.first = term_map[iter->first];
                pair.second = iter->first;

                pauli_data.emplace_back(std::make_pair(pair, iter->second));
            }
        }

        m_data = std::move(pauli_data);
    }

    void QPauliOperator::remapQubitIndex()
    {
        QIndexMap index_map;
        for (auto i = 0; i < m_data.size(); i++)
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
            m_index_map.insert(std::make_pair(cnt, iter->first));
            cnt++;
        }

        QPauli pauli_data;
        for (auto i = 0; i < m_data.size(); i++)
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

        m_data = std::move(pauli_data);
        m_max_index = cnt;
    }

}