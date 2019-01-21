#include <iostream>
#include <fstream>
#include "PauliOperator.h"

QPANDA_BEGIN
PauliOperator::PauliOperator(double value)
{
    complex_d c(value);
    m_data.push_back(std::make_pair(std::make_pair(QTerm(), ""), c));
}

PauliOperator::PauliOperator(const complex_d &complex)
{
    m_data.push_back(std::make_pair(std::make_pair(QTerm(), ""), complex));
}

PauliOperator::PauliOperator(const QPauliMap &map)
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

    reduceDuplicates();
}

PauliOperator::PauliOperator(const QHamiltonian &hamiltonian)
{
    for (size_t i = 0; i < hamiltonian.size(); i++)
    {
        auto item = hamiltonian[i];

        QPauliPair pair;
        pair.first = item.first;
        pair.second = QTerm2StdString(item.first);

        m_data.emplace_back(std::make_pair(pair, item.second));
    }

    reduceDuplicates();
}

PauliOperator::PauliOperator(PauliOperator &&op):
    m_data(op.m_data)
{
}

PauliOperator::PauliOperator(const PauliOperator &op):
    m_data(op.m_data)
{
}

PauliOperator::PauliOperator(QPauli &&pauli):
    m_data(pauli)
{
    reduceDuplicates();
}

PauliOperator::PauliOperator(const QPauli &pauli):
    m_data(pauli)
{
    reduceDuplicates();
}

PauliOperator & PauliOperator::operator=(const PauliOperator &op)
{
    m_data = op.m_data;
    return *this;
}

PauliOperator & PauliOperator::operator=(PauliOperator &&op)
{
    m_data = op.m_data;
    return *this;
}

PauliOperator PauliOperator::dagger() const
{
    auto tmp_data = m_data;
    for (size_t i = 0; i < tmp_data.size(); i++)
    {
        auto &item = tmp_data[i];
        item.second = complex_d(item.second.real(), -item.second.imag());
    }

    return PauliOperator(tmp_data);
}

PauliOperator PauliOperator::remapQubitIndex(std::map<size_t, size_t> &index_map)
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

    QPauli pauli_data;
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

    return PauliOperator(pauli_data);
}

size_t PauliOperator::getMaxIndex()
{
    size_t max_index = 0;

    for (size_t i = 0; i < m_data.size(); i++)
    {
        auto index_map = m_data[i].first.first;
        auto iter = index_map.rbegin();
        if (iter != index_map.rend())
        {
            if (iter->first > max_index)
            {
                max_index = iter->first;
            }
        }
    }

    if (max_index != 0)
    {
        max_index++;
    }

    return max_index;
}

bool PauliOperator::isAllPauliZorI()
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

QHamiltonian PauliOperator::toHamiltonian(bool *ok) const
{
    QHamiltonian hamiltonian;

    for (size_t i = 0; i < m_data.size(); i++)
    {
        auto item = m_data[i];
        auto pair = item.first;
        auto value = item.second;

        if (value.imag() > m_error_threshold)
        {
            std::cout << "QPauli data cannot convert to Hamiltonian." << std::endl;
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

std::string PauliOperator::toString() const
{
    std::string str = "{";
    for (size_t i = 0; i < m_data.size(); i++)
    {
        str += "\n";

        auto item = m_data[i];
        auto pair = item.first;
        auto value = item.second;

        str += "\"" + pair.second + "\" : ";

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

PauliOperator PauliOperator::operator+(const PauliOperator &rhs) const
{
    const QPauli &cdata = rhs.data();
    QPauli pauli_data = m_data;
    pauli_data.insert(pauli_data.end(), cdata.begin(), cdata.end());

    PauliOperator pauli_op(std::move(pauli_data));
    pauli_op.reduceDuplicates();

    return pauli_op;
}

PauliOperator PauliOperator::operator-(const PauliOperator &rhs) const
{
    PauliOperator tmp_pauli(rhs);
    tmp_pauli *= -1.0;

    const QPauli &cdata = tmp_pauli.data();
    QPauli pauli_data = m_data;
    pauli_data.insert(pauli_data.end(), cdata.begin(), cdata.end());

    PauliOperator pauli_op(std::move(pauli_data));
    pauli_op.reduceDuplicates();

    return pauli_op;
}

PauliOperator PauliOperator::operator*(const PauliOperator &rhs) const
{
    QPauli pauli_data;
    const QPauli &rhs_data = rhs.data();

    for (size_t i = 0; i < m_data.size(); i++)
    {
        auto &item_i = m_data[i];
        for (size_t j = 0; j < rhs_data.size(); j++)
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

    PauliOperator pauli(std::move(pauli_data));
    pauli.reduceDuplicates();

    return pauli;
}

PauliOperator &PauliOperator::operator+=(const PauliOperator &rhs)
{
    const QPauli &cdata = rhs.data();

    m_data.insert(m_data.end(), cdata.begin(), cdata.end());
    reduceDuplicates();

    return *this;
}

PauliOperator &PauliOperator::operator-=(const PauliOperator &rhs)
{
    PauliOperator tmp_pauli(rhs);
    tmp_pauli *= -1;

    const QPauli &cdata = tmp_pauli.data();

    m_data.insert(m_data.end(), cdata.begin(), cdata.end());
    reduceDuplicates();

    return *this;
}

PauliOperator &PauliOperator::operator*=(const PauliOperator &rhs)
{
    QPauli tmp_data;
    const QPauli &rhs_data = rhs.data();

    for (size_t i = 0; i < m_data.size(); i++)
    {
        auto &item_i = m_data[i];
        for (size_t j = 0; j < rhs_data.size(); j++)
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

    return *this;
}

QTermPair PauliOperator::genQTermPair(const QString &str) const
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

QPauliItem PauliOperator::genQPauliItem(const QTerm &map_i,
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
                std::string err = "Bad para in QPauli.";
                std::cout << err << std::endl;
                throw err;
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

std::string PauliOperator::QTerm2StdString(const QTerm &map) const
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

void PauliOperator::reduceDuplicates()
{
    std::map<std::string, complex_d> data_map;
    std::map<std::string, QTerm> term_map;

    for (size_t i = 0; i < m_data.size(); i++)
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

QPANDA_END
