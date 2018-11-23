#include "CSVIostream.h"

CSVOfstream::CSVOfstream()
{

}

CSVOfstream::CSVOfstream(const string &filename) :
    m_ofstream(filename)
{
}

bool CSVOfstream::is_open()
{
    return m_ofstream.is_open();
}

void CSVOfstream::close()
{
    m_ofstream.close();
}


CSVOfstream &CSVOfstream::operator<<(const vector<pair<string, int>> &container)
{
    for (auto &val : container)
    {
        m_ofstream << val.first << "," << to_string(val.second) << "\n";
    }

    return *this;
}

CSVOfstream &CSVOfstream::operator<<(const list<pair<string, int>> &container)
{
    for (auto &val : container)
    {
        m_ofstream << val.first << "," << to_string(val.second) << "\n";
    }

    return *this;
}

CSVOfstream &CSVOfstream::operator<<(const map<string, int> &container)
{
    for (auto &val : container)
    {
        m_ofstream << val.first << "," << to_string(val.second) << "\n";
    }

    return *this;
}

CSVOfstream::~CSVOfstream()
{
    m_ofstream.close();
}

void CSVOfstream::open(const string &filename)
{
    m_ofstream.open(filename, ios_base::out);
}

CSVIfstream::CSVIfstream()
{

}

CSVIfstream::CSVIfstream(const string &filename):
    m_ifstream(filename)
{
}

void CSVIfstream::open(const string &filename)
{
    m_ifstream.open(filename, ios_base::in);
}

bool CSVIfstream::is_open()
{
    return m_ifstream.is_open();
}

CSVIfstream &CSVIfstream::operator>>(vector<pair<string, int>> &container)
{
    container.clear();
    string sLine;
    while (getline(m_ifstream, sLine))
    {
        istringstream sin(sLine);
        vector<string> field_vec;
        string field;

        while (getline(sin, field, ','))
        {
            field_vec.emplace_back(field);
        }

        pair<string, int> data;
        data.first = field_vec[0].erase(0, field_vec[0].find_first_not_of(" \t\r\n"));
        string second = field_vec[1].erase(0, field_vec[1].find_first_not_of(" \t\r\n"));
        data.second = atoi(second.c_str());

        container.emplace_back(data);
    }

    return *this;
}

CSVIfstream &CSVIfstream::operator>>(list<pair<string, int>> &container)
{
    container.clear();
    string line;
    while (getline(m_ifstream, line))
    {
        istringstream sin(line);
        vector<string> field_vec;
        string filed;

        while (getline(sin, filed, ','))
        {
            field_vec.emplace_back(filed);
        }

        pair<string, int> data;
        data.first = field_vec[0].erase(0, field_vec[0].find_first_not_of(" \t\r\n"));
        string second = field_vec[1].erase(0, field_vec[1].find_first_not_of(" \t\r\n"));
        data.second = atoi(second.c_str());

        container.emplace_back(data);
    }

    return *this;
}

CSVIfstream &CSVIfstream::operator>>(map<string, int> &container)
{
    container.clear();
    string line;
    while (getline(m_ifstream, line))
    {
        istringstream sin(line);
        vector<string> field_vec;
        string field;

        while (getline(sin, field, ','))
        {
            field_vec.emplace_back(field);
        }

        pair<string, int> data;
        data.first = field_vec[0].erase(0, field_vec[0].find_first_not_of(" \t\r\n"));
        string second = field_vec[1].erase(0, field_vec[1].find_first_not_of(" \t\r\n"));
        data.second = atoi(second.c_str());

        container.insert(data);
    }

    return *this;
}

void CSVIfstream::close()
{
    m_ifstream.close();
}

CSVIfstream::~CSVIfstream()
{
    m_ifstream.close();
}
