/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QString.h

Author: LiYe
Created in 2018-09-18


*/

#ifndef QSTRING_H
#define QSTRING_H

#include <string>
#include <vector>
#include <ostream>
#include "Core/Utilities/QPandaNamespace.h"
QPANDA_BEGIN
class QString
{
public:
    enum SplitBehavior { KeepEmptyParts, SkipEmptyParts };
    enum BaseCovert { BIN, DEC, HEX };
public:
    QString();
    QString(char c):
        m_data(1, c)
    {}
    QString(const char *s):
        m_data(s)
    {}
    QString(const char *s, size_t n):
        m_data(s, n)
    {}
    QString(size_t n, char c):
        m_data(n, c)
    {}
    QString(const std::string &str):
        m_data(str)
    {}
    QString(const std::string &str, size_t pos, size_t len = std::string::npos):
        m_data(str, pos, len)
    {}
    template<class InputIterator>
    QString(InputIterator first, InputIterator last):
        m_data(first, last)
    {}
    QString(std::string &&str):
        m_data(str)
    {}
    QString(const QString &str):
        m_data(str.m_data)
    {
    }
    QString(QString &&str):
        m_data(str.m_data)
    {
    }

    QString &operator=(const char *s)
    {
        m_data = std::string(s);
        return *this;
    }
    QString &operator=(const std::string &str)
    {
        m_data = str;
        return *this;
    }
    QString &operator=(const QString &str)
    {
        m_data = str.m_data;
        return *this;
    }

    size_t size() const { return m_data.size(); }
    bool isEmpty() const { return m_data.empty(); }

    /*
     * The position of the first character of the first match.
     * If no matches were found, the function returns string::npos.
     */
    size_t find(const QString& sub_str, size_t pos = 0) const
    {
        return m_data.find(sub_str.m_data, pos);
    }

    char at(size_t i) const { return m_data.at(i); }
    char operator[](size_t i) const { return m_data[i]; }

    char front() const { return at(0); }
    char back() const { return at(size() - 1); }

    QString left(size_t n) const
    {
        if (n > size())
        {
            return *this;
        }
        return QString(m_data.substr(0, n));
    }
    QString right(size_t n) const
    {
        if (n > size())
        {
            return *this;
        }
        return QString(m_data.substr(size()-1 -n));
    }
    QString mid(size_t pos, size_t n = std::string::npos) const
    {
        return QString(m_data.substr(pos, n));
    }

    /*
     * split delimiter is one of the char in sep
     */
    std::vector<QString> split(
            const QString &sep,
            SplitBehavior behavior = KeepEmptyParts) const;

    /*
     * split delimiter is the sep
     */
    std::vector<QString> splitByStr(
            const QString &sep,
            SplitBehavior behavior = KeepEmptyParts) const;
    QString trimmed() const;

    QString toUpper() const;
    QString toLower() const;

    int toInt(bool *ok = nullptr, BaseCovert base = DEC) const;
    float toFloat(bool *ok = nullptr) const;
    double toDouble(bool *ok = nullptr) const;

    const std::string &data() const { return m_data; }

    bool operator==(const char *s) const
    {
        return m_data.compare(s) == 0;
    }

    bool operator!=(const char *s) const
    {
        return m_data.compare(s) != 0;
    }

    bool operator<(const char *s) const
    {
        return m_data.compare(s) < 0;
    }

    bool operator>(const char *s) const
    {
        return m_data.compare(s) > 0;
    }

    bool operator<=(const char *s) const
    {
        return m_data.compare(s) < 0 ||
                m_data.compare(s) == 0;
    }

    bool operator>=(const char *s) const
    {
        return m_data.compare(s) > 0 ||
                m_data.compare(s) == 0;
    }

    bool operator==(const std::string &s) const
    {
        return m_data.compare(s) == 0;
    }

    bool operator!=(const std::string &s) const
    {
        return m_data.compare(s) != 0;
    }

    bool operator<(const std::string &s) const
    {
        return m_data.compare(s) < 0;
    }

    bool operator>(const std::string &s) const
    {
        return m_data.compare(s) > 0;
    }

    bool operator<=(const std::string &s) const
    {
        return m_data.compare(s) < 0 ||
                m_data.compare(s) == 0;
    }

    bool operator>=(const std::string &s) const
    {
        return m_data.compare(s) > 0 ||
                m_data.compare(s) == 0;
    }

    friend bool operator==(const QString &s1, const QString &s2)
    {
        return s1.m_data.compare(s2.m_data) == 0;
    }

    friend bool operator!=(const QString &s1, const QString &s2)
    {
        return s1.m_data.compare(s2.m_data) != 0;
    }

    friend bool operator<(const QString &s1, const QString &s2)
    {
        return s1.m_data.compare(s2.m_data) < 0;
    }

    friend bool operator>(const QString &s1, const QString &s2)
    {
        return s1.m_data.compare(s2.m_data) > 0;
    }

    friend bool operator<=(const QString &s1, const QString &s2)
    {
        return s1.m_data.compare(s2.m_data) < 0 ||
                s1.m_data.compare(s2.m_data) == 0;
    }

    friend bool operator>=(const QString &s1, const QString &s2)
    {
        return s1.m_data.compare(s2.m_data) > 0 ||
                s1.m_data.compare(s2.m_data) == 0;
    }

    friend bool operator==(const char *s1, const QString &s2)
    {
        return s2.m_data.compare(s1) == 0;
    }

    friend bool operator!=(const char *s1, const QString &s2)
    {
        return std::string(s1).compare(s2.m_data) != 0;
    }

    friend bool operator<(const char *s1, const QString &s2)
    {
        return std::string(s1).compare(s2.m_data) < 0;
    }

    friend bool operator>(const char *s1, const QString &s2)
    {
        return std::string(s1).compare(s2.m_data) > 0;
    }

    friend bool operator<=(const char *s1, const QString &s2)
    {
        return std::string(s1).compare(s2.m_data) < 0 ||
                s2.m_data.compare(s1) == 0;
    }

    friend bool operator>=(const char *s1, const QString &s2)
    {
        return std::string(s1).compare(s2.m_data) > 0 ||
                s2.m_data.compare(s1) == 0;
    }

    friend bool operator==(const std::string &s1, const QString &s2)
    {
        return s1.compare(s2.m_data) == 0;
    }

    friend bool operator!=(const std::string &s1, const QString &s2)
    {
        return s1.compare(s2.m_data) != 0;
    }

    friend bool operator<(const std::string &s1, const QString &s2)
    {
        return s1.compare(s2.m_data) < 0;
    }

    friend bool operator>(const std::string &s1, const QString &s2)
    {
        return s1.compare(s2.m_data) > 0;
    }

    friend bool operator<=(const std::string &s1, const QString &s2)
    {
        return s1.compare(s2.m_data) < 0 ||
                s1.compare(s2.m_data) == 0;
    }

    friend bool operator>=(const std::string &s1, const QString &s2)
    {
        return s1.compare(s2.m_data) > 0 ||
                s1.compare(s2.m_data) == 0;
    }

    friend std::ostream &operator<<(std::ostream &out, const QString &str)
    {
        out << str.m_data;
        return out;
    }
private:
    std::string m_data;
};
QPANDA_END

#endif // QSTRING_H
