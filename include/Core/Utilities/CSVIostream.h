/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

CVSIotream.h
Author: Wangjing
Created in 2018-8-31

Classes for write and read CVS file

*/

#ifndef IO_CSV_STREAM_H
#define IO_CSV_STREAM_H

#include "Core/Utilities/QPandaNamespace.h"
#include <iostream>
#include <fstream>
#include <istream>
#include <sstream>
#include <vector>
#include <string>
#include <list>
#include <map>

QPANDA_BEGIN

class CSVOfstream
{
public:
    CSVOfstream();
    CSVOfstream(const std::string &filename);
    void open(const std::string &filename);
    bool is_open();

    CSVOfstream & operator<<(const std::vector<std::pair<std::string, int>> &container);
    CSVOfstream & operator<<(const std::list<std::pair<std::string, int>> &container);
    CSVOfstream & operator<<(const std::map<std::string, int> &container);

    void close();
    virtual ~CSVOfstream();
private:
    std::ofstream m_ofstream;
};

class CSVIfstream
{
public:
    CSVIfstream();
    CSVIfstream(const std::string &filename);
    void open(const std::string &filename);
    bool is_open();

    CSVIfstream & operator>>(std::vector<std::pair<std::string, int>> &container);
    CSVIfstream & operator>>(std::list<std::pair<std::string, int>> &container);
    CSVIfstream & operator>>(std::map<std::string, int> &container);

    void close();
    virtual ~CSVIfstream();
private:
    std::ifstream m_ifstream;
};
QPANDA_END
#endif // IOCVSSTREAM_H
