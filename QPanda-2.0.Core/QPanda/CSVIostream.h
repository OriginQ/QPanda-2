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

#include <iostream>
#include <fstream>
#include <istream>
#include <sstream>
#include <vector>
#include <string>
#include <list>
#include <map>

using namespace std;

class CSVOfstream
{
public:
    CSVOfstream();
    CSVOfstream(const string &filename);
    void open(const string &filename);
    bool is_open();

    CSVOfstream & operator<<(const vector<pair<string, int>> &container);
    CSVOfstream & operator<<(const list<pair<string, int>> &container);
    CSVOfstream & operator<<(const map<string, int> &container);

	void close();
    virtual ~CSVOfstream();
private:
    ofstream m_ofstream;
};

class CSVIfstream
{
public:
    CSVIfstream();
    CSVIfstream(const string &filename);
    void open(const string &filename);
    bool is_open();

    CSVIfstream & operator>>(vector<pair<string, int>> &container);
    CSVIfstream & operator>>(list<pair<string, int>> &container);
    CSVIfstream & operator>>(map<string, int> &container);

    void close();
    virtual ~CSVIfstream();
private:
    ifstream m_ifstream;
};

#endif // IOCVSSTREAM_H
