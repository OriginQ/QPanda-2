#pragma once

#include "ThirdParty/rabbit/rabbit.hpp"
#include "Core/Utilities/QPandaNamespace.h"

QPANDA_BEGIN

using std::map;
using std::vector;
using std::string;

template<typename T>
T get_parse_result(const std::string& result_json);

template<>
std::map<std::string, double> get_parse_result<std::map<std::string, double>>(const std::string& result_json);

template<>
std::vector<double> get_parse_result<std::vector<double>>(const std::string& result_json);

template<>
std::map<std::string, qcomplex_t> get_parse_result<std::map<std::string, qcomplex_t>>(const std::string& result_json);

template<>
qcomplex_t get_parse_result<qcomplex_t>(const std::string& result_json);

template<>
double get_parse_result<double>(const std::string& result_json);

template<>
std::vector<QStat> get_parse_result<std::vector<QStat>>(const std::string& result_json);

template<typename T>
void parse_result(const std::string& result_json, T& result)
{
    try
    {
        result = get_parse_result<T>(result_json);
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error(e.what());
    }
}

QPANDA_END
