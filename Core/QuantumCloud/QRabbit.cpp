#include "ThirdParty/rabbit/rabbit.hpp"
#include "Core/QuantumCloud/QRabbit.h"

USING_QPANDA

static std::string rabbit_json_extract(const rabbit::document& result_doc, std::string value)
{
    transform(value.begin(), value.end(), value.begin(), ::toupper);

    for (auto iter = result_doc.member_begin(); iter != result_doc.member_end(); ++iter)
    {
        std::string iter_name = iter->name();
        std::string iter_name_upper = iter->name();

        transform(iter_name_upper.begin(), iter_name_upper.end(), iter_name_upper.begin(), ::toupper);
        if (0 == strcmp(iter_name_upper.c_str(), value.c_str()))
            return std::string(iter_name);
    }

    QCERR_AND_THROW(std::runtime_error, "result json incorrect,no key or value found");
}

template<>
std::map<std::string, double> QPanda::get_parse_result<std::map<std::string, double>>(const std::string& recv_result_json)
{
    std::map<std::string, double> result;

    std::string result_json = recv_result_json;

    rabbit::document result_doc_parser;
    result_doc_parser.parse(result_json.c_str());

    if (result_doc_parser.is_array())
        result_json = result_doc_parser[0].str();

    rabbit::document result_doc;
    result_doc.parse(result_json.c_str());

    std::vector<std::string> key_list;
    std::vector<double> value_list;

    auto key_string = rabbit_json_extract(result_doc, "key");
    auto value_string = rabbit_json_extract(result_doc, "value");

    for (auto i = 0; i < result_doc[key_string.c_str()].size(); ++i)
        key_list.emplace_back(result_doc[key_string.c_str()][i].as_string());

    for (auto i = 0; i < result_doc[value_string.c_str()].size(); ++i)
    {
        auto val = result_doc[value_string.c_str()][i].is_double() ?
            result_doc[value_string.c_str()][i].as_double() : (double)result_doc[value_string.c_str()][i].as_int();
        value_list.emplace_back(val);
    }

    if (key_list.size() != value_list.size())
        QCERR_AND_THROW(std::runtime_error, "reasult json size incorrect");

    for (size_t i = 0; i < key_list.size(); i++)
        result.insert(make_pair(key_list[i], value_list[i]));

    return result;
}

template<>
std::vector<double> QPanda::get_parse_result<std::vector<double>>(const std::string& result_json)
{
    std::vector<double> result;

    rabbit::document result_doc;
    result_doc.parse(result_json.c_str());

    for (auto i = 0; i < result_doc.size(); ++i)
    {
        auto val = result_doc[i].is_double() ?
            result_doc[i].as_double() : (double)result_doc[i].as_int();
        result.emplace_back(val);
    }

    return result;
}

template<>
std::map<std::string, qcomplex_t> QPanda::get_parse_result<std::map<std::string, qcomplex_t>>(const std::string& result_json)
{
    std::map<std::string, qcomplex_t> result;

    rabbit::document result_doc;
    result_doc.parse(result_json.c_str());

    auto key_string = rabbit_json_extract(result_doc, "key");

    for (auto i = 0; i < result_doc[key_string.c_str()].size(); ++i)
    {
        auto key = result_doc[key_string.c_str()][i].as_string();
        auto val_real = result_doc["ValueReal"][i].is_double() ?
            result_doc["ValueReal"][i].as_double() : (double)result_doc["ValueReal"][i].as_int();
        auto val_imag = result_doc["ValueImag"][i].is_double() ?
            result_doc["ValueImag"][i].as_double() : (double)result_doc["ValueImag"][i].as_int();

        result.insert(make_pair(key, qcomplex_t(val_real, val_imag)));
    }

    return result;
}

template<>
qcomplex_t QPanda::get_parse_result<qcomplex_t>(const std::string& result_json)
{
    qcomplex_t result;

    rabbit::document result_doc;
    result_doc.parse(result_json.c_str());

    auto val_real = result_doc["ValueReal"][0].is_double() ?
        result_doc["ValueReal"][0].as_double() : (double)result_doc["ValueReal"][0].as_int();
    auto val_imag = result_doc["ValueImag"][0].is_double() ?
        result_doc["ValueImag"][0].as_double() : (double)result_doc["ValueImag"][0].as_int();

    result = qcomplex_t(val_real, val_imag);
    return result;
}

template<>
double QPanda::get_parse_result<double>(const std::string& result_json)
{
    double result;

    rabbit::document result_doc;
    result_doc.parse(result_json.c_str());

    auto exp_value_str = rabbit_json_extract(result_doc, "value");

    result = result_doc[exp_value_str.c_str()].is_double() ?
        result_doc[exp_value_str.c_str()].as_double() : (double)result_doc[exp_value_str.c_str()].as_int();

    return result;
}

template<>
std::vector<QStat> QPanda::get_parse_result<std::vector<QStat>>(const std::string& result_json)
{
    std::vector<QStat> result;

    rabbit::document qst_result_doc;
    qst_result_doc.parse(result_json.c_str());

    int rank = (int)std::sqrt(qst_result_doc.size());

    for (auto i = 0; i < rank; ++i)
    {
        QStat row_value;
        for (auto j = 0; j < rank; ++j)
        {
            auto qst_result_real_value = qst_result_doc[i*rank + j]["r"];
            auto qst_result_imag_value = qst_result_doc[i*rank + j]["i"];

            auto real_val = qst_result_real_value.is_double() ?
                qst_result_real_value.as_double() : (double)qst_result_real_value.as_int();
            auto imag_val = qst_result_imag_value.is_double() ?
                qst_result_imag_value.as_double() : (double)qst_result_imag_value.as_int();

            row_value.emplace_back(qcomplex_t(real_val, imag_val));
        }

        result.emplace_back(row_value);
    }

    return result;
}