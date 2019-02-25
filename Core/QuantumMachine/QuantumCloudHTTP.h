#ifndef QUANTUM_CLOUD_HTTP_H
#define QUANTUM_CLOUD_HTTP_H

#include <config.h>
#if 0
//#ifdef USE_CURL
#include <curl/curl.h>
#include <iostream>
#include <sstream>
#include <string>
#include "Core/QuantumMachine/Factory.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

//#define CURL_STATICLIB


QPANDA_BEGIN

class QuantumCloudHttp:public OriginQVM
{
private:
    std::string m_taskid;
    std::string m_APIKey;
    std::string m_computeAPI;
    std::string m_inqureAPI;
    std::string m_terminateAPI;

    int m_repeat_num;

    CURL *pCurl;
    CURLcode res;

    void configQuantumCloudHttp(const std::string& config_filepath);
    std::string postHttpJson(const std::string &sUrl, std::string & sJson);
    std::string QuantumCloudHttp::inqureResult(std::string task_typ);
    bool parserRecvJson(std::string recv_json, std::map<std::string, std::string>& recv_res);

    int PMeasureRun(QVec& qubit_vec);
    void MeasureRun(int repeat_num);

  public:
    QuantumCloudHttp();
    ~QuantumCloudHttp();

    void run();
    std::map<std::string, bool> getResultMap();

    std::vector<std::pair<size_t, double>> getProbTupleList(QVec &, int);
    std::vector<double> getProbList(QVec &, int);
    std::map<std::string, double> probRunDict(QProg &, QVec &, int);
    std::map<std::string, double> getProbDict(QVec &, int);
    std::vector<std::pair<size_t, double>> probRunTupleList(QProg &, QVec &, int);
    std::vector<double> probRunList(QProg &, QVec &, int);
    std::map<std::string, size_t> runWithConfiguration(QProg &, std::vector<ClassicalCondition> &, int);
    std::map<std::string, bool> directlyRun(QProg & qProg);

    std::map<std::string, size_t> quickMeasure(QVec &, size_t);
    std::string ResultToBinaryString(std::vector<ClassicalCondition>& cbit_vec);
};
QPANDA_END

#endif // USE_CURL

#endif // !_QUANTUM_CLOUD_HTTP_H
