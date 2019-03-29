#ifndef QUANTUM_CLOUD_HTTP_H
#define QUANTUM_CLOUD_HTTP_H
#include <config.h>
#ifdef USE_CURL

#include "QPanda.h"
#include <curl/curl.h>
#include "ThirdParty/rapidjson/document.h"
#include "ThirdParty/rapidjson/writer.h"
#include "ThirdParty/rapidjson/stringbuffer.h"
#include "include/Core/QuantumMachine/Factory.h"
QPANDA_BEGIN

class QuantumCloudHttp:public QVM
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

    void configQuantumCloudHttp(const std::string&);
    std::string postHttpJson(const std::string &, std::string &);
    std::string inqureResult(std::string);
    bool parserRecvJson(std::string , std::map<std::string, std::string>&);

    int PMeasureRun(QVec&);
    void MeasureRun(int);

  public:
    QuantumCloudHttp();
    ~QuantumCloudHttp();

    void run();
    std::map<std::string, bool> getResultMap();

    std::vector<double> getProbList(QVec &, int);
    std::vector<double> probRunList(QProg &, QVec &, int);

    std::map<std::string, double> getProbDict(QVec &, int);
    std::map<std::string, double> probRunDict(QProg &, QVec &, int);

    std::vector<std::pair<size_t, double>> getProbTupleList(QVec &, int);
    std::vector<std::pair<size_t, double>> probRunTupleList(QProg &, QVec &, int);

    std::map<std::string, size_t> runWithConfiguration(QProg &, std::vector<ClassicalCondition> &, int);
    std::map<std::string, bool> directlyRun(QProg & qProg);

    std::map<std::string, size_t> quickMeasure(QVec &, size_t);
    std::string ResultToBinaryString(std::vector<ClassicalCondition>& cbit_vec);
};
QPANDA_END

#endif // USE_CURL

#endif // !_QUANTUM_CLOUD_HTTP_H
