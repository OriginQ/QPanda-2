/*
* Copyright (c) 2019 Origin Quantum Computing. All Right Reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
* http://www.apache.org/licenses/LICENSE-2.0
*/
/*! \file QCloudMachine.h */
#ifndef QCLOUD_MACHINE_H
#define QCLOUD_MACHINE_H

#ifdef USE_CURL

#include <include/QPandaConfig.h>
//#ifdef USE_CURL

#include "QPanda.h"

#ifdef USE_CURL
#include <curl/curl.h>
#endif // USE_CURL

#include "ThirdParty/rapidjson/document.h"
#include "ThirdParty/rapidjson/writer.h"
#include "ThirdParty/rapidjson/stringbuffer.h"
#include "include/Core/QuantumMachine/Factory.h"
using namespace rapidjson;
QPANDA_BEGIN
/**
* @namespace QPanda
*/

/*
* @class QCloudMachine
* @brief Quantum Cloud Machine  for  connecting  QCloud server
* @ingroup QuantumMachine
* @note  QCloudMachine  also provides  python interface
*/
class QCloudMachine:public QVM
{
public:
    QCloudMachine();
    ~QCloudMachine();

    /**
    * @brief  Init  the quantum  machine environment
    * @return     void
    * @note   use  this at the begin
    */
    void init();

    /**
    * @brief  Run a measure quantum program with json or dict configuration
    * @param[in]  QProg& the reference to a quantum program
    * @param[in]  rapidjson::Document&   config  message
    * @return     std::string  QCloud taskid  and  task status
    * @see
        * @code
            rapidjson::Document doc1;
            doc1.SetObject();
            rapidjson::Document::AllocatorType &allocator1 = doc1.GetAllocator();
            doc1.AddMember("BackendType", QMachineType::CPU, allocator1);
            doc1.AddMember("RepeatNum", 1000, allocator1);
            doc1.AddMember("token", "E5CD3EA3CB534A5A9DA60280A52614E1", allocator1);
            std::cout << QCM->runWithConfiguration(qprog, doc1) << endl;
        * @endcode
    */
    std::string runWithConfiguration(QProg &,rapidjson::Document &);

    /**
    * @brief  Run a pmeasure quantum program with json or dict configuration
    * @param[in]  QProg& the reference to a quantum program
    * @param[in]  QVec qubits list
    * @param[in]  rapidjson::Document&   config  message
    * @return     std::string  QCloud taskid  and  task status
    * @see
        * @code
            rapidjson::Document doc;
            doc.SetObject();
            rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();

            doc.AddMember("BackendType", QMachineType::CPU, allocator);
            doc.AddMember("token", "E5CD3EA3CB534A5A9DA60280A52614E1", allocator);
            std::cout << QCM->probRunDict(qprog, qlist, doc) << endl;;
        * @endcode
    */
    std::string probRunDict(QProg &,QVec, rapidjson::Document &);

private:
    /** @brief TASK_TYPE enum, with inline docs */
    enum TASK_TYPE
    {
        MEASURE = 0, /**< enum value MEASURE. */
        PMEASURE     /**< enum value PMEASURE. */
    }; 

    /*
    	@brief  PostHttpJson
    	@author Yulei
    	@date   2019/04/08 13:38
    	@param[out] 
    	@param[in]  const std::string &  
    	@param[in]  std::string &  
    	@return     std::string  
    */
    std::string postHttpJson(const std::string &, std::string &);
};


/**
* @brief  Quamtum program tramsform to binary data
* @ingroup  Utilities
* @param[in]  size_t qubit num
* @param[in]  size_t cbit num 
* @param[in]  QProg the reference to a quantum program
* @return     std::string  binary data
*/
std::string QProgToBinary(size_t qubit_num, size_t cbit_num, QProg prog);


QPANDA_END
#endif // USE_CURL

#endif
