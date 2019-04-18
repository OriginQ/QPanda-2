#include "QPanda.h"
#include "gtest/gtest.h"
#include "QuantumMachine/QCloudMachine.h"
using namespace std;
using namespace rapidjson;
USING_QPANDA

TEST(CloudHttpTest, Post)
{
    auto QCM = new QCloudMachine();
    QCM->init();
    auto qlist = QCM->allocateQubits(10);
    auto clist = QCM->allocateCBits(10);
    auto qprog = QProg();

    for_each(qlist.begin(), qlist.end(), [&](Qubit *val) { qprog << H(val); });
    qprog << CZ(qlist[1], qlist[5])
        << CZ(qlist[3], qlist[7])
        << CZ(qlist[0], qlist[4])
        << RZ(qlist[7], PI / 4)
        << RX(qlist[5], PI / 4)
        << RX(qlist[4], PI / 4)
        << RY(qlist[3], PI / 4)
        << CZ(qlist[2], qlist[6])
        << RZ(qlist[3], PI / 4)
        << RZ(qlist[8], PI / 4)
        << CZ(qlist[9], qlist[5])
        << RY(qlist[2], PI / 4)
        << RZ(qlist[9], PI / 4)
        << CZ(qlist[2], qlist[3]);


    /******Test PMeasure***********/
    rapidjson::Document doc;
    doc.SetObject();
    rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();

    doc.AddMember("BackendType", QMachineType::CPU, allocator);
    doc.AddMember("token", "E5CD3EA3CB534A5A9DA60280A52614E1", allocator);
    std::cout << QCM->probRunDict(qprog, qlist, doc) << endl;;

    /******Test Measure***********/
    rapidjson::Document doc1;
    doc1.SetObject();
    rapidjson::Document::AllocatorType &allocator1 = doc1.GetAllocator();

    doc1.AddMember("BackendType", QMachineType::CPU, allocator1);
    doc1.AddMember("RepeatNum", 1000, allocator1);
    doc1.AddMember("token", "E5CD3EA3CB534A5A9DA60280A52614E1", allocator1);
    std::cout << QCM->runWithConfiguration(qprog, doc1) << endl;

    QCM->finalize();

    getchar();
}

#if 0

void QCloudMachine::configQuantumCloudHttp(const std::string& config_filepath)
{
    TiXmlDocument config_doc(config_filepath.c_str());
    bool loadSuccess = config_doc.LoadFile();
    if (!loadSuccess)
    {
        std::cout << "could not load the test file.Error:" << config_doc.ErrorDesc() << std::endl;
        throw std::invalid_argument("load failed");
    }

    TiXmlElement *RootElement = config_doc.RootElement();

    TiXmlNode *BetaAPIKEY = RootElement->FirstChildElement("QuantumCloudBetaConfig")
        ->FirstChildElement("BetaAPIKEY")->FirstChild();
    TiXmlNode *BetaComputeAPI = RootElement->FirstChildElement("QuantumCloudBetaConfig")
        ->FirstChildElement("BetaComputeAPI")->FirstChild();
    TiXmlNode *BetaInqureAPI = RootElement->FirstChildElement("QuantumCloudBetaConfig")
        ->FirstChildElement("BetaInqureAPI")->FirstChild();
    TiXmlNode *BetaTerminateAPI = RootElement->FirstChildElement("QuantumCloudBetaConfig")
        ->FirstChildElement("BetaTerminateAPI")->FirstChild();
    TiXmlNode *repeat_num = RootElement->FirstChildElement("QProg")
        ->FirstChildElement("MonteCarloRepeatNum");

    if (nullptr == BetaAPIKEY || nullptr == BetaComputeAPI || nullptr == BetaInqureAPI
        || nullptr == BetaTerminateAPI || nullptr == repeat_num)
    {
        QCERR("config Error");
        throw std::invalid_argument("config Error");
    }
    else
    {
        m_APIKey = BetaAPIKEY->Value();
        m_computeAPI = BetaComputeAPI->Value();
        m_inqureAPI = BetaInqureAPI->Value();
        m_terminateAPI = BetaTerminateAPI->Value();
        m_repeat_num = atoi(repeat_num->Value());
    }

}

QCloudMachine::QCloudMachine() :
    pCurl(nullptr),
    m_repeat_num(100)
{
    std::string config_filepath;
    char * QPanda_config_path = nullptr;
    QPanda_config_path = getenv("QPANDA_CONFIG_PATH");
    if (nullptr != QPanda_config_path)
    {
        config_filepath.append(QPanda_config_path);
        config_filepath.append("\\QuantumCloudConfig.xml");
        configQuantumCloudHttp(config_filepath);
    }
    else
    {
        m_APIKey = BETA_APIKEY;
        m_computeAPI = BETA_COMPUTEAPI;
        m_inqureAPI = BETA_INQUREAPI;
        m_terminateAPI = BETA_TERMINATEAPI;
    }

    curl_global_init(CURL_GLOBAL_ALL);
}


REGISTER_QUANTUM_MACHINE(QCloudMachine);

#endif

//TEST(CloudHttpTest, ConfigXML)
//{
//    TiXmlDocument mygetQubitNum()("D:\\QuantumCloudConfig.xml");//xml文档对象  
//    bool loadSuccess = mygetQubitNum().LoadFile();//加载文档  
//    if (!loadSuccess)
//    {
//        std::cout << "could not load the test file.Error:" << mygetQubitNum().ErrorDesc() << std::endl;
//        exit(1);
//    }
//    TiXmlElement *RootElement = mygetQubitNum().RootElement();  
//    TiXmlNode *BetaAPIKEY = RootElement->FirstChildElement("QuantumCloudBetaConfig")
//        ->FirstChildElement("BetaAPIKEY")->FirstChild();
//    TiXmlNode *BetaComputeAPI = RootElement->FirstChildElement("QuantumCloudBetaConfig")
//        ->FirstChildElement("BetaComputeAPI")->FirstChild();
//    TiXmlNode *BetaInqureAPI = RootElement->FirstChildElement("QuantumCloudBetaConfig")
//        ->FirstChildElement("BetaInqureAPI")->FirstChild();
//    TiXmlNode *BetaTerminateAPI = RootElement->FirstChildElement("QuantumCloudBetaConfig")
//        ->FirstChildElement("BetaTerminateAPI")->FirstChild();
//    TiXmlNode *repeatNum = RootElement->FirstChildElement("QProg")
//        ->FirstChildElement("MonteCarloRepeatNum");
//
//    cout << BetaAPIKEY->Value() << endl;
//    cout << BetaComputeAPI->Value()  << endl;
//    cout << BetaInqureAPI->Value() << endl;
//    cout << BetaTerminateAPI->Value() << endl;
//    cout << repeatNum->Value()<< endl;
//
//
//    //TiXmlDocument mygetQubitNum()("D:\\QuantumCloudConfig.xml");//xml文档对象  
//    //bool loadSuccess = mygetQubitNum().LoadFile();//加载文档  
//    //if (!loadSuccess)
//    //{
//    //    std::cout << "could not load the test file.Error:" << mygetQubitNum().ErrorDesc() << std::endl;
//    //    exit(1);
//    //}
//
//    //TiXmlElement *RootElement = mygetQubitNum().RootElement();  //根元素, Info  
//    //std::cout << "[root name]" << RootElement->Value() << " ";
//    //TiXmlAttribute *pattr = RootElement->FirstAttribute();//第一个属性
//    //while (NULL != pattr) //输出所有属性
//    //{
//    //    std::cout << pattr->Name() << "1:" << pattr->Value() << " ";
//    //    pattr = pattr->Next();
//    //}
//    //std::cout << std::endl;
//    //TiXmlElement *pEle = RootElement;
//
//    ////遍历该结点  
//    //for (TiXmlElement *StuElement = pEle->FirstChildElement();//第一个子元素  
//    //    StuElement != NULL;
//    //    StuElement = StuElement->NextSiblingElement())//下一个兄弟元素  
//    //{
//    //    // StuElement->Value() 节点名称  
//    //    std::cout << StuElement->Value() << " ";
//    //    TiXmlAttribute *pAttr = StuElement->FirstAttribute();//第一个属性  
//
//
//    //    while (NULL != pAttr) //输出所有属性  
//    //    {
//    //        std::cout << pAttr->Name() << "2:" << pAttr->Value() << " ";
//    //        pAttr = pAttr->Next();
//    //    }
//    //    std::cout << std::endl;
//
//    //    //输出子元素的值  
//    //    for (TiXmlElement *sonElement = StuElement->FirstChildElement();
//    //        sonElement != NULL;
//    //        sonElement = sonElement->NextSiblingElement())
//    //    {
//    //        std::cout << sonElement->Value() << "3:" << sonElement->FirstChild()->Value() << std::endl;
//    //    }
//    //}
//    getchar();
//}




