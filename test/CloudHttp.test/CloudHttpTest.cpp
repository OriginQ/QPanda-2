#include "QPanda.h"
#include "gtest/gtest.h"
#include "QuantumMachine/QuantumCloudHTTP.h"
using namespace std;
using namespace rapidjson;

USING_QPANDA

void HttpDemo()
{
    rapidjson::Document jsonDoc;
    rapidjson::Document::AllocatorType &allocator = jsonDoc.GetAllocator();
    rapidjson::Value root(kObjectType);

    root.AddMember("typ", "mcpr", allocator);

    root.AddMember("qprog", "QINIT 8\nCREG 2\nH 2\nMEASURE 0,$0\n", allocator);

    root.AddMember("token", "B8F1C7337CB9470E81697FCD4D9FB6DE", allocator);

    root.AddMember("repeat", 100, allocator);

    root.AddMember("meaarr", "{1,10}", allocator);

    root.AddMember("taskTyp", "2", allocator);


    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    root.Accept(writer);
    std::string strJson = buffer.GetString();

    std::cout << strJson << std::endl;;

    std::stringstream  out;

    curl_global_init(CURL_GLOBAL_ALL);

    CURL *pCurl = nullptr;
    CURLcode res;

    pCurl = curl_easy_init();


    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-type: application/json");
    headers = curl_slist_append(headers, "accept: application/json");
    headers = curl_slist_append(headers, "Charset: UTF-8");
    curl_easy_setopt(pCurl, CURLOPT_HTTPHEADER, headers);

    curl_easy_setopt(pCurl, CURLOPT_TIMEOUT, 3);

    curl_easy_setopt(pCurl, CURLOPT_CONNECTTIMEOUT, 3);

    curl_easy_setopt(pCurl, CURLOPT_URL, "https://qcode.qubitonline.cn/api/QCode/submitTask.json");

    curl_easy_setopt(pCurl, CURLOPT_HEADER, true);

    curl_easy_setopt(pCurl, CURLOPT_POST, true);

    curl_easy_setopt(pCurl, CURLOPT_SSL_VERIFYHOST, false);

    curl_easy_setopt(pCurl, CURLOPT_SSL_VERIFYPEER, false);

    curl_easy_setopt(pCurl, CURLOPT_READFUNCTION, NULL);

    curl_easy_setopt(pCurl, CURLOPT_NOSIGNAL, 1);

    curl_easy_setopt(pCurl, CURLOPT_POSTFIELDS, strJson.c_str());

    curl_easy_setopt(pCurl, CURLOPT_POSTFIELDSIZE, strJson.size());

    curl_easy_setopt(pCurl, CURLOPT_WRITEFUNCTION, recvJsonData);

    //curl_easy_setopt(pCurl, CURLOPT_VERBOSE, 1);

    curl_easy_setopt(pCurl, CURLOPT_WRITEDATA, &out);

    res = curl_easy_perform(pCurl);
    if (CURLE_OK != res)
    {
        std::cout << "post failed : " << curl_easy_strerror(res) << std::endl;
        throw std::invalid_argument("post failed");
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(pCurl);

    std::string s2(out.str().substr(out.str().find("{")));

    cout << s2 << endl;

    Document recvDoc;
    if (recvDoc.Parse(s2.c_str()).HasParseError() || !recvDoc.HasMember("obj"))
    {
        cout << recvDoc.GetParseError();
        cout << recvDoc.GetErrorOffset() << endl;
        throw std::invalid_argument("invalid argument");
    }
    else
    {
        const rapidjson::Value &Obj = recvDoc["obj"];
        if (!Obj.IsObject() || !Obj.HasMember("taskid"))
        {
            throw std::invalid_argument("invalid argument");
        }
        else
        {
            const rapidjson::Value &Taskid = Obj["taskid"];
            cout << Taskid.GetString() << endl;
        }
    }

    curl_global_cleanup();
    getchar();
}

TEST(CloudHttpTest, Post)
{
    auto  test = QuantumCloudHttp();

    init();
    test.init();

    auto prog = CreateEmptyQProg();
    auto qlist = qAllocMany(5);
    auto clist = cAllocMany(3);

    prog << H(qlist[0]) << H(qlist[1]) << H(qlist[2]) << H(qlist[2]);

    test.load(prog);

    auto result = test.getProbDict(qlist, 1);
    for (auto val : result)
    {
        cout << val.first << " : " << val.second << endl;
    }

    finalize();



    //auto prog = CreateEmptyQProg();
    //vector<ClassicalCondition> c;
    //QVec vec = qAllocMany(3);

    //std::map<std::string, bool> result;
    //std::map<std::string, double> result1;
    //std::map<std::string, size_t> result2;

    ////result2 = test.runWithConfiguration(prog, c, 100);

    //result1 = test.getProbDict(vec, 1);
    //for (auto val : result1)
    //{
    //    cout << val.first << " : " << val.second << endl;
    //}


    //result = test.getResultMap();
    //for (auto val : result)
    //{
    //    cout << val.first << " : " << val.second << endl;
    //}

    //result1 = test.getProbDict(vec,-1);


    getchar();
}


TEST(CloudHttpTest, ConfigXML)
{
    exit(0);

    TiXmlDocument mydoc("D:\\QuantumCloudConfig.xml");//xml文档对象  
    bool loadSuccess = mydoc.LoadFile();//加载文档  
    if (!loadSuccess)
    {
        std::cout << "could not load the test file.Error:" << mydoc.ErrorDesc() << std::endl;
        exit(1);
    }
    TiXmlElement *RootElement = mydoc.RootElement();  
    TiXmlNode *BetaAPIKEY = RootElement->FirstChildElement("QuantumCloudBetaConfig")
        ->FirstChildElement("BetaAPIKEY")->FirstChild();
    TiXmlNode *BetaComputeAPI = RootElement->FirstChildElement("QuantumCloudBetaConfig")
        ->FirstChildElement("BetaComputeAPI")->FirstChild();
    TiXmlNode *BetaInqureAPI = RootElement->FirstChildElement("QuantumCloudBetaConfig")
        ->FirstChildElement("BetaInqureAPI")->FirstChild();
    TiXmlNode *BetaTerminateAPI = RootElement->FirstChildElement("QuantumCloudBetaConfig")
        ->FirstChildElement("BetaTerminateAPI")->FirstChild();
    TiXmlNode *repeatNum = RootElement->FirstChildElement("QProg")
        ->FirstChildElement("MonteCarloRepeatNum");

    cout << BetaAPIKEY->Value() << endl;
    cout << BetaComputeAPI->Value()  << endl;
    cout << BetaInqureAPI->Value() << endl;
    cout << BetaTerminateAPI->Value() << endl;
    cout << repeatNum->Value()<< endl;


    //TiXmlDocument mydoc("D:\\QuantumCloudConfig.xml");//xml文档对象  
    //bool loadSuccess = mydoc.LoadFile();//加载文档  
    //if (!loadSuccess)
    //{
    //    std::cout << "could not load the test file.Error:" << mydoc.ErrorDesc() << std::endl;
    //    exit(1);
    //}

    //TiXmlElement *RootElement = mydoc.RootElement();  //根元素, Info  
    //std::cout << "[root name]" << RootElement->Value() << " ";
    //TiXmlAttribute *pattr = RootElement->FirstAttribute();//第一个属性
    //while (NULL != pattr) //输出所有属性
    //{
    //    std::cout << pattr->Name() << "1:" << pattr->Value() << " ";
    //    pattr = pattr->Next();
    //}
    //std::cout << std::endl;
    //TiXmlElement *pEle = RootElement;

    ////遍历该结点  
    //for (TiXmlElement *StuElement = pEle->FirstChildElement();//第一个子元素  
    //    StuElement != NULL;
    //    StuElement = StuElement->NextSiblingElement())//下一个兄弟元素  
    //{
    //    // StuElement->Value() 节点名称  
    //    std::cout << StuElement->Value() << " ";
    //    TiXmlAttribute *pAttr = StuElement->FirstAttribute();//第一个属性  


    //    while (NULL != pAttr) //输出所有属性  
    //    {
    //        std::cout << pAttr->Name() << "2:" << pAttr->Value() << " ";
    //        pAttr = pAttr->Next();
    //    }
    //    std::cout << std::endl;

    //    //输出子元素的值  
    //    for (TiXmlElement *sonElement = StuElement->FirstChildElement();
    //        sonElement != NULL;
    //        sonElement = sonElement->NextSiblingElement())
    //    {
    //        std::cout << sonElement->Value() << "3:" << sonElement->FirstChild()->Value() << std::endl;
    //    }
    //}
    getchar();
}

