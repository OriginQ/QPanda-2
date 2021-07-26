#include <iostream>
#include <limits>
#include "Core/Utilities/Tools/OriginCollection.h"
#include "Core/VirtualQuantumProcessor/NoiseQPU/NoiseModel.h"
#include "Core/VirtualQuantumProcessor/CPUImplQPU.h"
#include "QPanda.h"
#include "gtest/gtest.h"
#include "Core/QuantumCircuit/QNodeDeepCopy.h"
USING_QPANDA
using namespace std;

TEST(OriginCollection,CreateTest)
{
    OriginCollection test("./test");
    test={"key","value","value2"};


    test.insertValue("444", 0.89898,0.454543);
    test.insertValue(555, 0.89898,"akjsdhjahd");


    std::vector<int > cc = { 1, 2, 3, 4 };
    test.insertValue(666, 0.89898, cc );
    
    std::vector<std::string> key_n = {"key","value2"};
    test.insertValue(key_n, "888", 122);
    test.insertValue(key_n, 6564465, 345);
    
    auto value =test.getValue("value");
    test.write();

    
   /* for(auto & aiter : value)
    {
        std::cout<<aiter<<std::endl;
    }*/

    OriginCollection test2("test2");
    test2 = { "key","value" };
    std::map<std::string, bool> a;
    a.insert(std::make_pair("c0", true));
    a.insert(std::make_pair("c1", true));
    a.insert(std::make_pair("c2", true));
    for (auto aiter : a)
    {
        test2.insertValue( aiter.first, aiter.second);
    }

    //excepted_val = R"({"key":["c0","c1","c2"],"value":[true,true,true]})";
    //std::cout << test2.getJsonString() << std::endl;
    ASSERT_EQ(value[0],"0.89898");
    ASSERT_EQ(value[1],"0.89898");
    ASSERT_EQ(value[2],"0.89898");
    /*ASSERT_EQ(value[3], NULL);
    ASSERT_EQ(value[4], "");*/
    /*ASSERT_EQ(a.find("c1"), TRUE);
    ASSERT_EQ(a.find("c2"), TRUE);*/
    //cout << "OriginCollection tests over." << endl;

}

