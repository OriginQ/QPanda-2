#include "gtest/gtest.h"
#include "QAlg/ArithmeticUnit/ArithmeticUnit.h"
#include "QPanda.h"

USING_QPANDA
using namespace std;

void test_bind_data()
{
    int n, m;
    scanf("%d%d", &n, &m);
    auto a = qAllocMany(n);
    QProg prog;
    prog << bind_data(m, a);
    auto result = probRunDict(prog, a);
    for (auto& val : result)
    {
        if (val.second > 0)
            std::cout << val.first << ", " << val.second << std::endl;
    }
}

void test_bind_nonnegative_data()
{
    int n, m;
    scanf("%d%d", &n, &m);
    auto a = qAllocMany(n);
    QProg prog;
    prog << bind_nonnegative_data(m, a);
    auto result = probRunDict(prog, a);
    for (auto& val : result)
    {
        if (val.second > 0)
            std::cout << val.first << ", " << val.second << std::endl;
    }
}

void test_QAdder()
{
    int n, m;
    scanf("%d%d", &n, &m);
    const int len = 10;
    auto a = qAllocMany(len);
    auto b = qAllocMany(len);
    auto c = qAlloc();
    QProg prog;
    prog << bind_data(n, a) << bind_data(m, b)
        << QAdder(a, b, c);
    auto result = probRunDict(prog, a);
    for (auto& val : result)
    {
        if (val.second > 0)
            std::cout << val.first << ", " << val.second << std::endl;
    }
}

void test_QAdderWithCarry()
{
    int n, m;
    scanf("%d%d", &n, &m);
    const int len = 3;
    auto a = qAllocMany(len);
    auto b = qAllocMany(len);
    auto c = qAlloc();
    auto d = qAlloc();
    QVec t;
    t += a;
    t += d;
    QProg prog;
    prog << bind_data(n, a) << bind_data(m, b)
        << QAdder(a, b, c, d);
    auto result = probRunDict(prog, t);
    for (auto& val : result)
    {
        if (val.second > 0)
            std::cout << val.first << ", " << val.second << std::endl;
    }
}

void test_QAdd()
{
    int n, m;
    scanf("%d%d", &n, &m);
    const int len = 6;
    auto a = qAllocMany(len);
    auto b = qAllocMany(len);
    auto c = qAllocMany(len + 2);
    QProg prog;
    prog << bind_data(n, a) << bind_data(m, b)
        << QAdd(a, b, c);
    auto result = probRunDict(prog, a);
    for (auto& val : result)
    {
        if (val.second > 0)
            std::cout << val.first << ", " << val.second << std::endl;
    }
}

void test_QComplement()
{
    int n, m;
    scanf("%d%d", &n, &m);
    auto a = qAllocMany(n);
    auto k = qAllocMany(n + 2);
    QProg prog;
    prog << bind_data(m, a)
        << QComplement(a, k);
    auto result = probRunDict(prog, a);
    for (auto& val : result)
    {
        if (val.second > 0)
            std::cout << val.first << ", " << val.second << std::endl;
    }
}

void test_QSub()
{
    int n, m;
    scanf("%d%d", &n, &m);
    const int len = 6;
    auto a = qAllocMany(len);
    auto b = qAllocMany(len);
    auto c = qAllocMany(len + 2);
    QProg prog;
    prog << bind_data(n, a) << bind_data(m, b)
        << QSub(a, b, c);
    auto result = probRunDict(prog, a);
    for (auto& val : result)
    {
        if (val.second > 0)
            std::cout << val.first << ", " << val.second << std::endl;
    }
}

void test_QMultiplier()
{
    int n, m;
    scanf("%d%d", &n, &m);
    const int len = 4;
    auto a = qAllocMany(len);
    auto b = qAllocMany(len);
    auto k = qAllocMany(len + 1);
    auto d = qAllocMany(len * 2);
    QProg prog;
    prog << bind_data(n, a) << bind_data(m, b)
        << QMultiplier(a, b, k, d);
    auto result = probRunDict(prog, d);
    for (auto& val : result)
    {
        if (val.second > 0)
            std::cout << val.first << ", " << val.second << std::endl;
    }
}

void test_QMul()
{
    int n, m;
    scanf("%d%d", &n, &m);
    const int len = 4;
    auto a = qAllocMany(len);
    auto b = qAllocMany(len);
    auto k = qAllocMany(len);
    auto d = qAllocMany(len * 2 - 1);
    QProg prog;
    prog << bind_data(n, a) << bind_data(m, b)
        << QMul(a, b, k, d);
    auto result = probRunDict(prog, d);
    for (auto& val : result)
    {
        if (val.second > 0)
            std::cout << val.first << ", " << val.second << std::endl;
    }
}

void test_QDivider()
{
    int n, m;
    scanf("%d%d", &n, &m);
    const int len = 4;
    auto a = qAllocMany(len);
    auto b = qAllocMany(len);
    auto c = qAllocMany(len);
    auto k = qAllocMany(len * 2 + 2);
    auto t = cAlloc();
    QProg prog;
    prog << bind_data(n, a) << bind_data(m, b)
        << QDivider(a, b, c, k, t);
    auto result = probRunDict(prog, c);
    for (auto& val : result)
    {
        if (val.second > 0)
            std::cout << val.first << ", " << val.second << std::endl;
    }
}

void test_QDividerWithAccuracy()
{
    int n, m;
    scanf("%d%d", &n, &m);
    const int len = 3;
    auto a = qAllocMany(len);
    auto b = qAllocMany(len);
    auto c = qAllocMany(len);
    auto k = qAllocMany(len * 3 + 5);
    auto f = qAllocMany(3);
    auto t = cAllocMany(f.size()+2);
    QVec tt;
    tt += f;
    tt += c;
    QProg prog;
    prog << bind_data(n, a) << bind_data(m, b)
        << QDivider(a, b, c, k, f, t);
    auto result = probRunDict(prog, tt);
    for (auto& val : result)
    {
        if (val.second > 0)
            std::cout << val.first << ", " << val.second << std::endl;
    }
}

void test_QDiv()
{
    int n, m;
    scanf("%d%d", &n, &m);
    const int len = 4;
    auto a = qAllocMany(len);
    auto b = qAllocMany(len);
    auto c = qAllocMany(len);
    auto k = qAllocMany(len * 2 + 4);
    auto t = cAlloc();
    QProg prog;
    prog << bind_data(n, a) << bind_data(m, b)
        << QDiv(a, b, c, k, t);
    auto result = probRunDict(prog, c);
    for (auto& val : result)
    {
        if (val.second > 0)
            std::cout << val.first << ", " << val.second << std::endl;
    }
}

void test_QDivWithAccuracy()
{
    int n, m;
    scanf("%d%d", &n, &m);
    const int len = 3;
    auto a = qAllocMany(len);
    auto b = qAllocMany(len);
    auto c = qAllocMany(len);
    auto k = qAllocMany(len * 3 + 7);
    auto f = qAllocMany(3);
    auto t = cAllocMany(f.size()+2);
    QVec tt;
    tt += f;
    tt += c;
    QProg prog;
    prog << bind_data(n, a) << bind_data(m, b)
        << QDiv(a, b, c, k, f, t);
    auto result = probRunDict(prog, tt);
    for (auto& val : result)
    {
        if (val.second > 0)
            std::cout << val.first << ", " << val.second << std::endl;
    }
}

TEST(ArithmeticUnit, test1)
{
    do
    { 
        init();
        //test_bind_data();
        //test_bind_nonnegative_data();
        //test_QAdder();
        //test_QAdderWithCarry();
        //test_QAdd();
        //test_QComplement();
        //test_QSub();
        //test_QMultiplier();
        //test_QMul();
        //test_QDivider();
        //test_QDividerWithAccuracy();
        //test_QDiv();
        test_QDivWithAccuracy();
        finalize();
    } while (getchar() != 'q');
   
}