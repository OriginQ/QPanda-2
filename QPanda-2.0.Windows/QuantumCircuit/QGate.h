/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef _QGATE_H_
#define _QGATE_H_

#include <map>
#include <functional>
#include <vector>
#include <complex>
#define PI 3.14159265358979

#define SINGLE_GATE_TYPE 1
#define CU_TYPE 2
#define ISWAP 3

using namespace std;

typedef vector <complex<double>> QStat;
class QuantumGate
{
protected:
	int qOpNum;
public:
	QuantumGate();
    virtual ~QuantumGate() {};
    virtual double getAlpha() const = 0;
    virtual double getBeta() const = 0;
    virtual double getGamma() const = 0;
    virtual double getDelta() const = 0;
    virtual int getOpNum() const = 0;
    virtual void getMatrix(QStat & matrix) {};
};


typedef QuantumGate* (*CreateGate)(void);
typedef QuantumGate* (*CreateAngleGate)(double);
typedef QuantumGate* (*CreateSingleAndCUGate)(double, double, double, double);
typedef QuantumGate* (*CreateDoubleGate)(QStat &);

class QGateFactory
{
public:
    void registClass(string name, CreateGate method);
    void registClass(string name, CreateAngleGate method);
    void registClass(string name, CreateSingleAndCUGate method);
    void registClass(string name, CreateDoubleGate method);
    QuantumGate * getGateNode(std::string &);
    QuantumGate * getGateNode(std::string &, double angle);
    QuantumGate * getGateNode(std::string &, double alpha, double beta, double gamma, double delta);
    QuantumGate * getGateNode(std::string &, QStat&);

    static QGateFactory * getInstance()
    {
        static QGateFactory  s_Instance;
        return &s_Instance;
    }
private:
private:
    map<string, CreateGate> m_gateMap;
    map<string, CreateAngleGate> m_angleGateMap;
    map<string, CreateSingleAndCUGate> m_singleAndCUGateMap;
    map<string, CreateDoubleGate> m_DoubleGateMap;
    QGateFactory() {};

};

class RegisterAction {
public:
    RegisterAction(string className, CreateGate ptrCreateFn) {
        QGateFactory::getInstance()->registClass(className, ptrCreateFn);
    }
    RegisterAction(string className, CreateAngleGate ptrCreateFn) {
        QGateFactory::getInstance()->registClass(className, ptrCreateFn);
    }
    RegisterAction(string className, CreateSingleAndCUGate ptrCreateFn) {
        QGateFactory::getInstance()->registClass(className, ptrCreateFn);
    }
    RegisterAction(string className, CreateDoubleGate ptrCreateFn) {
        QGateFactory::getInstance()->registClass(className, ptrCreateFn);
    }
};


class QSingleGate : public QuantumGate
{
protected:
    double alpha;
    double beta;
    double gamma;
    double delta;

public:
	QSingleGate(QSingleGate&);
	QSingleGate();
	QSingleGate(double,double,double,double);

    inline double getAlpha()const
    {
        return alpha;
    }
    inline double getBeta() const
    {
        return this->beta;
    }
    inline double getGamma() const
    {
        return this->gamma;
    }
    inline double getDelta() const
    {
        return this->delta;
    }
    inline int getOpNum() const
    {
        return this->qOpNum;
    }
	//QSingleGate(double, double, double);
};

class XGate : public QSingleGate
{
public:
	XGate();
	XGate(double angle);
};


class YGate : public QSingleGate
{
public:
    YGate();
    YGate(double angle);
};
class ZGate : public QSingleGate
{
public:
    ZGate();
    ZGate(double angle);
};
class SGate : public QSingleGate
{
public:
    SGate();
};
class HadamardGate : public QSingleGate
{
public:
    HadamardGate();
};
//double quantum gates,contain CNOT ,CZ gates
class CU : public QuantumGate
{

protected:
    double alpha;
    double beta;
    double gamma;
    double delta;
public:
    CU(CU&);
    CU();
    CU(double, double, double,double);

    inline double getAlpha() const
    {
        return alpha;
    }

    inline double getBeta() const
    {
        return this->beta;
    }
    inline double getGamma() const
    {
        return this->gamma;
    }
    inline double getDelta() const
    {
        return this->delta;
    }

    inline int getOpNum() const
    {
        return this->qOpNum;
    }
    //QSingleGate(double, double, double);
};
class CNOTGate : public CU
{
public:
    CNOTGate();
};
class CZGate : public CU
{
public:
    CZGate();
};


class QDoubleGate : public QuantumGate 
{
public:
    QDoubleGate() {};
    QDoubleGate(const QDoubleGate & oldDouble);
    QDoubleGate (QStat & matrix);
    ~QDoubleGate() {};
    inline int getOpNum() const
    {
        return ISWAP;
    }
    inline double getAlpha()const
    {
        return 0;
    }
    inline double getBeta() const
    {
        return 0;
    }
    inline double getGamma() const
    {
        return 0;
    }
    inline double getDelta() const
    {
        return 0;
    }

    virtual void getMatrix(QStat & matrix) const;

protected:
    QStat m_matrix;
    int qOpNum;

};

class iSwapGate : public QDoubleGate
{
public:
    iSwapGate();
    ~iSwapGate() {};
    void getMatrix(QStat & matrix) const;
private:
    QStat m_matrix;
    int qOpNum;

};

#endif
