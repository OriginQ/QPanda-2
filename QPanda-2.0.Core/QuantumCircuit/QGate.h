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
#include "QGlobalVariable.h"
#include <complex>
#define PI 3.14159265358979

#define SINGLE_GATE_TYPE 1
#define CU_TYPE 2
//#define ISWAP 3

using namespace std;
typedef complex <double> COMPLEX;
typedef vector <complex<double>> QStat;


namespace QGATE_SPACE 
{
    class angleParameter
    {
    public:
        double theta;
        virtual double getParameter() const = 0;
    };
    /*
    class matrixParameter
    {
    public:
    QStat doublegatematrix;
    virtual  void getParameter(QStat&) const = 0;
    };
    */


    class QuantumGate
    {
    protected:
        int qOpNum;
        int gateType;
        QStat gatematrix;
       // double theta;
    public:
        QuantumGate();
        //QuantumGate(QuantumGate*);
        virtual ~QuantumGate() {};
        virtual double getAlpha() const = 0;
        virtual double getBeta() const = 0;
        virtual double getGamma() const = 0;
        virtual double getDelta() const = 0;
        virtual int getOpNum() const = 0;
        virtual void getMatrix(QStat & matrix) const = 0;
        virtual int getGateType()const = 0;
		//virtual double getParameter() const =0;
    };


    typedef QuantumGate* (*CreateGate)(void);
    typedef QuantumGate* (*CreateAngleGate)(double);
    typedef QuantumGate* (*CreateSingleAndCUGate)(double, double, double, double);
    typedef QuantumGate* (*CreateGateByMatrix)(QStat &);

    class QGateFactory
    {
    public:
        void registClass(string name, CreateGate method);
        void registClass(string name, CreateAngleGate method);
        void registClass(string name, CreateSingleAndCUGate method);
        void registClass(string name, CreateGateByMatrix method);
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
        map<string, CreateGateByMatrix> m_DoubleGateMap;
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
        RegisterAction(string className, CreateGateByMatrix ptrCreateFn) {
            QGateFactory::getInstance()->registClass(className, ptrCreateFn);
        }
    };


    class U4 : public QuantumGate
    {
    protected:
        double alpha;
        double beta;
        double gamma;
        double delta;
        inline double argc(COMPLEX num)
        {
            if (num.imag() >= 0)
            {
                return acos(num.real() / sqrt(num.real()*num.real() + num.imag()*num.imag()));
            }
            else
            {
                return -acos(num.real() / sqrt(num.real()*num.real() + num.imag()*num.imag()));
            }

        }

    public:
        U4();
        U4(U4&);
        U4(double, double, double, double);
        U4(QStat & matrix);      //initialize through matrix element 
        inline virtual int getGateType() const
        {
            return GATE_TYPE::U4_GATE;
        }
        inline double getAlpha()const
        {
            return this->alpha;
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
        void getMatrix(QStat & matrix) const;
        //U4(double, double, double);
    };

    class X :public U4
    {
    public:
        X();
        inline int getGateType() const
        {
            return GATE_TYPE::PAULI_X_GATE;
        }
    };
    class Y :public U4
    {
    public:
        Y();
        inline int getGateType() const
        {
            return GATE_TYPE::PAULI_Y_GATE;
        }
    };
    class Z :public U4
    {
    public:
        Z();
        inline int getGateType() const
        {
            return GATE_TYPE::PAULI_Z_GATE;
        }
    };

    class X1 :public U4
    {
    public:
        X1();
        inline int getGateType() const
        {
            return GATE_TYPE::X_HALF_PI;
        }
    };
    class Y1 :public U4
    {
    public:
        Y1();
        inline int getGateType() const
        {
            return GATE_TYPE::Y_HALF_PI;
        }
    };
    class Z1 :public U4
    {
    public:
        Z1();
        inline int getGateType() const
        {
            return GATE_TYPE::Z_HALF_PI;
        }
    };
    class H :public U4
    {
    public:
        H();
        inline int getGateType() const
        {
            return GATE_TYPE::HADAMARD_GATE;
        }
    };
    class T :public U4
    {
    public:
        T();
        inline int getGateType() const
        {
            return GATE_TYPE::T_GATE;
        }
    };
    class S :public U4
    {
    public:
        S();
        inline int getGateType() const
        {
            return GATE_TYPE::S_GATE;
        }
    };

    class RX :public U4, public angleParameter
    {
    public:
        RX(double);
        inline int getGateType() const
        {
            return GATE_TYPE::RX_GATE;
        }
        inline double getParameter() const
        {
            return this->theta;
        }
    };
    class RY :public U4, public angleParameter
    {
    public:
        RY(double);
        inline int getGateType() const
        {
            return GATE_TYPE::RY_GATE;
        }
        inline double getParameter() const
        {
            return this->theta;
        }
    };
    class RZ :public U4, public angleParameter
    {
    public:
        RZ(double);
        inline int getGateType() const
        {
            return GATE_TYPE::RZ_GATE;
        }
        inline double getParameter() const
        {
            return this->theta;
        }
    };
    //U1_GATE=[1 0;0 exp(i*theta)
    class U1 :public U4, public angleParameter
    {
    public:
        U1(double);
        inline int getGateType() const
        {
            return GATE_TYPE::RZ_GATE;
        }
        inline double getParameter() const
        {
            return this->theta;
        }
    };

    //double quantum gate 
    class QDoubleGate : public QuantumGate
    {
    protected:
        //QStat gatematrix;
    public:
        QDoubleGate();
        QDoubleGate(const QDoubleGate & oldDouble);
        QDoubleGate(QStat & matrix);
        ~QDoubleGate() {};

        inline int getGateType() const
        {
            return GATE_TYPE::TWO_QUBIT_GATE;
        }
        inline int getOpNum() const
        {
            return qOpNum;
        }
        inline virtual double getAlpha() const
        {
            return 0;
        }
        inline virtual double getBeta() const
        {
            return 0;
        }
        inline virtual double getGamma() const
        {
            return 0;
        }
        inline virtual double getDelta() const
        {
            return 0;
        }
        void getMatrix(QStat &) const;
        //virtual void getMatrix(QStat & matrix) const;

    protected:
        //QStat m_matrix;
        int qOpNum;

    };



    class CU :public QDoubleGate
    {
    protected:
        double alpha;
        double beta;
        double gamma;
        double delta;
        inline static double argc(COMPLEX num)
        {
            if (num.imag() >= 0)
            {
                return acos(num.real() / sqrt(num.real()*num.real() + num.imag()*num.imag()));
            }
            else
            {
                return -acos(num.real() / sqrt(num.real()*num.real() + num.imag()*num.imag()));
            }

        }
    public:
        CU();
        CU(CU&);
        CU(double, double, double, double);  //init (4,4) matrix 
        CU(QStat& matrix);
        inline virtual int getGateType() const
        {
            return GATE_TYPE::CU_GATE;
        }
        inline double getAlpha() const
        {
            return this->alpha;
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
        //U4(double, double, double);
    };
    //CNOT_GATE
    class CNOT :public CU
    {
    public:
        CNOT();
        inline int getGateType() const
        {
            return GATE_TYPE::CNOT_GATE;
        }
    };

    //control phase gate
    class CPhaseGate :public CU, public angleParameter
    {
    public:
        CPhaseGate() {};
        CPhaseGate(double);
        inline virtual int getGateType() const
        {
            return GATE_TYPE::CPHASE_GATE;
        }
        inline virtual double getParameter() const
        {
            return this->theta;
        }
    };

    class CZ :public CPhaseGate
    {
    public:
        CZ();
        inline int getGateType() const
        {
            return GATE_TYPE::CZ_GATE;
        }
    };

    class ISWAPTheta : public QDoubleGate, public angleParameter
    {
    public:
        ISWAPTheta() {};
        ISWAPTheta(double);
        inline virtual int getGateType() const
        {
            return GATE_TYPE::ISWAP_THETA_GATE;
        }
        inline double getParameter() const
        {
            return this->theta;
        }
    };
    class ISWAP : public ISWAPTheta
    {
    public:
        ISWAP();
        inline int getGateType() const
        {
            return GATE_TYPE::ISWAP_GATE;
        }
    };

    class SQISWAP : public ISWAPTheta
    {
    public:
        SQISWAP();
        inline int getGateType() const
        {
            return GATE_TYPE::SQISWAP_GATE;
        }
    };
}

#endif

