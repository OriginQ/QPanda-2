/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QGate.h
Author: Menghan.Dou
Created in 2018-6-30

Classes for QGate

Update@2018-8-30
Update by code specification
*/

#ifndef _QGATE_H
#define _QGATE_H
#include <map>
#include "QGlobalVariable.h"

namespace QGATE_SPACE 
{
    class angleParameter
    {
    public:
        double theta;
        virtual double getParameter() const = 0;
    };

    class QuantumGate
    {
    protected:
        int operation_num;
        int gate_type;
        size_t time = 0;
        QStat gate_matrix;
    public:
        QuantumGate();

        virtual ~QuantumGate() {};
        virtual double getAlpha() const = 0;
        virtual double getBeta() const = 0;
        virtual double getGamma() const = 0;
        virtual double getDelta() const = 0;
        virtual int getOperationNum() const = 0;
        virtual void getMatrix(QStat & matrix) const = 0;
        virtual int getGateType()const = 0;
    };




    typedef QuantumGate* (*CreateGate_cb)(void);
    typedef QuantumGate* (*CreateAngleGate_cb)(double);
    typedef QuantumGate* (*CreateSingleAndCUGate_cb)(double, double, double, double);
    typedef QuantumGate* (*CreateGateByMatrix_cb)(QStat &);

    class QGateFactory
    {
    public:
        void registClass(std::string name, CreateGate_cb method);
        void registClass(std::string name, CreateAngleGate_cb method);
        void registClass(std::string name, CreateSingleAndCUGate_cb method);
        void registClass(std::string name, CreateGateByMatrix_cb method);
        QuantumGate * getGateNode(const std::string &);
        QuantumGate * getGateNode(const std::string &, double angle);
        QuantumGate * getGateNode(const std::string &,
                                  const double alpha,
                                  const double beta,
                                  const double gamma,
                                  const double delta);
        QuantumGate * getGateNode(const std::string &, QStat&);

        static QGateFactory * getInstance()
        {
            static QGateFactory  instance;
            return &instance;
        }
    private:
    private:
        std::map<std::string, CreateGate_cb> m_gate_map;
        std::map<std::string, CreateAngleGate_cb> m_angle_gate_map;
        std::map<std::string, CreateSingleAndCUGate_cb> m_single_and_cu_gate_map;
        std::map<std::string, CreateGateByMatrix_cb> m_double_gate_map;
        QGateFactory() {};

    };

    class RegisterAction {
    public:
        RegisterAction(std::string class_name, CreateGate_cb create_qgate_callback) {
            QGateFactory::getInstance()->registClass(class_name, create_qgate_callback);
        }
        RegisterAction(std::string class_name, CreateAngleGate_cb create_qgate_callback) {
            QGateFactory::getInstance()->registClass(class_name, create_qgate_callback);
        }
        RegisterAction(std::string class_name, CreateSingleAndCUGate_cb create_qgate_callback) {
            QGateFactory::getInstance()->registClass(class_name, create_qgate_callback);
        }
        RegisterAction(std::string class_name, CreateGateByMatrix_cb create_qgate_callback) {
            QGateFactory::getInstance()->registClass(class_name, create_qgate_callback);
        }
    };


    class U4 : public QuantumGate
    {
    protected:
        double alpha;
        double beta;
        double gamma;
        double delta;
        inline double argc(qcomplex_t num)
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
            return GateType::U4_GATE;
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
        inline int getOperationNum() const
        {
            return this->operation_num;
        }
        void getMatrix(QStat & matrix) const;
    };

    class X :public U4
    {
    public:
        X();
        inline int getGateType() const
        {
            return GateType::PAULI_X_GATE;
        }
    };
    class Y :public U4
    {
    public:
        Y();
        inline int getGateType() const
        {
            return GateType::PAULI_Y_GATE;
        }
    };
    class Z :public U4
    {
    public:
        Z();
        inline int getGateType() const
        {
            return GateType::PAULI_Z_GATE;
        }
    };

    class X1 :public U4
    {
    public:
        X1();
        inline int getGateType() const
        {
            return GateType::X_HALF_PI;
        }
    };
    class Y1 :public U4
    {
    public:
        Y1();
        inline int getGateType() const
        {
            return GateType::Y_HALF_PI;
        }
    };
    class Z1 :public U4
    {
    public:
        Z1();
        inline int getGateType() const
        {
            return GateType::Z_HALF_PI;
        }
    };
    class H :public U4
    {
    public:
        H();
        inline int getGateType() const
        {
            return GateType::HADAMARD_GATE;
        }
    };
    class T :public U4
    {
    public:
        T();
        inline int getGateType() const
        {
            return GateType::T_GATE;
        }
    };
    class S :public U4
    {
    public:
        S();
        inline int getGateType() const
        {
            return GateType::S_GATE;
        }
    };

    class RX :public U4, public angleParameter
    {
    public:
        RX(double);
        inline int getGateType() const
        {
            return GateType::RX_GATE;
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
            return GateType::RY_GATE;
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
            return GateType::RZ_GATE;
        }
        inline double getParameter() const
        {
            return this->theta;
        }
    };

    class U1 :public U4, public angleParameter
    {
    public:
        U1(double);
        inline int getGateType() const
        {
            return GateType::U1_GATE;
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

    public:
        QDoubleGate();
        QDoubleGate(const QDoubleGate & oldDouble);
        QDoubleGate(QStat & matrix);
        ~QDoubleGate() {};

        inline int getGateType() const
        {
            return GateType::TWO_QUBIT_GATE;
        }
        inline int getOperationNum() const
        {
            return operation_num;
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


    protected:

        int operation_num;

    };



    class CU :public QDoubleGate
    {
    protected:
        double alpha;
        double beta;
        double gamma;
        double delta;
        inline static double argc(qcomplex_t num)
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
            return GateType::CU_GATE;
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
        inline int getOperationNum() const
        {
            return this->operation_num;
        }

    };
    //CNOT_GATE
    class CNOT :public CU
    {
    public:
        CNOT();
        inline int getGateType() const
        {
            return GateType::CNOT_GATE;
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
            return GateType::CPHASE_GATE;
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
            return GateType::CZ_GATE;
        }
    };

    class ISWAPTheta : public QDoubleGate, public angleParameter
    {
    public:
        ISWAPTheta() {};
        ISWAPTheta(double);
        inline virtual int getGateType() const
        {
            return GateType::ISWAP_THETA_GATE;
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
            return GateType::ISWAP_GATE;
        }
    };

    class SQISWAP : public ISWAPTheta
    {
    public:
        SQISWAP();
        inline int getGateType() const
        {
            return GateType::SQISWAP_GATE;
        }
    };
}

#endif
