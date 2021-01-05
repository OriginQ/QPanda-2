/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QGate.h
Author: Menghan.Dou
Created in 2018-6-30

Classes for QGate

Update@2018-8-30
Update by code specification
*/
/*! \file QuantumGate.h */
#ifndef _QUANTUM_GATE_H
#define _QUANTUM_GATE_H
#include <map>
#include <string>
#include  <string.h>
#include <functional>
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/Utilities/QPandaNamespace.h"

#ifdef __GNUC__
#include <cxxabi.h>
#endif
/**
* @namespace QGATE_SPACE
* @brief QGATE namespace
*/
namespace QGATE_SPACE 
{
    /**
    * @brief Quantum gate angle parameter  basic abstract class
	* @ingroup QuantumCircuit
    */
    class AbstractAngleParameter
    {
    public:
        virtual double getAlpha() const = 0;
        virtual double getBeta() const = 0;
        virtual double getGamma() const = 0;
        virtual double getDelta() const = 0;
    };

	class AbstractSingleAngleParameter
	{
	public:
		virtual double getParameter() const = 0;
	};

    /**
    * @class QuantumGate
    * @brief Quantum gate basic abstract class
	* @ingroup QuantumCircuit
    */
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
        virtual int getOperationNum() const = 0;
        virtual void getMatrix(QStat & matrix) const = 0;
		virtual int getGateType()const 
		{
			return gate_type;
		};
    };

	/**
    * @brief  Quantum Gate Factory
	* @ingroup QuantumCircuit
    */
	template<typename ...Targs>
    class QGateFactory
    {
    public:
		bool registClass(const std::string& type_name, std::function<QuantumGate*(Targs&&... args)> function)
		{
			if (nullptr == function)
			{
				return(false);
			}
			std::string real_type_name = type_name;

			bool reg = m_map_create_function.insert(std::make_pair(real_type_name, function)).second;
			return reg;
		}

		QuantumGate* getGateNode(const std::string& type_name, Targs&&... args)
		{
			auto iter = m_map_create_function.find(type_name);
			if (iter == m_map_create_function.end())
			{
				return(nullptr);
			}
			else
			{
				return(iter->second(std::forward<Targs>(args)...));
			}
		}

        static QGateFactory * getInstance()
        {
			if (nullptr == m_qgate_factory)
			{
				m_qgate_factory = new QGateFactory();
			}
			return m_qgate_factory;

        }
    private:
		std::unordered_map<std::string, std::function<QuantumGate*(Targs&&...)>> m_map_create_function;
		static QGateFactory<Targs...>* m_qgate_factory;
    };

	template<typename ...Targs>
	QGateFactory<Targs...>* QGateFactory<Targs...>::m_qgate_factory = nullptr;

	template<typename T, typename ...Targs>
	class DynamicCreator
	{
	public:
		struct Register
		{
			Register()
			{
				char* initial_name = nullptr;
				std::string type_name;
#ifdef __GNUC__
				initial_name = const_cast<char*>(abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr));
#else
				initial_name = const_cast<char*>(typeid(T).name());
#endif
				if (nullptr != initial_name)
				{
					char *real_pos = strstr(initial_name, "::");
					if (nullptr != real_pos)
					{
						type_name = real_pos + 2;
					}
					else
					{
						type_name = initial_name;
					}
#ifdef __GNUC__
					free((initial_name));
#endif
				}

				QGateFactory<Targs...>::getInstance()->registClass(type_name, CreateObject);
			}
			inline void do_nothing()const { };
		};
		DynamicCreator()
		{
			m_register.do_nothing();
		}
		virtual ~DynamicCreator() { m_register.do_nothing(); };

		static T* CreateObject(Targs&&... args)
		{
			return new T(std::forward<Targs>(args)...);
		}

		static Register m_register;
	};

	template<typename T, typename ...Targs>
	typename DynamicCreator<T, Targs...>::Register DynamicCreator<T, Targs...>::m_register;

    template<typename ...Targs>
    inline QuantumGate* create_quantum_gate(const std::string& type_name, Targs&&... args)
    {
        auto  p = QGateFactory<Targs...>::getInstance()->getGateNode(type_name, std::forward<Targs>(args)...);
        return p;
    }

	class OracularGate : public QuantumGate,
		public DynamicCreator<OracularGate,std::string&>,
		public DynamicCreator<OracularGate, QuantumGate *>
	{
	private:
		std::string oracle_name;
	public:
		OracularGate(std::string name)
			:oracle_name(name)
		{
			gate_type = GateType::ORACLE_GATE;
		}
		OracularGate(QuantumGate * qgate_old);
		std::string get_name() const { return oracle_name; }		
		inline int getOperationNum() const { return -1; }
		inline void getMatrix(QStat & matrix) const {}
	};

    class U4 : public QuantumGate,
        public AbstractAngleParameter,
		public DynamicCreator<U4,double&, double&, double&, double&>,
		public DynamicCreator<U4, QuantumGate*>,
		public DynamicCreator<U4, QStat&>
    {
    protected:
        double alpha;
        double beta;
        double gamma;
        double delta;
    public:
        U4();
        U4(U4&);
        U4(double, double, double, double);
        U4(QStat & matrix);      //initialize through matrix element 
		U4(QuantumGate  *);      
        virtual ~U4() {};

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

	class I :public U4,
		public DynamicCreator<I>,
		public DynamicCreator<I, QuantumGate*>
	{
	public:
		I();
		I(QuantumGate  * gate_old) :U4(gate_old)
		{
			if (gate_old->getGateType() != GateType::I_GATE)
			{
				QCERR("Parameter qgate_old error");
				throw std::invalid_argument("Parameter qgate_old error");
			}
			gate_type = gate_old->getGateType();

		};

	};

    class X :public U4,
		public DynamicCreator<X>,
		public DynamicCreator<X, QuantumGate*>
    {
    public:
        X();
		X(QuantumGate  * gate_old) :U4(gate_old) 
		{
			if (gate_old->getGateType() != GateType::PAULI_X_GATE)
			{
				QCERR("Parameter qgate_old error");
				throw std::invalid_argument("Parameter qgate_old error");
			}
			gate_type = gate_old->getGateType();

		};

    };
    class Y :public U4,
		public DynamicCreator<Y>,
		public DynamicCreator<Y, QuantumGate*>
    {
    public:
		Y(QuantumGate  * gate_old) :U4(gate_old) 
		{
			if (gate_old->getGateType() != GateType::PAULI_Y_GATE)
			{
				QCERR("Parameter qgate_old error");
				throw std::invalid_argument("Parameter qgate_old error");
			}
			gate_type = gate_old->getGateType();
		};
        Y();

    };
    class Z :public U4,
        public DynamicCreator<Z>,
		public DynamicCreator<Z, QuantumGate*>
    {
    public:
		Z(QuantumGate  * gate_old) :U4(gate_old) 
		{
			if (gate_old->getGateType() != GateType::PAULI_Z_GATE)
			{
				QCERR("Parameter qgate_old error");
				throw std::invalid_argument("Parameter qgate_old error");
			}
			gate_type = gate_old->getGateType();
		};
        Z();

    };

    class X1 :public U4,
        public DynamicCreator<X1>,
		public DynamicCreator<X1, QuantumGate*>
    {
    public:
		X1(QuantumGate  * gate_old) :U4(gate_old) 
		{
			if (gate_old->getGateType() != GateType::X_HALF_PI)
			{
				QCERR("Parameter qgate_old error");
				throw std::invalid_argument("Parameter qgate_old error");
			}
			gate_type = gate_old->getGateType();
		};
        X1();

    };
    class Y1 :public U4,
        public DynamicCreator<Y1>,
		public DynamicCreator<Y1, QuantumGate*>
    {
    public:
		Y1(QuantumGate  * gate_old) :U4(gate_old) 
		{
			if (gate_old->getGateType() != GateType::Y_HALF_PI)
			{
				QCERR("Parameter qgate_old error");
				throw std::invalid_argument("Parameter qgate_old error");
			}
			gate_type = gate_old->getGateType();
		};
        Y1();

    };
    class Z1 :public U4,
        public DynamicCreator<Z1>,
		public DynamicCreator<Z1, QuantumGate*>
    {
    public:
		Z1(QuantumGate  * gate_old) :U4(gate_old) 
		{
			if (gate_old->getGateType() != GateType::Z_HALF_PI)
			{
				QCERR("Parameter qgate_old error");
				throw std::invalid_argument("Parameter qgate_old error");
			}
			gate_type = gate_old->getGateType();
		};
        Z1();

    };
    class H :public U4,
        public DynamicCreator<H>,
		public DynamicCreator<H, QuantumGate*>
    {
    public:
		H(QuantumGate  * gate_old) :U4(gate_old) 
		{
			if (gate_old->getGateType() != GateType::HADAMARD_GATE)
			{
				QCERR("Parameter qgate_old error");
				throw std::invalid_argument("Parameter qgate_old error");
			}
			gate_type = gate_old->getGateType();
		};
        H();

    };

    #undef ECHO
	class ECHO :public U4,
		public DynamicCreator<ECHO>,
		public DynamicCreator<ECHO, QuantumGate*>
	{
	public:
		ECHO(QuantumGate  * gate_old) :U4(gate_old)
		{
			if (gate_old->getGateType() != GateType::ECHO_GATE)
			{
				QCERR("Parameter qgate_old error");
				throw std::invalid_argument("Parameter qgate_old error");
			}
			gate_type = gate_old->getGateType();
		};
		ECHO();

	};

    class BARRIER :public U4,
        public DynamicCreator<BARRIER>,
        public DynamicCreator<BARRIER, QuantumGate*>
    {
    public:
        BARRIER(QuantumGate  * gate_old) :U4(gate_old)
        {
            if (gate_old->getGateType() != GateType::BARRIER_GATE)
            {
                QCERR("Parameter qgate_old error");
                throw std::invalid_argument("Parameter qgate_old error");
            }
            gate_type = gate_old->getGateType();
        };
        BARRIER();

    };

    class T :public U4,
        public DynamicCreator<T>,
		public DynamicCreator<T, QuantumGate*>
    {
    public:
		T(QuantumGate  * gate_old) :U4(gate_old) 
		{
			if (gate_old->getGateType() != GateType::T_GATE)
			{
				QCERR("Parameter qgate_old error");
				throw std::invalid_argument("Parameter qgate_old error");
			}
			gate_type = gate_old->getGateType();
		};
        T();

    };
    class S :public U4, 
        public DynamicCreator<S>,
		public DynamicCreator<S, QuantumGate*>
    {
    public:
		S(QuantumGate  * gate_old) :U4(gate_old) 
		{
			if (gate_old->getGateType() != GateType::S_GATE)
			{
				QCERR("Parameter qgate_old error");
				throw std::invalid_argument("Parameter qgate_old error");
			}
			gate_type = gate_old->getGateType();
		};
        S();
    };

    class RX :public U4,
		public AbstractSingleAngleParameter,
		public DynamicCreator<RX,double&> ,
		public DynamicCreator<RX, QuantumGate*>
    {
    public:
		RX(QuantumGate  * gate_old) :U4(gate_old) 
		{
			if (gate_old->getGateType() != GateType::RX_GATE)
			{
				QCERR("Parameter qgate_old error");
				throw std::invalid_argument("Parameter qgate_old error");
			}
			gate_type = gate_old->getGateType();
		};
        RX(double);
		inline double getParameter() const
		{
			return this->gamma;
		}
    };
    class RY :public U4,
		public AbstractSingleAngleParameter,
		public DynamicCreator<RY,double&>,
		public DynamicCreator<RY, QuantumGate*>
    {
    public:
		RY(QuantumGate  * gate_old) :U4(gate_old) 
		{
			if (gate_old->getGateType() != GateType::RY_GATE)
			{
				QCERR("Parameter qgate_old error");
				throw std::invalid_argument("Parameter qgate_old error");
			}
			gate_type = gate_old->getGateType();
		};
        RY(double);
        inline double getParameter() const
        {
            return this->gamma;
        }
    };
    class RZ :public U4,
		public AbstractSingleAngleParameter,
		public DynamicCreator<RZ,double&>, 
		public DynamicCreator<RZ, QuantumGate*>
    {
    public:
		RZ(QuantumGate  * gate_old) :U4(gate_old) 
		{
			if (gate_old->getGateType() != GateType::RZ_GATE)
			{
				QCERR("Parameter qgate_old error");
				throw std::invalid_argument("Parameter qgate_old error");
			}
			gate_type = gate_old->getGateType();
		};
        RZ(double);
        inline double getParameter() const
        {
            return this->beta;
        }
    };

	class RPhi :public U4,
		public DynamicCreator<RPhi, double&, double&>,
		public DynamicCreator<RPhi, QuantumGate*>
	{
	public:
		RPhi(QuantumGate  * gate_old) :U4(gate_old)
		{
			if (gate_old->getGateType() != GateType::RPHI_GATE)
			{
				QCERR("Parameter qgate_old error");
				throw std::invalid_argument("Parameter qgate_old error");
			}
			gate_type = gate_old->getGateType();
			m_phi = dynamic_cast<QGATE_SPACE::RPhi*>(gate_old)->m_phi;
		};
		RPhi(double, double);
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
		inline double get_phi() const
		{
			return this->m_phi;
		}
		inline double get_theta() const
		{
			return this->beta;
		}
	private:
		double m_phi;
	};

    class U1 :public U4,
        public AbstractSingleAngleParameter,
		public DynamicCreator<U1,double&>,
		public DynamicCreator<U1, QuantumGate*>
    {
    public:
		U1(QuantumGate  * gate_old) :U4(gate_old) 
		{
			if (gate_old->getGateType() != GateType::U1_GATE)
			{
				QCERR("Parameter qgate_old error");
				throw std::invalid_argument("Parameter qgate_old error");
			}
			gate_type = gate_old->getGateType();
		};
        U1(double);
        inline double getParameter() const
        {
            return this->beta;
        }
    };

    class U2 :public U4,
        public DynamicCreator<U2, double&, double&>,
        public DynamicCreator<U2, QuantumGate*>
    {
    public:
        U2(QuantumGate  * gate_old) :U4(gate_old)
        {
            if (gate_old->getGateType() != GateType::U2_GATE)
            {
                QCERR("Parameter qgate_old error");
                throw std::invalid_argument("Parameter qgate_old error");
            }
            gate_type = gate_old->getGateType();
			m_phi = dynamic_cast<QGATE_SPACE::U2*>(gate_old)->m_phi;
			m_lambda = dynamic_cast<QGATE_SPACE::U2*>(gate_old)->m_lambda;
        };
        U2(double, double);

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

        inline double get_phi() const
        {
            return this->m_phi;
        }

        inline double get_lambda() const
        {
            return this->m_lambda;
        }
    private:
        double m_phi;
        double m_lambda;
    };

    class U3 :public U4,
        public DynamicCreator<U3, double&, double&, double&>,
        public DynamicCreator<U3, QuantumGate*>
    {
    public:
        U3(QuantumGate  * gate_old) :U4(gate_old)
        {
            if (gate_old->getGateType() != GateType::U3_GATE)
            {
                QCERR("Parameter qgate_old error");
                throw std::invalid_argument("Parameter qgate_old error");
            }
            gate_type = gate_old->getGateType();
			m_theta = dynamic_cast<QGATE_SPACE::U3*>(gate_old)->m_theta;
			m_phi = dynamic_cast<QGATE_SPACE::U3*>(gate_old)->m_phi;
			m_lambda = dynamic_cast<QGATE_SPACE::U3*>(gate_old)->m_lambda;
        };
        U3(double, double, double);

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

        inline double get_theta() const
        {
            return this->m_theta;
        }

        inline double get_phi() const
        {
            return this->m_phi;
        }

        inline double get_lambda() const
        {
            return this->m_lambda;
        }

    private:
        double m_theta;
        double m_phi;
        double m_lambda;
    };

    //double quantum gate 
    class QDoubleGate : 
		public QuantumGate,
		public DynamicCreator<QDoubleGate, QuantumGate*>,
		public DynamicCreator<QDoubleGate, QStat&>
    {
    protected:

    public:
		QDoubleGate(QuantumGate  * gate_old);
        QDoubleGate();
        QDoubleGate(const QDoubleGate & oldDouble);
        QDoubleGate(QStat & matrix);
        virtual ~QDoubleGate() {};

        inline int getOperationNum() const
        {
            return operation_num;
        }
        void getMatrix(QStat &) const;
    };



    class CU :public QDoubleGate,
        public AbstractAngleParameter,
		public DynamicCreator<CU,double&, double&, double&, double&>, 
		public DynamicCreator<CU, QStat&>,
		public DynamicCreator<CU, QuantumGate*>
    {
    protected:
        double alpha;
        double beta;
        double gamma;
        double delta;
    public:
		CU(QuantumGate  * gate_old);
        CU();
        CU(const CU&);
        CU(double, double, double, double);  //init (4,4) matrix 
        CU(QStat& matrix);
        virtual ~CU() {}
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
    class CNOT :public CU,
		public DynamicCreator<CNOT>,
		public DynamicCreator<CNOT, QuantumGate*>
    {
    public:
		CNOT(QuantumGate  * gate_old) :CU(gate_old)
		{
			if (gate_old->getGateType() != GateType::CNOT_GATE)
			{
				QCERR("Parameter qgate_old error");
				throw std::invalid_argument("Parameter qgate_old error");
			}
			gate_type = gate_old->getGateType();
		}
        CNOT();
        CNOT(const CNOT &);
    };

    //control phase gate
    class CPHASE :public CU,
		public AbstractSingleAngleParameter,
		public DynamicCreator<CPHASE,double&>,
		public DynamicCreator<CPHASE, QuantumGate*>
    {
    protected:
        CPHASE(){};
    public:
		CPHASE(QuantumGate  * gate_old) :CU(gate_old)
		{
			if (gate_old->getGateType() != GateType::CPHASE_GATE)
			{
				QCERR("Parameter qgate_old error");
				throw std::invalid_argument("Parameter qgate_old error");
			}
			gate_type = gate_old->getGateType();
		}
        CPHASE(double);
        inline virtual double getParameter() const
        {
            return this->beta;
        }
    };

    class CZ :public CU,
		public DynamicCreator<CZ>,
		public DynamicCreator<CZ, QuantumGate*>
    {
    public:
		CZ(QuantumGate  * gate_old) :CU(gate_old)
		{
			if (gate_old->getGateType() != GateType::CZ_GATE)
			{
				QCERR("Parameter qgate_old error");
				throw std::invalid_argument("Parameter qgate_old error");
			}
			gate_type = gate_old->getGateType();
		}

        CZ();
    };

    class ISWAPTheta : public QDoubleGate, 
		public AbstractSingleAngleParameter,
		public DynamicCreator<ISWAPTheta,double&>,
		public DynamicCreator<ISWAPTheta, QuantumGate*>
    {
    protected:
		ISWAPTheta() {};
        double theta;
    public:
		ISWAPTheta(QuantumGate  * gate_old);
        ISWAPTheta(double);
        inline double getParameter() const
        {
            return this->theta;
        }
    };
    class ISWAP : public QDoubleGate,
		public DynamicCreator<ISWAP>,
		public DynamicCreator<ISWAP, QuantumGate *>
    {
    public:
		ISWAP(QuantumGate  * gate_old) :QDoubleGate(gate_old)
		{
			if (gate_old->getGateType() != GateType::ISWAP_GATE)
			{
				QCERR("Parameter qgate_old error");
				throw std::invalid_argument("Parameter qgate_old error");
			}
			gate_type = gate_old->getGateType();
		}
        ISWAP();
    };

	class SQISWAP : public QDoubleGate,
		public DynamicCreator<SQISWAP>,
		public DynamicCreator<SQISWAP, QuantumGate *>
	{
	public:
		SQISWAP(QuantumGate  * gate_old) : QDoubleGate(gate_old)
		{
			if (gate_old->getGateType() != GateType::SQISWAP_GATE)
			{
				QCERR("Parameter qgate_old error");
				throw std::invalid_argument("Parameter qgate_old error");
			}

			gate_type = gate_old->getGateType();
			theta = PI / 4;
		}
		SQISWAP();
		double theta;
    };

    class SWAP : public QDoubleGate,
		public DynamicCreator<SWAP>,
		public DynamicCreator<SWAP, QuantumGate *>
    {
    public:
		SWAP(QuantumGate  * gate_old) :QDoubleGate(gate_old)
		{
			if (gate_old->getGateType() != GateType::SWAP_GATE)
			{
				QCERR("Parameter qgate_old error");
				throw std::invalid_argument("Parameter qgate_old error");
			}
			gate_type = gate_old->getGateType();
		}
        SWAP();
        inline int getGateType() const
        {
            return GateType::SWAP_GATE;
        }
    };
}

#endif
