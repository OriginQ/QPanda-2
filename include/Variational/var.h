/**
 * @file var.h
 * @author Agony5757 (Agony5757@github.com)
 * @brief Variational quantum-classical hybrid operations.
 * @date 2018-12-18
 *
 * @copyright Copyright Origin Quantum(c) 2018
 *
 */
#ifndef VAR_H
#define VAR_H

#include <iostream>
#include <vector>
#include <memory>
#include <utility>
#include <Eigen/Dense>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <type_traits>
#include "Core/QuantumCircuit/ClassicalProgram.h"
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "QAlg/Components/Operator/PauliOperator.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXd;
USING_QPANDA

namespace QPanda {
    namespace Variational {

        class var;

        /**
         * @brief enum class of operator types
         *
         */
        enum class op_type : int {
            plus,
            minus,
            multiply,
            divide,
            exponent,
            log,
            polynomial,
            dot,
            inverse,
            transpose,
            sum,
            stack,
            subscript,
            qop,
            qop_pmeasure,
            qop_real_chip,
            qop_pmeasure_real_chip,
            sigmoid,
            softmax,
            cross_entropy,
            dropout,
            none
        };

        int numOpArgs(op_type op);
    }
}

namespace std {

    /**
     * @brief hash function to enable unordered_map<var, _Ty>
     * and unordered_set<var>
     */
    template <> struct hash<QPanda::Variational::var> {
        size_t operator()(const QPanda::Variational::var&) const;
    };
}


namespace QPanda {
    namespace Variational {

        /**
         * @brief implementation class for the var. Impl only includes
         * classical operator with fixed number of arguments.
         *
         */
        struct impl {
        public:
            /**
             * @brief Construct from a Eigen matrix
             *
             */
            impl(const MatrixXd&);
            impl(const MatrixXd&, bool isDifferentiable);

            /**
             * @brief Construct from a operator
             *
             */
            impl(op_type, const std::vector<var>&);

            /**
             * @brief Internal value
             *
             */
            MatrixXd val;

            /**
             * @brief Placeholder/Variable
             *
             */
            bool m_is_differentiable;

            /**
             * @brief Operator type
             *
             */
            op_type op;

            /**
             * @brief Childrens. For example, c = a + b.
             * c is a and b's parent, a and b are c's children
             *
             */
            std::vector<var> children;

            /**
             * @brief Parents. For example, c = a + b.
             * c is a and b's parent, a and b are c's children
             *
             */
            std::vector<std::weak_ptr<impl>> parents;

            /**
            * @brief Internal value
            *
            */
            MatrixXd m_prob;

            /**
             * @brief Destroy the impl object
             *
             */
            virtual ~impl() = default;
        };
        /**
         * @brief Implementation class for the stack operation.
         *
         */
        struct impl_stack : public impl {
        public:

            /**
             * @brief Construct a new impl stack object by the axis
             * and children. y = stack(axis=0, [a,b,c,d]). It will
             * try to place a,b,c,d into one matrix with the same columns,
             * if axis==1, the same rows.
             *
             * @param axis the stack axis.
             */
            impl_stack(int axis, const std::vector<var>&);

            /**
             * @brief stack axis, should be either 0 or 1.
             *
             */
            int m_axis;
        };

        /**
         * @brief implementation for the subscript operation.
         */
        struct impl_subscript : public impl {
        public:

            /**
             * @brief Construct a new impl subscript object by child
             * and the subscript. c = a[i], subscript=i, a=children
             * and c=parent
             *
             * @param subscript the subscript.
             */
            impl_subscript(int subscript, const std::vector<var>&);

            /**
             * @brief the subscript
             *
             */
            int m_subscript;
        };

        /**
         * @brief The class denotes the variable
         *
         */
        class var {
        public:

            /**
             * @brief Construct a new var object by the impl object
             *
             */
            var(std::shared_ptr<impl>);

            /**
             * @brief Construct a new var object by a double.
             *
             */
            var(double);

            /**
             * @brief Construct a new var object by a Eigen matrix
             *
             */
            var(const MatrixXd&);

            var(double, bool);
            var(const MatrixXd&, bool);

            /**
             * @brief Construct a new var object by the operator type
             * and children
             *
             * @param op operator type
             * @param children children of the operator. For example,
             * c = a + b. c is a and b's parent, a and b are c's children
             */
            var(op_type op, const std::vector<var>& children);

            /**
             * @brief move constructor of var
             *
             */
            var(var&&);

            /**
             * @brief
             *
             * @return var&
             */
            var& operator=(var&&);
            var(const var&);
            var& operator=(const var&);
            var clone();
            virtual size_t getNumOpArgs();
            MatrixXd getValue() const;
            void setValue(const MatrixXd&);
            op_type getOp() const;
            void setOp(op_type);
            std::vector<var>& getChildren() const;
            std::vector<var> getParents() const;
            long getUseCount() const;
            bool getValueType() const;
            MatrixXd _eval();
            MatrixXd _back_single(const MatrixXd& dx, size_t op_idx);
            std::vector<MatrixXd> _back(const MatrixXd& dx, const std::unordered_set<var>& nonconsts);
            std::vector<MatrixXd> _back(const MatrixXd& dx);
            bool operator==(const var& rhs) const;
            friend struct std::hash<var>;
            template <typename... V>
            friend const var pack_expression(op_type, V&...);
            template <typename... V>
            friend const var pack_expression(op_type op, int axis, V&...);
            inline const var operator[](int subscript) {
                std::vector<std::shared_ptr<impl> > vimpl = { this->pimpl };
                std::vector<var> v;
                for (const std::shared_ptr<impl>& _impl : vimpl) {
                    v.emplace_back(_impl);
                }
                var res(std::make_shared<impl_subscript>(subscript, v));
                for (const std::shared_ptr<impl>& _impl : vimpl) {
                    _impl->parents.push_back(res.pimpl);
                }
                return res;
            }
            friend const var py_stack(int axis, std::vector<var>& args);
            std::shared_ptr<impl> pimpl;

            ~var();
        };

        class VariationalQuantumGate
        {
        public:
        protected:
            std::vector<var> m_vars;
            std::vector<double> m_constants;

        public:
            /**
             * @brief
             *
             * @return size_t the number of vars.
             */
            inline size_t n_var() { return m_vars.size(); }

            /**
             * @brief Get all variables for the VQG.
             *
             * @return std::vector<Variable>
             */
            const std::vector<var>& get_vars() { return m_vars; }

            const std::vector<double>& get_constants() { return m_constants; }

            /**
             * @brief Get the position for var in the m_vars.
             * If not existed, return -1. Otherwise, return the
             * position n, which is var == m_vars[n].
             *
             * @param var The corresponding variable.
             * @return int -1 if not existed, or position.
             */
            inline int var_pos(var _var) {
                for (size_t i = 0u; i < m_vars.size(); ++i)
                {
                    if (m_vars[i] == _var)
                        return (int)i;
                }
                return -1;
            }

            /**
             * @brief Copy Constructor for a new Variational
             *  Quantum Gate object
             *
             */
            VariationalQuantumGate(const VariationalQuantumGate&);

            /**
             * @brief Default Constructor for a new Variational
             *  Quantum Gate object
             *
             */
            VariationalQuantumGate() {}

            /**
             * @brief Interface to instantialize the QGate with
             * VQG
             *
             * @return QGate Instantiation
             */
            virtual QGate feed() const = 0;

            /**
             * @brief Interface to instantialize the QGate with
             * the "offset".
             *
             * @param offset <number of variable, offset>
             *
             * @return QGate
             */
            virtual QGate feed(std::map<size_t, double> offset) const
            {
                return this->feed();
            }

            /**
             * @brief Destroy the Variational Quantum Gate object
             *
             */
            virtual ~VariationalQuantumGate() {}

            /**
             * @brief Interface to copy the instance, and return a
             * shared_ptr for the object.
             *
             * @return std::shared_ptr<VariationalQuantumGate>
             */
            virtual std::shared_ptr<VariationalQuantumGate> copy() = 0;
        };

        class VariationalQuantumGate_H : public VariationalQuantumGate
        {
        private:
            Qubit* m_q;
        public:
            explicit VariationalQuantumGate_H(Qubit* q);
            inline QGate feed() const { return H(m_q); }
            inline std::shared_ptr<VariationalQuantumGate> copy()
            {
                return std::make_shared<VariationalQuantumGate_H>(m_q);
            }
        };

        class VariationalQuantumGate_X : public VariationalQuantumGate
        {
        private:
            Qubit* m_q;
        public:
            explicit VariationalQuantumGate_X(Qubit* q);
            inline QGate feed() const { return X(m_q); }
            inline std::shared_ptr<VariationalQuantumGate> copy()
            {
                return std::make_shared<VariationalQuantumGate_X>(m_q);
            }
        };

        class VariationalQuantumGate_RX : public VariationalQuantumGate
        {
        private:
            Qubit* m_q;
        public:
            explicit VariationalQuantumGate_RX(Qubit*, var);
            explicit VariationalQuantumGate_RX(Qubit*, double angle);
            QGate feed() const;
            QGate feed(std::map<size_t, double>) const;
            inline std::shared_ptr<VariationalQuantumGate> copy()
            {
                if (m_vars.size() == 0)
                    return std::make_shared<VariationalQuantumGate_RX>(m_q, m_constants[0]);
                else
                    return std::make_shared<VariationalQuantumGate_RX>(m_q, m_vars[0]);
            }
        };



        class VariationalQuantumGate_RY : public VariationalQuantumGate
        {
        private:
            Qubit* m_q;
        public:
            explicit VariationalQuantumGate_RY(Qubit*, var);
            explicit VariationalQuantumGate_RY(Qubit*, double angle);
            QGate feed() const;
            QGate feed(std::map<size_t, double>) const;
            inline std::shared_ptr<VariationalQuantumGate> copy()
            {
                if (m_vars.size() == 0)
                    return std::make_shared<VariationalQuantumGate_RY>(m_q, m_constants[0]);
                else
                    return std::make_shared<VariationalQuantumGate_RY>(m_q, m_vars[0]);
            }
        };

        class VariationalQuantumGate_RZ : public VariationalQuantumGate
        {
        private:
            Qubit* m_q;
        public:
            explicit VariationalQuantumGate_RZ(Qubit*, var);
            explicit VariationalQuantumGate_RZ(Qubit*, double angle);
            QGate feed() const;
            QGate feed(std::map<size_t, double>) const;
            inline std::shared_ptr<VariationalQuantumGate> copy()
            {
                if (m_vars.size() == 0)
                    return std::make_shared<VariationalQuantumGate_RZ>(m_q, m_constants[0]);
                else
                    return std::make_shared<VariationalQuantumGate_RZ>(m_q, m_vars[0]);
            }
        };
        class VariationalQuantumGate_CRX : public VariationalQuantumGate
        {
        private:
            Qubit* m_target;
            QVec m_control;
        public:
            explicit VariationalQuantumGate_CRX(Qubit*, QVec &, double angle);
            VariationalQuantumGate_CRX(VariationalQuantumGate_CRX & old);
            inline QGate feed() const;
            inline std::shared_ptr<VariationalQuantumGate> copy()
            {
                if (m_vars.size() == 0)
                    return std::make_shared<VariationalQuantumGate_CRX>(m_target, m_control, m_constants[0]);
            }
        };
        class VariationalQuantumGate_CRY : public VariationalQuantumGate
        {
        private:
            Qubit* m_target;
            QVec m_control;
        public:
            explicit VariationalQuantumGate_CRY(Qubit*, QVec &, double angle);
            VariationalQuantumGate_CRY(VariationalQuantumGate_CRY & old);
            inline QGate feed() const;
            inline std::shared_ptr<VariationalQuantumGate> copy()
            {
                if (m_vars.size() == 0)
                    return std::make_shared<VariationalQuantumGate_CRY>(m_target, m_control, m_constants[0]);


            }
        };
        class VariationalQuantumGate_CRZ : public VariationalQuantumGate
        {
        private:
            Qubit* m_target;
            QVec m_control;
        public:
            explicit VariationalQuantumGate_CRZ(Qubit*, QVec &, double angle);
            VariationalQuantumGate_CRZ(VariationalQuantumGate_CRZ & old);
            inline QGate feed() const;
            inline std::shared_ptr<VariationalQuantumGate> copy()
            {
                if (m_vars.size() == 0)
                    return std::make_shared<VariationalQuantumGate_CRZ>(m_target, m_control, m_constants[0]);
            }
        };
        class VariationalQuantumGate_CZ : public VariationalQuantumGate
        {
        private:
            Qubit* m_q1;
            Qubit* m_q2;
        public:
            explicit VariationalQuantumGate_CZ(Qubit*, Qubit*);
            inline QGate feed() const { return CZ(m_q1, m_q2); }
            inline std::shared_ptr<VariationalQuantumGate> copy()
            {
                return std::make_shared<VariationalQuantumGate_CZ>(m_q1, m_q2);
            }
        };

        class VariationalQuantumGate_CNOT : public VariationalQuantumGate
        {
        private:
            Qubit* m_q1;
            Qubit* m_q2;
        public:
            explicit VariationalQuantumGate_CNOT(Qubit*, Qubit*);
            inline QGate feed() const { return CNOT(m_q1, m_q2); }
            inline std::shared_ptr<VariationalQuantumGate> copy()
            {
                return std::make_shared<VariationalQuantumGate_CNOT>(m_q1, m_q2);
            }
        };

        class VariationalQuantumCircuit
        {
            std::vector<var> m_vars;
            std::vector<std::shared_ptr<VariationalQuantumGate>> m_gates;
            std::unordered_map<var,
                std::vector<std::weak_ptr<VariationalQuantumGate>>> m_var_in_which_gate;

            void _insert_copied_gate(std::shared_ptr<VariationalQuantumGate> gate);
        public:

            VariationalQuantumCircuit();
            VariationalQuantumCircuit(const VariationalQuantumCircuit&);
            VariationalQuantumCircuit(QCircuit);

            inline std::vector<var> &get_vars()
            {
                return m_vars;
            }

            QCircuit feed(const std::vector<
                std::tuple<std::weak_ptr<VariationalQuantumGate>, size_t, double>>) const;

            QCircuit feed() const;

            std::vector<std::weak_ptr<VariationalQuantumGate>> get_var_in_which_gate(const var&) const;

            template<typename VQG_Ty>
            VariationalQuantumCircuit& insert(VQG_Ty gate);
        private:
            static std::shared_ptr<VariationalQuantumGate> _cast_qg_vqg(QGate gate);
            static std::shared_ptr<VariationalQuantumGate> _cast_aqgn_vqg(AbstractQGateNode* gate);
            static VariationalQuantumCircuit _cast_qc_vqc(QCircuit q);
            static VariationalQuantumCircuit _cast_aqc_vqc(AbstractQuantumCircuit* q);

        };

        template<typename VQG_Ty>
        VariationalQuantumCircuit& VariationalQuantumCircuit::insert(VQG_Ty  gate)
        {
            static_assert(std::is_base_of<VariationalQuantumGate, VQG_Ty>::value, "Bad VQG Type");
            auto copy_gate = gate.copy();
            _insert_copied_gate(copy_gate);
            return *this;
        }

        template<>
        VariationalQuantumCircuit& VariationalQuantumCircuit::insert<std::shared_ptr<VariationalQuantumGate>>
            (std::shared_ptr<VariationalQuantumGate> gate);

        template<>
        VariationalQuantumCircuit& VariationalQuantumCircuit::insert<VariationalQuantumCircuit>
            (VariationalQuantumCircuit circuit);

        template<>
        VariationalQuantumCircuit& VariationalQuantumCircuit::insert<QGate&>(QGate &gate);

        template<>
        VariationalQuantumCircuit& VariationalQuantumCircuit::insert<QGate>(QGate gate);

        template<>
        VariationalQuantumCircuit& VariationalQuantumCircuit::insert<QCircuit>(QCircuit c);

        typedef VariationalQuantumGate_H VQG_H;
        typedef VariationalQuantumGate_X VQG_X;
        typedef VariationalQuantumGate_RX VQG_RX;
        typedef VariationalQuantumGate_RY VQG_RY;
        typedef VariationalQuantumGate_RZ VQG_RZ;
        typedef VariationalQuantumGate_CNOT VQG_CNOT;
        typedef VariationalQuantumGate_CZ VQG_CZ;
        typedef VariationalQuantumGate_CRX VQG_CRX;
        typedef VariationalQuantumGate_CRY VQG_CRY;
        typedef VariationalQuantumGate_CRZ VQG_CRZ;


        typedef VariationalQuantumGate VQG;
        typedef VariationalQuantumCircuit VQC;

        struct impl_vqp : public impl {
        public:
            impl_vqp(VariationalQuantumCircuit,
                PauliOperator,
                QuantumMachine*,
                std::vector<Qubit*>);

            impl_vqp(VariationalQuantumCircuit,
                PauliOperator,
                QuantumMachine*,
                std::map<size_t, Qubit*>);

            double _get_gradient(var _var);
            double _get_gradient_one_term(var _var, QTerm);
            double _get_expectation_one_term(QCircuit, QTerm);
            double _get_expectation();

        private:
            std::map<size_t, Qubit*> m_measure_qubits;
            PauliOperator m_op;
            QuantumMachine* m_machine;
            VariationalQuantumCircuit m_circuit;
        };

        struct impl_vqp_real_chip : public impl {
        public:
            impl_vqp_real_chip(VariationalQuantumCircuit,
                PauliOperator,
                QuantumMachine*,
                std::vector<Qubit*>,
                int shots);

            impl_vqp_real_chip(VariationalQuantumCircuit,
                PauliOperator,
                QuantumMachine*,
                std::map<size_t, Qubit*>,
                int shots);

            double _get_gradient(var _var);
            double _get_gradient_one_term(var _var, QTerm);
            double _get_expectation_one_term(QCircuit, QTerm);
            double _get_expectation();

        private:
            int m_shots;
            std::map<size_t, Qubit*> m_measure_qubits;
            PauliOperator m_op;
            QuantumMachine* m_machine;
            VariationalQuantumCircuit m_circuit;
        };

        struct impl_qop_pmeasure : public impl {
        public:
            impl_qop_pmeasure(VariationalQuantumCircuit,
                std::vector<size_t>,
                QuantumMachine*,
                std::vector<Qubit*>);

            std::vector<double> _get_gradient(var _var);
            std::vector<double> _get_value();
            std::vector<double> _get_circuit_value(QCircuit);

        private:
            std::vector<Qubit*> m_measure_qubits;
            std::vector<size_t> m_components;
            QuantumMachine* m_machine;
            VariationalQuantumCircuit m_circuit;
        };

        struct impl_qop_pmeasure_real_chip : public impl {
        public:
            impl_qop_pmeasure_real_chip(VariationalQuantumCircuit,
                std::vector<size_t>,
                QuantumMachine*,
                std::vector<Qubit*>,
                std::vector<ClassicalCondition>,
                size_t shots);

            std::vector<double> _get_gradient(var _var);
            std::vector<double> _get_value();
            std::vector<double> _get_circuit_value(QCircuit);

        private:
            size_t m_shots;
            std::vector<Qubit*> m_measure_qubits;
            std::vector<ClassicalCondition> m_cbits;
            std::vector<size_t> m_components;
            QuantumMachine* m_machine;
            VariationalQuantumCircuit m_circuit;
        };

        // Inline definitions of templated functions:
        template <typename... V>
        const var pack_expression(op_type op, V&... args) {
            std::vector<std::shared_ptr<impl> > vimpl = { args.pimpl... };
            std::vector<var> v;
            for (const std::shared_ptr<impl>& _impl : vimpl) {
                v.emplace_back(_impl);
            }
            var res(op, v);
            for (const std::shared_ptr<impl>& _impl : vimpl) {
                _impl->parents.push_back(res.pimpl);
            }
            return res;
        }

        template <typename... V>
        const var pack_expression(op_type op, int axis, V&... args) {
            std::vector<std::shared_ptr<impl> > vimpl = { args.pimpl... };
            std::vector<var> v;
            for (const std::shared_ptr<impl>& _impl : vimpl) {
                v.emplace_back(_impl);
            }
            var res(std::make_shared<impl_stack>(axis, v));
            for (const std::shared_ptr<impl>& _impl : vimpl) {
                _impl->parents.push_back(res.pimpl);
            }
            return res;
        }

        // We need const-ness in returns here to prevent things like:
        // a + b = c; which is obviously dumb

        inline const var operator+(var lhs, var rhs) {
            return pack_expression(op_type::plus, lhs, rhs);
        }

        inline const var operator-(var lhs, var rhs) {
            return pack_expression(op_type::minus, lhs, rhs);
        }

        inline const var operator*(var lhs, var rhs) {
            return pack_expression(op_type::multiply, lhs, rhs);
        }

        inline const var operator/(var lhs, var rhs) {
            return pack_expression(op_type::divide, lhs, rhs);
        }

        inline const var exp(var v) {
            return pack_expression(op_type::exponent, v);
        }
        inline const var sigmoid(var v) {
            return pack_expression(op_type::sigmoid, v);
        }
        inline const var log(var v) {
            return pack_expression(op_type::log, v);
        }

        inline const var poly(var v, var power) {
            return pack_expression(op_type::polynomial, v, power);
        }

        inline const var dot(var lhs, var rhs) {
            return pack_expression(op_type::dot, lhs, rhs);
        }

        inline const var inverse(var v) {
            return pack_expression(op_type::inverse, v);
        }

        inline const var transpose(var v) {
            return pack_expression(op_type::transpose, v);
        }

        inline const var sum(var v) {
            return pack_expression(op_type::sum, v);
        }

        inline const var softmax(var v) {
            return pack_expression(op_type::softmax, v);
        }

        inline const var crossEntropy(var lhs, var rhs) {
            return pack_expression(op_type::cross_entropy, lhs, rhs);
        }

        inline const var dropout(var lhs, var rhs) {
            return pack_expression(op_type::dropout, lhs, rhs);
        }
        template <typename ...T>
        inline const var stack(int axis, T&... v) {
            return pack_expression(op_type::stack, axis, v...);
        }

        // QOP Functions
        //                            SingleAmp    PartialAmp     FullAmp     RealChip     NoisyFullAmp    Cloud      Impl        UseShots?
        // QOP                     |     N      |     N        |     Y     |     N      |       N        |   N    |    PMeasure |     N     |
        // QOP_PMEASURE            |     Y      |     Y        |     N     |     N      |       N        |   N    |    PMeasure |     N     |
        // QOP_REAL_CHIP           |     N      |     N        |     N     |     Y      |       Y        |   Y    |    Run      |     Y     |
        // QOP_PMEASURE_REAL_CHIP  |     N      |     N        |     N     |     Y      |       Y        |   Y    |    Run      |     Y     |

        inline const var qop(VariationalQuantumCircuit& circuit,
            PauliOperator Hamiltonian,
            QuantumMachine* machine,
            std::vector<Qubit*> measure_qubits)
        {
            auto pimpl = std::make_shared<impl_vqp>(circuit, Hamiltonian, machine, measure_qubits);
            var res(pimpl);
            std::vector<var>& vars = circuit.get_vars();
            for (auto &var : vars)
            {
                var.pimpl->parents.push_back(res.pimpl);
            }
            return res;
        }

        inline const var qop_real_chip(VariationalQuantumCircuit& circuit,
            PauliOperator Hamiltonian,
            QuantumMachine* machine,
            std::vector<Qubit*> measure_qubits,
            int shots)
        {
            auto pimpl = std::make_shared<impl_vqp_real_chip>(circuit, Hamiltonian, machine, measure_qubits, shots);
            var res(pimpl);
            std::vector<var>& vars = circuit.get_vars();
            for (auto &var : vars)
            {
                var.pimpl->parents.push_back(res.pimpl);
            }
            return res;
        }

        inline const var qop(VariationalQuantumCircuit& circuit,
            PauliOperator Hamiltonian,
            QuantumMachine* machine,
            std::map<size_t, Qubit*> measure_qubits)
        {
            auto pimpl = std::make_shared<impl_vqp>(circuit, Hamiltonian, machine, measure_qubits);
            var res(pimpl);
            std::vector<var>& vars = circuit.get_vars();
            for (auto &var : vars)
            {
                var.pimpl->parents.push_back(res.pimpl);
            }
            return res;
        }

        inline const var qop_pmeasure(VariationalQuantumCircuit& circuit,
            std::vector<size_t> components,
            QuantumMachine* machine,
            std::vector<Qubit*> measure_qubits)
        {
            auto pimpl = std::make_shared<impl_qop_pmeasure>(circuit, components, machine, measure_qubits);
            var res(pimpl);
            std::vector<var>& vars = circuit.get_vars();
            for (auto &var : vars)
            {
                var.pimpl->parents.push_back(res.pimpl);
            }
            return res;
        }

        inline const var qop_pmeasure_real_chip(VariationalQuantumCircuit& circuit,
            std::vector<size_t> components,
            QuantumMachine* machine,
            std::vector<Qubit*> measure_qubits,
            std::vector<ClassicalCondition> cbits,
            size_t shots)
        {
            auto pimpl = std::make_shared<impl_qop_pmeasure_real_chip>(circuit, components, machine, measure_qubits, cbits, shots);
            var res(pimpl);
            std::vector<var>& vars = circuit.get_vars();
            for (auto &var : vars)
            {
                var.pimpl->parents.push_back(res.pimpl);
            }
            return res;
        }

        inline bool _is_scalar(const var& v) { return v.getValue().size() == 1; }
        inline bool _is_matrix(const var& v) { return v.getValue().cols() > 1 && v.getValue().rows() > 1; }
        inline bool _is_vector(const var& v) { return (v.getValue().cols() == 1) ^ (v.getValue().rows() == 1); }
        inline double _sval(const var& v) { return v.getValue()(0, 0); }
        inline MatrixXd _mval(const var& v) { return v.getValue(); }

        inline MatrixXd scalar(double num) {
            MatrixXd m(1, 1);
            m(0, 0) = num;
            return m;
        }

        inline MatrixXd vector2mat(std::vector<double> data) {
            MatrixXd m(1, data.size());
            for (size_t i = 0; i < data.size(); i++)
                m(0, i) = data[i];
            return m;
        }

        inline MatrixXd zeros_like(const MatrixXd& like) {
            return MatrixXd::Zero(like.rows(), like.cols());
        }

        inline MatrixXd zeros_like(const var& like) {
            return MatrixXd::Zero(like.getValue().rows(), like.getValue().cols());
        }

        inline MatrixXd ones_like(const MatrixXd& like) {
            return MatrixXd::Ones(like.rows(), like.cols());
        }

        inline MatrixXd ones_like(const var& like) {
            return MatrixXd::Ones(like.getValue().rows(), like.getValue().cols());
        }

    } // namespace Variational

    using complex_var = std::pair<Variational::var, Variational::var>;

    inline complex_var operator + (const complex_var &lhs, const complex_var &rhs)
    {
        return std::make_pair(lhs.first + rhs.first, lhs.second + rhs.second);
    }

    inline complex_var operator * (const complex_var &lhs, const complex_var &rhs)
    {
        return std::make_pair(lhs.first * rhs.first - lhs.second * rhs.second,
            lhs.first * rhs.second + lhs.second * rhs.first);
    }

    inline complex_var operator * (const complex_var &lhs, const double &rhs)
    {
        return std::make_pair(lhs.first * rhs, lhs.second * rhs);
    }

    inline complex_var operator * (const double &lhs, const complex_var &rhs)
    {
        return std::make_pair(lhs * rhs.first, lhs * rhs.second);
    }

} // namespace QPanda

#endif // ! VAR_H
