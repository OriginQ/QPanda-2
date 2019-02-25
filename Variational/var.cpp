#include "var.h"
#include <map>
#include "QPandaNamespace.h"


USING_QPANDA
using namespace QPanda::Variational;
using QPanda::QTerm;
using namespace std;

/* enum find */


int numOpArgs(op_type op){
    static const std::map<op_type, int> op_args = {
        { op_type::plus, 2 },
        { op_type::minus, 2 },
        { op_type::multiply, 2 },
        { op_type::divide, 2 },
        { op_type::exponent, 1 },
        { op_type::log, 1 },
        { op_type::polynomial, 2 },
        { op_type::dot, 2 },
        { op_type::inverse, 1 },
        { op_type::transpose, 1 },
        { op_type::sum, 1 },
		{ op_type::sigmoid, 1},
		{ op_type::softmax, 1},
		{ op_type::crossEntropy,2},
		{ op_type::dropout, 2},
        { op_type::none, 0 }
    };
    return op_args.find(op)->second;
};

var::var(var&&) = default;
var& var::operator=(var&&) = default;
var::~var() = default;
var::var(const var&) = default;
var& var::operator=(const var&) = default;
var var::clone(){
    return var(std::make_shared<impl>(*pimpl)); 
}

size_t var::getNumOpArgs()
{
    return getChildren().size();
}

var::var(std::shared_ptr<impl> _pimpl) : pimpl(_pimpl){};

var::var(const MatrixXd& _val) 
: pimpl(new impl(_val)) {}

var::var(double _val) 
: pimpl(new impl(scalar(_val))){}

var::var(const MatrixXd& _val, bool isDifferentiable)
    : pimpl(new impl(_val, isDifferentiable)) {}

var::var(double _val, bool isDifferentiable)
    : pimpl(new impl(scalar(_val), isDifferentiable)) {}

var::var(op_type _op, const std::vector<var>& _children)
: pimpl(new impl(_op, _children)){}

MatrixXd var::getValue() const{ return pimpl->val; }

void var::setValue(const MatrixXd& _val){ pimpl->val = _val; }

op_type var::getOp() const{ return pimpl->op; }

void var::setOp(op_type _op){ pimpl->op = _op; }

std::vector<var>& var::getChildren() const{ return pimpl->children; }

std::vector<var> var::getParents() const {
    std::vector<var> _parents;
    for( std::weak_ptr<impl> parent : pimpl->parents ){
        _parents.emplace_back( parent.lock() );
    } 
    return _parents;
}

long var::getUseCount() const{
    return pimpl.use_count();
}

bool QPanda::Variational::var::getValueType() const
{
    return pimpl->m_is_differentiable;
}

MatrixXd var::_eval()
{
    auto op = this->getOp();
    auto operands = this->getChildren();
    switch (op) {
    case op_type::plus: {
        if (_is_scalar(operands[0]))
            return _sval(operands[0]) + _mval(operands[1]).array();
        else if (_is_scalar(operands[1]))
            return _mval(operands[0]).array() + _sval(operands[1]);
        else
            return _mval(operands[0]).array() + _mval(operands[1]).array();
    }
    case op_type::minus: {
        if (_is_scalar(operands[0]))
            return _sval(operands[0]) - _mval(operands[1]).array();
        else if (_is_scalar(operands[1]))
            return _mval(operands[0]).array() - _sval(operands[1]);
        else
            return _mval(operands[0]).array() - _mval(operands[1]).array();
    }
    case op_type::multiply: {
        if (_is_scalar(operands[0]))
            return _sval(operands[0]) * _mval(operands[1]).array();
        else if (_is_scalar(operands[1]))
            return _mval(operands[0]).array() * _sval(operands[1]);
        else
            return _mval(operands[0]).array() * _mval(operands[1]).array();
    }
    case op_type::divide: {
        if (_is_scalar(operands[0]))
            return _sval(operands[0]) / _mval(operands[1]).array();
        else if (_is_scalar(operands[1]))
            return _mval(operands[0]).array() / _sval(operands[1]);
        else
            return _mval(operands[0]).array() / _mval(operands[1]).array();
    }
    case op_type::exponent: {
        return _mval(operands[0]).array().exp();
    }
    case op_type::log: {
        return _mval(operands[0]).array().log();
    }
    case op_type::polynomial: {
        return _mval(operands[0]).array().pow(_sval(operands[1]));
    }
    case op_type::dot: {
        return _mval(operands[0]) * _mval(operands[1]);
    }
    case op_type::inverse: {
        return _mval(operands[0]).inverse();
    }
    case op_type::transpose: {
        return _mval(operands[0]).transpose();
    }
    case op_type::sum: {
        return scalar(_mval(operands[0]).array().sum());
    }
    case op_type::stack: {        
        auto pimpl_stack = dynamic_pointer_cast<impl_stack>(pimpl);
        int axis = pimpl_stack->m_axis;        
        if (axis == 0){
            const auto &children = pimpl_stack->children;
            auto rows = children[0].pimpl->val.rows();
            Eigen::Index cols = 0;
            for (auto &child : children)
            {
                if (child.pimpl->val.rows() != rows)
                {
                    QCERR("Cannot stack, row not match");
                    throw std::invalid_argument("Cannot stack, row not match");
                }
                cols += child.pimpl->val.cols();
            }
            MatrixXd s(rows, cols);
            Eigen::Index current_col = 0;
            for (auto child : children)
            {
                auto child_col = child.getValue().cols();
                s.block(0, current_col, rows, child_col) = child.getValue();
                current_col += child_col;
            }
            return s;
        }
        if (axis == 1)
        {
            const auto &children = pimpl_stack->children;
            auto cols = children[0].pimpl->val.cols();
            Eigen::Index rows = 0;
            for (auto &child : children)
            {
                if (child.pimpl->val.cols() != cols)
                    throw std::invalid_argument("Cannot stack, row not match");
                rows += child.pimpl->val.rows();
            }
            MatrixXd s(rows, cols);
            Eigen::Index current_row = 0;
            for (auto child : children)
            {
                auto child_row = child.getValue().rows();
                s.block(current_row, 0, child_row, cols) = child.getValue();
                current_row += child_row;
            }
            return s;
        }
        else throw std::invalid_argument("Cannot stack, axis should be 0 or 1");
    }
    case op_type::subscript: {
        auto pimpl_subscript = dynamic_pointer_cast<impl_subscript>(pimpl);
        int subscript = pimpl_subscript->m_subscript;
        if (_is_scalar(operands[0])) {
            throw std::invalid_argument("Cannot index a scalar");
        }
        else if (_is_vector(operands[0])) {
            if (operands[0].getValue().rows() == 1)
                return scalar(operands[0].getValue()(0, subscript));
            else if (operands[0].getValue().cols() == 1)
                return scalar(operands[0].getValue()(subscript, 0));
            else throw exception();
        }
        else if (_is_matrix(operands[0])) {
            return operands[0].getValue().row(subscript);
        }
        else throw exception();
    }
    case op_type::vqp:
    {
        auto pimpl_vqp = dynamic_pointer_cast<impl_vqp>(pimpl);
        double expectation = pimpl_vqp->_get_expectation();
        return scalar(expectation);
    }
	case op_type::qop_pmeasure:
	{
		auto pimpl_pmeasure = dynamic_pointer_cast<impl_qop_pmeasure>(pimpl);
		std::vector<double> values = pimpl_pmeasure->_get_value();
		return vector2mat(values);
	}
	case op_type::sigmoid: {
		return 1 / (1 + (-1 * _mval(operands[0]).array()).exp());
	}
	case op_type::softmax: {
		return _mval(operands[0]).array().exp() / (_mval(operands[0]).array().exp()).sum();
	}
	case op_type::crossEntropy: {
		// Cross Entropy: H(p,q) p and q are probability distribution
		MatrixXd Hpq = _mval(operands[1]).array().log();
		return  -Hpq * _mval(operands[0]).transpose();
	}
	case op_type::dropout: {

		MatrixXd input_x = _mval(operands[0]);
		MatrixXd input_p = _mval(operands[1]);
		MatrixXd randomNumber = MatrixXd::Random(1, input_x.size());
		randomNumber = randomNumber.cwiseAbs();

		pimpl->m_prob = MatrixXd::Zero(1, input_x.size());
		for (auto i = 0; i < input_x.size(); i++) {
			if (randomNumber(i) < input_p(i))
				pimpl->m_prob(i) = 1 / input_p(i);
			else
				pimpl->m_prob(i) = 0;
		}
		return input_x.array() * pimpl->m_prob.array();
	}
    case op_type::none: 
        throw std::invalid_argument("Cannot have a non-leaf contain none-op.");    
    };
    throw exception();
}

MatrixXd var::_back_single(const MatrixXd & dx, size_t op_idx)
{
    auto op = getOp();
    auto &operands = getChildren();
    switch (op) {
    case op_type::plus: {
        if (!_is_scalar(operands[op_idx]))
            return dx;
        else
            return scalar(dx.array().sum());
    }
    case op_type::minus: {
        MatrixXd res = _is_scalar(operands[op_idx]) ? scalar(dx.array().sum()) : dx;
        if (op_idx == 0)
            return res;
        else
            return -1 * res.array();
    }
    case op_type::multiply: {
        if (_is_scalar(operands[op_idx]))
            return scalar((dx.array() * _mval(operands[1 - op_idx]).array()).sum());
        else if (_is_scalar(operands[1 - op_idx]))
            return dx.array() * _sval(operands[1 - op_idx]);
        else
            return dx.array() *
            _mval(operands[(1 - op_idx)]).array();
    }
    case op_type::divide: {
        if (op_idx == 0) {
            if (_is_scalar(operands[0])) // the scalar
                return scalar((dx.array() * (1 / _mval(operands[1]).array())).sum());
            else if (_is_scalar(operands[1]))
                return dx.array() * (1 / _sval(operands[1]));
            else
                return dx.array() *
                (1 / _mval(operands[1]).array());
        }
        else {
            if (_is_scalar(operands[1]))
                return scalar((dx.array() *
                (-_mval(operands[0]).array() /
                    std::pow(_sval(operands[1]), 2))).sum());
            else if (_is_scalar(operands[0]))
                return dx.array() * (-_sval(operands[0]) /
                    _mval(operands[1]).array().pow(2));
            else
                return dx.array() *
                (-_mval(operands[0]).array() /
                    _mval(operands[1]).array().pow(2));
        }
    }
    case op_type::exponent: {
        return dx.array() *
            _mval(operands[0]).array().exp();
    }
    case op_type::log: {
        return dx.array() *
            (1 / _mval(operands[0]).array());
    }
    case op_type::polynomial: {
        if (op_idx == 0)
            return dx.array() *
            _mval(operands[0]).array().pow(_sval(operands[1]) - 1) *
            _sval(operands[1]);
        else
            return scalar(0); // we don't support exponents other than e.
    }
    case op_type::dot: {
        if (op_idx == 0)
            return dx * _mval(operands[1]).transpose();
        else
            return _mval(operands[0]).transpose() * dx;
    }
    case op_type::inverse: {
        // (I)' = (AA^{-1})' = A(A^{-1}') + A'(A^{-1})
        // AA^{-1}' = -A'A^{-1}
        // A^{-1}AA^{-1}' = -A^{-1}A'A^{-1}
        // A^{-1}' = -A^{-1}A'A^{-1}
        // This means all of the next few chain rules are _nested_.
        // That makes this gradient way too hard. It doesn't work with our current
        // framework.
        throw std::invalid_argument("The derivative of an inverse is too hard.");
    }
    case op_type::transpose: {
        return dx.transpose();
    }
    case op_type::sum: {
        return _sval(dx) * ones_like(operands[0]).array();
    }
    case op_type::stack: {
        auto pimpl_stack = dynamic_pointer_cast<impl_stack>(pimpl);
        int axis = pimpl_stack->m_axis;
        if (axis == 0)
        {
            Eigen::Index pos = 0;
            for (int i = 0; i < op_idx; ++i) pos += getChildren()[i].getValue().cols();
            auto rows = getChildren()[op_idx].getValue().rows();
            auto columns = getChildren()[op_idx].getValue().cols();

            return dx.block(0, pos, rows, columns);
        }
        else if (axis == 1)
        {
            Eigen::Index pos = 0;
            for (int i = 0; i < op_idx; ++i) pos += getChildren()[i].getValue().rows();
            auto rows = getChildren()[op_idx].getValue().rows();
            auto columns = getChildren()[op_idx].getValue().cols();

            return dx.block(pos, 0, rows, columns);
        }
        else throw std::invalid_argument("axis can only be 0 or 1");
    }
    case op_type::subscript: {
        auto pimpl_subscript = dynamic_pointer_cast<impl_subscript>(pimpl);
        int subscript = pimpl_subscript->m_subscript;
        if (_is_scalar(operands[0])) {
            throw std::invalid_argument("Cannot index a scalar");
        }
        else if (_is_vector(operands[0])) {
            auto s = zeros_like(operands[0]);
            if (s.rows() == 1) {
                s(0, subscript) = dx(0, 0);
            }
            else if (s.cols() == 1) {
                s(subscript, 0) = dx(0, 0);
            }
            else throw exception();
            return s;
        }
        else if (_is_matrix(operands[0])) {
            auto s = zeros_like(operands[0]);
            s.row(subscript) = dx;
            return s;
        }
        else throw exception();        
    }
    case op_type::vqp: {
        auto pimpl_vqp = dynamic_pointer_cast<impl_vqp>(pimpl);
        return scalar(_sval(dx)*pimpl_vqp->_get_gradient(pimpl_vqp->children[op_idx]));
    }
	case op_type::qop_pmeasure: {
		auto pimpl_pmeasure = dynamic_pointer_cast<impl_qop_pmeasure>(pimpl);
		auto deriv = vector2mat(pimpl_pmeasure->
			_get_gradient(pimpl_pmeasure->children[op_idx]));
		return scalar((dx.cwiseProduct(deriv)).sum());
	}
	case op_type::sigmoid: {
		return dx.array() *
			1 / (1 + (-1 * _mval(operands[0]).array()).exp()) *
			(1 - 1 / (1 + (-1 * _mval(operands[0]).array()).exp()));
	}
	case op_type::softmax: {
		if (_is_scalar(dx))
			throw std::invalid_argument("invalid dx");
		auto m = _mval(operands[0]);
		auto size = m.rows() > m.cols() ? m.rows() : m.cols();
		auto softmaxSum = _mval(operands[0]).array().exp().sum();

		MatrixXd Vec2DiagMat = MatrixXd::Zero(size, size);
		for (auto i = 0; i < size; i++)
		{
			if (m.rows() > m.cols()) {
				Vec2DiagMat.row(i)[i] = std::exp(m.row(i)[0]) / softmaxSum;
			}
			else {
			    Vec2DiagMat.row(i)[i] = std::exp(m.row(0)[i]) / softmaxSum;
			}
		}
		MatrixXd operd = _mval(operands[0]).array().exp() / _mval(operands[0]).array().exp().sum();
	
		if (m.cols() > m.rows()) {
			return dx
				*(Vec2DiagMat - operd.transpose() * operd);
		}
		else {
			return dx
				* (Vec2DiagMat - operd * operd.transpose());
		}
	}
	case op_type::crossEntropy: {
		// Cross Entropy: H(p,q) 
		// dHdp = dH/dp; dHdq = dH/dq
		MatrixXd dHdp = _mval(operands[1]).array().log();
		MatrixXd dHdq = _mval(operands[0]).array() / _mval(operands[1]).array();

		if (op_idx == 0)
			return -dx * dHdp;
		else
			return -dx * dHdq;
	}
	case op_type::dropout: {
		return dx.array() * pimpl->m_prob.array();
	}
    case op_type::none: {
        throw std::invalid_argument("Cannot have a non-leaf contain none-op.");
    }
    };
    throw exception();
}

std::vector<MatrixXd> var::_back(const MatrixXd & dx, const std::unordered_set<var>& nonconsts)
{
    auto operands = getChildren();
    std::vector<MatrixXd> derivatives;
    for (size_t i = 0; i < operands.size(); i++) {
        if (nonconsts.find(operands[i]) == nonconsts.end())
            derivatives.push_back(zeros_like(operands[i])); // no gradient flow.
        else
            derivatives.push_back(_back_single(dx, i));
    }
    return derivatives;
}

std::vector<MatrixXd> var::_back(const MatrixXd & dx)
{
    auto operands = getChildren();
    std::vector<MatrixXd> derivatives;
	try {
		for (size_t i = 0; i < operands.size(); i++) {
			derivatives.push_back(_back_single(dx, i));
		}
	}
	catch (exception e)
	{
		cout << e.what();
	}
    return derivatives;
}

/* hash/comparisons */
bool var::operator==(const var& rhs) const{ return pimpl.get() == rhs.pimpl.get(); }

/* et::var::impl funcs: */
impl::impl(const MatrixXd& _val) : 
    val(_val), 
    m_is_differentiable(false),
    op(op_type::none){}

impl::impl(const MatrixXd& _val,bool is_differentiable) :
    val(_val),
    m_is_differentiable(is_differentiable),
    op(op_type::none) {}

impl::impl(op_type _op, const std::vector<var>& _children)
: op(_op) {
    for(const var& v : _children){
        children.emplace_back(v);
    }
    bool is_differentiable = false;
    for (auto iter : _children)
    {
        if (iter.getValueType())
        {
            is_differentiable = true;
        }
    }
    m_is_differentiable = is_differentiable;
}

impl_stack::impl_stack(int axis, const std::vector<var>& vars)
    :impl(op_type::stack, vars), m_axis(axis)
{    
}

impl_subscript::impl_subscript(int subscript, const std::vector<var>& vars)
    :impl(op_type::subscript, vars), m_subscript(subscript)
{     
}

static vector<size_t> is_gate_match(const std::vector<
    std::tuple<weak_ptr<VariationalQuantumGate>, size_t, double>> &gate_offsets,
    weak_ptr<VariationalQuantumGate> the_gate);

VariationalQuantumGate::VariationalQuantumGate(const VariationalQuantumGate &gate)
{
    m_vars.assign(gate.m_vars.begin(), gate.m_vars.end());
}

VariationalQuantumGate_H::VariationalQuantumGate_H(Qubit* q)
{
    m_q = q;
}

VariationalQuantumGate_RX::VariationalQuantumGate_RX(Qubit *q, var _var)
{
    m_q = q;
    m_vars.push_back(_var);
}

VariationalQuantumGate_RX::VariationalQuantumGate_RX(Qubit *q, double _var)
{
    m_q = q;
    m_constants.push_back(_var);
}

QGate VariationalQuantumGate_RX::feed() const
{
    if (m_vars.size() == 1)
        return RX(m_q, _sval(m_vars[0]));
    else if (m_constants.size() == 1)
        return RX(m_q, m_constants[0]);
    else throw exception();
}

QGate VariationalQuantumGate_RX::feed(map<size_t, double> offset) const
{
    if (offset.find(0) == offset.end())
        throw exception();
    return RX(m_q, _sval(m_vars[0]) + offset[0]);
}

VariationalQuantumGate_RY::VariationalQuantumGate_RY(Qubit *q, var _var)
{
    m_q = q;
    m_vars.push_back(_var);
}

VariationalQuantumGate_RY::VariationalQuantumGate_RY(Qubit *q, double _var)
{
    m_q = q;
    m_constants.push_back(_var);
}

QGate VariationalQuantumGate_RY::feed() const
{
    if (m_vars.size() == 1)
        return RY(m_q, _sval(m_vars[0]));
    else if (m_constants.size() == 1)
        return RY(m_q, m_constants[0]);
    else throw exception();
}

QGate VariationalQuantumGate_RY::feed(map<size_t, double> offset) const
{
    if (offset.find(0) == offset.end())
        throw exception();
    return RY(m_q, _sval(m_vars[0]) + offset[0]);
}

VariationalQuantumGate_RZ::VariationalQuantumGate_RZ(Qubit *q, var _var)
{
    m_q = q;
    m_vars.push_back(_var);
}

VariationalQuantumGate_RZ::VariationalQuantumGate_RZ(Qubit *q, double _var)
{
    m_q = q;
    m_constants.push_back(_var);
}

QGate VariationalQuantumGate_RZ::feed() const
{
    if (m_vars.size() == 1)
        return RZ(m_q, _sval(m_vars[0]));
    else if (m_constants.size() == 1)
        return RZ(m_q, m_constants[0]);
    else throw exception();
}

QGate VariationalQuantumGate_RZ::feed(map<size_t, double> offset) const
{
    if (offset.find(0) == offset.end())
        throw exception();
    return RZ(m_q, _sval(m_vars[0]) + offset[0]);
}

VariationalQuantumGate_CZ::VariationalQuantumGate_CZ(Qubit* q1, Qubit* q2)
    :m_q1(q1), m_q2(q2)
{}

VariationalQuantumGate_CNOT::VariationalQuantumGate_CNOT(Qubit* q1, Qubit* q2)
    : m_q1(q1), m_q2(q2)
{}

void VariationalQuantumCircuit::_insert_copied_gate(std::shared_ptr<VariationalQuantumGate> gate)
{
    m_gates.push_back(gate);

    auto vars = gate->get_vars();
    for (auto _var : vars)
    {
        auto var_gate_iter = m_var_in_which_gate.find(_var);
        if (var_gate_iter == m_var_in_which_gate.end())
        {
            m_var_in_which_gate.insert(
                make_pair(_var, vector<weak_ptr<VariationalQuantumGate>>(1, gate))
            );
            m_vars.push_back(_var);
        }
        else
        {
            var_gate_iter->second.push_back(gate);
        }
    }
}

VariationalQuantumCircuit::VariationalQuantumCircuit()
{
}

VariationalQuantumCircuit::VariationalQuantumCircuit(const VariationalQuantumCircuit &circuit)
{
    auto gates = circuit.m_gates;
    for (auto gate : gates)
    {
        auto copy_gate = gate->copy();
        m_gates.push_back(copy_gate);

        auto vars = gate->get_vars();
        for (auto _var : vars)
        {
            auto var_gate_iter = m_var_in_which_gate.find(_var);
            if (var_gate_iter == m_var_in_which_gate.end())
            {
                m_var_in_which_gate.insert(
                    make_pair(_var, vector<weak_ptr<VariationalQuantumGate>>(1, copy_gate))
                );
                m_vars.push_back(_var);
            }
            else
            {
                var_gate_iter->second.push_back(copy_gate);
            }
        }
    }
}

VariationalQuantumCircuit::VariationalQuantumCircuit(QCircuit c)
    :VariationalQuantumCircuit(_cast_qc_vqc(c))
{
}

QCircuit VariationalQuantumCircuit::feed
(const std::vector<std::tuple<weak_ptr<VariationalQuantumGate>, size_t, double>> gate_offsets) const
{
    QCircuit c;
    for (auto & gate : m_gates)
    {
        auto i_offsets = is_gate_match(gate_offsets, gate);
        if (i_offsets.size() == 0)
        {
            c << gate->feed();
        }
        else
        {
            map<size_t, double> gate_offset;
            for (auto i : i_offsets)
            {
                gate_offset.insert(make_pair(
                    std::get<1>(gate_offsets[i]),
                    std::get<2>(gate_offsets[i])
                ));
            }
            c << gate->feed(gate_offset);
        }
    }
    return c;
}

QCircuit VariationalQuantumCircuit::feed() const
{
    vector<tuple<weak_ptr<VariationalQuantumGate>, size_t, double>> empty_offset;
    return this->feed(empty_offset);
}

vector<weak_ptr<VariationalQuantumGate>>
VariationalQuantumCircuit::get_var_in_which_gate(const var& _var) const
{
    auto iter = m_var_in_which_gate.find(_var);
    if (iter == m_var_in_which_gate.end())
        throw(invalid_argument("Cannot find the Variable"));
    return vector<weak_ptr<VariationalQuantumGate>>(
        iter->second.begin(), iter->second.end());
}



static bool _compare_weak_ptr(weak_ptr<VariationalQuantumGate> w1,
    weak_ptr<VariationalQuantumGate> w2)
{
    auto s1 = shared_ptr<VariationalQuantumGate>(w1);
    auto s2 = shared_ptr<VariationalQuantumGate>(w2);
    return s1.get() == s2.get();
}

static vector<size_t> is_gate_match(const std::vector<
    std::tuple<weak_ptr<VariationalQuantumGate>, size_t, double>> &gate_offsets,
    weak_ptr<VariationalQuantumGate> the_gate)
{
    vector<size_t> i_offsets;
    for (size_t i = 0u; i < gate_offsets.size(); ++i)
    {
        auto gate = std::get<0>(gate_offsets[i]);
        if (_compare_weak_ptr(gate, the_gate))
            i_offsets.push_back(i);
    }
    return i_offsets;
}

namespace std{

// Template specialize hash for vars
size_t hash<var>::operator()(const var& v) const{
    return std::hash<std::shared_ptr<impl> >{}(v.pimpl);
}
}

impl_vqp::impl_vqp(VariationalQuantumCircuit circuit, 
    QPanda::PauliOperator op, 
    QuantumMachine* machine,
    std::vector<Qubit*> qubits)
    :
    impl(op_type::vqp, circuit.get_vars()),
    m_circuit(circuit), m_op(op), m_machine(machine)
{
    for (int i = 0; i < qubits.size(); ++i) m_measure_qubits[i] = qubits[i];
}

impl_vqp::impl_vqp(VariationalQuantumCircuit circuit,
    QPanda::PauliOperator op,
    QuantumMachine* machine,
    std::map<size_t, Qubit*> qubits)
    :
    impl(op_type::vqp, circuit.get_vars()),
    m_circuit(circuit), m_op(op), m_machine(machine),
    m_measure_qubits(qubits.begin(),qubits.end())
{
}

double impl_vqp::_get_gradient(var _var)
{

    double grad = 0;

    auto hamiltonian = m_op.data();
    for (auto term : hamiltonian)
    {
        auto coefficient = term.second;
        double coefficient_real = 0;
        if (coefficient.imag() < m_op.error_threshold()
            &&
            coefficient.imag() > - m_op.error_threshold()
            )
        {
            coefficient_real = coefficient.real();
        }
        else
            throw(invalid_argument("Hamiltonian has imagine parts"));

        grad += (coefficient_real*
            _get_gradient_one_term(_var, term.first.first));
    }
    return grad;
}

double impl_vqp::_get_gradient_one_term(var _var, QTerm hamiltonian_term)
{
    auto gates = m_circuit.get_var_in_which_gate(_var);
    double grad = 0;
    for (auto gate : gates)
    {
        int pos = shared_ptr<VariationalQuantumGate>(gate)
            ->var_pos(_var);
        if (pos < 0) throw(invalid_argument("Error VQG"));

        vector<tuple<
            weak_ptr<VariationalQuantumGate>, size_t, double>> plus;
        plus.push_back(make_tuple(gate, pos, PI / 2));

        QCircuit plus_circuit = m_circuit.feed(plus);
        auto plus_expectation = _get_expectation_one_term(
            plus_circuit, hamiltonian_term);

        vector<tuple<
            weak_ptr<VariationalQuantumGate>, size_t, double>> minus;
        minus.push_back(make_tuple(gate, pos, -PI / 2));

        QCircuit minus_circuit = m_circuit.feed(minus);
        auto minus_expectation = _get_expectation_one_term(
            minus_circuit, hamiltonian_term);

        grad += ((plus_expectation - minus_expectation) / 2);
    }
    return grad;
}

static bool _parity_check(size_t number)
{
    bool label = true;
    size_t i = 0;
    while ((number >> i) != 0)
    {
        if ((number >> i) % 2 == 1)
        {
            label = !label;
        }
        ++i;
    }
    return label;
}

double impl_vqp::_get_expectation_one_term(QCircuit c,
    QTerm term)
{
    double expectation = 0;
    auto qprog = CreateEmptyQProg();
    qprog << c;
    vector<Qubit *> vqubit;
    for (auto iter : term)
    {
        vqubit.push_back(m_machine->Allocate_Qubit(iter.first));
        if (iter.second == 'X')
        {
            qprog << H(m_measure_qubits[iter.first]);
        }
        else if (iter.second == 'Y')
        {
            qprog << RX(m_measure_qubits[iter.first], PI / 2);
        }
    }
    m_machine->directlyRun(qprog);

    auto result = m_machine->PMeasure(vqubit,-1);


    for (auto i = 0; i < result.size(); i++)
    {
        if (_parity_check(result[i].first))
            expectation += result[i].second;
        else
            expectation -= result[i].second;
    }
    return expectation;
}

double impl_vqp::_get_expectation()
{
    auto c = m_circuit.feed();
    double expectation = 0;
    auto terms = m_op.data();
    for (auto term : terms)
    {
        auto coefficient = term.second;
        double coefficient_real = 0;
        if (coefficient.imag() < m_op.error_threshold()
            &&
            coefficient.imag() > -m_op.error_threshold()
            )
        {
            coefficient_real = coefficient.real();
        }
        else 
            throw(invalid_argument("Hamiltonian has imagine parts"));

        expectation += (coefficient_real * 
            _get_expectation_one_term(c, term.first.first));
    }
    return expectation;
}

impl_qop_pmeasure::impl_qop_pmeasure(
	VariationalQuantumCircuit circuit,
	std::vector<size_t> components,
	QuantumMachine *machine,
	std::vector<Qubit*> qubits)
	:
	impl(op_type::qop_pmeasure, circuit.get_vars()),
	m_circuit(circuit),
	m_machine(machine),
	m_components(components),
	m_measure_qubits(qubits.begin(), qubits.end())
{
}

std::vector<double> impl_qop_pmeasure::_get_value()
{
	auto c = m_circuit.feed();
	return _get_circuit_value(c);
}

std::vector<double> impl_qop_pmeasure::_get_circuit_value(QCircuit c)
{	
    auto temp = dynamic_cast<IdealMachineInterface *>(m_machine);
    if (nullptr == temp)
    {
        QCERR("m_machine is error");
        throw runtime_error("m_machine is error");
    }
	vector<double> probs =
        temp->probRunList(QProg() << c, m_measure_qubits, -1);
	vector<double> values;
	for (auto component : m_components)
		values.push_back(probs[component]);
	return values;
}

std::vector<double> impl_qop_pmeasure::_get_gradient(var _var)
{
	auto gates = m_circuit.get_var_in_which_gate(_var);
	vector<double> grad;
	grad.resize(m_components.size());
	for (auto gate : gates)
	{
		int pos = shared_ptr<VariationalQuantumGate>(gate)
			->var_pos(_var);
		if (pos < 0) throw(invalid_argument("Error VQG"));

		vector<tuple<
			weak_ptr<VariationalQuantumGate>, size_t, double>> plus;
		plus.push_back(make_tuple(gate, pos, PI / 2));

		QCircuit plus_circuit = m_circuit.feed(plus);
		auto plus_expectation = _get_circuit_value(plus_circuit);

		vector<tuple<
			weak_ptr<VariationalQuantumGate>, size_t, double>> minus;
		minus.push_back(make_tuple(gate, pos, -PI / 2));

		QCircuit minus_circuit = m_circuit.feed(minus);
		auto minus_expectation = _get_circuit_value(minus_circuit);
		for (size_t i = 0; i < m_components.size(); ++i)
			grad[i] += ((plus_expectation[i] - minus_expectation[i]) / 2);
	}
	return grad;
}

template<>
VariationalQuantumCircuit&  VariationalQuantumCircuit::insert<std::shared_ptr<VariationalQuantumGate>>(std::shared_ptr<VariationalQuantumGate> gate)
{
    auto copy_gate = gate->copy();
    _insert_copied_gate(copy_gate);
    return *this;
}

template <>
VariationalQuantumCircuit& VariationalQuantumCircuit::insert<VariationalQuantumCircuit>(VariationalQuantumCircuit circuit)
{
    for (auto gate : circuit.m_gates)
    {
        _insert_copied_gate(gate->copy());
    }
    return *this;
}
template< >
VariationalQuantumCircuit&  VariationalQuantumCircuit::insert<QGate &>(QGate  &gate)
{
    auto copy_gate = _cast_qg_vqg(gate);
    _insert_copied_gate(copy_gate);
    return *this;
}

template< >
VariationalQuantumCircuit&  VariationalQuantumCircuit::insert<QGate >(QGate  gate)
{
    auto copy_gate = _cast_qg_vqg(gate);
    _insert_copied_gate(copy_gate);
    return *this;
}
    
template<>
VariationalQuantumCircuit&  VariationalQuantumCircuit::insert<QCircuit>(QCircuit c)
{
    VariationalQuantumCircuit circuit = _cast_qc_vqc(c);
    this->insert(circuit);
    return *this;
}

shared_ptr<VariationalQuantumGate> VariationalQuantumCircuit::_cast_qg_vqg(QGate gate)
{
    QuantumGate* qgate = gate.getQGate();
    int gate_type = qgate->getGateType();
    QVec op_qubit;
    gate.getQuBitVector(op_qubit);
    QGATE_SPACE::RX* rx;
    QGATE_SPACE::RY* ry;
    QGATE_SPACE::RZ* rz;
    QGATE_SPACE::H* h;
    switch (gate_type)
    {
    case GateType::HADAMARD_GATE:
        h = dynamic_cast<QGATE_SPACE::H*>(qgate);
        return std::make_shared<VariationalQuantumGate_H>(op_qubit[0]);
    case GateType::RX_GATE:
        rx = dynamic_cast<QGATE_SPACE::RX*>(qgate);
        return std::make_shared<VariationalQuantumGate_RX>(op_qubit[0], rx->theta);
    case GateType::RY_GATE:
        ry = dynamic_cast<QGATE_SPACE::RY*>(qgate);
        return std::make_shared<VariationalQuantumGate_RY>(op_qubit[0], ry->theta);
    case GateType::RZ_GATE:
        rz = dynamic_cast<QGATE_SPACE::RZ*>(qgate);
        return std::make_shared<VariationalQuantumGate_RZ>(op_qubit[0], rz->theta);
    case GateType::CNOT_GATE:
        return std::make_shared<VariationalQuantumGate_CNOT>(op_qubit[0], op_qubit[1]);
    case GateType::CZ_GATE:
        return std::make_shared<VariationalQuantumGate_CZ>(op_qubit[0], op_qubit[1]);
    default: 
        throw runtime_error("Unsupported VQG type");
    }
}

shared_ptr<VariationalQuantumGate>  VariationalQuantumCircuit::_cast_aqgn_vqg(AbstractQGateNode* gate)
{
    QuantumGate* qgate = gate->getQGate();
    int gate_type = qgate->getGateType();
    QVec op_qubit;
    gate->getQuBitVector(op_qubit);
    QGATE_SPACE::RX* rx;
    QGATE_SPACE::RY* ry;
    QGATE_SPACE::RZ* rz;
    QGATE_SPACE::H* h;
    switch (gate_type)
    {
    case GateType::HADAMARD_GATE:
        h = dynamic_cast<QGATE_SPACE::H*>(qgate);
        return std::make_shared<VariationalQuantumGate_H>(op_qubit[0]);
    case GateType::RX_GATE:
        rx = dynamic_cast<QGATE_SPACE::RX*>(qgate);
        return std::make_shared<VariationalQuantumGate_RX>(op_qubit[0], rx->theta);
    case GateType::RY_GATE:
        ry = dynamic_cast<QGATE_SPACE::RY*>(qgate);
        return std::make_shared<VariationalQuantumGate_RY>(op_qubit[0], ry->theta);
    case GateType::RZ_GATE:
        rz = dynamic_cast<QGATE_SPACE::RZ*>(qgate);
        return std::make_shared<VariationalQuantumGate_RZ>(op_qubit[0], rz->theta);
    case GateType::CNOT_GATE:
        return std::make_shared<VariationalQuantumGate_CNOT>(op_qubit[0], op_qubit[1]);
    case GateType::CZ_GATE:
        return std::make_shared<VariationalQuantumGate_CZ>(op_qubit[0], op_qubit[1]);
    default:
        throw runtime_error("Unsupported VQG type");
    }
}

VariationalQuantumCircuit  VariationalQuantumCircuit::_cast_qc_vqc(QCircuit q)
{
    VariationalQuantumCircuit new_vqc;
    for (auto iter = q.getFirstNodeIter(); iter != q.getEndNodeIter(); ++iter)
    {
        NodeType node_type = (*(iter))->getNodeType();
        if (node_type == NodeType::CIRCUIT_NODE)
        {
            AbstractQuantumCircuit* qc = dynamic_pointer_cast<AbstractQuantumCircuit>(*iter).get();
            new_vqc.insert(_cast_aqc_vqc(qc));
        }
        else if (node_type == NodeType::GATE_NODE)
        {
            AbstractQGateNode* qg = dynamic_pointer_cast<AbstractQGateNode>(*iter).get();
            new_vqc.insert(_cast_aqgn_vqg(qg));
        }
        else throw runtime_error("Unsupported VQG type");
    }
    return new_vqc;
}

VariationalQuantumCircuit  VariationalQuantumCircuit::_cast_aqc_vqc(AbstractQuantumCircuit* q)
{
    VariationalQuantumCircuit new_vqc;
    for (auto iter = q->getFirstNodeIter(); iter != q->getEndNodeIter(); ++iter)
    {
        NodeType node_type = (*(iter))->getNodeType();
        if (node_type == NodeType::CIRCUIT_NODE)
        {
            AbstractQuantumCircuit* qc = dynamic_pointer_cast<AbstractQuantumCircuit>(*iter).get();
            new_vqc.insert(_cast_aqc_vqc(qc));
        }
        else if (node_type == NodeType::GATE_NODE)
        {
            AbstractQGateNode* qg = dynamic_pointer_cast<AbstractQGateNode>(*iter).get();
            new_vqc.insert(_cast_aqgn_vqg(qg));
        }
        else throw runtime_error("Unsupported VQG type");
    }
    return new_vqc;
}
