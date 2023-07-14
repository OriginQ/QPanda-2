
#include "Core/Variational/Optimizer.h"
#include "QPandaNamespace.h"


using namespace QPanda::Variational;

Optimizer::Optimizer(var lost_function, double learning_rate) :
    m_cost_function(lost_function),
    m_learning_rate(learning_rate)
{

}

std::vector<var> Optimizer::get_variables()
{
    return m_cost_function.findVariables();
}

std::unordered_map<var, Eigen::MatrixXd>
    Optimizer::compute_gradients(std::vector<var> &var_set)
{
    std::unordered_map<var, MatrixXd> gradient;
    for (auto iter : var_set)
    {
        gradient[iter] = zeros_like(iter);
    }

    auto leaf_set = m_cost_function.findNonConsts(var_set);
    auto var = m_cost_function.getRoot();

    eval(var, true);
    back(m_cost_function, gradient, leaf_set);

    return gradient;
}

double Optimizer::get_loss()
{
    var value = m_cost_function.getRoot();
    eval(value,1);
    if (_is_scalar(value))
    {
        return _sval(value);
    }
    else
    {
        throw std::invalid_argument("not match");
    }
}

VanillaGradientDescentOptimizer::VanillaGradientDescentOptimizer(
        var cost_function,
        double learning_rate,
        double stop_condition,
        OptimizerMode mode) :
    Optimizer(cost_function, learning_rate),
    m_stop_condition(stop_condition),
    m_mode(mode)
{
}

std::shared_ptr<Optimizer> VanillaGradientDescentOptimizer::minimize(
        var lost_function,
        double learning_rate,
        double stop_condition)
{
    return std::make_shared<VanillaGradientDescentOptimizer>
            (lost_function, learning_rate, stop_condition);
}

std::vector<var> VanillaGradientDescentOptimizer::get_variables()
{
    return Optimizer::get_variables();
}

std::unordered_map<var, MatrixXd>
    VanillaGradientDescentOptimizer::compute_gradients(
        std::vector<var> &var_set)
{
    return Optimizer::compute_gradients(var_set);
}

double VanillaGradientDescentOptimizer::get_loss()
{
    return Optimizer::get_loss();
}

bool VanillaGradientDescentOptimizer::run(
        std::vector<var> &leaves,
        size_t t)
{    
    (void)(t);

    std::unordered_map<var, MatrixXd> gradient;
    for (auto iter : leaves)
    {
        gradient[iter] = zeros_like(iter);
    }

    auto leaf_set = m_cost_function.findNonConsts(leaves);
    auto var = m_cost_function.getRoot();

    eval(var, true);
    back(m_cost_function, gradient, leaf_set);
    for (auto iter : leaves)
    {
        iter.setValue(iter.getValue() - m_learning_rate * gradient[iter]);
    }

    return true;
}

MomentumOptimizer::MomentumOptimizer(
        var lost,
        double learning_rate,
        double momentum):
    Optimizer(lost, learning_rate),
    m_momentum(momentum)
{

}

std::shared_ptr<Optimizer> MomentumOptimizer::minimize(
        var &lost,
        double learning_rate,
        double momentum)
{
    return std::make_shared<MomentumOptimizer>(lost, learning_rate, momentum);
}

std::unordered_map<var, Eigen::MatrixXd>
    MomentumOptimizer::compute_gradients(std::vector<var> &var_set)
{
    return Optimizer::compute_gradients(var_set);
}

std::vector<var> MomentumOptimizer::get_variables()
{
    return Optimizer::get_variables();
}

double MomentumOptimizer::get_loss()
{
    return Optimizer::get_loss();
}

bool MomentumOptimizer::run(std::vector<var> &leaves, size_t t)
{
    (void)(t);

    std::unordered_map<var, MatrixXd> gradient;
    for (auto iter : leaves)
    {
        gradient[iter] = zeros_like(iter);
    }

    auto leaf_set = m_cost_function.findNonConsts(leaves);
    auto var = m_cost_function.getRoot();

    eval(var, true);
    back(m_cost_function, gradient, leaf_set);
    for (auto iter : leaves)
    {
        MatrixXd m;
        auto find_result = m_momentum_map.find(iter);
        if (find_result == m_momentum_map.end())
        {
            m = zeros_like(iter);
        }
        else
        {
            m = std::move(find_result->second);
        }

        m = m_momentum * m.array() + m_learning_rate * gradient[iter].array();

        iter.setValue(iter.getValue() - m);
        m_momentum_map[iter] = m;
    }

    return true;
}

AdaGradOptimizer::AdaGradOptimizer(
        var lost,
        double learning_rate,
        double initial_accumulator_value,
        double epsilon):
    Optimizer(lost, learning_rate),
    m_initial_accumulator_value(initial_accumulator_value),
    m_epsilon(epsilon)
{

}

std::shared_ptr<Optimizer> AdaGradOptimizer::minimize(
        var &lost,
        double learning_rate,
        double initial_accumulator_value,
        double epsilon)
{
    return std::make_shared<AdaGradOptimizer>(
                lost, learning_rate, initial_accumulator_value, epsilon);
}

std::unordered_map<var, Eigen::MatrixXd>
    AdaGradOptimizer::compute_gradients(std::vector<var> &var_set)
{
    return Optimizer::compute_gradients(var_set);
}

std::vector<var> AdaGradOptimizer::get_variables()
{
    return Optimizer::get_variables();
}

double AdaGradOptimizer::get_loss()
{
    return Optimizer::get_loss();
}

bool AdaGradOptimizer::run(std::vector<var> &leaves, size_t t)
{
    (void)(t);

    std::unordered_map<var, MatrixXd> gradient;
    for (auto iter : leaves)
    {
        gradient[iter] = zeros_like(iter);
    }

    auto leaf_set = m_cost_function.findNonConsts(leaves);
    auto var = m_cost_function.getRoot();

    eval(var, true);
    back(m_cost_function, gradient, leaf_set);
    for (auto iter : leaves)
    {
        MatrixXd s;
        auto find_result = m_adagrad_map.find(iter);
        if (find_result == m_adagrad_map.end())
        {
            s = zeros_like(iter).array() + m_initial_accumulator_value;
        }
        else
        {
            s = std::move(find_result->second);
        }

        s = s.array() + gradient[iter].array().pow(2);

        iter.setValue(iter.getValue().array()
                      - m_learning_rate* gradient[iter].array()
                      /(s.array().sqrt() + m_epsilon));
        m_adagrad_map[iter] = s;
    }

    return true;
}

RMSPropOptimizer::RMSPropOptimizer(
        var lost,
        double learning_rate,
        double decay,
        double epsilon):
    Optimizer(lost, learning_rate),
    m_decay(decay),
    m_epsilon(epsilon)
{

}

std::shared_ptr<Optimizer> RMSPropOptimizer::minimize(
        var &lost,
        double learning_rate,
        double decay,
        double epsilon)
{
    return std::make_shared<RMSPropOptimizer>(
                lost, learning_rate, decay, epsilon);
}

std::unordered_map<var, Eigen::MatrixXd>
    RMSPropOptimizer::compute_gradients(std::vector<var> &var_set)
{
    return Optimizer::compute_gradients(var_set);
}

std::vector<var> RMSPropOptimizer::get_variables()
{
    return Optimizer::get_variables();
}

double RMSPropOptimizer::get_loss()
{
    return Optimizer::get_loss();
}

bool RMSPropOptimizer::run(std::vector<var> &leaves, size_t t)
{
    (void)(t);

    std::unordered_map<var, MatrixXd> gradient;
    for (auto iter : leaves)
    {
        gradient[iter] = zeros_like(iter);
    }

    auto leaf_set = m_cost_function.findNonConsts(leaves);
    auto var = m_cost_function.getRoot();

    eval(var, true);
    back(m_cost_function, gradient, leaf_set);
    for (auto iter : leaves)
    {
        MatrixXd s;
        auto find_result = m_rmsprop_map.find(iter);
        if (find_result == m_rmsprop_map.end())
        {
            s = zeros_like(iter).array();
        }
        else
        {
            s = std::move(find_result->second);
        }

        s = m_decay * s.array() + (1 - m_decay) * gradient[iter].array().pow(2);

        iter.setValue(iter.getValue().array()
                      - m_learning_rate * gradient[iter].array()
                      /(s.array().sqrt() + m_epsilon));
        m_rmsprop_map[iter] = s;
    }

    return true;
}

AdamOptimizer::AdamOptimizer(
        var lost,
        double learning_rate,
        double beta1,
        double beta2,
        double epsilon):
    Optimizer(lost, learning_rate),
    m_beta1(beta1),
    m_beta2(beta2),
    m_epsilon(epsilon)
{

}

std::shared_ptr<Optimizer> AdamOptimizer::minimize(
        var &lost,
        double learning_rate,
        double beta1,
        double beta2,
        double epsilon)
{
    return std::make_shared<AdamOptimizer>(
                lost, learning_rate, beta1, beta2, epsilon);
}

std::unordered_map<var, Eigen::MatrixXd>
    AdamOptimizer::compute_gradients(std::vector<var> &var_set)
{
    return Optimizer::compute_gradients(var_set);
}

std::vector<var> AdamOptimizer::get_variables()
{
    return Optimizer::get_variables();
}

double AdamOptimizer::get_loss()
{
    return Optimizer::get_loss();
}

bool AdamOptimizer::run(std::vector<var> &leaves, size_t t)
{
    t = t+1;

    std::unordered_map<var, MatrixXd> gradient;
    for (auto iter : leaves)
    {
        gradient[iter] = zeros_like(iter);
    }

    auto leaf_set = m_cost_function.findNonConsts(leaves);
    auto var = m_cost_function.getRoot();

    eval(var, true);
    back(m_cost_function, gradient, leaf_set);
    for (auto iter : leaves)
    {
        MatrixXd m;
        MatrixXd v;
        auto find_first = m_first_map.find(iter);
        auto find_second = m_second_map.find(iter);
        if (find_first == m_first_map.end())
        {
            m = zeros_like(iter).array();
            v = m;
        }
        else
        {
            m = std::move(find_first->second);
            v = std::move(find_second->second);
        }

        m = m_beta1 * m.array() + (1 - m_beta1) * gradient[iter].array();
        v = m_beta2 * v.array() + (1 - m_beta2) * gradient[iter].array().pow(2);

        auto m_d = m.array()/(1 - std::pow(m_beta1, t));
        auto v_d = v.array()/(1 - std::pow(m_beta2, t));

        iter.setValue(iter.getValue().array()
                      - m_learning_rate * m_d/(v_d.sqrt() + m_epsilon));

        m_first_map[iter] = m;
        m_second_map[iter] = v;
    }

    return true;
}
