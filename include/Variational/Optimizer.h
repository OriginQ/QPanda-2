#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Variational/var.h"
#include "Variational/expression.h"
#include "Variational/utils.h"

namespace QPanda {
namespace Variational
{

class Optimizer
{
public:
    Optimizer(var lost_function, double learning_rate = 0.01);
    virtual ~Optimizer(){}
    virtual std::unordered_set<var> get_variables() = 0;
    virtual std::unordered_map<var, MatrixXd>
        compute_gradients(std::unordered_set<var> &var_set) = 0;
    virtual double get_loss() = 0;

    /* t is the time step */
    virtual bool run(std::unordered_set<var> &leaves, size_t t = 0) = 0;

protected:
    expression m_cost_function;
    double m_learning_rate;
};

enum class OptimizerMode
{
    MINIMIZE,
    MAXIMIZE,
};

class VanillaGradientDescentOptimizer : public Optimizer
{
public:    
    VanillaGradientDescentOptimizer(
        var lost_function,
        double learning_rate = 0.01,
        double stop_condition = 1.e-6,
        OptimizerMode mode = OptimizerMode::MINIMIZE);
    static std::shared_ptr<Optimizer>
        minimize(var,double,double);

    virtual std::unordered_set<var> get_variables();
    virtual std::unordered_map<var, MatrixXd>
        compute_gradients(std::unordered_set<var> &var_set);
    virtual double get_loss();
    virtual bool run(std::unordered_set<var> &leaves, size_t t = 0);



private:
    double m_stop_condition;
    OptimizerMode m_mode;
};

/*

  The momentum method(Polyak, 1964), which we refer to as classical momentum(CM)
  , is a technique for accelerating gradient descent that accumulates a velocity
  vector in directions of persistent reduction in the objective across
  iterations.

  http://www.cs.toronto.edu/~hinton/absps/momentum.pdf
*/

class MomentumOptimizer : public Optimizer
{
public:
    MomentumOptimizer(var lost,
                      double learning_rate = 0.01,
                      double momentum = 0.9);

    static std::shared_ptr<Optimizer>
        minimize(var &lost,
                 double learning_rate = 0.01,
                 double momentum = 0.9);

    virtual std::unordered_map<var, MatrixXd>
        compute_gradients(std::unordered_set<var> &var_set);
    virtual std::unordered_set<var> get_variables();
    virtual double get_loss();
    virtual bool run(std::unordered_set<var> &leaves, size_t t = 0);

private:
    double m_momentum;
    std::unordered_map<var, MatrixXd> m_momentum_map;
};

/*

  AdaGrad (for adaptive gradient algorithm) is a modified stochastic gradient
  descent with per-parameter learning rate, first published in 2011.Informally,
  this increases the learning rate for more sparse parameters and decreases the
  learning rate for less sparse ones. This strategy often improves convergence
  performance over standard stochastic gradient descent in settings where data
  is sparse and sparse parameters are more informative.

  Examples of such applications include natural language processing and image
  recognition.

*/
class AdaGradOptimizer : public Optimizer
{
public:
    /*
     * initial_accumulator_value: A floating point value.
     *   Starting value for the accumulators, must be positive.
     * epsilon: Small value to avoid zero denominator.
     */
    AdaGradOptimizer(var lost,
                     double learning_rate = 0.01,
                     double initial_accumulator_value = 0.0,
                     double epsilon=1e-10);

    static std::shared_ptr<Optimizer>
        minimize(var &lost,
                 double learning_rate = 0.01,
                 double initial_accumulator_value = 0.0,
                 double epsilon = 1e-10);

    virtual std::unordered_map<var, MatrixXd>
        compute_gradients(std::unordered_set<var> &var_set);
    virtual std::unordered_set<var> get_variables();
    virtual double get_loss();
    virtual bool run(std::unordered_set<var> &leaves, size_t t = 0);

private:
    double m_initial_accumulator_value;
    double m_epsilon;
    std::unordered_map<var, MatrixXd> m_adagrad_map;
};

/*

  RMSProp (for Root Mean Square Propagation) is also a method in which
  the learning rate is adapted for each of the parameters. The idea is
  to divide the learning rate for a weight by a running average of the
  magnitudes of recent gradients for that weight.

  See the [paper]
  (http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).

*/
class RMSPropOptimizer : public Optimizer
{
public:
    /*
     * decay: Discounting factor for the history/coming gradient.
     * epsilon: Small value to avoid zero denominator.
     */
    RMSPropOptimizer(var lost,
                     double learning_rate = 0.001,
                     double decay = 0.9,
                     double epsilon = 1e-10);

    static std::shared_ptr<Optimizer>
        minimize(var &lost,
                 double learning_rate = 0.001,
                 double decay = 0.9,
                 double epsilon = 1e-10);

    virtual std::unordered_map<var, MatrixXd>
        compute_gradients(std::unordered_set<var> &var_set);
    virtual std::unordered_set<var> get_variables();
    virtual double get_loss();
    virtual bool run(std::unordered_set<var> &leaves, size_t t = 0);

private:
    double m_decay;
    double m_epsilon; 
    std::unordered_map<var, MatrixXd> m_rmsprop_map;
};

/*
  Adam(short for Adaptive Moment Estimation) is an update to the RMSProp
  optimizer. In this optimization algorithm, running averages of both the
  gradients and the second moments of the gradients are used.

  See [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
  ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).

*/
class AdamOptimizer : public Optimizer
{
public:
    /*
     * ¦Â1, ¦Â2 ¡Ê [0, 1): Exponential decay rates for the moment estimates
     * epsilon: Small value to avoid zero denominator.
     */
    AdamOptimizer(var lost,
                  double learning_rate = 0.001,
                  double beta1 = 0.9,
                  double beta2 = 0.999,
                  double epsilon = 1e-8);

    static std::shared_ptr<Optimizer>
        minimize(var &lost,
                 double learning_rate = 0.001,
                 double beta1 = 0.9,
                 double beta2 = 0.999,
                 double epsilon = 1e-10);

    virtual std::unordered_map<var, MatrixXd>
        compute_gradients(std::unordered_set<var> &var_set);
    virtual std::unordered_set<var> get_variables();
    virtual double get_loss();
    virtual bool run(std::unordered_set<var> &leaves, size_t t = 0);

private:
    double m_beta1;
    double m_beta2;
    double m_epsilon;
    std::unordered_map<var, MatrixXd> m_first_map;
    std::unordered_map<var, MatrixXd> m_second_map;
};

}
}


#endif // ! OPTIMIZER_H
