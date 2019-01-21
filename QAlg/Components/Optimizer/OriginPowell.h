/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

OriginPowell.h

Author: LiYe
Created in 2018-09-26


*/

#ifndef ORIGINPOWELL_H
#define ORIGINPOWELL_H

#include "Eigen/Dense"
#include "AbstractOptimizer.h"

namespace QPanda
{

    /*

    Minimization of scalar function of one or more variables using the
    Powell algorithm.

    */
    class Brent;
    class OriginPowell : public AbstractOptimizer
    {
    public:
        OriginPowell();
        OriginPowell(const OriginPowell &) = delete;
        OriginPowell& operator = (const OriginPowell &) = delete;

        virtual void exec();
        virtual QOptimizationResult getResult();
    private:
        bool init();
        void adaptTerminationPara();

        QResultPair callFunc(const Eigen::VectorXd &para);

        /*

        Line-search algorithm using fminbound.
        Find the minimium of the function func(x0+ alpha*direc).
        
        */
        QResultPair linesearch(
            Eigen::VectorXd &x0, 
            Eigen::VectorXd &direc);

        void dispResult();
        void writeToFile();
    private:
        double m_nonzdelt;
        double m_zdelt;

        size_t m_fcalls;
        size_t m_iter;
        size_t m_n;

        QResultPair m_fval;
        Eigen::VectorXd m_x;
        Eigen::MatrixXd m_direc;
    };

    /*
    
    Given a function of one-variable, return the local minimum 
    of the function isolated to a fractional precision of tol.

    Notes:
        Uses inverse parabolic interpolation when possible to speed up
        convergence of golden section method.
    */
    class Brent
    {
        using Func = std::function<QResultPair(double)>;
        using Vec3Pair = std::vector<std::pair<double, QResultPair>>;

    public:
        Brent(const Func &func, double tol = 1.48e-8, size_t maxiter = 500);
        Brent(const Brent &) = delete;
        Brent& operator = (const Brent &) = delete;

        void optimize();

        std::pair<double, QResultPair> getResult();
    private:
        /*
        Bracket the minimum of the function.

        Given a function and distinct initial points, search in the
        downhill direction (as defined by the initital points) and return
        new points xa, xb, xc that bracket the minimum of the function
        f(xa) > f(xb) < f(xc). It doesn't always mean that obtained
        solution will satisfy xa<=x<=xb

        param:
            func: callable f(x)
                Objective function to minimize.
            xa, xb: float, optional
                Bracketing interval. Defaults `xa` to 0.0, and `xb` to 1.0.
            grow_limit: float, optional
                Maximum grow limit.  Defaults to 110.0
            maxiter: int, optional
                Maximum number of iterations to perform. Defaults to 1000.
        return:
            Vec3Pair
            xa, xb, xc : [double] Bracket.
            fa, fb, fc : [QResultPair] Objective function values in bracket.
        */
        Vec3Pair bracket(
            const Func &func,
            double xa = 0.0, 
            double xb = 1.0,
            double grow_limit = 110.0,
            size_t maxiter = 1000);

        Vec3Pair genVec3Pair(
            double xa, double xb, double xc,
            QResultPair fa, QResultPair fb, QResultPair fc);
    private:
        Func m_func;
        
        double m_tol;
        double m_xmin;

        QResultPair m_fval;

        size_t m_maxiter;
    };
}

#endif // ORIGINPOWELL_H
