#include <iostream>
#include <fstream>
#include "Components/Optimizer/OriginPowell.h"
#include "Core/Utilities/OriginCollection.h"
#include "QString.h"

const std::string POWELL_CACHE_HEADER = "POWELL CACHE FILE";

namespace QPanda
{
    bool operator < (const QResultPair &p1, const QResultPair &p2)
    {
        return p1.second < p2.second;
    }

    bool operator < (const QResultPair &p1, const double &coef)
    {
        return p1.second < coef;
    }

    bool operator <= (const QResultPair &p1, const QResultPair &p2)
    {
        return p1.second <= p2.second;
    }

    bool operator <= (const QResultPair &p1, const double &coef)
    {
        return p1.second <= coef;
    }

    bool operator > (const QResultPair &p1, const QResultPair &p2)
    {
        return p1.second > p2.second;
    }

    bool operator >= (const QResultPair &p1, const QResultPair &p2)
    {
        return p1.second >= p2.second;
    }

    double operator - (const QResultPair &p1, const QResultPair &p2)
    {
        return p1.second - p2.second;
    }

    double operator + (const QResultPair &p1, const QResultPair &p2)
    {
        return p1.second + p2.second;
    }

    double operator * (const double &coef, const QResultPair &p)
    {
        return p.second * coef;
    }

    double operator * (const QResultPair &p, const double &coef)
    {
        return p.second * coef;
    }

    OriginPowell::OriginPowell() :
        m_nonzdelt(0.05),
        m_zdelt(0.00025),
        m_fcalls(0),
        m_iter(0)
    {

    }

    void OriginPowell::exec()
    {
        init();

        auto x1 = m_x;
        while (true)
        {
            auto fx = m_fval;
            size_t bigind = 0;
            double delta = 0.0;

            Eigen::VectorXd direc1;
            for (size_t i = 0; i < m_n; i++)
            {
                direc1 = m_direc.row(i);
                auto fx2 = m_fval;
                m_fval = linesearch(m_x, direc1);


                if ((fx2 - m_fval) > delta)
                {
                    delta = fx2 - m_fval;
                    bigind = i;
                }
            }

            m_iter++;

            auto bnd = m_fatol * (fabs(fx.second) + fabs(m_fval.second)) 
                       + 1e-20;
            
            if (2.0 * (fx - m_fval) <= bnd)
            {                
                break;
            }

            if (m_fcalls >= m_max_fcalls)
            {
                break;
            }

            if (m_iter >= m_max_iter)
            {
                break;
            }

            // Construct the extrapolated point
            direc1 = m_x - x1;
            auto x2 = 2 * m_x - x1;
            x1 = m_x;
            auto fx2 = callFunc(x2);

            if (fx > fx2)
            {
                auto t = 2.0*(fx + fx2 - 2.0*m_fval);
                auto temp = fx - m_fval - delta;
                t *= temp * temp;
                temp = fx - fx2;
                t -= delta * temp * temp;

                if (t < 0.0)
                {
                    m_fval = linesearch(m_x, direc1);
                    m_direc.row(bigind) = m_direc.row(m_n - 1);
                    m_direc.row(m_n - 1) = direc1;
                }
            }

            dispResult();
            writeToFile();

            saveParaToCache();
        }
    }

    QOptimizationResult OriginPowell::getResult()
    {
        if (m_fcalls >= m_max_fcalls)
        {
            m_result.message = DEF_OPTI_STATUS_MAX_FEV;
            std::cout << DEF_WARING + m_result.message
                << std::endl;
        }
        else if (m_iter >= m_max_iter)
        {
            m_result.message = DEF_OPTI_STATUS_MAX_ITER;
            std::cout << DEF_WARING + m_result.message
                << std::endl;
        }
        else
        {
            m_result.message = DEF_OPTI_STATUS_SUCCESS;
            dispResult();
        }

        m_result.key = m_fval.first;
        m_result.fun_val = m_fval.second;
        m_result.fcalls = m_fcalls;
        m_result.iters = m_iter;
        m_result.para.resize(m_n);
        memcpy(m_result.para.data(),
            m_x.data(),
            m_n * sizeof(m_x[0]));

        return m_result;
    }

    bool OriginPowell::init()
    {
#ifdef _MSC_VER
        using convert_typeX = std::codecvt_utf8<wchar_t>;
        std::wstring_convert<convert_typeX, wchar_t> converterX;

        auto w_file = converterX.from_bytes(m_cache_file);
        if (m_restore_from_cache_file &&
            (_waccess(w_file.c_str(), 0) != -1))
#else
        struct stat buffer;
        if (m_restore_from_cache_file &&
            (stat(m_cache_file.c_str(), &buffer) == 0))
#endif // WIN32
        {
            if (!restoreParaFromCache())
            {
                return false;
            }

            m_n = m_x.size();
        }
        else
        {
            m_fcalls = 0;
            m_iter = 0;
            m_n = m_optimized_para.size();
            if (0 == m_n)
            {
                std::cout << "optimized para size is 0." << std::endl;
            }

            m_x = Eigen::VectorXd(m_n);
            memcpy(m_x.data(),
                m_optimized_para.data(),
                m_optimized_para.size() * sizeof(m_optimized_para[0]));

            m_fval = callFunc(m_x);
            m_direc = Eigen::MatrixXd::Identity(m_n, m_n);
        }

        adaptTerminationPara();
        m_result.message = DEF_OPTI_STATUS_CALCULATING;

        return true;
    }

    void OriginPowell::adaptTerminationPara()
    {
        if (0 == m_max_iter)
        {
            m_max_iter = m_n * 1000;
        }

        if (0 == m_max_fcalls)
        {
            m_max_fcalls = m_n * 1000;
        }
    }

    QResultPair OriginPowell::callFunc(const Eigen::VectorXd &para)
    {
        m_fcalls++;

        vector_d optimized_para(para.data(), para.data() + para.size());
        return m_func(optimized_para);
    }

    QResultPair OriginPowell::linesearch(
        Eigen::VectorXd &x0,
        Eigen::VectorXd &direc)
    {
        auto func = [=](double alpha)
        {
            return callFunc(x0 + alpha * direc);
        };

        Brent brent(func);
        brent.optimize();
        auto result = brent.getResult();

        direc = result.first * direc;
        x0 = x0 + direc;

        return result.second;
    }

    void OriginPowell::dispResult()
    {
        if (m_disp)
        {
            std::cout << m_result.message << std::endl;
            std::cout << "         Current function value: "
                << m_fval.second << std::endl;
            std::cout << "         Key: "
                << m_fval.first << std::endl;
            std::cout << "         Iterations: "
                << m_iter << std::endl;
            std::cout << "         Function evaluations: "
                << m_fcalls << std::endl;

            std::cout << "         Optimized para: " << std::endl;
            for (auto i = 0; i < m_n; i++)
            {
                std::cout << "             " << m_x[i] << std::endl;
            }
        }
    }

    void OriginPowell::writeToFile()
    {
        if (!m_para_file.empty())
        {
            if (fabs(m_fval.second - m_test_value) < m_fatol)
            {
                std::fstream f(m_para_file, std::ios::app);
                if (f.fail())
                {
                    std::cout << "Open file failed! " << m_para_file
                        << std::endl;

                    return;
                }

                for (size_t i = 0; i < m_n; i++)
                {
                    if (i != 0)
                    {
                        f << "\t";
                    }

                    f << m_x[i];
                }

                f << std::endl;
                f.close();

                exit(0);
            }
        }
    }

    void OriginPowell::saveParaToCache()
    {
        OriginCollection collection(m_cache_file, false);
        collection = { "index", "tag",
                       "fval", "x","direc","iter", "fcalls" };

        std::string tmp_fval = std::to_string(m_fval.second);


        std::string tmp_x;
        for (size_t i = 0; i < m_x.size(); i++)
        {
            if (i == 0)
            {
                tmp_x = std::to_string(m_x[i]);
            }
            else
            {
                tmp_x += "," + std::to_string(m_x[i]);
            }
        }

        std::string tmp_direc;
        for (size_t i = 0; i < m_n; i++)
        {
            if (i != 0)
            {
                tmp_direc += ";";
            }

            for (size_t j = 0; j < m_n; j++)
            {
                if (j == 0)
                {
                    tmp_direc += std::to_string(m_direc.row(i)[j]);
                }
                else
                {
                    tmp_direc += "," + std::to_string(m_direc.row(i)[j]);
                }
            }
        }

        collection.insertValue(0, POWELL_CACHE_HEADER, tmp_fval,
            tmp_x, tmp_direc, m_iter, m_fcalls);
        collection.write();
    }

    bool OriginPowell::restoreParaFromCache()
    {
        OriginCollection cache_file;
        if (!cache_file.open(m_cache_file))
        {
            std::cout << std::string("Open file failed! filename: ") + m_cache_file;
            return false;
        }


        std::string tag = cache_file.getValue("tag")[0];
        if (tag != POWELL_CACHE_HEADER)
        {
            std::cout << "It is not a POWELL cache file! Tag: " << tag
                << std::endl;
            return false;
        }

        m_fval.second = QString(cache_file.getValue("fval")[0]).toDouble();
   

        QString tmp_x = cache_file.getValue("x")[0];
        auto x_list = tmp_x.split(",", QString::SkipEmptyParts);
        m_x = Eigen::VectorXd::Zero(x_list.size());
        for (auto i = 0u; i < x_list.size(); i++)
        {
            m_x[i] = x_list[i].toDouble();
        }

        QString tmp_direc = cache_file.getValue("direc")[0];
        auto direc_list = tmp_direc.split(";", QString::SkipEmptyParts);
        m_direc = Eigen::MatrixXd::Identity(direc_list.size(), direc_list.size());
        for (auto i = 0u; i < direc_list.size(); i++)
        {
            auto item_list = direc_list[i].split(",", QString::SkipEmptyParts);
            for (auto j = 0u; j < item_list.size(); j++)
            {
                m_direc.row(i)[j] = item_list[j].toDouble();
            }
        }

        m_iter = QString(cache_file.getValue("iter")[0]).toInt();
        m_fcalls = QString(cache_file.getValue("fcalls")[0]).toInt();

        return true;
    }

    Brent::Brent(const Func &func, double tol, size_t maxiter) :
        m_func(func),
        m_tol(tol),
        m_maxiter(maxiter)
    {
    }

    void Brent::optimize()
    {
        const double kMinTol{ 1.0e-11 };
        const double kCg{ 0.3819660 };

        Vec3Pair vec = bracket(m_func);
        auto xa = vec[0].first;
//        auto fa = vec[0].second;
        auto xb = vec[1].first;
//        auto fb = vec[1].second;
        auto xc = vec[2].first;
//        auto fc = vec[2].second;

        auto x = xb;
        auto w = xb;
        auto v = xb;

        auto fx = m_func(x);
        auto fw = fx;
        auto fv = fx;

        auto a = xa < xc ? xa : xc;
        auto b = xa < xc ? xc : xa;

        double deltax = 0.0;
        size_t iter = 0;
        while (iter < m_maxiter)
        {
            auto tol1 = m_tol * fabs(x) + kMinTol;
            auto tol2 = 2.0 * tol1;
            auto xmid = 0.5 * (a + b);
            double rat = 0.0;
            // check for convergence
            if (fabs(x - xmid) < (tol2 - 0.5 * (b - a)))
            {
                break;
            }

            if (fabs(deltax) <= tol1) 
            { // do a golden section step
                deltax = x >= xmid ? a - x : b - x;
                rat = kCg * deltax;
            }
            else 
            { // do a parabolic step
                auto tmp1 = (x - w) * (fx - fv);
                auto tmp2 = (x - v) * (fx - fw);
                auto p = (x - v) * tmp2 - (x - w) * tmp1;
                tmp2 = 2.0 * (tmp2 - tmp1);
                
                if (tmp2 > 0.0)
                {
                    p = -p;
                }
                tmp2 = fabs(tmp2);
                auto dx_temp = deltax;
                deltax = rat;

                // check parabolic fit
                if ((p > tmp2 * (a - x)) &&
                    (p < tmp2 * (b - x)) &&
                    (fabs(p) < fabs(0.5 * tmp2 * dx_temp)))
                { // if parabolic step is useful
                    rat = p * 1.0 / tmp2;
                    auto u = x + rat;
                    if ((u - a) < tol2 ||
                        (b - u) < tol2)
                    {
                        rat = xmid - x >= 0 ? tol1 : -tol1;
                    }
                }
                else
                { // if it's not do a golden section step
                    deltax = x >= xmid ? a - x : b - x;
                    rat = kCg * deltax;
                }
            }

            double u = 0.0;
            if (fabs(rat) < tol1) // update by at least tol1
            {
                u = rat >= 0 ? x + tol1 : x - tol1;
            }
            else
            {
                u = x + rat;
            }
            auto fu = m_func(u); // calculate new output value

            if (fu > fx) // if it's bigger than current
            {
                if (u < x)
                {
                    a = u;
                }
                else
                {
                    b = u;
                }

                if (fu <= fw || w == x)
                {
                    v = w;
                    w = u;
                    fv = fw;
                    fw = fu;
                }
                else if (fu <= fv || v == x || v == w)
                {
                    v = u;
                    fv = fu;
                }
            }
            else
            {
                if (u >= x)
                {
                    a = x;
                }
                else
                {
                    b = x;
                }

                v = w;
                w = x;
                x = u;
                fv = fw;
                fw = fx;
                fx = fu;
            }

            iter++;
        }

        m_xmin = x;
        m_fval = fx;
    }

    std::pair<double, QResultPair> Brent::getResult()
    {
        return std::make_pair(m_xmin, m_fval);
    }

    Brent::Vec3Pair    Brent::bracket(
            const Func &func,
            double xa,
            double xb,
            double grow_limit,
            size_t maxiter)
    {
        const double kGold{ 1.618034 }; // golden ratio: (1.0+sqrt(5.0))/2.0
        const double kVerySmallNum{ 1e-21 };

        auto fa = func(xa);
        auto fb = func(xb);

        if (fa < fb)
        {
            auto tmp_xa = xa;
            auto tmp_fa = fa;

            xa = xb;
            fa = fb;
            xb = tmp_xa;
            fb = tmp_fa;
        }

        auto xc = xb + kGold * (xb - xa);
        auto fc = func(xc);

        size_t iter = 0;
        while (fc < fb)
        {
            auto tmp1 = (xb - xa) * (fb - fc);
            auto tmp2 = (xb - xc) * (fb - fa);
            auto val = tmp2 - tmp1;

            auto denom = (fabs(val) < kVerySmallNum) ?
                2.0 * kVerySmallNum : 2.0 * val;

            auto w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom;
            auto wlim = xb + grow_limit * (xc - xb);

            if (iter > maxiter)
            {
                std::cout << "Too many iterations." << std::endl;
                break;
            }

            iter++;

            QResultPair fw;
            if ((w - xc) * (xb - w) > 0.0)
            {
                fw = func(w);
                if (fw < fc)
                {
                    xa = xb;
                    xb = w;
                    fa = fb;
                    fb = fw;

                    return genVec3Pair(xa, xb, xc, fa, fb, fc);
                }
                else if (fw > fb)
                {
                    xc = w;
                    fc = fw;
                    
                    return genVec3Pair(xa, xb, xc, fa, fb, fc);                        
                }

                w = xc + kGold * (xc - xb);
                fw = func(w);
            }
            else if ((w - wlim)*(wlim - xc) >= 0.0)
            {
                w = wlim;
                fw = func(w);
            }
            else if ((w - wlim)*(xc - w) > 0.0)
            {
                fw = func(w);
                if (fw < fc)
                {
                    xb = xc;
                    xc = w;
                    w = xc + kGold * (xc - xb);
                    fb = fc;
                    fc = fw;
                    fw = func(w);
                }    
            }
            else
            {
                w = xc + kGold * (xc - xb);
                fw = func(w);
            }

            xa = xb;
            xb = xc;
            xc = w;
            fa = fb;
            fb = fc;
            fc = fw;
        }

        return genVec3Pair(xa, xb, xc, fa, fb, fc);
    }

    Brent::Vec3Pair Brent::genVec3Pair(
        double xa, 
        double xb, 
        double xc, 
        QResultPair fa,
        QResultPair fb,
        QResultPair fc)
    {
        Vec3Pair vec;
        vec.push_back(std::make_pair(xa, fa));
        vec.push_back(std::make_pair(xb, fb));
        vec.push_back(std::make_pair(xc, fc));

        return vec;
    }
}
