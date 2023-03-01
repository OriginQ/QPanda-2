#include <iostream>
#include "Core/Utilities/Tools/OriginCollection.h"
#include "Core/Utilities/Tools/QString.h"
#include "Components/Optimizer/OriginGradient.h"
#include "Eigen/Dense"

const std::string GD_CACHE_HEADER = "GRADIENT CACHE FILE";

QPANDA_BEGIN

OriginGradient::OriginGradient():
	m_fcalls(0),
    m_iter(0),
    m_learning_rate(0.1)
{
}

void OriginGradient::exec()
{
	init();

	// no-convergence test
	while ((m_fcalls < m_max_fcalls) && (m_iter < m_max_iter))
	{
		// the domain and function-value convergence test
		if (testTermination())
		{
			break;
		}

		//vector_d optimized_para(m_sim.row(0).data(), 
		//	m_sim.row(0).data() + m_sim.row(0).size());
        std::vector<double> optimized_para(m_sim.row(0).size());
		//optimized_para.resize(m_sim.row(0).size());
        for (size_t i = 0; i < optimized_para.size(); i++)
		{
			optimized_para[i] = m_sim.row(0)[i];
		}
		auto result = m_func(optimized_para, m_gradient, ++m_iter, ++m_fcalls);

		m_sim.row(1) = m_sim.row(0);
		m_fsim[1] = m_fsim[0];
		m_fsim[0] = result.second;
		if (m_fsim[0] < m_fsim[2])
		{
			m_fsim[2] = m_fsim[0];
			m_sim.row(2) = m_sim.row(0);
		}

        for (size_t i = 0; i < m_gradient.size(); i++)
		{
			m_sim.row(0)[i] -= m_learning_rate * m_gradient[i];
		}
		
		saveParaToCache();
		dispResult();
	}
}

QOptimizationResult OriginGradient::getResult()
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

	m_result.fun_val = m_fsim[2];
	m_result.fcalls = m_fcalls;
	m_result.iters = m_iter;
	m_result.para.resize(m_sim.row(2).size());

	for (auto i = 0u; i < m_result.para.size(); i++)
	{
		m_result.para[i] = m_sim.row(2)[i];
	}

	return m_result;
}

void OriginGradient::init()
{
	bool success_flag = false;
	if (m_restore_from_cache_file)
	{
#ifdef _MSC_VER
		using convert_typeX = std::codecvt_utf8<wchar_t>;
		std::wstring_convert<convert_typeX, wchar_t> converterX;

		auto w_file = converterX.from_bytes(m_cache_file);
		if (_waccess(w_file.c_str(), 0) != -1)
#else
		struct stat buffer;
		if (stat(m_cache_file.c_str(), &buffer) == 0)
#endif // WIN32
		{
			if (!restoreParaFromCache())
			{
				QCERR("Restore from cache file failed!");
			}
			else 
			{
				success_flag = true;
			}

		}
		else
		{
			QCERR("Restore from cache file failed: no cache file found!");
		}
	}

	if (!success_flag)
	{
		m_sim = Eigen::MatrixXd::Zero(3, m_optimized_para.size());
        for (size_t i = 0; i < m_optimized_para.size(); i++)
		{
			m_sim.row(0)[i] = m_optimized_para[i];
		}
		//memcpy(m_sim.row(0).data(),
		//	m_optimized_para.data(),
		//	m_optimized_para.size() * sizeof(m_optimized_para[0]));

		m_fsim = Eigen::VectorXd::Ones(3)* std::numeric_limits<double>::max();
		m_fsim[0] = 0.0;
	}

	auto iter = m_optional_para.find(DEF_LEARNING_RATE);
	if (iter != m_optional_para.end())
	{
		m_learning_rate = std::stod(iter->second);
	}
	m_gradient.resize(m_optimized_para.size());
}

bool OriginGradient::testTermination()
{
	auto tmp_sim = (m_sim.row(0) - m_sim.row(1)).cwiseAbs();
	auto tmp_fsim = fabs(m_fsim[0] - m_fsim[1]);

	double t_s = tmp_sim.maxCoeff();
	double t_f = tmp_fsim;

	if ((t_s <= m_xatol)&&(t_f <= m_fatol))
	{
		std::cout << "go into here" << std::endl;
		return true;
	}
	else
	{
		return false;
	}

	// return (tmp_sim.maxCoeff() <= m_xatol) && (tmp_fsim <= m_fatol);
}

void OriginGradient::dispResult()
{
	if (m_disp)
	{
		//std::cout << m_result.message << std::endl;
		//std::cout << "         Current function value: "
		//	<< m_fsim[0] << std::endl;
		//std::cout << "         Best function value: "
		//	<< m_fsim[2] << std::endl;
		//std::cout << "         Iterations: "
		//	<< m_iter << std::endl;
		//std::cout << "         Function evaluations: "
		//	<< m_fcalls << std::endl;

		//std::cout << "         Optimized para: " << std::endl;
		for (auto i = 0u; i < m_sim.row(0).size(); i++)
		{
			std::cout << "             " << m_sim.row(0)[i] << std::endl;
		}
	}
}

bool OriginGradient::saveParaToCache()
{
	OriginCollection collection(m_cache_file, false);
	collection = { "index", "tag",
				   "fsim","sim","iter", "fcalls" };


	std::string tmp_fsim;
    for (auto i = 0; i < m_fsim.size(); i++)
	{
		if (i == 0)
		{
			tmp_fsim = QString(m_fsim[i]).data();
		}
		else
		{
			tmp_fsim += "," + QString(m_fsim[i]).data();
		}
	}

	std::string tmp_sim;
    for (auto i = 0; i < m_sim.rows(); i++)
	{
		if (i != 0)
		{
			tmp_sim += ";";
		}

        for (auto j = 0; j < m_sim.cols(); j++)
		{
			if (j == 0)
			{
				tmp_sim += QString(m_sim.row(i)[j]).data();
			}
			else
			{
				tmp_sim += "," + QString(m_sim.row(i)[j]).data();
			}
		}
	}

	collection.insertValue(0, GD_CACHE_HEADER,
		tmp_fsim, tmp_sim, m_iter, m_fcalls);
	return collection.write();
}

bool OriginGradient::restoreParaFromCache()
{
	OriginCollection cache_file;
	if (!cache_file.open(m_cache_file))
	{
		QCERR(std::string("Open file failed! filename: ") + m_cache_file);
		return false;
	}

	std::string tag = cache_file.getValue("tag")[0];
	if (tag != GD_CACHE_HEADER)
	{
		QCERR(std::string("It is not a Gradient cache file! Tag: ") + tag);
		return false;
	}

	QString tmp_fsim = cache_file.getValue("fsim")[0];
	auto fsim_list = tmp_fsim.split(",", QString::SkipEmptyParts);
	m_fsim = Eigen::VectorXd::Zero(fsim_list.size());
	for (auto i = 0u; i < fsim_list.size(); i++)
	{
		m_fsim[i] = fsim_list[i].toDouble();
	}

	QString tmp_sim = cache_file.getValue("sim")[0];
	auto sim_list = tmp_sim.split(";", QString::SkipEmptyParts);
	size_t col_size = 0;
	if (sim_list.size() > 0)
	{
		auto item_list = sim_list[0].split(",", QString::SkipEmptyParts);
		col_size = item_list.size();
	}
	
	m_sim = Eigen::MatrixXd::Zero(fsim_list.size(), col_size);
	for (auto i = 0u; i < sim_list.size(); i++)
	{
		auto item_list = sim_list[i].split(",", QString::SkipEmptyParts);
		for (auto j = 0u; j < item_list.size(); j++)
		{
			m_sim.row(i)[j] = item_list[j].toDouble();
		}
	}

	m_iter = QString(cache_file.getValue("iter")[0]).toInt();
	m_fcalls = QString(cache_file.getValue("fcalls")[0]).toInt();

	return true;
}

QPANDA_END
