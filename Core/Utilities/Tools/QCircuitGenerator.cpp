#include "Core/Utilities/Tools/QCircuitGenerator.h"
#include "Core/Utilities/Tools/PraseExpressionStr.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"

USING_QPANDA
using namespace std;

/*******************************************************************
*                      class ArbitrarilyAnglePrase
********************************************************************/
class ArbitrarilyAnglePrase : public ParseExpressionStr
{
public:
	ArbitrarilyAnglePrase(const std::vector<double>& angle_vec)
		:m_angle_vec(angle_vec)
	{}
	~ArbitrarilyAnglePrase() {}

	double string_to_double(const std::string src_str) override {
		double ret_val = 0.0;
		auto p_angle_str = strstr(src_str.c_str(), "theta_");
		if (nullptr != p_angle_str)
		{
			const auto i = atoi(p_angle_str + 6);
			if (m_angle_vec.size() < i) {
				QCERR_AND_THROW(run_fail, "Error: angle config error.");
				
			}
			ret_val = m_angle_vec[i - 1];
			if (src_str.at(0) == '-'){
				ret_val *= -1.0;
			}

		}
		else{
			return ParseExpressionStr::string_to_double(src_str);
		}

		return ret_val;
	}

private:
	const std::vector<double>& m_angle_vec;
};

/*******************************************************************
*                      class QCircuitGenerator
********************************************************************/
QCircuit QCircuitGenerator::get_cir()
{
	QCircuit cir;
	for (auto& cir_node : m_cir_node_vec)
	{
		const auto angle_size = cir_node->m_angle.size();
		const auto qubit_size = cir_node->m_target_q.size();
		QVec q;
		for (const auto& i : cir_node->m_target_q){
			q.emplace_back(m_qubit[i]);
		}

		switch (angle_size)
		{
		case 0:
		{
			cir << QGateNodeFactory::getInstance()->getGateNode(cir_node->m_op, q);
		}
			break;

		case 1:
		{
			if (cir_node->m_angle.size() < 1){
				QCERR_AND_THROW(run_fail, "Error: unknow circuit node error, no angle for rotation gate.");
			}

			auto _angle = angle_str_to_double(cir_node->m_angle[0]);
			cir <<  QGateNodeFactory::getInstance()->getGateNode(cir_node->m_op, q, _angle);
		}
			break;

		case 2:
		{
			if (cir_node->m_angle.size() < 1) {
				QCERR_AND_THROW(run_fail, "Error: unknow circuit node error, no enough angles for double rotation gate.");
			}

			auto _angle_0 = angle_str_to_double(cir_node->m_angle[0]);
			auto _angle_1 = angle_str_to_double(cir_node->m_angle[1]);
			cir << QGateNodeFactory::getInstance()->getGateNode(cir_node->m_op, q, _angle_0, _angle_1);
		}
			break;

		case 3:
		{
			if (cir_node->m_angle.size() < 1) {
				QCERR_AND_THROW(run_fail, "Error: unknow circuit node error, no enough angles for three rotation gate.");
			}

			auto _angle_0 = angle_str_to_double(cir_node->m_angle[0]);
			auto _angle_1 = angle_str_to_double(cir_node->m_angle[1]);
			auto _angle_2 = angle_str_to_double(cir_node->m_angle[2]);
			cir << QGateNodeFactory::getInstance()->getGateNode(cir_node->m_op, q, _angle_0, _angle_1, _angle_2);
		}
			break;

		default:
			QCERR_AND_THROW(run_fail, "Error: unknow circuit node error, too many angles.");
			break;
		}
	}

	return cir;
}

double QCircuitGenerator::angle_str_to_double(const string& angle_str){
	return ArbitrarilyAnglePrase(m_angle_vec).parse(angle_str);
}