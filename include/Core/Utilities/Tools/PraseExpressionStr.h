#ifndef PARSE_EXPRESSION_STR_H
#define PARSE_EXPRESSION_STR_H
#include "Core/Utilities/QPandaNamespace.h"
#include <iostream>
#include <complex>
#include <vector>
#include <list>

QPANDA_BEGIN

class ParseExpressionStr
{
//public:
	enum OperatorType
	{
		OP_ADD = '+',
		OP_SUB = '-',
		OP_MUL = '*',
		OP_DIV = '/'
	};

	enum ListNodeT
	{
		DATA_T = 0,
		OP_T
	};

	struct StrNode
	{
		StrNode(const std::string& str, ListNodeT t)
			:m_str(str), m_type(t)
		{}

		std::string m_str;
		ListNodeT m_type;
	};

public:
	ParseExpressionStr() {}
	virtual ~ParseExpressionStr() {}

	virtual double parse(const std::string& src_str) {
		char tmp_buf[128] = "";
		for (size_t i = 0; i < src_str.size(); ++i)
		{
			if (src_str.at(i) == OP_SUB)
			{
				if (strlen(tmp_buf) == 0)
				{
					tmp_buf[0] = OP_SUB;
					continue;
				}
				auto p_node = std::make_shared<StrNode>(tmp_buf, DATA_T);
				m_expression_list.push_back(p_node);
				m_expression_list.push_back(std::make_shared<StrNode>(std::string(1,OP_SUB), OP_T));
				memset(tmp_buf, 0, sizeof(tmp_buf));
			}
			else if (src_str.at(i) == OP_ADD)
			{
				auto p_node = std::make_shared<StrNode>(tmp_buf, DATA_T);
				m_expression_list.push_back(p_node);
				m_expression_list.push_back(std::make_shared<StrNode>(std::string(1, OP_ADD), OP_T));
				memset(tmp_buf, 0, sizeof(tmp_buf));
			}
			else if (src_str.at(i) == OP_MUL)
			{
				auto p_node = std::make_shared<StrNode>(tmp_buf, DATA_T);
				m_expression_list.push_back(p_node);
				m_expression_list.push_back(std::make_shared<StrNode>(std::string(1, OP_MUL), OP_T));
				memset(tmp_buf, 0, sizeof(tmp_buf));
			}
			else if (src_str.at(i) == OP_DIV)
			{
				auto p_node = std::make_shared<StrNode>(tmp_buf, DATA_T);
				m_expression_list.push_back(p_node);
				m_expression_list.push_back(std::make_shared<StrNode>(std::string(1, OP_DIV), OP_T));
				memset(tmp_buf, 0, sizeof(tmp_buf));
			}
			else
			{
				tmp_buf[strlen(tmp_buf)] = src_str.at(i);
			}
		}

		auto p_node = std::make_shared<StrNode>(tmp_buf, DATA_T);
		m_expression_list.push_back(p_node);

		return calc_expression();
	}

protected:
	virtual double string_to_double(const std::string src_str) {
		double ret_val = 0.0;
		if (nullptr != strstr(src_str.c_str(), "PI"))
		{
			if (src_str.at(0) == '-')
			{
				ret_val = (-1.0 * PI);
			}
			else
			{
				ret_val = PI;
			}
		}
		else
		{
			ret_val = atof(src_str.c_str());
		}

		return ret_val;
	}

	double calc_expression(
		std::list<std::shared_ptr<StrNode>>::iterator start_itr = std::list<std::shared_ptr<StrNode>>::iterator(),
		bool b_start = true) {
		double ret_val = 0.0;
		if (b_start){
			start_itr = m_expression_list.begin();
		}

		auto get_data_node_val = [&](const StrNode& data_node) ->double{
			if (DATA_T != data_node.m_type)
			{
				QCERR_AND_THROW_ERRSTR(run_fail, "Error: failed to calc_expression, node type error.");
			}
			return string_to_double(data_node.m_str);
		};

		for (; start_itr != m_expression_list.end(); ++start_itr)
		{
			const auto& node = (*start_itr);
			if ((DATA_T == node->m_type))
			{
				ret_val = string_to_double(node->m_str);
				continue;
			}

			else if (OP_T == node->m_type)
			{
				if (node->m_str.size() != 1)
				{
					QCERR_AND_THROW_ERRSTR(run_fail, "Error: nuknow error on parse expression string.");
				}

				switch (node->m_str.at(0))
				{
				case OP_ADD:
					return ret_val + calc_expression(++start_itr, false);
					//break;

				case OP_SUB:
					return ret_val - calc_expression(++start_itr, false);
					//break;

				case OP_MUL:
				{
					const auto next_val = get_data_node_val(*(*(++start_itr)));
					ret_val = (ret_val * next_val);
				}
					break;

				case OP_DIV:
				{
					const auto next_val = get_data_node_val(*(*(++start_itr)));
					ret_val = (ret_val / next_val);
				}
					break;

				default:
					QCERR_AND_THROW_ERRSTR(run_fail, "Error: nuknow error on parse expression string, wrong opertor type.");
					break;
				}
			}
			else
			{
				QCERR_AND_THROW_ERRSTR(run_fail, "Error: unknow error on parse string to double val.");
			}
		}

		return ret_val;
	}

private:
	std::list<std::shared_ptr<StrNode>> m_expression_list;
};

QPANDA_END
#endif // PARSE_EXPRESSION_STR_H
