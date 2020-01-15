#include "Core/Utilities/QProgInfo/QProgToMatrix.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/GraphMatch.h"
#include "Core/Core.h"

USING_QPANDA
using namespace std;

#define PRINT_TRACE 0
#if PRINT_TRACE
#define PTrace printf
#define PTraceMat(mat) (std::cout << (mat) << endl)
#else
#define PTrace
#define PTraceMat(mat)
#endif

QProgToMatrix::MatrixOfOneLayer::MatrixOfOneLayer(SequenceLayer& layer, const QProgDAG& prog_dag, std::vector<int> &qubits_in_use)
	:m_qubits_in_use(qubits_in_use), m_mat_I{ 1, 0, 0, 1 }
{
	for (auto &layer_item : layer)
	{
		auto p_node = prog_dag.get_vertex(layer_item.first.m_vertex_num);
		auto p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(p_node);
		QVec qubits_vector;
		p_gate->getQuBitVector(qubits_vector);
		QVec control_qubits_vector;
		p_gate->getControlVector(control_qubits_vector);
		if (control_qubits_vector.size() > 0)
		{
			qubits_vector.insert(qubits_vector.end(), control_qubits_vector.begin(), control_qubits_vector.end());
			std::sort(qubits_vector.begin(), qubits_vector.end(), [](Qubit* a, Qubit* b) {
				return a->getPhysicalQubitPtr()->getQubitAddr() < b->getPhysicalQubitPtr()->getQubitAddr();
			});

			std::vector<int> tmp_vec;
			tmp_vec.push_back(qubits_vector.front()->getPhysicalQubitPtr()->getQubitAddr());
			tmp_vec.push_back(qubits_vector.back()->getPhysicalQubitPtr()->getQubitAddr());

			m_controled_gates.push_back(std::pair<std::shared_ptr<AbstractQGateNode>, std::vector<int>>(p_gate, tmp_vec));
			continue;
		}

		if (qubits_vector.size() == 2)
		{
			std::vector<int> quBits;
			for (auto _val : qubits_vector)
			{
				quBits.push_back(_val->getPhysicalQubitPtr()->getQubitAddr());
			}

			m_double_qubit_gates.push_back(std::pair<std::shared_ptr<AbstractQGateNode>, std::vector<int>>(p_gate, quBits));
		}
		else if (qubits_vector.size() == 1)
		{
			std::vector<int> quBits;
			quBits.push_back(qubits_vector.front()->getPhysicalQubitPtr()->getQubitAddr());
			m_single_qubit_gates.push_back(std::pair<std::shared_ptr<AbstractQGateNode>, std::vector<int>>(p_gate, quBits));
		}
		else
		{
			QCERR_AND_THROW_ERRSTR(runtime_error, "Error: QGate type error.");
		}
	}

	//sort by qubit address spacing
	auto sorfFun = [](gateAndQubitsItem_t &a, gateAndQubitsItem_t &b) { return (abs(a.second.front() - a.second.back())) < (abs(b.second.front() - b.second.back())); };
	std::sort(m_controled_gates.begin(), m_controled_gates.end(), sorfFun);
	std::sort(m_double_qubit_gates.begin(), m_double_qubit_gates.end(), sorfFun);
	std::sort(m_single_qubit_gates.begin(), m_single_qubit_gates.end(), sorfFun);
}

QStat QProgToMatrix::MatrixOfOneLayer::reverse_ctrl_gate_matrix_CX(QStat& src_mat)
{
	init(QMachineType::CPU);
	auto q = qAllocMany(6);
	QGate gate_H = H(q[0]);
	QStat mat_H;
	gate_H.getQGate()->getMatrix(mat_H);
	finalize();

	QStat result_mat;
	QStat mat_of_zhang_multp_two_H = QPanda::tensor(mat_H, mat_H);

	result_mat = (mat_of_zhang_multp_two_H * src_mat);
	result_mat = (result_mat * mat_of_zhang_multp_two_H);

	PTrace("reverse_ctrl_gate_matrix_CX: ");
	PTraceMat(result_mat);
	return result_mat;
}

QStat QProgToMatrix::MatrixOfOneLayer::reverse_ctrl_gate_matrix_CU(QStat& src_mat)
{
	init(QMachineType::CPU);
	auto q = qAllocMany(6);
	QGate gate_swap = SWAP(q[0], q[1]);
	QStat mat_swap;
	gate_swap.getQGate()->getMatrix(mat_swap);
	finalize();

	QStat result_mat;

	result_mat = (mat_swap * src_mat * mat_swap);

	PTrace("reverse_ctrl_gate_matrix_CX: ");
	PTraceMat(result_mat);
	return result_mat;
}

void QProgToMatrix::MatrixOfOneLayer::merge_two_crossed_matrix(const calcUintItem_t& calc_unit_1, const calcUintItem_t& calc_unit_2, calcUintItem_t& result)
{
	int qubit_start = (calc_unit_1.second[0] < calc_unit_2.second[0]) ? calc_unit_1.second[0] : calc_unit_2.second[0];
	int qubit_end = (calc_unit_1.second[1] > calc_unit_2.second[1]) ? calc_unit_1.second[1] : calc_unit_2.second[1];
	QStat tensored_calc_unit_1;
	QStat tensored_calc_unit_2;

	auto tensor_func = [this](const size_t &qubit_index, const calcUintItem_t &calc_unit, QStat &tensor_result) {
		if (qubit_index < calc_unit.second[0])
		{
			tensor_by_matrix(tensor_result, m_mat_I);
		}
		else if (qubit_index == calc_unit.second[0])
		{
			tensor_by_matrix(tensor_result, calc_unit.first);
		}
		else if (qubit_index > calc_unit.second[1])
		{
			tensor_by_matrix(tensor_result, m_mat_I);
		}
	};

	for (size_t i = qubit_start; i < qubit_end + 1; ++i)
	{
		tensor_func(i, calc_unit_1, tensored_calc_unit_1);
		tensor_func(i, calc_unit_2, tensored_calc_unit_2);
	}

	result.first = tensored_calc_unit_1 * tensored_calc_unit_2;
	result.second.push_back(qubit_start);
	result.second.push_back(qubit_end);
}

//return true on cross, or else return false
bool QProgToMatrix::MatrixOfOneLayer::check_cross_calc_unit(calcUnitVec_t& calc_unit_vec, calcUnitVec_t::iterator target_calc_unit_itr)
{
	const auto& target_calc_qubits = target_calc_unit_itr->second;
	for (auto itr_calc_unit = calc_unit_vec.begin(); itr_calc_unit < calc_unit_vec.end(); ++itr_calc_unit)
	{
		if (((target_calc_qubits[0] > itr_calc_unit->second.front()) && (target_calc_qubits[0] < itr_calc_unit->second.back()))
			||
			((target_calc_qubits[1] > itr_calc_unit->second.front()) && (target_calc_qubits[1] < itr_calc_unit->second.back())))
		{
			//merge two crossed matrix
			calcUintItem_t merge_result_calc_unit;
			merge_two_crossed_matrix(*itr_calc_unit, *target_calc_unit_itr, merge_result_calc_unit);

			itr_calc_unit->first.swap(merge_result_calc_unit.first);
			itr_calc_unit->second.swap(merge_result_calc_unit.second);

			return true;
		}
	}

	return false;
}

void QProgToMatrix::MatrixOfOneLayer::tensor_by_QGate(QStat& src_mat, std::shared_ptr<AbstractQGateNode> &pGate)
{
	if (nullptr == pGate)
	{
		return;
	}

	if (src_mat.empty())
	{
		pGate->getQGate()->getMatrix(src_mat);
		if (pGate->isDagger())
		{
			dagger(src_mat);
		}
	}
	else
	{
		QStat single_gate_mat;
		pGate->getQGate()->getMatrix(single_gate_mat);
		if (pGate->isDagger())
		{
			dagger(single_gate_mat);
		}
		src_mat = QPanda::tensor(src_mat, single_gate_mat);
	}
}

void QProgToMatrix::MatrixOfOneLayer::tensor_by_matrix(QStat& src_mat, const QStat& tensor_mat)
{
	if (src_mat.empty())
	{
		src_mat = tensor_mat;
	}
	else
	{
		src_mat = QPanda::tensor(src_mat, tensor_mat);
	}
}

void QProgToMatrix::MatrixOfOneLayer::get_stride_over_qubits(const std::vector<int> &qgate_used_qubits, std::vector<int> &stride_over_qubits)
{
	stride_over_qubits.clear();

	for (auto &qubit_val : m_qubits_in_use)
	{
		if ((qubit_val > qgate_used_qubits.front()) && (qubit_val < qgate_used_qubits.back()))
		{
			stride_over_qubits.push_back(qubit_val);
		}
	}
}

void QProgToMatrix::MatrixOfOneLayer::merge_to_calc_unit(std::vector<int>& qubits, QStat& gate_mat, calcUnitVec_t &calc_unit_vec, gateQubitInfo_t &single_qubit_gates)
{
	//auto &qubits = curGareItem.second;
	std::sort(qubits.begin(), qubits.end(), [](int &a, int &b) {return a < b; });
	std::vector<int> stride_over_qubits;
	get_stride_over_qubits(qubits, stride_over_qubits);
	if (stride_over_qubits.empty())
	{
		//serial qubits
		calc_unit_vec.insert(calc_unit_vec.begin(), std::pair<QStat, std::vector<int>>(gate_mat, qubits));
	}
	else
	{
		//get crossed CalcUnits;
		calcUnitVec_t crossed_calc_units;
		for (auto itr_calc_unit = calc_unit_vec.begin(); itr_calc_unit < calc_unit_vec.end();)
		{
			if ((qubits[0] < itr_calc_unit->second.front()) && (qubits[1] > itr_calc_unit->second.back()))
			{
				/*if the current double qubit gate has crossed the itr_calc_unit,
				  calc two crossed matrix, and replease the current itr_calc_unit
				*/
				check_cross_calc_unit(crossed_calc_units, itr_calc_unit);
				crossed_calc_units.push_back(*itr_calc_unit);

				itr_calc_unit = calc_unit_vec.erase(itr_calc_unit);
				continue;

			}

			++itr_calc_unit;
		}

		//get crossed SingleQubitGates;
		gateQubitInfo_t crossed_single_qubit_gates;
		for (auto itr_single_gate = single_qubit_gates.begin(); itr_single_gate < single_qubit_gates.end();)
		{
			const int qubit_val = itr_single_gate->second.front();
			if ((qubit_val > qubits[0]) && (qubit_val < qubits[1]))
			{
				crossed_single_qubit_gates.push_back(*itr_single_gate);

				itr_single_gate = single_qubit_gates.erase(itr_single_gate);
				continue;
			}

			++itr_single_gate;
		}

		//zhang multiply
		QStat filled_matrix;
		for (auto &in_used_qubit_val : m_qubits_in_use)
		{
			if (in_used_qubit_val > qubits[0])
			{
				if (in_used_qubit_val >= qubits[1])
				{
					break;
				}

				bool b_no_qGate_on_this_qubit = true;

				//find current qubit_val in crossed_single_qubit_gates
				std::shared_ptr<AbstractQGateNode> pGate;
				for (auto itr_crossed_single_gate = crossed_single_qubit_gates.begin();
					itr_crossed_single_gate != crossed_single_qubit_gates.end(); itr_crossed_single_gate++)
				{
					if (in_used_qubit_val == itr_crossed_single_gate->second.front())
					{
						b_no_qGate_on_this_qubit = false;

						tensor_by_QGate(filled_matrix, itr_crossed_single_gate->first);
						crossed_single_qubit_gates.erase(itr_crossed_single_gate);
						break;
					}
				}

				//find current qubit_val in crossed_calc_units
				for (auto itr_crossed_calc_unit = crossed_calc_units.begin();
					itr_crossed_calc_unit != crossed_calc_units.end(); itr_crossed_calc_unit++)
				{
					if ((in_used_qubit_val >= itr_crossed_calc_unit->second.front()) && (in_used_qubit_val <= itr_crossed_calc_unit->second.back()))
					{
						b_no_qGate_on_this_qubit = false;

						if (in_used_qubit_val == itr_crossed_calc_unit->second.front())
						{
							tensor_by_matrix(filled_matrix, itr_crossed_calc_unit->first);
							//just break, CANN'T erase itr_crossed_calc_unit here
							break;
						}
					}
				}

				//No handle on this qubit
				if (b_no_qGate_on_this_qubit)
				{
					tensor_by_matrix(filled_matrix, m_mat_I);
				}
			}
		}

		//blockMultip
		QStat filled_double_gate_matrix;
		blockedMatrix_t blocked_mat;
		partition(gate_mat, 2, 2, blocked_mat);
		blockMultip(filled_matrix, blocked_mat, filled_double_gate_matrix);

		//insert into calc_unit_vec
		calc_unit_vec.insert(calc_unit_vec.begin(), std::pair<QStat, std::vector<int>>(filled_double_gate_matrix, qubits));
	}
}

void  QProgToMatrix::MatrixOfOneLayer::merge_double_gate()
{
	GateType gate_T = GATE_UNDEFINED;
	for (auto &double_gate : m_double_qubit_gates)
	{
		QStat gate_mat;
		gate_T = (GateType)(double_gate.first->getQGate()->getGateType());
		if (2 == double_gate.second.size())
		{
			auto &qubits = double_gate.second;

			double_gate.first->getQGate()->getMatrix(gate_mat);

			if (qubits[0] > qubits[1])
			{
				if (CNOT_GATE == gate_T)
				{
					// transf base matrix
					auto transformed_mat = reverse_ctrl_gate_matrix_CX(gate_mat);
					gate_mat.swap(transformed_mat);
				}
				else if (CU_GATE == gate_T)
				{
					auto transformed_mat = reverse_ctrl_gate_matrix_CU(gate_mat);
					gate_mat.swap(transformed_mat);
				}
			}

			if (double_gate.first->isDagger())
			{
				dagger(gate_mat);
			}
		}
		else
		{
			QCERR_AND_THROW_ERRSTR(runtime_error, "Error: Qubits number error.");
		}

		merge_to_calc_unit(double_gate.second, gate_mat, m_calc_unit_vec, m_single_qubit_gates);
	}
}

void  QProgToMatrix::MatrixOfOneLayer::merge_calc_unit()
{
	for (auto &itr_calc_unit_vec : m_calc_unit_vec)
	{
		//calc all the qubits to get the final matrix
		QStat final_mat_of_one_calc_unit;
		for (auto &in_used_qubit_val : m_qubits_in_use)
		{
			bool b_no_gate_on_this_qubit = true;
			for (auto itr_single_gate = m_single_qubit_gates.begin(); itr_single_gate < m_single_qubit_gates.end();)
			{
				const int qubit_val = itr_single_gate->second.front();
				if (qubit_val == in_used_qubit_val)
				{
					b_no_gate_on_this_qubit = false;
					tensor_by_QGate(final_mat_of_one_calc_unit, itr_single_gate->first);

					itr_single_gate = m_single_qubit_gates.erase(itr_single_gate);
					continue;
				}

				++itr_single_gate;
			}

			if (itr_calc_unit_vec.second.front() == in_used_qubit_val)
			{
				b_no_gate_on_this_qubit = false;
				tensor_by_matrix(final_mat_of_one_calc_unit, itr_calc_unit_vec.first);
			}

			if ((itr_calc_unit_vec.second.front() <= in_used_qubit_val) && (itr_calc_unit_vec.second.back() >= in_used_qubit_val))
			{
				continue;
			}

			if (b_no_gate_on_this_qubit)
			{
				tensor_by_matrix(final_mat_of_one_calc_unit, m_mat_I);
			}
		}

		//Multiply, NOT tensor
		if (m_current_layer_mat.empty())
		{
			m_current_layer_mat = final_mat_of_one_calc_unit;
		}
		else
		{
			m_current_layer_mat = final_mat_of_one_calc_unit * m_current_layer_mat;
		}
	}
}

void QProgToMatrix::MatrixOfOneLayer::reverse_ctrl_gate_matrix(QStat& src_mat, const GateType &gate_T)
{
	QStat result;
	switch (gate_T)
	{
	case CNOT_GATE:
		result = reverse_ctrl_gate_matrix_CX(src_mat);
		break;

	case CU_GATE:
		result = reverse_ctrl_gate_matrix_CU(src_mat);
		break;

	default:
		QCERR_AND_THROW_ERRSTR(runtime_error, "Error: reverse_ctrl_gate_matrix error, unsupport type.");
		break;
	}

	src_mat.swap(result);
}

void  QProgToMatrix::MatrixOfOneLayer::merge_controled_gate()
{
	if (m_controled_gates.size() == 0)
	{
		return;
	}

	for (auto& controled_gate : m_controled_gates)
	{
		const GateType gate_T = (GateType)(controled_gate.first->getQGate()->getGateType());
		QVec gate_qubits;
		controled_gate.first->getQuBitVector(gate_qubits);
		QVec control_gate_qubits;
		controled_gate.first->getControlVector(control_gate_qubits);

		//get base matrix
		QStat base_gate_mat;
		controled_gate.first->getQGate()->getMatrix(base_gate_mat);

		//build standard controled gate matrix
		std::vector<int> all_gate_qubits_vec;
		for (auto& itr : control_gate_qubits)
		{
			all_gate_qubits_vec.push_back(itr->getPhysicalQubitPtr()->getQubitAddr());
		}
		for (auto& itr : gate_qubits)
		{
			all_gate_qubits_vec.push_back(itr->getPhysicalQubitPtr()->getQubitAddr());
		}

		std::vector<int> tmp_vec = all_gate_qubits_vec;
		sort(tmp_vec.begin(), tmp_vec.end(), [](const int &a, const int &b) {return a < b; });
		tmp_vec.erase(unique(tmp_vec.begin(), tmp_vec.end()), tmp_vec.end());
		if (tmp_vec.size() != all_gate_qubits_vec.size())
		{
			QCERR_AND_THROW_ERRSTR(runtime_error, "Error: Conflict between control qubits and target qubits.");
		}

		int all_control_gate_qubits = all_gate_qubits_vec.size();
		QStat standard_mat;
		build_standard_control_gate_matrix(base_gate_mat, all_control_gate_qubits, standard_mat);

		//tensor
		int idle_qubits = m_qubits_in_use.size() - all_control_gate_qubits;
		QStat tmp_idle_mat;
		if (idle_qubits > 0)
		{
			for (size_t i = 0; i < idle_qubits; i++)
			{
				tensor_by_matrix(tmp_idle_mat, m_mat_I);
			}
			standard_mat = tensor(tmp_idle_mat, standard_mat);
		}

#if PRINT_TRACE
		cout << "tmp_idle_mat:" << endl;
		cout << tmp_idle_mat << endl;

		cout << "tensored standard matrix:" << endl;
		cout << standard_mat << endl;
#endif // PRINT_TRACE

		//swap
		auto iter_after_last_control_qubits = --(all_gate_qubits_vec.end());
		if ((gate_T == ISWAP_THETA_GATE) || (gate_T == ISWAP_GATE) || (gate_T == SQISWAP_GATE) || (gate_T == SWAP_GATE))
		{
			--iter_after_last_control_qubits;

			//sort the target qubits
			sort(iter_after_last_control_qubits, all_gate_qubits_vec.end(), [](const int &a, const int &b) {return a < b; });
	    }
		sort(all_gate_qubits_vec.begin(), iter_after_last_control_qubits, [](const int &a, const int &b) {return a < b; });

		auto used_qubit_iter = m_qubits_in_use.begin() + idle_qubits;
		std::vector<int> used_qubits;
		used_qubits.assign(used_qubit_iter, m_qubits_in_use.end());

		remove_same_control_qubits(used_qubits, all_gate_qubits_vec, iter_after_last_control_qubits - all_gate_qubits_vec.begin());

		auto gate_qubit_item = all_gate_qubits_vec.begin();
		used_qubit_iter = used_qubits.begin();
		for (; gate_qubit_item != all_gate_qubits_vec.end(); ++gate_qubit_item, ++used_qubit_iter)
		{
			const auto& gate_qubit_tmp = *gate_qubit_item;
			const auto& maped_gate_qubit = *used_qubit_iter;
			if (gate_qubit_tmp != maped_gate_qubit)
			{
				swap_two_qubit_on_matrix(standard_mat, controled_gate.second[0], controled_gate.second[1], gate_qubit_tmp, maped_gate_qubit);
				auto tmp_itr = used_qubit_iter;
				for (++tmp_itr; tmp_itr != used_qubits.end(); ++tmp_itr)
				{
					if (gate_qubit_tmp == (*tmp_itr))
					{
						*tmp_itr = maped_gate_qubit;
					}
				}

				*used_qubit_iter = gate_qubit_tmp;
			}
		}

		//merge to current layer matrix directly
		if (m_current_layer_mat.empty())
		{
			m_current_layer_mat = standard_mat;
		}
		else
		{
			m_current_layer_mat = standard_mat * m_current_layer_mat;
		}
	}
}

void QProgToMatrix::MatrixOfOneLayer::remove_same_control_qubits(std::vector<int>& qubits_in_standard_mat, std::vector<int>& gate_qubits, const size_t control_qubits_cnt)
{
	int i = 0;
	int j = 0;
	auto tmp_cnt = control_qubits_cnt;
	while ((i < tmp_cnt) && (j < tmp_cnt))
	{
		if (qubits_in_standard_mat.at(i) == gate_qubits.at(j))
		{
			qubits_in_standard_mat.erase(qubits_in_standard_mat.begin() + i);
			gate_qubits.erase(gate_qubits.begin() + j);
			--tmp_cnt;
		}
		else if (qubits_in_standard_mat.at(i) < gate_qubits.at(j))
		{
			++i;
		}
		else
		{
			++j;
		}
	}
}

void QProgToMatrix::MatrixOfOneLayer::swap_two_qubit_on_matrix(QStat& src_mat, const int mat_qubit_start, const int mat_qubit_end, const int qubit_1, const int qubit_2)
{
	if (qubit_1 == qubit_2)
	{
		return;
	}

	auto machine = initQuantumMachine(QMachineType::CPU);
	auto q = machine->allocateQubits(4);
	auto c = machine->allocateCBits(4);
	auto swap_gate = SWAP(q[0], q[1]);
	QStat swap_gate_matrix;
	swap_gate.getQGate()->getMatrix(swap_gate_matrix);
	destroyQuantumMachine(machine);

	QStat tmp_tensor_mat;
	int tensor_start_qubit = qubit_1 < qubit_2 ? qubit_1 : qubit_2;
	int tensor_end_qubit = qubit_1 < qubit_2 ? qubit_2 : qubit_1;

	for (auto &used_qubit_item : m_qubits_in_use)
	{
		if ((used_qubit_item > tensor_start_qubit) && (used_qubit_item < tensor_end_qubit))
		{
			tensor_by_matrix(tmp_tensor_mat, m_mat_I);
		}
	}

#if PRINT_TRACE
	cout << "tmp_tensor_mat:" << endl;
	cout << tmp_tensor_mat << endl;
#endif // PRINT_TRACE

	//blockMultip
	QStat tensored_swap_gate_matrix;
	if (tmp_tensor_mat.size() > 0)
	{
		blockedMatrix_t blocked_mat;
		partition(swap_gate_matrix, 2, 2, blocked_mat);
		blockMultip(tmp_tensor_mat, blocked_mat, tensored_swap_gate_matrix);
	}
	else
	{
		tensored_swap_gate_matrix.swap(swap_gate_matrix);
	}

	QStat tmp_mat;
	for (auto used_qubit_itr = m_qubits_in_use.begin(); used_qubit_itr != m_qubits_in_use.end(); ++used_qubit_itr)
	{
		const auto& qubit_tmp = *used_qubit_itr;
		if ((qubit_tmp < tensor_start_qubit) || (qubit_tmp > tensor_end_qubit))
		{
			tensor_by_matrix(tmp_mat, m_mat_I);
		}
		else if (qubit_tmp == tensor_start_qubit)
		{
			tensor_by_matrix(tmp_mat, tensored_swap_gate_matrix);
		}
	}

	src_mat = tmp_mat * src_mat * tmp_mat;
}

void QProgToMatrix::MatrixOfOneLayer::build_standard_control_gate_matrix(const QStat& src_mat, const int qubit_number, QStat& result_mat)
{
	size_t rows = 1; // rows of the standard matrix
	size_t columns = 1;// columns of the standard matrix
	for (size_t i = 0; i < qubit_number; i++)
	{
		rows *= 2;
	}
	columns = rows;

	result_mat.resize(rows * columns);

	size_t src_mat_colums = sqrt(src_mat.size());
	size_t src_mat_rows = src_mat_colums;
	size_t item_index = 0;
	for (size_t i = 0; i < rows; ++i)
	{
		for (size_t j = 0; j < columns; ++j)
		{
			item_index = i * rows + j;
			if (((rows - i) <= src_mat_rows) && ((columns - j) <= src_mat_colums))
			{
				result_mat[item_index] = src_mat[(src_mat_rows - (rows - i)) * src_mat_rows + src_mat_colums - (columns - j)];
			}
			else if (i == j)
			{
				result_mat[item_index] = 1;
			}
			else
			{
				result_mat[item_index] = 0;
			}
		}
	}

#if PRINT_TRACE
	cout << result_mat << endl;
#endif // PRINT_TRACE
}

void QProgToMatrix::MatrixOfOneLayer::merge_sing_gate()
{
	if (m_single_qubit_gates.size() > 0)
	{
		QStat all_single_gate_matrix;
		for (auto &in_used_qubit_val : m_qubits_in_use)
		{
			bool b_no_gate_on_this_qubit = true;
			for (auto itr_single_gate = m_single_qubit_gates.begin(); itr_single_gate != m_single_qubit_gates.end();)
			{
				const int qubit_val = itr_single_gate->second.front();
				if (qubit_val == in_used_qubit_val)
				{
					b_no_gate_on_this_qubit = false;
					tensor_by_QGate(all_single_gate_matrix, itr_single_gate->first);

					itr_single_gate = m_single_qubit_gates.erase(itr_single_gate);
					continue;
				}

				++itr_single_gate;
			}

			if (b_no_gate_on_this_qubit)
			{
				tensor_by_matrix(all_single_gate_matrix, m_mat_I);
			}
		}

		if (m_current_layer_mat.empty())
		{
			m_current_layer_mat = all_single_gate_matrix;
		}
		else
		{
			m_current_layer_mat = all_single_gate_matrix * m_current_layer_mat;
		}
	}
}

QStat QProgToMatrix::get_matrix()
{
	QStat result_matrix;

	//get quantumBits number
	get_all_used_qubits(m_prog, m_qubits_in_use);

	//layer
	GraphMatch match;
	TopologicalSequence seq;
	match.get_topological_sequence(m_prog, seq);
	const QProgDAG& prog_dag = match.getProgDAG();
	for (auto &seqItem : seq)
	{
		//each layer
		if (result_matrix.size() == 0)
		{
			result_matrix = get_matrix_of_one_layer(seqItem, prog_dag);
		}
		else
		{
			result_matrix = (get_matrix_of_one_layer(seqItem, prog_dag)) * result_matrix;
		}
	}

	return result_matrix;
}

QStat QProgToMatrix::get_matrix_of_one_layer(SequenceLayer& layer, const QProgDAG& prog_dag)
{
	MatrixOfOneLayer get_one_layer_matrix(layer, prog_dag, m_qubits_in_use);

	get_one_layer_matrix.merge_controled_gate();

	get_one_layer_matrix.merge_double_gate();

	get_one_layer_matrix.merge_calc_unit();

	get_one_layer_matrix.merge_sing_gate();

	return get_one_layer_matrix.m_current_layer_mat;
}
