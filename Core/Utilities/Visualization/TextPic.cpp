#include "Core/Utilities/Visualization/TextPic.h"
#include "TranformQGateTypeStringAndEnum.h"
#include <iostream>
#include <codecvt>
#include <locale>
#include "Core/Utilities/Visualization/charsTransform.h"

USING_QPANDA
using namespace std;

#define WIRE_HEAD_LEN 6
#define OUTPUT_TMP_FILE ("QCircuitTextPic.txt")

#define BOX_RIGHT_CONNECT_CHAR              0XE2949C     /* UNICODE CHAR:    ©À    */
#define BOX_LEFT_CONNECT_CHAR               0XE294A4     /* UNICODE CHAR:    ©È    */
#define BOX_UP_CONNECT_CHAR                 0XE294B4     /* UNICODE CHAR:    ©Ø    */
#define BOX_DOWN_CONNECT_CHAR               0XE294AC     /* UNICODE CHAR:    ©Ð    */
#define BOX_LEFT_DOUBLE_CONNECT_CHAR        0XE295A1     /* UNICODE CHAR:    ¨e    */
#define BOX_RIGHT_DOUBLE_CONNECT_CHAR       0XE2959E     /* UNICODE CHAR:    ¨b    */
#define BOX_DOWN_DOUBLE_CONNECT_CHAR        0XE295A5     /* UNICODE CHAR:    ¨i    */
#define BOX_LEFT_TOP_CHAR                   0XE2948C     /* UNICODE CHAR:    ©°    */
#define BOX_RIGHT_TOP_CHAR                  0XE29490     /* UNICODE CHAR:    ©´    */
#define BOX_LEFT_BOTTOM_CHAR                0XE29494     /* UNICODE CHAR:    ©¸    */
#define BOX_RIGHT_BOTTOM_CHAR               0XE29498     /* UNICODE CHAR:    ©¼    */

#define SINGLE_VERTICAL_LINE                0XE29482     /* UNICODE CHAR:    ©¦    */
#define DOUBLE_VERTICAL_LINE                0XE29591     /* UNICODE CHAR:    ¨U    */
#define SINGLE_HORIZONTAL_LINE              0XE29480     /* UNICODE CHAR:    ©¤    */
#define DOUBLE_HORIZONTAL_LINE              0XE29590     /* UNICODE CHAR:    ¨T    */
#define CROSS_CHAR                          0XE294BC     /* UNICODE CHAR:    ©à    */
#define DOUBLE_CROSS_CHAR                   0XE295AC     /* UNICODE CHAR:    ¨p    */
#define BLACK_SQUARE_CHAR                   0XE296A0     /* UNICODE CHAR:    ¡ö    */    
#define DOUBLE_LINE_UP_CONNECT_CHAR         0XE295A9     /* UNICODE CHAR:    ¨m    */
#define SINGLE_LINE_ACROSS_DOUBLE_LINE      0XE295AB     /* UNICODE CHAR:    ¨o    */
#define DOUBLE_LINE_ACROSS_SINGLE_LINE      0XE295AA     /* UNICODE CHAR:    ¨n    */

class MeasureTo : public DrawBox
{
public:
	MeasureTo()
		:DrawBox(
			std::string(" ") + ulongToUtf8(DOUBLE_VERTICAL_LINE) + std::string(" "),
			ulongToUtf8(DOUBLE_HORIZONTAL_LINE) + ulongToUtf8(DOUBLE_LINE_UP_CONNECT_CHAR) + ulongToUtf8(DOUBLE_HORIZONTAL_LINE),
			std::string("   "))
	{}
	~MeasureTo() {}

	int getLen() const { return 3; }

private:
};

class MeasureFrom : public DrawBox
{
public:
	MeasureFrom()
		:DrawBox(
			/*std::string("©°©¤©´") + */ulongToUtf8(BOX_LEFT_TOP_CHAR) + ulongToUtf8(SINGLE_HORIZONTAL_LINE) + ulongToUtf8(BOX_RIGHT_TOP_CHAR),
			/*std::string("©ÈM©À")*/ulongToUtf8(BOX_LEFT_CONNECT_CHAR) + std::string("M") + ulongToUtf8(BOX_RIGHT_CONNECT_CHAR),
			/*std::string("©¸¨i©¼")*/ulongToUtf8(BOX_LEFT_BOTTOM_CHAR) + ulongToUtf8(BOX_DOWN_DOUBLE_CONNECT_CHAR) + ulongToUtf8(BOX_RIGHT_BOTTOM_CHAR))
	{}
	~MeasureFrom() {}

	int getLen() const { return 3; }

private:
};

class SwapFrom : public DrawBox
{
public:
	SwapFrom()
		:DrawBox(
			std::string(" "),
			std::string("X"),
			/*std::string("©¦")*/ulongToUtf8(SINGLE_VERTICAL_LINE))
	{}

	int getLen() const { return 1; }

private:
};

class SwapTo : public DrawBox
{
public:
	SwapTo()
		:DrawBox(
			/*std::string("©¦")*/ulongToUtf8(SINGLE_VERTICAL_LINE),
			std::string("X"),
			std::string(" "))
	{}

	int getLen() const { return 1; }

private:
};

class ControlQuBit : public DrawBox
{
public:
	ControlQuBit()
		:DrawBox(
			std::string(" "),
			/*std::string("¡ö")*/ulongToUtf8(BLACK_SQUARE_CHAR),
			std::string(" "))
	{}
	~ControlQuBit() {}

	int getLen() const { return 1; }

	void setTopConnected() {
		m_top_format = /*std::string("©¦")*/ulongToUtf8(SINGLE_VERTICAL_LINE);
	}

	void setBotConnected() {
		m_bot_format = /*std::string("©¦")*/ulongToUtf8(SINGLE_VERTICAL_LINE);
	}

private:
};

class ControlLine : public DrawBox
{
public:
	ControlLine()
		:DrawBox(
			/*std::string("©¦")*/ulongToUtf8(SINGLE_VERTICAL_LINE),
			/*std::string("©à")*/ulongToUtf8(CROSS_CHAR),
			/*std::string("©¦")*/ulongToUtf8(SINGLE_VERTICAL_LINE))
	{}

	int getLen() const { return 1; }

private:
};

class MeasureLine : public DrawBox
{
public:
	MeasureLine(const std::string& midFormatStr)
		:DrawBox(
			/*std::string("¨U")*/ulongToUtf8(DOUBLE_VERTICAL_LINE),
			std::string(midFormatStr),
			/*std::string("¨U")*/ulongToUtf8(DOUBLE_VERTICAL_LINE))
	{}

	static const std::string getMeasureLineCrossClWire() { return ulongToUtf8(DOUBLE_CROSS_CHAR); }
	static const std::string getMeasureLineCrossQuWire() { return ulongToUtf8(SINGLE_LINE_ACROSS_DOUBLE_LINE); }

	int getLen() const { return 1; }

private:
};

class BoxOnWire : public DrawBox
{
public:
	BoxOnWire(const std::string &topFormatStr, const std::string &midFormatStr, const std::string &botFormatStr, const std::string &padStr)
		:DrawBox(
			std::string(topFormatStr),
			std::string(midFormatStr),
			std::string(botFormatStr))
		, m_pad_str(padStr), m_len(0)
	{}

	void setName(const std::string& name) {
		std::string pad_str;
		for (size_t i = 0; i < name.size(); i++)
		{
			pad_str.append(m_pad_str);
		}
		char buf[128] = "";
		sprintf(buf, m_top_format.c_str(), pad_str.c_str());
		m_top_format = buf;

		sprintf(buf, m_mid_format.c_str(), name.c_str());
		m_mid_format = buf;

		sprintf(buf, m_bot_format.c_str(), pad_str.c_str());
		m_bot_format = buf;

		m_len = name.size() + 2; // 2 = sizeof("©°©´") / sizeof("©°")
	}

	virtual int getLen() const { return m_len; }

	virtual void setStr(std::string &targetStr, const int pos, const std::string str) {
		int char_start_pos = pos * 3; //because the size of utf8-char is 3 Bytes
		for (size_t i = 0; i < str.length(); i++)
		{
			targetStr.at(char_start_pos + i) = str.at(i);
		}
	}

protected:
	const std::string m_pad_str;
	int m_len;
};

class BoxOnClWire : public BoxOnWire
{
public:
	BoxOnClWire(const std::string& name)
		:BoxOnWire(
			/*std::string("©°%s©´")*/ulongToUtf8(BOX_LEFT_TOP_CHAR) + std::string("%s") + ulongToUtf8(BOX_RIGHT_TOP_CHAR),
			/*std::string("¨e%s¨b")*/ulongToUtf8(BOX_LEFT_DOUBLE_CONNECT_CHAR) + std::string("%s") + ulongToUtf8(BOX_RIGHT_DOUBLE_CONNECT_CHAR),
			/*std::string("©¸%s©¼")*/ulongToUtf8(BOX_LEFT_BOTTOM_CHAR) + std::string("%s") + ulongToUtf8(BOX_RIGHT_BOTTOM_CHAR),
			/*std::string("©¤")*/ulongToUtf8(SINGLE_HORIZONTAL_LINE))
		, m_name(name)
	{
		setName(m_name);
	}
	~BoxOnClWire() {}

private:
	const std::string &m_name;
};

class BoxOnQuWire : public BoxOnWire
{
public:
	BoxOnQuWire(const std::string& name)
		:BoxOnWire(
			/*std::string("©°%s©´")*/ulongToUtf8(BOX_LEFT_TOP_CHAR) + std::string("%s") + ulongToUtf8(BOX_RIGHT_TOP_CHAR),
			/*std::string("©È%s©À")*/ulongToUtf8(BOX_LEFT_CONNECT_CHAR) + std::string("%s") + ulongToUtf8(BOX_RIGHT_CONNECT_CHAR),
			/*std::string("©¸%s©¼"*/ulongToUtf8(BOX_LEFT_BOTTOM_CHAR) + std::string("%s") + ulongToUtf8(BOX_RIGHT_BOTTOM_CHAR),
			/*std::string("©¤")*/ulongToUtf8(SINGLE_HORIZONTAL_LINE))
		, m_name(name)
		, m_top_connector(ulongToUtf8(BOX_UP_CONNECT_CHAR))
		, m_bot_connector(ulongToUtf8(BOX_DOWN_CONNECT_CHAR))
	{
		setName(m_name);
	}
	~BoxOnQuWire() {}

	void setTopConnected() {
		setStr(m_top_format, m_len / 2, m_top_connector);
	}

	void setBotConnected() {
		setStr(m_bot_format, m_len / 2, m_bot_connector);
	}

private:
	const std::string m_top_connector;
	const std::string m_bot_connector;
	const std::string &m_name;
};

class ClassWire : public Wire
{
public:
	ClassWire()
		:Wire(ulongToUtf8(DOUBLE_HORIZONTAL_LINE))
	{}
	~ClassWire() {}

private:
};

class QuantumWire : public Wire
{
public:
	QuantumWire()
		:Wire(ulongToUtf8(SINGLE_HORIZONTAL_LINE))
	{}
	~QuantumWire() {}

private:
};

TextPic::TextPic(TopologincalSequence &layerInfo, const QProgDAG& progDag)
	:m_layer_info(layerInfo), m_prog_dag(progDag), m_text_len(0)
{
}

TextPic::~TextPic()
{
}

void TextPic::appendMeasure(std::shared_ptr<AbstractQuantumMeasure> pMeasure)
{
	int qubit_index = pMeasure->getQuBit()->getPhysicalQubitPtr()->getQubitAddr();
	int c_bit_index = pMeasure->getCBit()->getValue();
	
	auto start_quBit = m_quantum_bit_wires.find(qubit_index);
	auto end_quBit = m_quantum_bit_wires.end();
	int append_pos = getMaxQuWireLength(start_quBit, end_quBit);

	MeasureFrom box_measure_from;
	append_pos = start_quBit->second->append(box_measure_from, append_pos);
	

	MeasureTo box_measure_to;
	m_class_bit_wires[c_bit_index]->append(box_measure_to, (append_pos - (box_measure_to.getLen())));

	MeasureLine measure_line_on_qu_wire(MeasureLine::getMeasureLineCrossQuWire());
	int offset = (box_measure_from.getLen() - measure_line_on_qu_wire.getLen()) / 2 + measure_line_on_qu_wire.getLen();
	for (auto itr = ++start_quBit; itr != m_quantum_bit_wires.end(); itr++)
	{
		itr->second->append(measure_line_on_qu_wire, (append_pos - offset));
	}

	MeasureLine measure_line_on_cl_wire(MeasureLine::getMeasureLineCrossClWire());
	offset = (box_measure_from.getLen() - measure_line_on_cl_wire.getLen()) / 2 + measure_line_on_cl_wire.getLen();
	for (size_t i = 0; i < c_bit_index; i++)
	{
		if (m_class_bit_wires.find(i) != m_class_bit_wires.end())
		{
			m_class_bit_wires[i]->append(measure_line_on_cl_wire, (append_pos - (offset)));
		}
	}
}

void TextPic::appendCtrlGate(string gateName, QVec &qubitsVector)
{
	int ctrl_qubit_index = qubitsVector.front()->getPhysicalQubitPtr()->getQubitAddr();
	int target_qubit_index = qubitsVector.back()->getPhysicalQubitPtr()->getQubitAddr();

	//find the max length of QuWires between control QuBit and target QuBit
	auto start_quBit = ctrl_qubit_index < target_qubit_index ? ctrl_qubit_index : target_qubit_index;
	auto end_quBit = ctrl_qubit_index < target_qubit_index ? target_qubit_index : ctrl_qubit_index;
	int append_pos = getMaxQuWireLength(m_quantum_bit_wires.find(start_quBit), ++(m_quantum_bit_wires.find(end_quBit)));

	//append target qubit
	BoxOnQuWire quTargetBox(gateName);
	if (ctrl_qubit_index < target_qubit_index)
	{
		quTargetBox.setTopConnected();
	}
	else
	{
		quTargetBox.setBotConnected();
	}

	append_pos = m_quantum_bit_wires.at(target_qubit_index)->append(quTargetBox, append_pos);

	//append control qubit
	ControlQuBit quControlBox;
	if (ctrl_qubit_index > target_qubit_index)
	{
		quControlBox.setTopConnected();
	}
	else
	{
		quControlBox.setBotConnected();
	}
	m_quantum_bit_wires.at(ctrl_qubit_index)->append(quControlBox, (append_pos - (quTargetBox.getLen() / 2)));

	//append qubits between control and target qubit
	ControlLine ctr_line;
	for (size_t i = start_quBit + 1; i < end_quBit; i++)
	{
		if (m_quantum_bit_wires.find(i) != m_quantum_bit_wires.end())
		{
			m_quantum_bit_wires[i]->append(ctr_line, (append_pos - (quTargetBox.getLen() / 2)));
		}
	}
}

void TextPic::appendSwapGate(string gateName, QVec &qubitsVector)
{
	int swap_from_qubit_index = qubitsVector.front()->getPhysicalQubitPtr()->getQubitAddr();
	int swap_to_qubit_index = qubitsVector.back()->getPhysicalQubitPtr()->getQubitAddr();

	//find the max length of QuWires between control QuBit and target QuBit
	auto start_quBit = swap_from_qubit_index < swap_to_qubit_index ? swap_from_qubit_index : swap_to_qubit_index;
	auto end_quBit = swap_from_qubit_index < swap_to_qubit_index ? swap_to_qubit_index : swap_from_qubit_index;
	int append_pos = getMaxQuWireLength(m_quantum_bit_wires.find(start_quBit), ++(m_quantum_bit_wires.find(end_quBit)));

	SwapFrom swap_from;
	append_pos = m_quantum_bit_wires[start_quBit]->append(swap_from, append_pos);

	SwapTo swap_to;
	m_quantum_bit_wires[end_quBit]->append(swap_to, append_pos - swap_to.getLen());

	//append qubits between control and target qubit
	ControlLine ctr_line;
	for (size_t i = start_quBit + 1; i < end_quBit; i++)
	{
		if (m_quantum_bit_wires.find(i) != m_quantum_bit_wires.end())
		{
			m_quantum_bit_wires[i]->append(ctr_line, (append_pos - (swap_from.getLen())));
		}
	}
}

void TextPic::drawBack()
{
	for (auto &seq_item : m_layer_info)
	{
		for (auto &seq_node_item : seq_item)
		{
			SequenceNode n = seq_node_item.first;
			if (-1 == n.m_node_type)
			{
				std::shared_ptr<AbstractQuantumMeasure> p_measure = dynamic_pointer_cast<AbstractQuantumMeasure>(m_prog_dag.getVertex(n.m_vertex_num));
				appendMeasure(p_measure);
			}
			else
			{
				QVec qubits_vector;
				string gate_name;
				std::shared_ptr<AbstractQGateNode> p_gate = dynamic_pointer_cast<AbstractQGateNode>(m_prog_dag.getVertex(n.m_vertex_num));
				p_gate->getQuBitVector(qubits_vector);
				gate_name = TransformQGateType::getInstance()[(GateType)(n.m_node_type)];

				if (1 == qubits_vector.size())
				{
					// single gate
					int qubit_index = qubits_vector.front()->getPhysicalQubitPtr()->getQubitAddr();
					BoxOnQuWire quBox(gate_name);
					m_quantum_bit_wires[qubit_index]->append(quBox, 0);
				}
				else if (2 == qubits_vector.size())
				{
					//double gate
					switch ((GateType)(n.m_node_type))
					{
					case ISWAP_THETA_GATE:
					case ISWAP_GATE:
					case SQISWAP_GATE:
					case SWAP_GATE:
						appendSwapGate(gate_name, qubits_vector);
						break;

					case CU_GATE:
					case CNOT_GATE:
					case CZ_GATE:
					case CPHASE_GATE:
						appendCtrlGate(gate_name, qubits_vector);
						break;

					default:
						break;
					}
					
				}
				else
				{
					//other gate type
				}
			}
		}

		//update m_text_len
		updateTextPicLen();
	}

	//merge line
	mergeLine();
}

int TextPic::getMaxQuWireLength(wireIter startQuBitWire, wireIter endQuBitWire)
{
	int max_length = -1;
	int tmp_length = 0;
	for (auto itr = startQuBitWire; itr != endQuBitWire; ++itr)
	{
		tmp_length = itr->second->getWireLength();
		if (tmp_length > max_length)
		{
			max_length = tmp_length;
		}
	}

	return max_length;
}

void TextPic::updateTextPicLen()
{
	auto max_len = getMaxQuWireLength(m_quantum_bit_wires.begin(), m_quantum_bit_wires.end());
	for (auto &itr : m_quantum_bit_wires)
	{
		itr.second->updateWireLen(max_len);
	}

	m_text_len = max_len;
}

void TextPic::present()
{
	// init console first
	initConsole();

	/* write to file */
	ofstream outfile(OUTPUT_TMP_FILE, ios::out | ios::binary);
	if (!outfile.is_open())
	{
		throw runtime_error("Can NOT open the output file");
	}

	for (auto &itr : m_quantum_bit_wires)
	{
		itr.second->draw(outfile, 0);
	}

	for (auto &itr : m_class_bit_wires)
	{
		itr.second->draw(outfile, 0);
	}

	outfile.close();
}

void TextPic::init(std::vector<int>& quBits, std::vector<int>& clBits)
{
	const std::string quantum_wire_pad = string("|0>") + ulongToUtf8(SINGLE_HORIZONTAL_LINE);
	const std::string class_wire_pad = string(" 0 ") + ulongToUtf8(DOUBLE_HORIZONTAL_LINE);
	char head_buf[WIRE_HEAD_LEN + 2] = "";
	for (auto i : quBits)
	{
		auto p = std::make_shared<QuantumWire>();
		sprintf(head_buf, "q_%d:", i);
		for (size_t j = strlen(head_buf); j < WIRE_HEAD_LEN; j++)
		{
			head_buf[j] = ' ';
		}

		string name = string(head_buf) + quantum_wire_pad;
		p->setName(name, name.size() -2); //because the size of utf8-char is 3 Bytes
		m_quantum_bit_wires.insert(wireElement(i, p));
	}

	memset(head_buf, 0, sizeof(head_buf));
	for (auto i : clBits)
	{
		auto p = std::make_shared<ClassWire>();
		sprintf(head_buf, " c_%d:", i);
		for (size_t j = strlen(head_buf); j < WIRE_HEAD_LEN; j++)
		{
			head_buf[j] = ' ';
		}

		string name = string(head_buf) + class_wire_pad;
		p->setName(name, name.size() -2);//because the size of utf8-char is 3 Bytes
		m_class_bit_wires.insert(wireElement(i, p));
	}

	m_text_len = m_quantum_bit_wires[0]->getWireLength();
}

void TextPic::mergeLine()
{
	std::shared_ptr<Wire> upside_wire = m_quantum_bit_wires.begin()->second;
	for (auto downside_wire = ++(m_quantum_bit_wires.begin()); downside_wire != m_quantum_bit_wires.end(); ++downside_wire)
	{
		merge(upside_wire->getBotLine(), const_cast<std::string&>((downside_wire->second)->getTopLine()));
		upside_wire->setMergedFlag(true);
		upside_wire = (downside_wire->second);
	}

	for (auto downside_wire = m_class_bit_wires.begin(); downside_wire != m_class_bit_wires.end(); ++downside_wire)
	{
		merge(upside_wire->getBotLine(), const_cast<std::string&>((downside_wire->second)->getTopLine()));
		upside_wire->setMergedFlag(true);
		upside_wire = (downside_wire->second);
	}
}

void TextPic::merge(const std::string& upWire, std::string& downWire)
{
	const char* p_upside_str = upWire.c_str();
	const char* p_downside_str = /*const_cast<char*>*/(downWire.c_str());
	unsigned long upside_wide_char = 0;
	unsigned long downside_wide_char = 0;
	std::string tmp_str;
	char wide_char_buf[4] = "";
	size_t upside_char_index = 0;
	size_t downside_char_index = 0;

	for (size_t upside_char_index = 0; upside_char_index < upWire.size(); ++upside_char_index, ++downside_char_index)
	{
		if (downWire.size() == downside_char_index)
		{
			tmp_str.append(p_upside_str + upside_char_index);
			break;
		}

		if ((p_upside_str[upside_char_index] == ' ') && (' ' == p_downside_str[downside_char_index]))
		{
			tmp_str.append(" ");
		}
		else if (p_upside_str[upside_char_index] == ' ')
		{
			wide_char_buf[0] = p_downside_str[downside_char_index];
			wide_char_buf[1] = p_downside_str[++downside_char_index];
			wide_char_buf[2] = p_downside_str[++downside_char_index];
			tmp_str.append(wide_char_buf);
			continue;
		}
		else if (' ' == p_downside_str[downside_char_index])
		{
			wide_char_buf[0] = p_upside_str[upside_char_index];
			wide_char_buf[1] = p_upside_str[++upside_char_index];
			wide_char_buf[2] = p_upside_str[++upside_char_index];
			tmp_str.append(wide_char_buf);
			continue;
		}
		else
		{
			upside_wide_char = getWideCharVal((unsigned char*)p_upside_str + upside_char_index);
			downside_wide_char = getWideCharVal((unsigned char*)p_downside_str + downside_char_index);

			if (upside_wide_char == downside_wide_char)
			{
				wide_char_buf[0] = p_downside_str[downside_char_index];
				wide_char_buf[1] = p_downside_str[++downside_char_index];
				wide_char_buf[2] = p_downside_str[++downside_char_index];
				upside_char_index += 2;
				tmp_str.append(wide_char_buf);
				continue;
			}

			switch (upside_wide_char)
			{
			case BOX_LEFT_BOTTOM_CHAR:
			{
				if (BOX_LEFT_TOP_CHAR == downside_wide_char)
				{
					tmp_str.append(ulongToUtf8(BOX_RIGHT_CONNECT_CHAR));
				}
				else if (BOX_RIGHT_TOP_CHAR == downside_wide_char)
				{
					tmp_str.append(ulongToUtf8(CROSS_CHAR));
				}
				else if (SINGLE_HORIZONTAL_LINE == downside_wide_char)
				{
					tmp_str.append(ulongToUtf8(BOX_UP_CONNECT_CHAR));
				}
			}
				break;

			case BOX_RIGHT_BOTTOM_CHAR:
			{
				if (BOX_RIGHT_TOP_CHAR == downside_wide_char)
				{
					tmp_str.append(ulongToUtf8(BOX_LEFT_CONNECT_CHAR));
				}
				else if (BOX_LEFT_TOP_CHAR == downside_wide_char)
				{
					tmp_str.append(ulongToUtf8(CROSS_CHAR));
				}
				else if (SINGLE_HORIZONTAL_LINE == downside_wide_char)
				{
					tmp_str.append(ulongToUtf8(BOX_UP_CONNECT_CHAR));
				}
			}
				break;

			case SINGLE_VERTICAL_LINE:
			{
				if (BOX_UP_CONNECT_CHAR == downside_wide_char)
				{
					tmp_str.append(ulongToUtf8(BOX_UP_CONNECT_CHAR));
				}
			}
				break;

			case BOX_DOWN_CONNECT_CHAR:
			{
				if (SINGLE_VERTICAL_LINE == downside_wide_char)
				{
					tmp_str.append(ulongToUtf8(BOX_DOWN_CONNECT_CHAR));
				}
			}
				break;

			case SINGLE_HORIZONTAL_LINE:
			{
				if (BOX_RIGHT_TOP_CHAR == downside_wide_char)
				{
					tmp_str.append(ulongToUtf8(BOX_DOWN_CONNECT_CHAR));
				}
			}
					break;

			case BOX_DOWN_DOUBLE_CONNECT_CHAR:
			{
				if (DOUBLE_VERTICAL_LINE == downside_wide_char)
				{
					tmp_str.append(ulongToUtf8(BOX_DOWN_DOUBLE_CONNECT_CHAR));
				}
			}
				break;

			default:
				tmp_str.append("@"); //This symbol indicates that an error occurred
				break;
			}

			upside_char_index += 2;
			downside_char_index += 2;
		}
	}

	if (downWire.size() > downside_char_index)
	{
		tmp_str.append(p_downside_str + downside_char_index);
	}

	downWire = tmp_str;
}

