#pragma once
#include "Core/Utilities/QPandaNamespace.h"
#include <string>
#include <list>
#include <vector>
#include <fstream>
#include "Core/Utilities/QProgToDAG/GraphMatch.h"

QPANDA_BEGIN

class DrawBox
{
public:
	DrawBox(const std::string &topFormatStr, const std::string &midFormatStr, const std::string &botFormatStr)
		:m_top_format(topFormatStr), m_mid_format(midFormatStr), m_bot_format(botFormatStr)
	{}

	virtual const std::string& getTopStr() const {
		return m_top_format;
	}

	virtual const std::string& getMidStr() const {
		return m_mid_format;
	}

	virtual const std::string& getBotStr() const {
		return m_bot_format;
	}

	virtual int getLen() const = 0;

protected:
	std::string m_top_format;
	std::string m_mid_format;
	std::string m_bot_format;
};

class Wire
{
public:
	Wire(const std::string& connectStr)
		:m_connect_str(connectStr), m_cur_len(0), m_b_merged_bot_line(false)
	{}
	virtual ~Wire() {}

	virtual void setName(const std::string& name, size_t nameLen) {
		for (size_t i = 0; i < nameLen; i++)
		{
			m_top_line.append(" ");
			m_bot_line.append(" ");
		}
		m_mid_line.append(name);
		m_cur_len = nameLen;
	}

	virtual int append(const DrawBox& box, const int boxPos) {
		if (boxPos > m_cur_len)
		{
			for (size_t i = m_cur_len; i < boxPos; i++)
			{
				m_top_line.append(" ");
				m_mid_line.append(m_connect_str);
				m_bot_line.append(" ");
				++m_cur_len;
			}
		}

		m_top_line.append(box.getTopStr());
		m_mid_line.append(box.getMidStr());
		m_bot_line.append(box.getBotStr());

		m_cur_len += box.getLen();
		return (boxPos + box.getLen());
	}

	virtual int getWireLength() { return m_cur_len; }

	virtual int draw(std::ofstream& fd, int srartRow) {
		fd << m_top_line;
		std::cout << m_top_line << std::endl;
		fd << "\n";

		fd << m_mid_line.data();
		std::cout << m_mid_line << std::endl;
		fd << "\n";

		if (!m_b_merged_bot_line)
		{
			fd << m_bot_line;
			std::cout << m_bot_line << std::endl;
			fd << "\n";
			return srartRow + 3;
		}

		return srartRow + 2;
	}

	virtual void updateWireLen(const int len) {
		for (size_t i = m_cur_len; i < len; i++)
		{
			m_top_line.append(" ");
			m_mid_line.append(m_connect_str);
			m_bot_line.append(" ");
		}

		m_cur_len = len;
	}

	virtual void setMergedFlag(bool b) { m_b_merged_bot_line = b; }

	virtual const std::string& getTopLine() const { return m_top_line; }

	virtual const std::string& getMidLine() const { return m_mid_line; }

	virtual const std::string& getBotLine() const { return m_bot_line; }

protected:
	const std::string m_connect_str;
	std::string m_top_line;
	std::string m_mid_line;
	std::string m_bot_line;
	int m_cur_len;
	bool m_b_merged_bot_line;
};

class TextPic
{
	using wireElement = std::pair<int, std::shared_ptr<Wire>>;
	using wireIter = std::map<int, std::shared_ptr<Wire>>::iterator;

public:
	TextPic(TopologincalSequence &layerInfo, const QProgDAG& progDag);
	~TextPic();

	void init(std::vector<int>& quBits, std::vector<int>& clBits);
	void drawBack();
	void present();
	void mergeLine();
	int getMaxQuWireLength(wireIter startQuBitWire, wireIter endQuBitWire);
	void updateTextPicLen();
	unsigned long getWideCharVal(const unsigned char* wideChar) {
		if (nullptr == wideChar)
		{
			return 0;
		}
		return (((unsigned long)(wideChar[0]) << 16) | ((unsigned long)(wideChar[1]) << 8) | (((unsigned long)(wideChar[2]))));
	}

protected:
	void appendMeasure(std::shared_ptr<AbstractQuantumMeasure> pMeasure);
	void appendCtrlGate(std::string gateName, QVec &qubitsVector);
	void appendSwapGate(std::string gateName, QVec &qubitsVector);
	void merge(const std::string& upWire, std::string& downWire);

private:
	std::map<int, std::shared_ptr<Wire>> m_quantum_bit_wires;
	std::map<int, std::shared_ptr<Wire>> m_class_bit_wires;
	TopologincalSequence &m_layer_info;
	const QProgDAG& m_prog_dag;
	int m_text_len;
};

QPANDA_END