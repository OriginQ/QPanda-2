
#ifndef  _SEARCH_DATA_TYPE_H
#define  _SEARCH_DATA_TYPE_H

QPANDA_BEGIN

#define PRINT_TRACE 0
#if PRINT_TRACE
#define PTrace printf
#define PTraceMat(mat) (std::cout << (mat) << endl)
#define PTraceQCircuit(string, cir) (std::cout << string << endl << cir << endl)
#else
#define PTrace
#define PTraceMat(mat)
#define PTraceMat(string, cir)
#endif

class AbstractSearchData
{
public:
	virtual bool operator < (const AbstractSearchData &other) const = 0;
	virtual bool operator <= (const AbstractSearchData &other) const = 0;
	virtual bool operator > (const AbstractSearchData &other) const = 0;
	virtual bool operator >= (const AbstractSearchData &other) const = 0;
	virtual bool operator == (const AbstractSearchData &&other) const = 0;
	virtual AbstractSearchData& operator - (const AbstractSearchData &other) = 0;
	virtual QCircuit build_to_circuit(QVec &used_qubits, size_t use_qubit_cnt, const AbstractSearchData &mini_data) const = 0;
	virtual QCircuit build_to_condition_circuit(QVec &used_qubits, QCircuit cir_mark, const AbstractSearchData &mini_data) = 0;
	virtual size_t check_max_need_qubits() = 0;
	virtual AbstractSearchData& set_val(const char* p_val) = 0;
};

class SearchDataByUInt : public AbstractSearchData
{
public:
	SearchDataByUInt(unsigned int val = 0)
		:m_data(val)
	{}

	bool operator < (const AbstractSearchData &other) const {
		return m_data < (dynamic_cast<const SearchDataByUInt&>(other)).m_data;
	}

	bool operator <= (const AbstractSearchData &other) const {
		return m_data <= (dynamic_cast<const SearchDataByUInt&>(other)).m_data;
	}

	bool operator > (const AbstractSearchData &other) const {
		return m_data > (dynamic_cast<const SearchDataByUInt&>(other)).m_data;
	}

	bool operator >= (const AbstractSearchData &other) const {
		return m_data >= (dynamic_cast<const SearchDataByUInt&>(other)).m_data;
	}

	bool operator == (const AbstractSearchData &&other) const {
		return m_data == (dynamic_cast<const SearchDataByUInt&>(other)).m_data;
	}

	AbstractSearchData& operator - (const AbstractSearchData &other) override {
		m_data -= (dynamic_cast<const SearchDataByUInt&>(other)).m_data;
		return *this;
	}

	SearchDataByUInt& operator = (const AbstractSearchData &other) {
		m_data = (dynamic_cast<const SearchDataByUInt&>(other)).m_data;
		return *this;
	}

	size_t check_max_need_qubits() override {
		size_t cnt = 1;
		unsigned int temp_data = m_data + 1;
		for (;0 != (temp_data /= 2); ++cnt){}

		return cnt;
	}

	AbstractSearchData& set_val(const char* p_val) override {
		m_data = atol(p_val);
		return *this;
	}

	QCircuit build_to_circuit(QVec &oracle_qubits, size_t use_qubit_cnt, const AbstractSearchData &mini_data) const override {
		unsigned int weight_data = m_data - (dynamic_cast<const SearchDataByUInt&>(mini_data)).m_data + 1;
		QCircuit ret_cir;
		for (size_t i = 0; i < use_qubit_cnt; ++i)
		{
			if (0 != weight_data % 2)
			{
				ret_cir << X(oracle_qubits[i]);
			}

			weight_data /= 2;
		}

		return ret_cir;
	}

	QCircuit build_to_condition_circuit(QVec &oracle_qubits, QCircuit cir_mark, const AbstractSearchData &mini_data) override {
		int weight_data = m_data - (dynamic_cast<const SearchDataByUInt&>(mini_data)).m_data + 1;
		
		QCircuit ret_cir;
		if ((weight_data < 1) || (weight_data >= pow(2, oracle_qubits.size())))
		{
			return ret_cir;
		}
		
		//auto ancilla_gate = X(ancilla_qubit);
		cir_mark.setControl(oracle_qubits);

		QCircuit search_cir;
		for (size_t i = 0; i < oracle_qubits.size(); ++i)
		{
			if (0 == weight_data % 2)
			{
				search_cir << X(oracle_qubits[i]);
			}

			weight_data /= 2;
		}

		ret_cir << search_cir << cir_mark << search_cir;
		//PTraceQCircuit("ret_cir", ret_cir);
		return ret_cir;
	}

private:
	unsigned int m_data;
};

QPANDA_END

#endif