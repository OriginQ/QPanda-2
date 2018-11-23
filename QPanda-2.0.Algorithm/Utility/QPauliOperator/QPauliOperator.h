/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QPauliOperator.h

Author: LiYe
Created in 2018-09-20


*/

#ifndef QPAULIOPERATOR_H
#define QPAULIOPERATOR_H

#include <complex>
#include <ostream>
#include "../QAlgDataStruct.h"
#include "Utility/QString.h"

namespace QPanda
{
	/*
	Hamiltonian expressed in Pauli Operator
	*/
	class QPauliOperator
	{
	public:
		QPauliOperator(const QPauliMap &map);
		QPauliOperator(const QHamiltonian &hamiltonian);
		QPauliOperator(QPauliOperator &&op);
		QPauliOperator(const QPauliOperator &op);
		QPauliOperator(QPauli &&pauli);
		QPauliOperator(const QPauli &pauli);
		QPauliOperator &operator = (const QPauliOperator &op);
		QPauliOperator &operator = (QPauliOperator &&op);

		size_t getMaxIndex() { return m_max_index;  }
		QIndexMap getIndexMap() { return m_index_map; }

		bool isAllPauliZorI();

		void setErrorThreshold(double threshold) 
		{ 
			m_error_threshold = threshold; 
		}

		QHamiltonian toHamiltonian(bool *ok = nullptr) const;
		std::string  toString() const;

		const QPauli &data() const { return m_data; }

		QPauliOperator  operator + (const complex_d &rhs) const;
		QPauliOperator  operator - (const complex_d &rhs) const;
		QPauliOperator  operator * (const complex_d &rhs) const;
		QPauliOperator &operator +=(const complex_d &rhs);
		QPauliOperator &operator -=(const complex_d &rhs);
		QPauliOperator &operator *=(const complex_d &rhs);
		QPauliOperator  operator + (const QPauliOperator &rhs) const;
		QPauliOperator  operator - (const QPauliOperator &rhs) const;
		QPauliOperator  operator * (const QPauliOperator &rhs) const;
		QPauliOperator &operator +=(const QPauliOperator &rhs);
		QPauliOperator &operator -=(const QPauliOperator &rhs);
		QPauliOperator &operator *=(const QPauliOperator &rhs);

		QPauliOperator  operator + (const QHamiltonian &rhs) const;
		QPauliOperator  operator - (const QHamiltonian &rhs) const;
		QPauliOperator  operator * (const QHamiltonian &rhs) const;
		QPauliOperator &operator +=(const QHamiltonian &rhs);
		QPauliOperator &operator -=(const QHamiltonian &rhs);
		QPauliOperator &operator *=(const QHamiltonian &rhs);

		friend QPauliOperator operator + (const complex_d &lhs,
			                              const QPauliOperator &rhs);
		friend QPauliOperator operator - (const complex_d &lhs, 
			                              const QPauliOperator &rhs);
		friend QPauliOperator operator * (const complex_d &lhs, 
			                              const QPauliOperator &rhs);
		friend QPauliOperator operator + (const QHamiltonian &lhs, 
			                              const QPauliOperator &rhs);
		friend QPauliOperator operator - (const QHamiltonian &lhs, 
			                              const QPauliOperator &rhs);
		friend QPauliOperator operator * (const QHamiltonian &lhs, 
			                              const QPauliOperator &rhs);
		friend std::ostream  &operator <<(std::ostream &out,
			                              const QPauliOperator &rhs);
		friend std::ostream  &operator <<(std::ostream &out,
			                              const QHamiltonian &rhs);
		friend std::ostream  &operator <<(std::ostream &out,
			                              const QPauliMap &rhs);
		friend std::ostream  &operator <<(std::ostream &out,
			                              const QPauli &rhs);
	private:
		QTermPair   genQTermPair(const QString &str) const;
		QPauliItem  genQPauliItem(const QTerm &map_i,
			                      const QTerm &map_j, 
			                      const complex_d &value) const;
		std::string QTerm2StdString(const QTerm &map) const;

		void reduceDuplicates();
		void remapQubitIndex();
	private:
		QPauli m_data;
		QIndexMap m_index_map;

		size_t m_max_index{ 0 };
		double m_error_threshold{1e-6};
	};
}

#endif // QPAULIOPERATOR_H