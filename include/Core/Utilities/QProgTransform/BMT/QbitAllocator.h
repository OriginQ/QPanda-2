#ifndef __EFD_QBIT_ALLOCATOR_H__
#define __EFD_QBIT_ALLOCATOR_H__

#include "Core/Utilities/Tools/ArchGraph.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"

using QNodeRef = std::shared_ptr<QPanda::QNode> ;

namespace BMT {
	/// \brief Defines the type used for mapping the qubits.
	typedef std::vector<uint32_t> Mapping;
	typedef std::vector<uint32_t> InverseMap;

	/// \brief Constant should be used as an undefined in a mapping.
	static const uint32_t _undef = std::numeric_limits<uint32_t>::max();

	/// \brief Struct used for representing a swap between two qubits;
	struct Swap {
		uint32_t u;
		uint32_t v;
	};

	/// \brief Two \p Swap objects are considered equal if they occur
	/// on equal qubits (order is irrelevant).
	inline bool operator==(const Swap& lhs, const Swap& rhs) {
		return (lhs.u == rhs.u && lhs.v == rhs.v) ||
			(lhs.u == rhs.v && lhs.v == rhs.u);
	}

	inline bool operator!=(const Swap& lhs, const Swap& rhs) {
		return !(lhs == rhs);
	}

	typedef std::vector<Swap> SwapSeq;

	using GateWeightMap = std::map<std::string, uint32_t>;

    /// \brief Base abstract class that allocates the qbits used in the program to
    /// the qbits that are in the physical architecture.
    class QbitAllocator /*: public PassT<Mapping>*/ 
	{
        public:
            typedef QbitAllocator* Ref;
            typedef std::unique_ptr<QbitAllocator> uRef;

        private:
			Mapping m_mapping;
            uint32_t m_CX_cost;
			uint32_t m_CZ_cost;
            uint32_t m_u3_cost;

        protected:
			QPanda::ArchGraph::sRef mArchGraph;
            GateWeightMap mGateWeightMap;

            uint32_t mVQubits;
            uint32_t mPQubits;
			QPanda::QProg m_mapped_prog;

            QbitAllocator(QPanda::ArchGraph::sRef archGraph);

            /// \brief Executes the allocation algorithm after the preprocessing.
            virtual Mapping allocate(QPanda::QProg prog, QPanda::QuantumMachine *qvm) = 0;

            /// \brief Returns the cost of a \em CNOT gate, based on the defined weights.
            uint32_t get_CX_cost(uint32_t u, uint32_t v);

			/// \brief Returns the cost of a \em CZ gate, based on the defined weights.
			uint32_t get_CZ_cost(uint32_t u, uint32_t v);

            /// \brief Returns the cost of a \em SWAP gate, based on the defined weights.
            uint32_t getSwapCost(uint32_t u, uint32_t v);

        public:
            bool run(QPanda::QProg prog, QPanda::QuantumMachine *qvm);

            /// \brief Sets the weights to be used for each gate.
			void setGateWeightMap(const GateWeightMap& weightMap) { mGateWeightMap = weightMap; }

			Mapping get_mapping() const { return m_mapping; }
			QPanda::QProg get_mapped_prog() const { return m_mapped_prog; }
    };

    /// \brief Generates an assignment mapping (maps the architecture's qubits
    /// to the logical ones) of size \p archQ.
    InverseMap InvertMapping(uint32_t archQ, Mapping mapping, bool fill = true);

    /// \brief Fills the unmapped qubits with the ones missing.
    void Fill(uint32_t archQ, Mapping& mapping);
    void Fill(Mapping& mapping, InverseMap& inv);

    /// \brief Returns an identity mapping.
    Mapping IdentityMapping(uint32_t progQ);

    /// \brief Prints the mapping \p m to a string and returns it.
    std::string MappingToString(Mapping m);
}

#endif
