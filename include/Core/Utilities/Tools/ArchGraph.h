#ifndef __EFD_ARCH_GRAPH_H__
#define __EFD_ARCH_GRAPH_H__

#include "Graph.h"

QPANDA_BEGIN

/// \brief This is the base class for the architectures that this project will
/// be supporting.
class ArchGraph : public WeightedGraph<double> 
{
public:
    typedef ArchGraph* Ref;
    typedef std::unique_ptr<ArchGraph> uRef;
    typedef std::shared_ptr<ArchGraph> sRef;

    typedef std::vector<std::pair<std::string, uint32_t>> RegsVector;
    typedef RegsVector::iterator RegsIterator;

protected:
    RegsVector mRegs;

    std::vector<std::string> mId; 
    std::unordered_map<std::string, uint32_t> mStrToId;

    bool mGeneric;
    uint32_t mVID;

    ArchGraph(uint32_t n, bool isGeneric = true);
 
public:
    /// \brief Creates a string vertex and puts it in the vector.
    uint32_t putVertex(std::string s);

    /// \brief Register the register.
    void putReg(std::string id, std::string size);

    /// \brief Returns the vertex number.
	uint32_t get_vertex_count() { return mId.size(); }

	std::vector<std::vector<int>> get_adjacent_matrix();
	std::vector<std::vector<double>> get_adj_weight_matrix();

    /// \brief Returns true if this is a generic architechture graph,
    /// i.e.: it was not created by any of the architechtures compiled within
    /// the program.
    bool isGeneric();

    /// \brief The begin iterator for the \p mRegs.
    RegsIterator reg_begin();

    /// \brief The end iterator for the \p mRegs.
    RegsIterator reg_end();

    /// \brief Returns true if \p g is of this type.
    static bool ClassOf(const Graph* g);

    /// \brief Encapsulates the creation of a new ArchGraph.
    static uRef Create(uint32_t n);
};

template <> struct JsonFields<ArchGraph> {
	static const std::string _quantum_chip_arch_label;
    static const std::string _qubits_label;
    static const std::string _name_label;
    static const std::string _adj_list_label;
    static const std::string _v_label;
    static const std::string _weight_label;
};

template <> struct JsonBackendParser<ArchGraph> {
    static std::unique_ptr<ArchGraph> Parse(const rapidjson::Value& root);
};

QPANDA_END
#endif
