#ifndef __EFD_GRAPH_H__
#define __EFD_GRAPH_H__

#include "Core/Utilities/QPandaNamespace.h"
#include "ThirdParty/rapidjson/rapidjson.h"
#include "ThirdParty/rapidjson/rapidjson.h"
#include "ThirdParty/rapidjson/document.h"
#include "ThirdParty/rapidjson/writer.h"
#include "ThirdParty/rapidjson/prettywriter.h"
#include "ThirdParty/rapidjson/stringbuffer.h"
#include "Core/Utilities/Tools/QPandaException.h"
#include <set>
#include <vector>
#include <unordered_map>
#include <memory>
#include <map>
#include <fstream>
#include <sstream>

QPANDA_BEGIN

/// \brief Centralizes the field names for each type.
template <class T> struct JsonFields {};

/// \brief Gets a \em T instance (wrapped in a \em std::unique_ptr) from the \em Json::Value.
template <class T> struct JsonBackendParser {
	static std::unique_ptr<T> Parse(const rapidjson::Value& root) {
		QCERR_AND_THROW(QPanda::run_fail, "Parse method not implemented for '" << typeid(T).name() << "'.");
	}
};

/// \brief Frontend to the json parser.
template <class T> 
class JsonParser 
{
private:
	static std::unique_ptr<T> ParseInputStream(const std::string& in) {
		rapidjson::Document m_doc;
		if (m_doc.Parse(in.c_str()).HasParseError())
		{
			QCERR_AND_THROW(QPanda::run_fail, "Error: failed to parse the config data.");
		}

		return JsonBackendParser<T>::Parse(m_doc);
	}

public:
	/// \brief Parses a json \em std::string.
	static std::unique_ptr<T> ParseString(std::string str) {
		return ParseInputStream(str);
	}

	/// \brief Parses a json file.
	static std::unique_ptr<T> ParseFile(std::string filename) {
		std::ifstream ifs(filename.c_str());
		if (!ifs.is_open()){
			QCERR_AND_THROW(QPanda::run_fail, "Error: failed to parse the config file.");
		}

		return ParseInputStream(std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>()));
	}
};

/// \brief Graph representation.
class Graph 
{
    public:
        typedef Graph* Ref;
        typedef std::unique_ptr<Graph> uRef;
        typedef std::shared_ptr<Graph> sRef;

        enum Kind {
            K_GRAPH,
            K_WEIGHTED,
            K_ARCH
        };

        enum Type { Directed, Undirected };

    protected:
        Kind mK;
        uint32_t mN;
        Type mTy;

        std::vector<std::set<uint32_t>> mSuccessors;
        std::vector<std::set<uint32_t>> mPredecessors;

        Graph(Kind k, uint32_t n, Type ty = Undirected);

        virtual std::string vertexToString(uint32_t i) const;
        virtual std::string edgeToString(uint32_t i, uint32_t j, std::string op) const;

    public:
        virtual ~Graph() = default;
        Graph(uint32_t n, Type ty = Undirected);

        /// \brief Return the degree entering the vertex \p i.
        uint32_t inDegree(uint32_t i) const;
        /// \brief Return the degree leaving the vertex \p i.
        uint32_t outDegree(uint32_t i) const; 
        /// \brief Return the number of vertices.
        uint32_t size() const;
    
        /// \brief Return the set of succesors of some vertex \p i.
        std::set<uint32_t>& succ(uint32_t i);
		const std::set<uint32_t>& c_succ(uint32_t i) const;
        /// \brief Return the set of predecessors of some vertex \p i.
        std::set<uint32_t>& pred(uint32_t i); 
        /// \brief Return the set of adjacent vertices of some vertex \p i.
        std::set<uint32_t> adj(uint32_t i) const;
    
        /// \brief Inserts an edge (i, j) in the successor's list and
        /// an edge (j, i) in the predecessor's list.
        void putEdge(uint32_t i, uint32_t j);
        /// \brief Returns true whether it has an edge (i, j).
        bool hasEdge(uint32_t i, uint32_t j) const;

        /// \brief Returns true if this is a weighted graph.
		bool isWeighted() const { return (mK == K_WEIGHTED) || (mK == K_ARCH); }

        /// \brief Returns true if this is an architecture graph.
		bool isArch() const { return mK == K_ARCH; }

        /// \brief Returns true if this is a directed graph.
		bool isDirectedGraph() const { return mTy == Directed; }

        /// \brief Converts itself to a 'dot' graph representation.
        std::string dotify(std::string name = "Dump") const;

        /// \brief Returns true if \p g is of this type.
        static bool ClassOf(const Graph* g){ return true; }
    
        /// \brief Encapsulates the creation of a new Graph.
		static uRef Create(uint32_t n, Type ty = Undirected) { return std::unique_ptr<Graph>(new Graph(K_GRAPH, n, ty)); }
};

template <> struct JsonFields<Graph> {
    static const std::string _VerticesLabel_;
    static const std::string _AdjListLabel_;
    static const std::string _TypeLabel_;
    static const std::string _VLabel_;
};

template <> struct JsonBackendParser<Graph> {
    static std::unique_ptr<Graph> Parse(const rapidjson::Value& root);
};


template <typename T>
class WeightedGraph : public Graph 
{
public:
	typedef WeightedGraph<T>* Ref;
	typedef std::unique_ptr<WeightedGraph<T>> uRef;
	typedef std::shared_ptr<WeightedGraph<T>> sRef;

private:
	std::map<std::pair<uint32_t, uint32_t>, T> mW;

protected:
	std::string edgeToString(uint32_t i, uint32_t j, std::string op) const override {
		return vertexToString(i) + " " + op + " " + vertexToString(j) +
			"[label=" + std::to_string(getW(i, j)) + "]";
	}

	/// \brief Constructor to be used by whoever inherits this class.
	WeightedGraph(Kind k, uint32_t n, Type ty = Undirected) : Graph(k, n, ty) {}

public:
	WeightedGraph(uint32_t n, Type ty = Undirected) : Graph(K_WEIGHTED, n, ty) {}

	/// \brief Insert the edge (i, j) with weight w(i, j) = \p w.
	void putEdge(uint32_t i, uint32_t j, T w) {
		Graph::putEdge(i, j);

		mW[std::make_pair(i, j)] = w;
		if (!isDirectedGraph()) {
			mW[std::make_pair(j, i)] = w;
		}
	}

	/// \brief Sets the weight of an edge (i, j).
	void setW(uint32_t i, uint32_t j, T w) {
		auto pair = std::make_pair(i, j);
		if (mW.find(pair) == mW.end())
		{
			QCERR_AND_THROW(run_fail, "Edge not found: `(" << i << ", " << j << ")`.");
		}

		mW[pair] = w;
	}

	/// \brief Gets the weight of an edge (i, j).
	T getW(uint32_t i, uint32_t j) const {
		auto pair = std::make_pair(i, j);

		if (mW.find(pair) == mW.end())
		{
			QCERR_AND_THROW(run_fail, "Edge weight not found for edge: `(" << i << ", " << j << ")`.");
		}

		return mW.at(pair);
	}

	/// \brief Returns true if \p g is of this type.
	static bool ClassOf(const Graph* g) {
		return g->isWeighted();
	}

	/// \brief Encapsulates the creation of a new Graph.
	static uRef Create(uint32_t n, Type ty = Undirected) {
		return std::unique_ptr<WeightedGraph<T>>(new WeightedGraph<T>(n, ty));
	}
};

template <class T> struct JsonFields<WeightedGraph<T>> {
	static std::string _WeightLabel_;
};

template <class T> struct JsonBackendParser<WeightedGraph<T>> {
	static T ParseWeight(const rapidjson::Value& v);
	//static std::vector<Json::ValueType> GetTysForT();
	static std::unique_ptr<WeightedGraph<T>> Parse(const rapidjson::Value& root);
};

template <> 
int32_t JsonBackendParser<WeightedGraph<int32_t>>::ParseWeight(const rapidjson::Value& v);

template <> 
uint32_t JsonBackendParser<WeightedGraph<uint32_t>>::ParseWeight(const rapidjson::Value& v);

template <> 
double JsonBackendParser<WeightedGraph<double>>::ParseWeight(const rapidjson::Value& v);

// ----------------------------- JsonFields -------------------------------
template <class T>
std::string JsonFields<WeightedGraph<T>>::_WeightLabel_ = "w";

// ----------------------------- JsonBackendParser -------------------------------
template <class T>
T JsonBackendParser<WeightedGraph<T>>::ParseWeight(const rapidjson::Value& v) {
	QCERR_AND_THROW(init_fail, "ParseWeight not implemented for `" << typeid(T).name() << "`.");
}

QPANDA_END
#endif
