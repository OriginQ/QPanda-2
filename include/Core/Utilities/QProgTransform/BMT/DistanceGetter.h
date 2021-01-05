#ifndef __EFD_DISTANCE_GETTER_H__
#define __EFD_DISTANCE_GETTER_H__

#include "Core/Utilities/Tools/Graph.h"

QPANDA_BEGIN
/// \brief Interface for calculating the distance of the path between two vertices.
template <typename T>
class DistanceGetter 
{
public:
    typedef DistanceGetter* Ref;
    typedef std::shared_ptr<DistanceGetter> sRef;
    typedef std::unique_ptr<DistanceGetter> uRef;

protected:
    Graph::Ref mG;

	virtual void initImpl() {}
    virtual T getImpl(uint32_t u, uint32_t v) = 0;

private:
    void checkInitialized() {
		if (nullptr == mG)
		{
			QCERR_AND_THROW(run_fail, "Set `Graph` for the DistanceGetter!");
		}
	}

    void checkVertexInGraph(uint32_t u) {
		if (mG->size() <= u)
		{
			QCERR_AND_THROW(run_fail, "Out of Bounds: can't calculate distance for: `" << u << "`");
		}
	}

public:
    DistanceGetter() : mG(nullptr) {}
    virtual ~DistanceGetter() = default;

    void init(Graph::Ref graph) {
		if (graph != nullptr) {
			mG = graph;
			initImpl();
		}
	}

    T get(uint32_t u, uint32_t v) {
		checkInitialized();
		checkVertexInGraph(u);
		checkVertexInGraph(v);
		return getImpl(u, v);
	}
};

QPANDA_END
#endif
