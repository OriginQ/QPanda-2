#ifndef __EFD_BFS_CACHED_DISTANCE_H__
#define __EFD_BFS_CACHED_DISTANCE_H__

#include "DistanceGetter.h"

QPANDA_BEGIN
/// \brief Calculates the distance by applying BFS.
class BFSCachedDistance : public DistanceGetter<uint32_t>
{
public:
    typedef BFSCachedDistance* Ref;
    typedef std::shared_ptr<BFSCachedDistance> sRef;
    typedef std::unique_ptr<BFSCachedDistance> uRef;

private:
    typedef std::vector<uint32_t> VecUInt32;
    typedef std::vector<VecUInt32> MatrixUInt32;

    void cacheDistanceFrom(uint32_t u);

protected:
    void initImpl() override;
    uint32_t getImpl(uint32_t u, uint32_t v) override;

private:
    MatrixUInt32 mDistance;

public:
    BFSCachedDistance();

    /// \brief Instantiate one object of this type.
    static uRef Create();
};

QPANDA_END
#endif
