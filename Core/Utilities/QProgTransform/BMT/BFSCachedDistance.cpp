#include "Core/Utilities/QProgTransform/BMT//BFSCachedDistance.h"
#include "Core/Utilities/QProgTransform/BMT/QbitAllocator.h"
#include <queue>

using namespace QPanda;

BFSCachedDistance::BFSCachedDistance()
    : DistanceGetter() {}

void BFSCachedDistance::cacheDistanceFrom(uint32_t u) {
    auto& distance = mDistance[u];
    distance.assign(mG->size(), BMT::_undef);

    std::queue<uint32_t> q;
    std::vector<bool> visited(mG->size(), false);

    q.push(u);
    visited[u] = true;
    distance[u] = 0;

    while (!q.empty()) {
        uint32_t u = q.front();
        q.pop();

        for (uint32_t v : mG->adj(u)) {
            if (!visited[v]) {
                visited[v] = true;
                distance[v] = distance[u] + 1;
                q.push(v);
            }
        }
    }
}

void BFSCachedDistance::initImpl() {
    mDistance.assign(mG->size(), VecUInt32());
}

uint32_t BFSCachedDistance::getImpl(uint32_t u, uint32_t v) {
    if (!mDistance[u].empty()) return mDistance[u][v];
    if (!mDistance[v].empty()) return mDistance[v][u];
    cacheDistanceFrom(u);
    return mDistance[u][v];
}

BFSCachedDistance::uRef BFSCachedDistance::Create() {
    return uRef(new BFSCachedDistance());
}
