#include "Core/Debugger/Debug.h"

QPANDA_BEGIN

std::shared_ptr<OriginDebug> g_origin_debug = std::make_shared<OriginDebug>();

QDebug Debug()
{
    return QDebug(g_origin_debug);
}

QPANDA_END