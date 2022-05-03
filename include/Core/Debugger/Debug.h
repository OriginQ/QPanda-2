#pragma once

#include "OriginDebug.h"
#include "QDebug.h"

QPANDA_BEGIN

extern std::shared_ptr<OriginDebug> g_origin_debug;

QDebug Debug();

QPANDA_END