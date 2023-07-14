#pragma once

#include <ctime>
#include <string>
#include <sstream>
#include "QPandaConfig.h"

std::string qcloud_signature(const std::string& apikey);
