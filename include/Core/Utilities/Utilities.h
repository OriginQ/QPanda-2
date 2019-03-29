#include"QPandaNamespace.h"
#include <iostream>
#include <map>
QPANDA_BEGIN
std::string dec2bin(unsigned n, size_t size);
double RandomNumberGenerator();
void add_up_a_map(std::map<std::string, size_t> &meas_result, std::string key);
QPANDA_END