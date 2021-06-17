#ifndef SLSQP_H
#define SLSQP_H

#include "nlopt.h"
#include "nlopt-util.h"
#include "ThirdParty/rapidjson/document.h"
#include "ThirdParty/rapidjson/writer.h"
#include "ThirdParty/rapidjson/filereadstream.h"
#include "ThirdParty/rapidjson/filewritestream.h"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdarg>

using namespace rapidjson;



	nlopt_result nlopt_slsqp(unsigned n, nlopt_func f, void* f_data,
		unsigned m, nlopt_constraint* fc,
		unsigned p, nlopt_constraint* h,
		const double* lb, const double* ub,
		double* x, double* minf,
		nlopt_stopping* stop, bool restore_flag = false,
		std::string save_file_name = "slsqp_break_point.json");



#endif