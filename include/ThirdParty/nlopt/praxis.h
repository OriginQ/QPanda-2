#ifndef PRAXIS_H
#define PRAXIS_H

#include "nlopt-util.h"
#include "nlopt.h"



typedef double (*praxis_func)(int n, const double *x, void *f_data);

nlopt_result praxis_(double t0, double machep, double h0,
		     int n, double *x, praxis_func f, void *f_data, 
		     nlopt_stopping *stop, double *minf);



#endif /* PRAXIS_H */
