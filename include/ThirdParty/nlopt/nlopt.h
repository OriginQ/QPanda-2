/* Copyright (c) 2007-2014 Massachusetts Institute of Technology
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef NLOPT_H
#define NLOPT_H

 /* The size of `unsigned int', as computed by sizeof. */
#define SIZEOF_UNSIGNED_INT 4

/* The size of `unsigned long', as computed by sizeof. */
#define SIZEOF_UNSIGNED_LONG 4

 /* Major version number. */
#define MAJOR_VERSION 2

/* Minor version number. */
#define MINOR_VERSION 6

/* Bugfix version number. */
#define BUGFIX_VERSION 2

#include <cstddef>             /* for ptrdiff_t and size_t */
#include <string>
#include <functional>
#include <vector>

    //typedef double (*nlopt_func) (unsigned n, const double* x,
    //    double* gradient, /* NULL if not needed */
    //    void* func_data);
using vector_data = std::vector<double>;
using nlopt_func = std::function<double(unsigned, const double*, double*, void*)>;

//typedef void (*nlopt_mfunc) (unsigned m, double* result, unsigned n, const double* x,
//    double* gradient, /* NULL if not needed */
//    void* func_data);
using nlopt_mfunc = std::function<void(unsigned, double*, unsigned, const double*, double*, void*)>;

/* A preconditioner, which preconditions v at x to return vpre.
   (The meaning of "preconditioning" is algorithm-dependent.) */
   //typedef void (*nlopt_precond) (unsigned n, const double* x, const double* v, double* vpre, void* data);
using nlopt_precond = std::function<void(unsigned, const double*, const double*, double*, void*)>;

enum class nlopt_algorithm {
    /* Naming conventions:

       NLOPT_{G/L}{D/N}_*
       = global/local derivative/no-derivative optimization,
       respectively

       *_RAND algorithms involve some randomization.

       *_NOSCAL algorithms are *not* scaled to a unit hypercube
       (i.e. they are sensitive to the units of x)
     */

    NLOPT_LD_LBFGSB,
    NLOPT_LN_COBYLA,
    NLOPT_LD_SLSQP,

    NLOPT_NUM_ALGORITHMS        /* not an algorithm, just the number of them */
};

const char* nlopt_algorithm_name(nlopt_algorithm a);

/* nlopt_algorithm enum <-> string conversion */
const char* nlopt_algorithm_to_string(nlopt_algorithm algorithm);
nlopt_algorithm nlopt_algorithm_from_string(const char* name);

typedef enum class nlopt_result {
    NLOPT_FAILURE = -1,         /* generic failure code */
    NLOPT_INVALID_ARGS = -2,
    NLOPT_OUT_OF_MEMORY = -3,
    NLOPT_ROUNDOFF_LIMITED = -4,
    NLOPT_FORCED_STOP = -5,
    NLOPT_SUCCESS = 1,          /* generic success code */
    NLOPT_STOPVAL_REACHED = 2,
    NLOPT_FTOL_REACHED = 3,
    NLOPT_XTOL_REACHED = 4,
    NLOPT_MAXEVAL_REACHED = 5,
    NLOPT_MAXTIME_REACHED = 6,
    NLOPT_MAXITER_REACHED = 7,
    NLOPT_NUM_RESULTS           /* not a result, just the number of them */
};

/* nlopt_result enum <-> string conversion */
const char* nlopt_result_to_string(nlopt_result algorithm);
nlopt_result nlopt_result_from_string(const char* name);

#define NLOPT_MINF_MAX_REACHED NLOPT_STOPVAL_REACHED

void nlopt_srand(unsigned long seed);
void nlopt_srand_time(void);

void nlopt_version(int* major, int* minor, int* bugfix);

/*************************** OBJECT-ORIENTED API **************************/
/* The style here is that we create an nlopt_opt "object" (an opaque pointer),
   then set various optimization parameters, and then execute the
   algorithm.  In this way, we can add more and more optimization parameters
   (including algorithm-specific ones) without breaking backwards
   compatibility, having functions with zillions of parameters, or
   relying non-reentrantly on global variables.*/

struct nlopt_opt_s;             /* opaque structure, defined internally */
typedef struct nlopt_opt_s* nlopt_opt;

/* the only immutable parameters of an optimization are the algorithm and
   the dimension n of the problem, since changing either of these could
   have side-effects on lots of other parameters */
nlopt_opt_s nlopt_create(nlopt_algorithm algorithm, unsigned n);
void nlopt_destroy(nlopt_opt_s opt);
nlopt_opt nlopt_copy(const nlopt_opt opt);

nlopt_result nlopt_optimize(nlopt_opt opt, double* x, double* opt_f,
    bool restore_flag = false, std::string save_file_name = "");

nlopt_result nlopt_set_min_objective(nlopt_opt opt, nlopt_func f, void* f_data);
nlopt_result nlopt_set_max_objective(nlopt_opt opt, nlopt_func f, void* f_data);

nlopt_result nlopt_set_precond_min_objective(nlopt_opt opt, nlopt_func f, nlopt_precond pre, void* f_data);
nlopt_result nlopt_set_precond_max_objective(nlopt_opt opt, nlopt_func f, nlopt_precond pre, void* f_data);

nlopt_algorithm nlopt_get_algorithm(const nlopt_opt opt);
unsigned nlopt_get_dimension(const nlopt_opt opt);

const char* nlopt_get_errmsg(nlopt_opt opt);


/* constraints: */

nlopt_result nlopt_set_lower_bounds(nlopt_opt opt, const double* lb);
nlopt_result nlopt_set_lower_bounds1(nlopt_opt opt, double lb);
nlopt_result nlopt_set_lower_bound(nlopt_opt opt, int i, double lb);
nlopt_result nlopt_get_lower_bounds(const nlopt_opt opt, double* lb);
nlopt_result nlopt_set_upper_bounds(nlopt_opt opt, const double* ub);
nlopt_result nlopt_set_upper_bounds1(nlopt_opt opt, double ub);
nlopt_result nlopt_set_upper_bound(nlopt_opt opt, int i, double ub);
nlopt_result nlopt_get_upper_bounds(const nlopt_opt opt, double* ub);

nlopt_result nlopt_remove_inequality_constraints(nlopt_opt opt);
nlopt_result nlopt_add_inequality_constraint(nlopt_opt opt, nlopt_func fc, void* fc_data, double tol);
nlopt_result nlopt_add_precond_inequality_constraint(nlopt_opt opt, nlopt_func fc, nlopt_precond pre, void* fc_data, double tol);
nlopt_result nlopt_add_inequality_mconstraint(nlopt_opt opt, unsigned m, nlopt_mfunc fc, void* fc_data, const double* tol);

nlopt_result nlopt_remove_equality_constraints(nlopt_opt opt);
nlopt_result nlopt_add_equality_constraint(nlopt_opt opt, nlopt_func h, void* h_data, double tol);
nlopt_result nlopt_add_precond_equality_constraint(nlopt_opt opt, nlopt_func h, nlopt_precond pre, void* h_data, double tol);
nlopt_result nlopt_add_equality_mconstraint(nlopt_opt opt, unsigned m, nlopt_mfunc h, void* h_data, const double* tol);

/* stopping criteria: */

nlopt_result nlopt_set_stopval(nlopt_opt opt, double stopval);
double nlopt_get_stopval(const nlopt_opt opt);

nlopt_result nlopt_set_ftol_rel(nlopt_opt opt, double tol);
double nlopt_get_ftol_rel(const nlopt_opt opt);
nlopt_result nlopt_set_ftol_abs(nlopt_opt opt, double tol);
double nlopt_get_ftol_abs(const nlopt_opt opt);

nlopt_result nlopt_set_xtol_rel(nlopt_opt opt, double tol);
double nlopt_get_xtol_rel(const nlopt_opt opt);
nlopt_result nlopt_set_xtol_abs1(nlopt_opt opt, double tol);
nlopt_result nlopt_set_xtol_abs(nlopt_opt opt, const double* tol);
nlopt_result nlopt_get_xtol_abs(const nlopt_opt opt, double* tol);
nlopt_result nlopt_set_x_weights1(nlopt_opt opt, double w);
nlopt_result nlopt_set_x_weights(nlopt_opt opt, const double* w);
nlopt_result nlopt_get_x_weights(const nlopt_opt opt, double* w);

nlopt_result nlopt_set_maxeval(nlopt_opt opt, int maxeval);
int nlopt_get_maxeval(const nlopt_opt opt);
nlopt_result nlopt_set_maxiter(nlopt_opt opt, int maxiter);
int nlopt_get_maxiter(const nlopt_opt opt);

int nlopt_get_numevals(const nlopt_opt opt);
int nlopt_get_numiters(const nlopt_opt opt);

nlopt_result nlopt_set_maxtime(nlopt_opt opt, double maxtime);
double nlopt_get_maxtime(const nlopt_opt opt);

nlopt_result nlopt_force_stop(nlopt_opt opt);
nlopt_result nlopt_set_force_stop(nlopt_opt opt, int val);
int nlopt_get_force_stop(const nlopt_opt opt);

/* more algorithm-specific parameters */

nlopt_result nlopt_set_local_optimizer(nlopt_opt opt, const nlopt_opt local_opt);

nlopt_result nlopt_set_population(nlopt_opt opt, unsigned pop);
unsigned nlopt_get_population(const nlopt_opt opt);

nlopt_result nlopt_set_vector_storage(nlopt_opt opt, unsigned dim);
unsigned nlopt_get_vector_storage(const nlopt_opt opt);

nlopt_result nlopt_set_default_initial_step(nlopt_opt opt, const double* x);
nlopt_result nlopt_set_initial_step(nlopt_opt opt, const double* dx);
nlopt_result nlopt_set_initial_step1(nlopt_opt opt, double dx);
nlopt_result nlopt_get_initial_step(const nlopt_opt opt, const double* x, double* dx);

/* the following are functions mainly designed to be used internally
   by the Fortran and SWIG wrappers, allow us to tel nlopt_destroy and
   nlopt_copy to do something to the f_data pointers (e.g. free or
   duplicate them, respectively) */
typedef void* (*nlopt_munge) (void* p);
void nlopt_set_munge(nlopt_opt opt, nlopt_munge munge_on_destroy, nlopt_munge munge_on_copy);
typedef void* (*nlopt_munge2) (void* p, void* data);
void nlopt_munge_data(nlopt_opt opt, nlopt_munge2 munge, void* data);

/*************************** DEPRECATED API **************************/
/* The new "object-oriented" API is preferred, since it allows us to
   gracefully add new features and algorithm-specific options in a
   re-entrant way, and we can automatically assume reasonable defaults
   for unspecified parameters. */

   /* Where possible (e.g. for gcc >= 3.1), enable a compiler warning
      for code that uses a deprecated function */
#if defined(__GNUC__) && (__GNUC__ > 3 || (__GNUC__==3 && __GNUC_MINOR__ > 0))
#  define NLOPT_DEPRECATED __attribute__((deprecated))
#else
#  define NLOPT_DEPRECATED
#endif

typedef double (*nlopt_func_old) (int n, const double* x, double* gradient,     /* NULL if not needed */
    void* func_data);

nlopt_result nlopt_minimize(nlopt_algorithm algorithm, int n, nlopt_func_old f, void* f_data,
    const double* lb, const double* ub, /* bounds */
    double* x,    /* in: initial guess, out: minimizer */
    double* minf, /* out: minimum */
    double minf_max, double ftol_rel, double ftol_abs, double xtol_rel, const double* xtol_abs, int maxeval, int maxiter, double maxtime) NLOPT_DEPRECATED;

nlopt_result nlopt_minimize_constrained(nlopt_algorithm algorithm, int n, nlopt_func_old f, void* f_data, int m, nlopt_func_old fc, void* fc_data, ptrdiff_t fc_datum_size,
    const double* lb, const double* ub,   /* bounds */
    double* x,        /* in: initial guess, out: minimizer */
    double* minf,     /* out: minimum */
    double minf_max, double ftol_rel, double ftol_abs, double xtol_rel, const double* xtol_abs, int maxeval, int maxiter, double maxtime) NLOPT_DEPRECATED;

nlopt_result nlopt_minimize_econstrained(nlopt_algorithm algorithm, int n, nlopt_func_old f, void* f_data, int m, nlopt_func_old fc, void* fc_data, ptrdiff_t fc_datum_size, int p, nlopt_func_old h, void* h_data, ptrdiff_t h_datum_size,
    const double* lb, const double* ub,   /* bounds */
    double* x,       /* in: initial guess, out: minimizer */
    double* minf,    /* out: minimum */
    double minf_max, double ftol_rel, double ftol_abs,
    double xtol_rel, const double* xtol_abs, double htol_rel, double htol_abs, int maxeval, int maxiter, double maxtime) NLOPT_DEPRECATED;

void nlopt_get_local_search_algorithm(nlopt_algorithm* deriv, nlopt_algorithm* nonderiv, int* maxeval, int *maxiter) NLOPT_DEPRECATED;
void nlopt_set_local_search_algorithm(nlopt_algorithm deriv, nlopt_algorithm nonderiv, int maxeval, int maxiter) NLOPT_DEPRECATED;

int nlopt_get_stochastic_population(void) NLOPT_DEPRECATED;
void nlopt_set_stochastic_population(int pop) NLOPT_DEPRECATED;

/*********************************************************************/
#endif
