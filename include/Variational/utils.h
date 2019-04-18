#pragma once

#include "Variational/expression.h"
#include <set>

namespace QPanda {
namespace Variational {

// The utils file is a list of functions that
// could be commonly used by the user.
//
// So far, we support eval(), and back().
//
// The general format is for the user to
// input a specific flag into the functions.
// The user _should not_ have to know the internals.

// Provides an interface for the et::expression evaluation
// pipeline. This is to abstract away the construction of
// an expression and choose the method of evaluation.
MatrixXd eval(var v, bool iter);

// Provides an interface for the et::expression backprop
// pipeline.

enum class back_flags {
    const_qualify
};

void back(const var&, std::unordered_map<var, MatrixXd>&);
void back(expression&, std::unordered_map<var, MatrixXd>&);
void back(const var&, std::unordered_map<var, MatrixXd>&, const std::unordered_set<var>&);
void back(expression&, std::unordered_map<var, MatrixXd>&, const std::unordered_set<var>&);

}
}
