
#include "Core/Utilities/Compiler/operations/Expression.hpp"

namespace sym {

Variable::Variable(const std::string& name) {
  const auto it = registered.find(name);
  if (it != registered.end()) {
    id = it->second;
  } else {
    registered[name] = nextId;
    names[nextId] = name;
    id = nextId;
    ++nextId;
  }
}

std::string Variable::getName() const { return names[id]; }

std::ostream& operator<<(std::ostream& os, const Variable& var) {
  os << var.getName();
  return os;
}
} // namespace sym
