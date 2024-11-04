
#include "Core/Utilities/Compiler/QASMCompiler/parser/Types.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace qasm {

template <typename T>
DesignatedType<T>::DesignatedType(DesignatedTy ty)
    : type(ty), designator(nullptr) {}
template <>
bool DesignatedType<std::shared_ptr<Expression>>::fits(
    const Type<std::shared_ptr<Expression>>& other) {
  if (const auto* o = dynamic_cast<const DesignatedType*>(&other)) {
    if (type == Int && o->type == Uint) {
      return true;
    }
    if (type == Float && (o->type == Int || o->type == Uint)) {
      return true;
    }

    return type == o->type;
  }
  return false;
}

template <> bool DesignatedType<uint64_t>::fits(const Type<uint64_t>& other) {
  if (const auto* o = dynamic_cast<const DesignatedType*>(&other)) {
    bool typeFits = type == o->type;
    if (type == Int && o->type == Uint) {
      typeFits = true;
    }
    if (type == Float && (o->type == Int || o->type == Uint)) {
      typeFits = true;
    }

    return typeFits && designator >= o->designator;
  }
  return false;
}

template <>
DesignatedType<uint64_t>::DesignatedType(DesignatedTy ty)
    : type(ty), designator(0) {
  switch (ty) {
  case QBit:
  case Bit:
    designator = 1;
    break;
  case Int:
  case Uint:
    designator = 32;
    break;
  case Float:
  case Angle:
    designator = 64;
    break;
  }
}

template <>
std::string DesignatedType<std::shared_ptr<Expression>>::designatorToString() {
  return "expr";
}

template <> std::string DesignatedType<uint64_t>::designatorToString() {
  return std::to_string(designator);
}

} // namespace qasm
