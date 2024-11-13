#pragma once

#include "Core/Utilities/Compiler/QASMCompiler/parser/InstVisitor.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace qasm {
class Expression;

template <typename T> class Type;
using TypeExpr = Type<std::shared_ptr<Expression>>;
using ResolvedType = Type<uint64_t>;

template <typename T> class Type {
public:
  virtual ~Type() = default;

  [[nodiscard]] virtual bool operator==(const Type<T>& other) const = 0;
  [[nodiscard]] bool operator!=(const Type<T>& other) const {
    return !(*this == other);
  }
  [[nodiscard]] virtual bool allowsDesignator() const = 0;

  virtual void setDesignator(T /*designator*/) {
    throw std::runtime_error("Type does not allow designator");
  }

  [[nodiscard]] virtual T getDesignator() = 0;

  virtual std::shared_ptr<ResolvedType> accept(TypeVisitor<T>* visitor) = 0;

  virtual bool isBool() { return false; }
  virtual bool isNumber() { return false; }
  virtual bool isFP() { return false; }
  virtual bool isUint() { return false; }
  virtual bool isBit() { return false; }

  virtual bool fits(const Type<T>& other) { return *this == other; }

  virtual std::string toString() = 0;
};

enum DesignatedTy {
  QBit,
  Bit,
  Int,
  Uint,
  Float,
  Angle,
};

template <typename T> class DesignatedType : public Type<T> {
public:
  ~DesignatedType() override = default;

  DesignatedTy type;

  T designator;

  DesignatedType(const DesignatedTy ty, T design)
      : type(ty), designator(std::move(design)) {}

  explicit DesignatedType(DesignatedTy ty);

  bool operator==(const Type<T>& other) const override {
    if (const auto* o = dynamic_cast<const DesignatedType<T>*>(&other)) {
      return type == o->type && designator == o->designator;
    }
    return false;
  }

  [[nodiscard]] bool allowsDesignator() const override { return true; }

  static std::shared_ptr<DesignatedType> getQubitTy(T designator) {
    return std::make_shared<DesignatedType>(QBit, designator);
  }
  static std::shared_ptr<DesignatedType> getBitTy(T designator) {
    return std::make_shared<DesignatedType>(Bit, designator);
  }
  static std::shared_ptr<DesignatedType> getIntTy(T designator) {
    return std::make_shared<DesignatedType>(Int, designator);
  }
  static std::shared_ptr<DesignatedType> getUintTy(T designator) {
    return std::make_shared<DesignatedType>(Uint, designator);
  }
  static std::shared_ptr<DesignatedType> getFloatTy(T designator) {
    return std::make_shared<DesignatedType>(Float, designator);
  }
  static std::shared_ptr<DesignatedType> getAngleTy(T designator) {
    return std::make_shared<DesignatedType>(Angle, designator);
  }

  void setDesignator(T d) override { this->designator = std::move(d); }

  T getDesignator() override { return designator; }

  std::shared_ptr<ResolvedType> accept(TypeVisitor<T>* visitor) override {
    return visitor->visitDesignatedType(this);
  }

  bool isNumber() override {
    return type == Int || type == Uint || type == Bit || type == Float;
  }

  bool isUint() override { return type == Uint; }

  bool isBit() override { return type == Bit; }

  bool isFP() override { return type == Float; }

  bool fits(const Type<T>& other) override;

  std::string toString() override {
    switch (type) {
    case QBit:
      return "qubit[" + designatorToString() + "]";
    case Bit:
      return "bit[" + designatorToString() + "]";
    case Int:
      return "int[" + designatorToString() + "]";
    case Uint:
      return "uint[" + designatorToString() + "]";
    case Float:
      return "float[" + designatorToString() + "]";
    case Angle:
      return "angle[" + designatorToString() + "]";
    }
    throw std::runtime_error("Unhandled type");
  }

  std::string designatorToString();
};

enum UnsizedTy { Bool, Duration };

template <typename T> class UnsizedType : public Type<T> {
public:
  ~UnsizedType() override = default;

  UnsizedTy type;

  explicit UnsizedType(const UnsizedTy ty) : type(ty) {}

  bool operator==(const Type<T>& other) const override {
    if (const auto* o = dynamic_cast<const UnsizedType*>(&other)) {
      return type == o->type;
    }
    return false;
  }
  [[nodiscard]] bool allowsDesignator() const override { return false; }

  static std::shared_ptr<Type<T>> getBoolTy() {
    return std::make_shared<UnsizedType>(Bool);
  }
  static std::shared_ptr<Type<T>> getDurationTy() {
    return std::make_shared<UnsizedType>(Duration);
  }

  T getDesignator() override {
    throw std::runtime_error("Unsized types do not have designators");
  }

  std::shared_ptr<ResolvedType> accept(TypeVisitor<T>* visitor) override {
    return visitor->visitUnsizedType(this);
  }

  bool isBool() override { return type == Bool; }

  std::string toString() override {
    switch (type) {
    case Bool:
      return "bool";
    case Duration:
      return "duration";
    }
    throw std::runtime_error("Unhandled type");
  }
};

template <typename T> class ArrayType : public Type<T> {
public:
  std::shared_ptr<Type<T>> type;
  T size;

  ArrayType(std::shared_ptr<Type<T>> ty, T sz)
      : type(std::move(ty)), size(sz) {}
  ~ArrayType() override = default;

  bool operator==(const Type<T>& other) const override {
    if (const auto* o = dynamic_cast<const ArrayType*>(&other)) {
      return *type == *o->type && size == o->size;
    }
    return false;
  }
  [[nodiscard]] bool allowsDesignator() const override { return true; }

  T getDesignator() override { return size; }

  std::shared_ptr<ResolvedType> accept(TypeVisitor<T>* visitor) override {
    return visitor->visitArrayType(this);
  }

  bool fits(const Type<T>& other) override {
    if (const auto* o = dynamic_cast<const ArrayType*>(&other)) {
      return type->fits(*o->type) && size == o->size;
    }
    return false;
  }

  std::string toString() override {
    if constexpr (std::is_same_v<T, size_t>) {
      return type->toString() + "[" + std::to_string(size) + "]";
    } else {
      return type->toString() + "[<expression>]";
    }
  }
};

} // namespace qasm
