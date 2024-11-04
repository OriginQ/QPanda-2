
// Generated from .\pyquil.g4 by ANTLR 4.8

#pragma once


#include "antlr4-runtime.h"
#include "Core/Utilities/Compiler/PyquilCompiler/pyquilVisitor.h"


/**
 * This class provides an empty implementation of pyquilVisitor, which can be
 * extended to create a visitor which only needs to handle a subset of the available methods.
 */
class  pyquilBaseVisitor : public pyquilVisitor {
public:

  virtual antlrcpp::Any visitProg(pyquilParser::ProgContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitCode_block(pyquilParser::Code_blockContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitLoop(pyquilParser::LoopContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitLoop_start(pyquilParser::Loop_startContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitLoop_end(pyquilParser::Loop_endContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitLoop_if_continue(pyquilParser::Loop_if_continueContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitOperation(pyquilParser::OperationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDeclare(pyquilParser::DeclareContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitMeasure(pyquilParser::MeasureContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitMove(pyquilParser::MoveContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSub(pyquilParser::SubContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitVar_name(pyquilParser::Var_nameContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitVar_mem(pyquilParser::Var_memContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQbit(pyquilParser::QbitContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitGate(pyquilParser::GateContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitBool_val(pyquilParser::Bool_valContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitParam(pyquilParser::ParamContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitExpr(pyquilParser::ExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitArray_item(pyquilParser::Array_itemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitArrayname(pyquilParser::ArraynameContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitIdx(pyquilParser::IdxContext *ctx) override {
    return visitChildren(ctx);
  }


};

