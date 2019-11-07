
// Generated from .\qasm.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"
#include "qasmVisitor.h"


/**
 * This class provides an empty implementation of qasmVisitor, which can be
 * extended to create a visitor which only needs to handle a subset of the available methods.
 */
class  qasmBaseVisitor : public qasmVisitor {
public:

  virtual antlrcpp::Any visitMainprogram(qasmParser::MainprogramContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitHead_decl(qasmParser::Head_declContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitVersion_decl(qasmParser::Version_declContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitInclude_decl(qasmParser::Include_declContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitStatement(qasmParser::StatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitReg_decl(qasmParser::Reg_declContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitOpaque_decl(qasmParser::Opaque_declContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitIf_decl(qasmParser::If_declContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitBarrier_decl(qasmParser::Barrier_declContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitGate_decl(qasmParser::Gate_declContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitGoplist(qasmParser::GoplistContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitBop(qasmParser::BopContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQop(qasmParser::QopContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitUop(qasmParser::UopContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitAnylist(qasmParser::AnylistContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitIdlist(qasmParser::IdlistContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitId_index(qasmParser::Id_indexContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitArgument(qasmParser::ArgumentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitExplist(qasmParser::ExplistContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitExp(qasmParser::ExpContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitId(qasmParser::IdContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitInteger(qasmParser::IntegerContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDecimal(qasmParser::DecimalContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitFilename(qasmParser::FilenameContext *ctx) override {
    return visitChildren(ctx);
  }


};

