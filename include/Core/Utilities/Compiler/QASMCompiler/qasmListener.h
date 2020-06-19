
// Generated from .\qasm.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"
#include "qasmParser.h"


/**
 * This interface defines an abstract listener for a parse tree produced by qasmParser.
 */
class  qasmListener : public antlr4::tree::ParseTreeListener {
public:

  virtual void enterMainprogram(qasmParser::MainprogramContext *ctx) = 0;
  virtual void exitMainprogram(qasmParser::MainprogramContext *ctx) = 0;

  virtual void enterHead_decl(qasmParser::Head_declContext *ctx) = 0;
  virtual void exitHead_decl(qasmParser::Head_declContext *ctx) = 0;

  virtual void enterVersion_decl(qasmParser::Version_declContext *ctx) = 0;
  virtual void exitVersion_decl(qasmParser::Version_declContext *ctx) = 0;

  virtual void enterInclude_decl(qasmParser::Include_declContext *ctx) = 0;
  virtual void exitInclude_decl(qasmParser::Include_declContext *ctx) = 0;

  virtual void enterStatement(qasmParser::StatementContext *ctx) = 0;
  virtual void exitStatement(qasmParser::StatementContext *ctx) = 0;

  virtual void enterReg_decl(qasmParser::Reg_declContext *ctx) = 0;
  virtual void exitReg_decl(qasmParser::Reg_declContext *ctx) = 0;

  virtual void enterOpaque_decl(qasmParser::Opaque_declContext *ctx) = 0;
  virtual void exitOpaque_decl(qasmParser::Opaque_declContext *ctx) = 0;

  virtual void enterIf_decl(qasmParser::If_declContext *ctx) = 0;
  virtual void exitIf_decl(qasmParser::If_declContext *ctx) = 0;

  virtual void enterBarrier_decl(qasmParser::Barrier_declContext *ctx) = 0;
  virtual void exitBarrier_decl(qasmParser::Barrier_declContext *ctx) = 0;

  virtual void enterGate_decl(qasmParser::Gate_declContext *ctx) = 0;
  virtual void exitGate_decl(qasmParser::Gate_declContext *ctx) = 0;

  virtual void enterGoplist(qasmParser::GoplistContext *ctx) = 0;
  virtual void exitGoplist(qasmParser::GoplistContext *ctx) = 0;

  virtual void enterBop(qasmParser::BopContext *ctx) = 0;
  virtual void exitBop(qasmParser::BopContext *ctx) = 0;

  virtual void enterQop(qasmParser::QopContext *ctx) = 0;
  virtual void exitQop(qasmParser::QopContext *ctx) = 0;

  virtual void enterUop(qasmParser::UopContext *ctx) = 0;
  virtual void exitUop(qasmParser::UopContext *ctx) = 0;

  virtual void enterAnylist(qasmParser::AnylistContext *ctx) = 0;
  virtual void exitAnylist(qasmParser::AnylistContext *ctx) = 0;

  virtual void enterIdlist(qasmParser::IdlistContext *ctx) = 0;
  virtual void exitIdlist(qasmParser::IdlistContext *ctx) = 0;

  virtual void enterId_index(qasmParser::Id_indexContext *ctx) = 0;
  virtual void exitId_index(qasmParser::Id_indexContext *ctx) = 0;

  virtual void enterArgument(qasmParser::ArgumentContext *ctx) = 0;
  virtual void exitArgument(qasmParser::ArgumentContext *ctx) = 0;

  virtual void enterExplist(qasmParser::ExplistContext *ctx) = 0;
  virtual void exitExplist(qasmParser::ExplistContext *ctx) = 0;

  virtual void enterExp(qasmParser::ExpContext *ctx) = 0;
  virtual void exitExp(qasmParser::ExpContext *ctx) = 0;

  virtual void enterId(qasmParser::IdContext *ctx) = 0;
  virtual void exitId(qasmParser::IdContext *ctx) = 0;

  virtual void enterReal(qasmParser::RealContext *ctx) = 0;
  virtual void exitReal(qasmParser::RealContext *ctx) = 0;

  virtual void enterInteger(qasmParser::IntegerContext *ctx) = 0;
  virtual void exitInteger(qasmParser::IntegerContext *ctx) = 0;

  virtual void enterDecimal(qasmParser::DecimalContext *ctx) = 0;
  virtual void exitDecimal(qasmParser::DecimalContext *ctx) = 0;

  virtual void enterFilename(qasmParser::FilenameContext *ctx) = 0;
  virtual void exitFilename(qasmParser::FilenameContext *ctx) = 0;


};

