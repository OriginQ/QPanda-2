
// Generated from .\pyquil.g4 by ANTLR 4.8

#pragma once


#include "antlr4-runtime.h"
#include "Core/Utilities/Compiler/PyquilCompiler/pyquilParser.h"


/**
 * This interface defines an abstract listener for a parse tree produced by pyquilParser.
 */
class  pyquilListener : public antlr4::tree::ParseTreeListener {
public:

  virtual void enterProg(pyquilParser::ProgContext *ctx) = 0;
  virtual void exitProg(pyquilParser::ProgContext *ctx) = 0;

  virtual void enterCode_block(pyquilParser::Code_blockContext *ctx) = 0;
  virtual void exitCode_block(pyquilParser::Code_blockContext *ctx) = 0;

  virtual void enterLoop(pyquilParser::LoopContext *ctx) = 0;
  virtual void exitLoop(pyquilParser::LoopContext *ctx) = 0;

  virtual void enterLoop_start(pyquilParser::Loop_startContext *ctx) = 0;
  virtual void exitLoop_start(pyquilParser::Loop_startContext *ctx) = 0;

  virtual void enterLoop_end(pyquilParser::Loop_endContext *ctx) = 0;
  virtual void exitLoop_end(pyquilParser::Loop_endContext *ctx) = 0;

  virtual void enterLoop_if_continue(pyquilParser::Loop_if_continueContext *ctx) = 0;
  virtual void exitLoop_if_continue(pyquilParser::Loop_if_continueContext *ctx) = 0;

  virtual void enterOperation(pyquilParser::OperationContext *ctx) = 0;
  virtual void exitOperation(pyquilParser::OperationContext *ctx) = 0;

  virtual void enterDeclare(pyquilParser::DeclareContext *ctx) = 0;
  virtual void exitDeclare(pyquilParser::DeclareContext *ctx) = 0;

  virtual void enterMeasure(pyquilParser::MeasureContext *ctx) = 0;
  virtual void exitMeasure(pyquilParser::MeasureContext *ctx) = 0;

  virtual void enterMove(pyquilParser::MoveContext *ctx) = 0;
  virtual void exitMove(pyquilParser::MoveContext *ctx) = 0;

  virtual void enterSub(pyquilParser::SubContext *ctx) = 0;
  virtual void exitSub(pyquilParser::SubContext *ctx) = 0;

  virtual void enterVar_name(pyquilParser::Var_nameContext *ctx) = 0;
  virtual void exitVar_name(pyquilParser::Var_nameContext *ctx) = 0;

  virtual void enterVar_mem(pyquilParser::Var_memContext *ctx) = 0;
  virtual void exitVar_mem(pyquilParser::Var_memContext *ctx) = 0;

  virtual void enterQbit(pyquilParser::QbitContext *ctx) = 0;
  virtual void exitQbit(pyquilParser::QbitContext *ctx) = 0;

  virtual void enterGate(pyquilParser::GateContext *ctx) = 0;
  virtual void exitGate(pyquilParser::GateContext *ctx) = 0;

  virtual void enterBool_val(pyquilParser::Bool_valContext *ctx) = 0;
  virtual void exitBool_val(pyquilParser::Bool_valContext *ctx) = 0;

  virtual void enterParam(pyquilParser::ParamContext *ctx) = 0;
  virtual void exitParam(pyquilParser::ParamContext *ctx) = 0;

  virtual void enterExpr(pyquilParser::ExprContext *ctx) = 0;
  virtual void exitExpr(pyquilParser::ExprContext *ctx) = 0;

  virtual void enterArray_item(pyquilParser::Array_itemContext *ctx) = 0;
  virtual void exitArray_item(pyquilParser::Array_itemContext *ctx) = 0;

  virtual void enterArrayname(pyquilParser::ArraynameContext *ctx) = 0;
  virtual void exitArrayname(pyquilParser::ArraynameContext *ctx) = 0;

  virtual void enterIdx(pyquilParser::IdxContext *ctx) = 0;
  virtual void exitIdx(pyquilParser::IdxContext *ctx) = 0;


};

