
// Generated from .\pyquil.g4 by ANTLR 4.8

#pragma once


#include "antlr4-runtime.h"
#include "Core/Utilities/Compiler/PyquilCompiler/pyquilListener.h"


/**
 * This class provides an empty implementation of pyquilListener,
 * which can be extended to create a listener which only needs to handle a subset
 * of the available methods.
 */
class  pyquilBaseListener : public pyquilListener {
public:

  virtual void enterProg(pyquilParser::ProgContext * /*ctx*/) override { }
  virtual void exitProg(pyquilParser::ProgContext * /*ctx*/) override { }

  virtual void enterCode_block(pyquilParser::Code_blockContext * /*ctx*/) override { }
  virtual void exitCode_block(pyquilParser::Code_blockContext * /*ctx*/) override { }

  virtual void enterLoop(pyquilParser::LoopContext * /*ctx*/) override { }
  virtual void exitLoop(pyquilParser::LoopContext * /*ctx*/) override { }

  virtual void enterLoop_start(pyquilParser::Loop_startContext * /*ctx*/) override { }
  virtual void exitLoop_start(pyquilParser::Loop_startContext * /*ctx*/) override { }

  virtual void enterLoop_end(pyquilParser::Loop_endContext * /*ctx*/) override { }
  virtual void exitLoop_end(pyquilParser::Loop_endContext * /*ctx*/) override { }

  virtual void enterLoop_if_continue(pyquilParser::Loop_if_continueContext * /*ctx*/) override { }
  virtual void exitLoop_if_continue(pyquilParser::Loop_if_continueContext * /*ctx*/) override { }

  virtual void enterOperation(pyquilParser::OperationContext * /*ctx*/) override { }
  virtual void exitOperation(pyquilParser::OperationContext * /*ctx*/) override { }

  virtual void enterDeclare(pyquilParser::DeclareContext * /*ctx*/) override { }
  virtual void exitDeclare(pyquilParser::DeclareContext * /*ctx*/) override { }

  virtual void enterMeasure(pyquilParser::MeasureContext * /*ctx*/) override { }
  virtual void exitMeasure(pyquilParser::MeasureContext * /*ctx*/) override { }

  virtual void enterMove(pyquilParser::MoveContext * /*ctx*/) override { }
  virtual void exitMove(pyquilParser::MoveContext * /*ctx*/) override { }

  virtual void enterSub(pyquilParser::SubContext * /*ctx*/) override { }
  virtual void exitSub(pyquilParser::SubContext * /*ctx*/) override { }

  virtual void enterVar_name(pyquilParser::Var_nameContext * /*ctx*/) override { }
  virtual void exitVar_name(pyquilParser::Var_nameContext * /*ctx*/) override { }

  virtual void enterVar_mem(pyquilParser::Var_memContext * /*ctx*/) override { }
  virtual void exitVar_mem(pyquilParser::Var_memContext * /*ctx*/) override { }

  virtual void enterQbit(pyquilParser::QbitContext * /*ctx*/) override { }
  virtual void exitQbit(pyquilParser::QbitContext * /*ctx*/) override { }

  virtual void enterGate(pyquilParser::GateContext * /*ctx*/) override { }
  virtual void exitGate(pyquilParser::GateContext * /*ctx*/) override { }

  virtual void enterBool_val(pyquilParser::Bool_valContext * /*ctx*/) override { }
  virtual void exitBool_val(pyquilParser::Bool_valContext * /*ctx*/) override { }

  virtual void enterParam(pyquilParser::ParamContext * /*ctx*/) override { }
  virtual void exitParam(pyquilParser::ParamContext * /*ctx*/) override { }

  virtual void enterExpr(pyquilParser::ExprContext * /*ctx*/) override { }
  virtual void exitExpr(pyquilParser::ExprContext * /*ctx*/) override { }

  virtual void enterArray_item(pyquilParser::Array_itemContext * /*ctx*/) override { }
  virtual void exitArray_item(pyquilParser::Array_itemContext * /*ctx*/) override { }

  virtual void enterArrayname(pyquilParser::ArraynameContext * /*ctx*/) override { }
  virtual void exitArrayname(pyquilParser::ArraynameContext * /*ctx*/) override { }

  virtual void enterIdx(pyquilParser::IdxContext * /*ctx*/) override { }
  virtual void exitIdx(pyquilParser::IdxContext * /*ctx*/) override { }


  virtual void enterEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void exitEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void visitTerminal(antlr4::tree::TerminalNode * /*node*/) override { }
  virtual void visitErrorNode(antlr4::tree::ErrorNode * /*node*/) override { }

};

