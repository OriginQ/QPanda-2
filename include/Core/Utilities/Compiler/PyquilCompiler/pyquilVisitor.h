
// Generated from .\pyquil.g4 by ANTLR 4.8

#pragma once


#include "antlr4-runtime.h"
#include "Core/Utilities/Compiler/PyquilCompiler/pyquilParser.h"



/**
 * This class defines an abstract visitor for a parse tree
 * produced by pyquilParser.
 */
class  pyquilVisitor : public antlr4::tree::AbstractParseTreeVisitor {
public:

  /**
   * Visit parse trees produced by pyquilParser.
   */
    virtual antlrcpp::Any visitProg(pyquilParser::ProgContext *context) = 0;

    virtual antlrcpp::Any visitCode_block(pyquilParser::Code_blockContext *context) = 0;

    virtual antlrcpp::Any visitLoop(pyquilParser::LoopContext *context) = 0;

    virtual antlrcpp::Any visitLoop_start(pyquilParser::Loop_startContext *context) = 0;

    virtual antlrcpp::Any visitLoop_end(pyquilParser::Loop_endContext *context) = 0;

    virtual antlrcpp::Any visitLoop_if_continue(pyquilParser::Loop_if_continueContext *context) = 0;

    virtual antlrcpp::Any visitOperation(pyquilParser::OperationContext *context) = 0;

    virtual antlrcpp::Any visitDeclare(pyquilParser::DeclareContext *context) = 0;

    virtual antlrcpp::Any visitMeasure(pyquilParser::MeasureContext *context) = 0;

    virtual antlrcpp::Any visitMove(pyquilParser::MoveContext *context) = 0;

    virtual antlrcpp::Any visitSub(pyquilParser::SubContext *context) = 0;

    virtual antlrcpp::Any visitVar_name(pyquilParser::Var_nameContext *context) = 0;

    virtual antlrcpp::Any visitVar_mem(pyquilParser::Var_memContext *context) = 0;

    virtual antlrcpp::Any visitQbit(pyquilParser::QbitContext *context) = 0;

    virtual antlrcpp::Any visitGate(pyquilParser::GateContext *context) = 0;

    virtual antlrcpp::Any visitBool_val(pyquilParser::Bool_valContext *context) = 0;

    virtual antlrcpp::Any visitParam(pyquilParser::ParamContext *context) = 0;

    virtual antlrcpp::Any visitExpr(pyquilParser::ExprContext *context) = 0;

    virtual antlrcpp::Any visitArray_item(pyquilParser::Array_itemContext *context) = 0;

    virtual antlrcpp::Any visitArrayname(pyquilParser::ArraynameContext *context) = 0;

    virtual antlrcpp::Any visitIdx(pyquilParser::IdxContext *context) = 0;


};

