
// Generated from .\qasm.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"
#include "qasmParser.h"



/**
 * This class defines an abstract visitor for a parse tree
 * produced by qasmParser.
 */
class  qasmVisitor : public antlr4::tree::AbstractParseTreeVisitor {
public:

  /**
   * Visit parse trees produced by qasmParser.
   */
    virtual antlrcpp::Any visitMainprogram(qasmParser::MainprogramContext *context) = 0;

    virtual antlrcpp::Any visitHead_decl(qasmParser::Head_declContext *context) = 0;

    virtual antlrcpp::Any visitVersion_decl(qasmParser::Version_declContext *context) = 0;

    virtual antlrcpp::Any visitInclude_decl(qasmParser::Include_declContext *context) = 0;

    virtual antlrcpp::Any visitStatement(qasmParser::StatementContext *context) = 0;

    virtual antlrcpp::Any visitReg_decl(qasmParser::Reg_declContext *context) = 0;

    virtual antlrcpp::Any visitOpaque_decl(qasmParser::Opaque_declContext *context) = 0;

    virtual antlrcpp::Any visitIf_decl(qasmParser::If_declContext *context) = 0;

    virtual antlrcpp::Any visitBarrier_decl(qasmParser::Barrier_declContext *context) = 0;

    virtual antlrcpp::Any visitGate_decl(qasmParser::Gate_declContext *context) = 0;

    virtual antlrcpp::Any visitGoplist(qasmParser::GoplistContext *context) = 0;

    virtual antlrcpp::Any visitBop(qasmParser::BopContext *context) = 0;

    virtual antlrcpp::Any visitQop(qasmParser::QopContext *context) = 0;

    virtual antlrcpp::Any visitUop(qasmParser::UopContext *context) = 0;

    virtual antlrcpp::Any visitAnylist(qasmParser::AnylistContext *context) = 0;

    virtual antlrcpp::Any visitIdlist(qasmParser::IdlistContext *context) = 0;

    virtual antlrcpp::Any visitId_index(qasmParser::Id_indexContext *context) = 0;

    virtual antlrcpp::Any visitArgument(qasmParser::ArgumentContext *context) = 0;

    virtual antlrcpp::Any visitExplist(qasmParser::ExplistContext *context) = 0;

    virtual antlrcpp::Any visitExp(qasmParser::ExpContext *context) = 0;

    virtual antlrcpp::Any visitId(qasmParser::IdContext *context) = 0;

    virtual antlrcpp::Any visitInteger(qasmParser::IntegerContext *context) = 0;

    virtual antlrcpp::Any visitDecimal(qasmParser::DecimalContext *context) = 0;

    virtual antlrcpp::Any visitFilename(qasmParser::FilenameContext *context) = 0;


};

