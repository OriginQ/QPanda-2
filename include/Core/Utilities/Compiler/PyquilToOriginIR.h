/*
Copyright (c) 2017-2024 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
/*! \file QASMToQPorg.h */

#ifndef  _PYQUIL_TO_ORIGINIR_H
#define  _PYQUIL_TO_ORIGINIR_H
#include "ThirdParty/antlr4/runtime/src/antlr4-runtime.h"
#include "Core/Utilities/Compiler/PyquilCompiler/pyquilLexer.h"
#include "Core/Utilities/Compiler/PyquilCompiler/pyquilParser.h"
#include "Core/Utilities/Compiler/PyquilCompiler/pyquilVisitor.h"
#include "Core/Utilities/Compiler/PyquilCompiler/pyquilBaseVisitor.h"


#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QReset.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/QuantumMeasure.h"

#include "Core/Utilities/Compiler/Definitions.hpp"
#include "Core/Utilities/Tools/MatrixDecomposition.h"
#include "Core/Utilities/QProgTransform/TransformDecomposition.h"
#include "Core/Utilities/Compiler/QProgToOriginIR.h"
#include "Core/Utilities/Compiler/UmatrixToOriginIR.h"

#include <cmath>
#include <complex>



namespace QPanda {
    extern QProg convert_originir_string_to_qprog(std::string str_originir, QuantumMachine* qm);
    extern QProg convert_originir_string_to_qprog(std::string str_originir, QuantumMachine* qm, QVec& qv, std::vector<ClassicalCondition>& cv);
    class PyquilToOriginIR : public pyquilBaseVisitor
    {
    private:
        std::stringstream ss_dec;
        std::stringstream ss_cdblk;
        std::unordered_map<std::string, double> var_name_val;
        std::unordered_map<std::string, uint32_t> var_name_cbit_idx;
        uint32_t cbit_max_idx;
        uint32_t qbit_max_idx;
    public:
        PyquilToOriginIR() {
            cbit_max_idx = 0;
            qbit_max_idx = 0;
        }
        ~PyquilToOriginIR() {}
        virtual antlrcpp::Any visitProg(pyquilParser::ProgContext* ctx) override {
            for (auto& dec : ctx->declare()) {
                visitDeclare(dec);
            }
            for (auto& cb : ctx->code_block()) {
                visitCode_block(cb);
            }
            ss_dec << "QINIT " << qbit_max_idx + 1 << "\n";
            ss_dec << "CREG " << cbit_max_idx + 1 << "\n";
            return ss_dec.str() + ss_cdblk.str();
        }

        virtual antlrcpp::Any visitCode_block(pyquilParser::Code_blockContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitLoop(pyquilParser::LoopContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitLoop_start(pyquilParser::Loop_startContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitLoop_end(pyquilParser::Loop_endContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitLoop_if_continue(pyquilParser::Loop_if_continueContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitOperation(pyquilParser::OperationContext* ctx) override {
            return  visitChildren(ctx);;
        }

        virtual antlrcpp::Any visitDeclare(pyquilParser::DeclareContext* ctx) override {
            std::string var_name = ctx->children[2]->getText();
            uint32_t var_mem_size = std::stoi(ctx->children[4]->children[2]->getText());
            for (int i = 0; i < var_mem_size; i++) {
                std::stringstream ss;
                cbit_max_idx += 1;
                ss << var_name << "[" << cbit_max_idx << "]";
                var_name_cbit_idx[ss.str()] = cbit_max_idx;
                var_name_val[ss.str()] = 0;
            }

            return nullptr;
        }

        virtual antlrcpp::Any visitMeasure(pyquilParser::MeasureContext* ctx) override {
            ss_cdblk << "MEASURE" << " ";
            ss_cdblk<<"q["<<(int)visitQbit(dynamic_cast<pyquilParser::QbitContext*>(ctx->children[2]))<<"]";
            std::string array_item = ctx->children[4]->getText();
            int mem_idx = var_name_cbit_idx[array_item];
            ss_cdblk << "," << "c[" << mem_idx << "]\n";
            return nullptr;
        }

        virtual antlrcpp::Any visitMove(pyquilParser::MoveContext* ctx) override {
            var_name_val[ctx->array_item()->getText()] = (double)visitExpr(ctx->expr());
            ss_cdblk << "c[" << var_name_cbit_idx[ctx->array_item()->getText()] << "]=" << (double)visitExpr(ctx->expr()) << "\n";
            return nullptr;
        }

        virtual antlrcpp::Any visitSub(pyquilParser::SubContext* ctx) override {
            var_name_val[ctx->array_item()->getText()] -= (double)visitExpr(ctx->expr());
            ss_cdblk << "c[" << var_name_cbit_idx[ctx->array_item()->getText()] << "]=" << var_name_val[ctx->array_item()->getText()] << "-" << (double)visitExpr(ctx->expr()) << "\n";
            return nullptr;
        }

        virtual antlrcpp::Any visitVar_name(pyquilParser::Var_nameContext* ctx) override {
            return ctx->getText();
        }

        virtual antlrcpp::Any visitVar_mem(pyquilParser::Var_memContext* ctx) override {
            std::stringstream ss;
            for (int i = 0; i < ctx->children.size(); i++) {
                if (i == 2) {
                    ss<<(int)visitIdx(dynamic_cast<pyquilParser::IdxContext*>(ctx->children[i]));
                }
                else {
                    ss << ctx->children[i];
                }
            }
            return ss.str();
        }

        virtual antlrcpp::Any visitQbit(pyquilParser::QbitContext* ctx) override {
            int idx = std::stoi(ctx->getText());
            if (idx > qbit_max_idx) {
                qbit_max_idx = idx;
            }
            return idx;
        }

        std::string gate_contex2originir(pyquilParser::GateContext* ctx);
        std::vector<unsigned int> gate_contex_to_qbit_idxs(pyquilParser::GateContext* ctx);
        std::vector<double> gate_contex_to_params(pyquilParser::GateContext* ctx);
        QStat gate_contex_to_matrix(pyquilParser::GateContext* ctx);
        std::string gate_contex2originir_by_umatrix(pyquilParser::GateContext* ctx);
        static std::string gate_contex2originir_by_irsupported(const std::string& gatename, const std::vector<unsigned int> qbit_idxs, const std::vector<double>params);
        
        virtual antlrcpp::Any visitGate(pyquilParser::GateContext* ctx) override {
            std::string gatename = ctx->children[0]->getText();
            auto qbit_idxs = gate_contex_to_qbit_idxs(ctx);
            auto params = gate_contex_to_params(ctx);
          
            std::string originir_str;
            const std::set<std::string> convert_need_umatrix = {"SQISW","PSWAP","ISWAP","XY","FSIM","PHASEDFSIM"};//pyquil程序库当前时间实际不支持SQISW、FSIM和PHASEDFSIM，这三个门的Pyquil指令串解析对应的词法规则暂未定义
            const std::set<std::string> gate_originir_supported = {"I","Z","Y","X","H","S","T","CZ","CNOT","SWAP"};
            if (convert_need_umatrix.count(gatename)) {
                originir_str = gate_contex2originir_by_umatrix(ctx);
            }
            else if(gate_originir_supported.count(gatename)){
                originir_str = gate_contex2originir_by_irsupported(gatename, qbit_idxs, params);
            }
            else if (gatename == "PHASE") {
                originir_str = gate_contex2originir_by_irsupported("P", qbit_idxs, params);
            }
            else {
                std::cerr << "Error[PyquilToOriginIR ]: don't support gate:" << gatename << std::endl;
            }
            ss_cdblk << originir_str;
            return nullptr;
        }

        virtual antlrcpp::Any visitBool_val(pyquilParser::Bool_valContext* ctx) override {
            std::cout << ctx->getText();
            return ctx->getText();
        }

        virtual antlrcpp::Any visitParam(pyquilParser::ParamContext* ctx) override {
            return (double)visitExpr(ctx->expr());

        }

        virtual antlrcpp::Any visitExpr(pyquilParser::ExprContext* ctx) override {
            double res = 0;
            if (ctx->children.size() == 1) {
                if (ctx->INT()) {
                    res= std::stod(ctx->INT()->getText());
                }
                else if (ctx->FLOAT()) {
                    res= std::stod(ctx->FLOAT()->getText());
                }
                else if (ctx->array_item()) {
                    res= var_name_val[ctx->array_item()->getText()];
                }
                else {
                    std::cerr << "expr error. Exit(-1). " << std::endl;
                    exit(-1);
                }
            }
            else if (ctx->children.size() == 2) {
                if (ctx->getStart()->getText() == "-") {
                    res = 0 - (double)visitExpr(dynamic_cast<pyquilParser::ExprContext*>(ctx->children[1]));
                }
                else {
                    std::cerr << "expr error. Exit(-1). " << std::endl;
                    exit(-1);
                }
            }
            else if (ctx->children.size() == 3) {
                if (ctx->getStart()->getText() == "(" && ctx->getStop()->getText() == ")") {
                    res = (double)visitExpr(dynamic_cast<pyquilParser::ExprContext*>(ctx->children[1]));
                }

                else if (ctx->children[1]->getText() == "*") {
                    res = (double)visitExpr(dynamic_cast<pyquilParser::ExprContext*>(ctx->children[0])) * (double)visitExpr(dynamic_cast<pyquilParser::ExprContext*>(ctx->children[2]));
                }
                else if (ctx->children[1]->getText() == "/") {
                    res = (double)visitExpr(dynamic_cast<pyquilParser::ExprContext*>(ctx->children[0])) / (double)visitExpr(dynamic_cast<pyquilParser::ExprContext*>(ctx->children[2]));
                }
                else if (ctx->children[1]->getText() == "+") {
                    res = (double)visitExpr(dynamic_cast<pyquilParser::ExprContext*>(ctx->children[0])) + (double)visitExpr(dynamic_cast<pyquilParser::ExprContext*>(ctx->children[2]));
                }
                else if (ctx->children[1]->getText() == "-") {
                    res = (double)visitExpr(dynamic_cast<pyquilParser::ExprContext*>(ctx->children[0])) - (double)visitExpr(dynamic_cast<pyquilParser::ExprContext*>(ctx->children[2]));
                }
                else {
                    std::cerr << "expr error. Exit(-1). " << std::endl;
                    exit(-1);
                }
            }

            else {
                std::cerr << "expr error. Exit(-1). " << std::endl;
                exit(-1);
            }
            return res;
        }

        virtual antlrcpp::Any visitArray_item(pyquilParser::Array_itemContext* ctx) override {
            ss_cdblk << ctx->getText();
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitArrayname(pyquilParser::ArraynameContext* ctx) override {
            return ctx->getText();
        }

        virtual antlrcpp::Any visitIdx(pyquilParser::IdxContext* ctx) override {
            return std::stoi(ctx->getText());
        }

      
    private:

    };

    std::string convert_pyquil_string_to_originir(std::string pyquil_str);
    QProg convert_pyquil_string_to_qprog(std::string pyquil_str, QuantumMachine* qm);
    QProg convert_pyquil_string_to_qprog(std::string pyquil_str, QuantumMachine* qm, QVec& qv, std::vector<ClassicalCondition>& cv);
    std::string convert_pyquil_file_to_originir(std::string pyquil_filepath);
    QProg convert_pyquil_file_to_qprog(std::string pyquil_filepath, QuantumMachine* qm);
    QProg convert_pyquil_file_to_qprog(std::string pyquil_filepath, QuantumMachine* qm, QVec& qv, std::vector<ClassicalCondition>& cv);
};
#endif //!_QASMTOQPORG_H