#include "Core/Utilities/Compiler/PyquilToOriginIR.h"



namespace QPanda{
    
    
	std::string convert_pyquil_string_to_originir(std::string pyquil_str)
	{
		antlr4::ANTLRInputStream input(pyquil_str);
		pyquilLexer lexer(&input);
		antlr4::CommonTokenStream tokens(&lexer);
		pyquilParser parser(&tokens);

		pyquilParser::ProgContext*  tree = parser.prog();
		PyquilToOriginIR pyquilToOriginIR;
		return pyquilToOriginIR.visitProg(tree);
	}
    
    QProg convert_pyquil_string_to_qprog(std::string pyquil_str, QuantumMachine* qm)
    {
        return convert_originir_string_to_qprog(convert_pyquil_string_to_originir(pyquil_str),qm);
    }

    QProg convert_pyquil_string_to_qprog(std::string pyquil_str, QuantumMachine* qm, QVec& qv, std::vector<ClassicalCondition>& cv)
    {
        return convert_originir_string_to_qprog(convert_pyquil_string_to_originir(pyquil_str), qm, qv,cv);
    }

    std::string convert_pyquil_file_to_originir(std::string pyquil_filepath) {
        std::ifstream ifs(pyquil_filepath);
        std::stringstream ss;
        if (ifs.is_open()) {
            ss << ifs.rdbuf();
            ifs.close();
            return convert_pyquil_string_to_originir(ss.str());
        }
        else {
            std::cerr << "Error [PyquilToOriginIR convert_pyquil_file_to_originir] can't open file:" << pyquil_filepath << std::endl;
            exit(-1);
        }
        return "";
    }

    QProg convert_pyquil_file_to_qprog(std::string pyquil_filepath, QuantumMachine* qm)
    {
        return convert_originir_string_to_qprog(convert_pyquil_file_to_originir(pyquil_filepath), qm);
    }
    QProg convert_pyquil_file_to_qprog(std::string pyquil_filepath, QuantumMachine* qm, QVec& qv, std::vector<ClassicalCondition>& cv)
    {
        return convert_originir_string_to_qprog(convert_pyquil_file_to_originir(pyquil_filepath), qm, qv, cv);
    }

    std::vector<unsigned int> PyquilToOriginIR::gate_contex_to_qbit_idxs(pyquilParser::GateContext* ctx) {
        std::vector<unsigned int> qbit_idxs;
        std::vector<unsigned int> token_idxs;
        if (ctx->GATE1Q()) {
            token_idxs = { 2 };
        }else if (ctx->GATE1Q1P()) {
            token_idxs = { 5 };
        }
        else if (ctx->GATE2Q()) {
            token_idxs = { 2,4 };
        }
        else if (ctx->GATE2Q1P()) {
            token_idxs = { 5,7 };
        }
        else if (ctx->GATE3Q()) {
            token_idxs = { 2,4,6 };
        }
        else {
            std::cerr << "PyauilToOriginIR£¬gate_contex_to_qbit_idxs£¬gate not supported!" << std::endl;
            return { };
        }
        for (unsigned int i : token_idxs) {
            qbit_idxs.push_back((int)visitQbit(dynamic_cast<pyquilParser::QbitContext*>(ctx->children[i])));
        }
        return qbit_idxs;
    }
    std::vector<double> PyquilToOriginIR::gate_contex_to_params(pyquilParser::GateContext* ctx) {
        std::vector<double> params;
        std::vector<unsigned int> token_idxs;
        if (ctx->GATE1Q1P()|| ctx->GATE2Q1P()) {
            token_idxs = { 2 };
        }
        for (auto i : token_idxs) {
            params.push_back((double)visitParam(dynamic_cast<pyquilParser::ParamContext*>(ctx->children[i])));
        }
        return params;
    }
    std::string PyquilToOriginIR::gate_contex2originir_by_umatrix(pyquilParser::GateContext* ctx) {
        const std::string gatename = ctx->children[0]->getText();
        const auto qbit_idxs = gate_contex_to_qbit_idxs(ctx);
        auto umatrix = gate_contex_to_matrix(ctx);
        return UTIR::convert_matrix_to_originir_without_declare(umatrix, qbit_idxs);
    }
    QStat PyquilToOriginIR::gate_contex_to_matrix(pyquilParser::GateContext* ctx) {
        std::string gatename = ctx->children[0]->getText();
        const auto params = gate_contex_to_params(ctx);
        QStat umatrix;
        if (gatename == "CSWAP") {
            umatrix = std::get<1>(UTIR::CSWAP());
        }
        else if (gatename == "ISWAP") {
            umatrix = std::get<1>(UTIR::ISWAP());
        }
        else if (gatename == "SQISW") {
            umatrix = std::get<1>(UTIR::SQISW());
        }
        else if (gatename == "PSWAP") {
            umatrix = std::get<1>(UTIR::PSWAP(params));
        }
        else if (gatename == "XY") {
            umatrix = std::get<1>(UTIR::XY(params));
        }
        else if (gatename == "FSIM") {
            umatrix = std::get<1>(UTIR::FSIM(params));
        }
        else if (gatename == "PHASEDFSIM") {
            umatrix = std::get<1>(UTIR::PHASEDFSIM(params));
        }
        else {

        }
        return umatrix;
    }

    std::string PyquilToOriginIR::gate_contex2originir_by_irsupported(const std::string& gatename, const std::vector<unsigned int> qbit_idxs, const std::vector<double>params) {
        std::stringstream ss;
        ss << gatename << " " << "q[" << qbit_idxs[0] << "]";
        for (int i = 1; i < qbit_idxs.size(); i++) {
            ss << ",q[" << qbit_idxs[i] << "]";
        }
        if (params.size() > 0) {
            ss << ",(" << params[0];
            for (int i = 1; i < params.size(); i++) {
                ss << "," << params[i];
            }
            ss << ")";
        }
        ss << "\n";
        return ss.str();
    }
};