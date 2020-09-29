#include"Core/Utilities/Compiler/Micro-architecture/instructions.h"
#include "Core/Utilities/Tools/QPandaException.h"
USING_QPANDA
using namespace std;

static config_map ins_config;
Instructions *Instructions::m_manager = new Instructions;

Instructions *Instructions::getInstance()
{
        return m_manager;
}

static uint32_t extract_imm(uint32_t imm, uint32_t l, uint32_t h)
{
    return  (imm >> l)&((1 << (h + 1)) - 1);
}

uint32_t LOAD(uint32_t rs1, uint32_t rd, uint32_t imm)
{
    return 0 | extract_imm(imm, 0, 16) << 15
             | rs1 << 10
             | rd << 5
             | ins_config["load"]["opcode_4_3"] << 3
             | ins_config["load"]["opcode_2_1"] << 1;
}

uint32_t STORE(uint32_t rs1, uint32_t rs2, uint32_t imm)
{
    return 0 | extract_imm(imm, 14, 16) << 29 
             | rs2 << 24 
             | extract_imm(imm, 0, 8) << 15 
             | rs1 << 10 
             | extract_imm(imm, 9, 13) << 5
             | ins_config["store"]["opcode_4_3"] << 3
             | ins_config["store"]["opcode_2_1"] << 1;
}

uint32_t BEQ(uint32_t rs1, uint32_t rs2, uint32_t imm)
{
    return 0 | ins_config["branch"]["BEQ_func_3"] << 29
             | rs2 << 24
             | extract_imm(imm, 0, 8) << 15
             | rs1 << 10
             | extract_imm(imm, 9, 13) << 5
             | ins_config["branch"]["opcode_4_3"] << 3
             | ins_config["branch"]["opcode_2_1"] << 1;
}

uint32_t BNE(uint32_t rs1, uint32_t rs2, uint32_t imm)
{
    return 0 | ins_config["branch"]["BNE_func_3"] << 29
             | rs2 << 24
             | extract_imm(imm, 0, 8) << 15
             | rs1 << 10
             | extract_imm(imm, 9, 13) << 5
             | ins_config["branch"]["opcode_4_3"] << 3
             | ins_config["branch"]["opcode_2_1"] << 1;
}

uint32_t BLT(uint32_t rs1, uint32_t rs2, uint32_t imm)
{
    return 0 | ins_config["branch"]["BLT_func_3"] << 29
             | rs2 << 24
             | extract_imm(imm, 0, 8) << 15
             | rs1 << 10
             | extract_imm(imm, 9, 13) << 5
             | ins_config["branch"]["opcode_4_3"] << 3
             | ins_config["branch"]["opcode_2_1"] << 1;
}

uint32_t BGT(uint32_t rs1, uint32_t rs2, uint32_t imm)
{
    return 0 | ins_config["branch"]["BGT_func_3"] << 29
             | rs2 << 24
             | extract_imm(imm, 0, 8) << 15
             | rs1 << 10
             | extract_imm(imm, 9, 13) << 5
             | ins_config["branch"]["opcode_4_3"] << 3
             | ins_config["branch"]["opcode_2_1"] << 1;
}

uint32_t ADDI(uint32_t rs1, uint32_t rd, uint32_t imm)
{
    return 0 | ins_config["op_imm"]["ADDI_func_3"] << 29
             | extract_imm(imm, 9, 13) << 24
             | extract_imm(imm, 0, 8) << 15
             | rs1 << 10
             | rd << 5
             | ins_config["op_imm"]["opcode_4_3"] << 3
             | ins_config["op_imm"]["opcode_2_1"] << 1;
}

uint32_t ANDI(uint32_t rs1, uint32_t rd, uint32_t imm)
{
    return 0 | ins_config["op_imm"]["ANDI_func_3"] << 29
             | extract_imm(imm, 9, 13) << 24
             | extract_imm(imm, 0, 8) << 15
             | rs1 << 10
             | rd << 5
             | ins_config["op_imm"]["opcode_4_3"] << 3
             | ins_config["op_imm"]["opcode_2_1"] << 1;
}

uint32_t XORI(uint32_t rs1, uint32_t rd, uint32_t imm)
{
    return 0 | ins_config["op_imm"]["XORI_func_3"] << 29
             | extract_imm(imm, 9, 13) << 24
             | extract_imm(imm, 0, 8) << 15
             | rs1 << 10
             | rd << 5
             | ins_config["op_imm"]["opcode_4_3"] << 3
             | ins_config["op_imm"]["opcode_2_1"] << 1;
}

uint32_t ORI(uint32_t rs1, uint32_t rd, uint32_t imm)
{
    return 0 | ins_config["op_imm"]["ORI_func_3"] << 29
             | extract_imm(imm, 9, 13) << 24
             | extract_imm(imm, 0, 8) << 15
             | rs1 << 10
             | rd << 5
             | ins_config["op_imm"]["opcode_4_3"] << 3
             | ins_config["op_imm"]["opcode_2_1"] << 1;
}

uint32_t ADD(uint32_t rs1, uint32_t rs2, uint32_t rd)
{
    return 0 | ins_config["op"]["ADD_fun_3"] << 29
             | rs2 << 24
             | 0 << 15
             | rs1 << 10
             | rd << 5
             | ins_config["op"]["opcode_4_3"] << 3
             | ins_config["op"]["opcode_2_1"] << 1;
}

uint32_t AND(uint32_t rs1, uint32_t rs2, uint32_t rd)
{
    return 0 | ins_config["op"]["AND_fun_3"] << 29
             | rs2 << 24
             | 0 << 15
             | rs1 << 10
             | rd << 5
             | ins_config["op"]["opcode_4_3"] << 3
             | ins_config["op"]["opcode_2_1"] << 1;
}

uint32_t XOR(uint32_t rs1, uint32_t rs2, uint32_t rd)
{
    return 0 | ins_config["op"]["XOR_fun_3"] << 29
             | rs2 << 24
             | 0 << 15
             | rs1 << 10
             | rd << 5
             | ins_config["op"]["opcode_4_3"] << 3
             | ins_config["op"]["opcode_2_1"] << 1;
}

uint32_t OR(uint32_t rs1, uint32_t rs2, uint32_t rd)
{
    return 0 | ins_config["op"]["OR_fun_3"] << 29
             | rs2 << 24
             | 0 << 15
             | rs1 << 10
             | rd << 5
             | ins_config["op"]["opcode_4_3"] << 3
             | ins_config["op"]["opcode_2_1"] << 1;
}

uint32_t QWAITI(uint32_t imm)
{
    return 0 | extract_imm(imm, 24, 26) << 29
             | extract_imm(imm, 14, 18) << 24
             | extract_imm(imm, 0, 8) << 15
             | extract_imm(imm, 9, 13) << 10
             | extract_imm(imm, 19, 23) << 5
             | ins_config["qwait"]["opcode_4_3"] << 3
             | ins_config["qwait"]["opcode_2_1"] << 1;
}

uint32_t FMR(uint32_t rs1, uint32_t rd)
{
    return 0 | ins_config["fmr"]["FMR_fun_3"] << 29
             | 0 << 24
             | 0 << 15
             | rs1 << 10
             | rd << 5
             | ins_config["fmr"]["opcode_4_3"] << 3
             | ins_config["fmr"]["opcode_2_1"] << 1;
}

uint32_t SMIS(uint32_t rd, uint32_t imm)
{
    return 0 | extract_imm(imm, 19, 21) << 29
             | extract_imm(imm, 14, 18) << 24
             | extract_imm(imm, 0, 8) << 15
             | extract_imm(imm, 9, 13) << 10
             | rd << 5
             | ins_config["smist"]["opcode_4_3"] << 3
             | ins_config["smist"]["opcode_2_1"] << 1;
}

uint32_t QI(uint32_t rs1, uint32_t rs2, uint32_t pi,
            uint32_t opcode1, uint32_t opcode2)
{
    return 0 | pi << 29
             | rs2 << 24
             | opcode2 << 15
             | rs1 << 10
             | opcode1 << 1
             | 1 << 0;
}

uint32_t MEASURE(uint32_t rs1, uint32_t pi)
{
    return 0 | pi << 29
             | 0 << 15
             | rs1 << 10
             | 1023 << 1;
}

void Instructions::instructionConfig()
{
	JsonConfigParam config;
    if (!config.load_config(CONFIG_PATH))
    {
        throw run_fail("config");
    }
    config.getInstructionConfig(ins_config);
}
