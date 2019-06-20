#ifndef  _INSTRUCTIONS_H_
#define  _INSTRUCTIONS_H_
#include <map>
#include <iostream>
#include"Core/Utilities/XMLConfigParam.h"

using config_map = std::map<std::string, std::map<std::string, uint32_t>> ;

uint32_t LOAD(uint32_t rs1, uint32_t rd, uint32_t imm);
uint32_t STORE(uint32_t rs1, uint32_t rd, uint32_t imm);

uint32_t BEQ(uint32_t rs1, uint32_t rs2, uint32_t imm);
uint32_t BNE(uint32_t rs1, uint32_t rs2, uint32_t imm);
uint32_t BLT(uint32_t rs1, uint32_t rs2, uint32_t imm);
uint32_t BGT(uint32_t rs1, uint32_t rs2, uint32_t imm);

uint32_t ADDI(uint32_t rs1, uint32_t rd, uint32_t imm);
uint32_t ANDI(uint32_t rs1, uint32_t rd, uint32_t imm);
uint32_t XORI(uint32_t rs1, uint32_t rd, uint32_t imm);
uint32_t ORI(uint32_t rs1, uint32_t rd, uint32_t imm);

uint32_t ADD(uint32_t rs1, uint32_t rs2, uint32_t rd);
uint32_t AND(uint32_t rs1, uint32_t rs2, uint32_t rd);
uint32_t XOR(uint32_t rs1, uint32_t rs2, uint32_t rd);
uint32_t OR(uint32_t rs1, uint32_t rs2, uint32_t rd);

uint32_t QWAITI(uint32_t imm);
uint32_t FMR(uint32_t rs1, uint32_t rd);
uint32_t SMIS(uint32_t rd, uint32_t imm);

uint32_t QI(uint32_t rs1, uint32_t rs2, uint32_t PI, uint32_t opcode1, uint32_t opcode2);
uint32_t MEASURE(uint32_t rs1, uint32_t PI);


class Instructions 
{
private:
    Instructions()
    {
        instructionConfig();
    };
    static void instructionConfig();
    static Instructions *m_manager;
public:
    static Instructions *getInstance();
};

#endif