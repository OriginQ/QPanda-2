
// Generated from .\originir.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"




class  originirLexer : public antlr4::Lexer {
public:
  enum {
    PI = 1, QINIT_KEY = 2, CREG_KEY = 3, Q_KEY = 4, C_KEY = 5, BARRIER_KEY = 6, 
    ECHO_GATE = 7, H_GATE = 8, X_GATE = 9, NOT_GATE = 10, T_GATE = 11, S_GATE = 12, 
    Y_GATE = 13, Z_GATE = 14, X1_GATE = 15, Y1_GATE = 16, Z1_GATE = 17, 
    I_GATE = 18, U2_GATE = 19, RPHI_GATE = 20, U3_GATE = 21, U4_GATE = 22, 
    RX_GATE = 23, RY_GATE = 24, RZ_GATE = 25, U1_GATE = 26, CNOT_GATE = 27, 
    CZ_GATE = 28, CU_GATE = 29, ISWAP_GATE = 30, SQISWAP_GATE = 31, SWAPZ1_GATE = 32, 
    ISWAPTHETA_GATE = 33, CR_GATE = 34, TOFFOLI_GATE = 35, DAGGER_KEY = 36, 
    ENDDAGGER_KEY = 37, CONTROL_KEY = 38, ENDCONTROL_KEY = 39, QIF_KEY = 40, 
    ELSE_KEY = 41, ENDIF_KEY = 42, QWHILE_KEY = 43, ENDQWHILE_KEY = 44, 
    MEASURE_KEY = 45, RESET_KEY = 46, ASSIGN = 47, GT = 48, LT = 49, NOT = 50, 
    EQ = 51, LEQ = 52, GEQ = 53, NE = 54, AND = 55, OR = 56, PLUS = 57, 
    MINUS = 58, MUL = 59, DIV = 60, COMMA = 61, LPAREN = 62, RPAREN = 63, 
    LBRACK = 64, RBRACK = 65, NEWLINE = 66, Identifier = 67, Integer_Literal = 68, 
    Double_Literal = 69, Digit_Sequence = 70, REALEXP = 71, WhiteSpace = 72, 
    SingleLineComment = 73
  };

  originirLexer(antlr4::CharStream *input);
  ~originirLexer();

  virtual std::string getGrammarFileName() const override;
  virtual const std::vector<std::string>& getRuleNames() const override;

  virtual const std::vector<std::string>& getChannelNames() const override;
  virtual const std::vector<std::string>& getModeNames() const override;
  virtual const std::vector<std::string>& getTokenNames() const override; // deprecated, use vocabulary instead
  virtual antlr4::dfa::Vocabulary& getVocabulary() const override;

  virtual const std::vector<uint16_t> getSerializedATN() const override;
  virtual const antlr4::atn::ATN& getATN() const override;

private:
  static std::vector<antlr4::dfa::DFA> _decisionToDFA;
  static antlr4::atn::PredictionContextCache _sharedContextCache;
  static std::vector<std::string> _ruleNames;
  static std::vector<std::string> _tokenNames;
  static std::vector<std::string> _channelNames;
  static std::vector<std::string> _modeNames;

  static std::vector<std::string> _literalNames;
  static std::vector<std::string> _symbolicNames;
  static antlr4::dfa::Vocabulary _vocabulary;
  static antlr4::atn::ATN _atn;
  static std::vector<uint16_t> _serializedATN;


  // Individual action functions triggered by action() above.

  // Individual semantic predicate functions triggered by sempred() above.

  struct Initializer {
    Initializer();
  };
  static Initializer _init;
};

