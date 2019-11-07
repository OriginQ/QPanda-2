
// Generated from .\originir.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"




class  originirLexer : public antlr4::Lexer {
public:
  enum {
    PI = 1, QINIT_KEY = 2, CREG_KEY = 3, Q_KEY = 4, C_KEY = 5, H_GATE = 6, 
    X_GATE = 7, NOT_GATE = 8, T_GATE = 9, S_GATE = 10, Y_GATE = 11, Z_GATE = 12, 
    X1_GATE = 13, Y1_GATE = 14, Z1_GATE = 15, I_GATE = 16, U2_GATE = 17, 
    U3_GATE = 18, U4_GATE = 19, RX_GATE = 20, RY_GATE = 21, RZ_GATE = 22, 
    U1_GATE = 23, CNOT_GATE = 24, CZ_GATE = 25, CU_GATE = 26, ISWAP_GATE = 27, 
    SQISWAP_GATE = 28, SWAPZ1_GATE = 29, ISWAPTHETA_GATE = 30, CR_GATE = 31, 
    TOFFOLI_GATE = 32, DAGGER_KEY = 33, ENDDAGGER_KEY = 34, CONTROL_KEY = 35, 
    ENDCONTROL_KEY = 36, QIF_KEY = 37, ELSE_KEY = 38, ENDIF_KEY = 39, QWHILE_KEY = 40, 
    ENDQWHILE_KEY = 41, MEASURE_KEY = 42, PMEASURE_KEY = 43, ASSIGN = 44, 
    GT = 45, LT = 46, NOT = 47, EQ = 48, LEQ = 49, GEQ = 50, NE = 51, AND = 52, 
    OR = 53, PLUS = 54, MINUS = 55, MUL = 56, DIV = 57, COMMA = 58, LPAREN = 59, 
    RPAREN = 60, LBRACK = 61, RBRACK = 62, NEWLINE = 63, Identifier = 64, 
    Integer_Literal = 65, Double_Literal = 66, Digit_Sequence = 67, WhiteSpace = 68, 
    SingleLineComment = 69
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

