
// Generated from .\originir.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"




class  originirLexer : public antlr4::Lexer {
public:
  enum {
    PI = 1, QINIT_KEY = 2, CREG_KEY = 3, Q_KEY = 4, C_KEY = 5, H_GATE = 6, 
    X_GATE = 7, NOT_GATE = 8, T_GATE = 9, S_GATE = 10, Y_GATE = 11, Z_GATE = 12, 
    X1_GATE = 13, Y1_GATE = 14, Z1_GATE = 15, U4_GATE = 16, RX_GATE = 17, 
    RY_GATE = 18, RZ_GATE = 19, U1_GATE = 20, CNOT_GATE = 21, CZ_GATE = 22, 
    CU_GATE = 23, ISWAP_GATE = 24, SQISWAP_GATE = 25, SWAPZ1_GATE = 26, 
    ISWAPTHETA_GATE = 27, CR_GATE = 28, TOFFOLI_GATE = 29, DAGGER_KEY = 30, 
    ENDDAGGER_KEY = 31, CONTROL_KEY = 32, ENDCONTROL_KEY = 33, QIF_KEY = 34, 
    ELSE_KEY = 35, ENDIF_KEY = 36, QWHILE_KEY = 37, ENDQWHILE_KEY = 38, 
    MEASURE_KEY = 39, ASSIGN = 40, GT = 41, LT = 42, NOT = 43, EQ = 44, 
    LEQ = 45, GEQ = 46, NE = 47, AND = 48, OR = 49, PLUS = 50, MINUS = 51, 
    MUL = 52, DIV = 53, COMMA = 54, LPAREN = 55, RPAREN = 56, LBRACK = 57, 
    RBRACK = 58, NEWLINE = 59, Identifier = 60, Integer_Literal = 61, Double_Literal = 62, 
    Digit_Sequence = 63, WhiteSpace = 64, SingleLineComment = 65
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

