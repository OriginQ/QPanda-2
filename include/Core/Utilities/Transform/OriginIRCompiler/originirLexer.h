
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
    ISWAPTHETA_GATE = 27, CR_GATE = 28, DAGGER_KEY = 29, ENDDAGGER_KEY = 30, 
    CONTROL_KEY = 31, ENDCONTROL_KEY = 32, QIF_KEY = 33, ELSE_KEY = 34, 
    ENDIF_KEY = 35, QWHILE_KEY = 36, ENDQWHILE_KEY = 37, MEASURE_KEY = 38, 
    ASSIGN = 39, GT = 40, LT = 41, NOT = 42, EQ = 43, LEQ = 44, GEQ = 45, 
    NE = 46, AND = 47, OR = 48, PLUS = 49, MINUS = 50, MUL = 51, DIV = 52, 
    COMMA = 53, LPAREN = 54, RPAREN = 55, LBRACK = 56, RBRACK = 57, NEWLINE = 58, 
    Identifier = 59, Integer_Literal = 60, Double_Literal = 61, Digit_Sequence = 62, 
    WhiteSpace = 63, SingleLineComment = 64
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

