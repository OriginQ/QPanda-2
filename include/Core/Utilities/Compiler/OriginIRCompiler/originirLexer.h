
// Generated from .\originir.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"




class  originirLexer : public antlr4::Lexer {
public:
  enum {
    PI = 1, QINIT_KEY = 2, CREG_KEY = 3, Q_KEY = 4, C_KEY = 5, BARRIER_KEY = 6, 
    QGATE_KEY = 7, ENDQGATE_KEY = 8, ECHO_GATE = 9, H_GATE = 10, X_GATE = 11, 
    NOT_GATE = 12, T_GATE = 13, S_GATE = 14, Y_GATE = 15, Z_GATE = 16, X1_GATE = 17, 
    Y1_GATE = 18, Z1_GATE = 19, I_GATE = 20, U2_GATE = 21, RPHI_GATE = 22, 
    U3_GATE = 23, U4_GATE = 24, RX_GATE = 25, RY_GATE = 26, RZ_GATE = 27, 
    U1_GATE = 28, CNOT_GATE = 29, CZ_GATE = 30, CU_GATE = 31, ISWAP_GATE = 32, 
    SQISWAP_GATE = 33, SWAPZ1_GATE = 34, ISWAPTHETA_GATE = 35, CR_GATE = 36, 
    TOFFOLI_GATE = 37, DAGGER_KEY = 38, ENDDAGGER_KEY = 39, CONTROL_KEY = 40, 
    ENDCONTROL_KEY = 41, QIF_KEY = 42, ELSE_KEY = 43, ENDIF_KEY = 44, QWHILE_KEY = 45, 
    ENDQWHILE_KEY = 46, MEASURE_KEY = 47, RESET_KEY = 48, ASSIGN = 49, GT = 50, 
    LT = 51, NOT = 52, EQ = 53, LEQ = 54, GEQ = 55, NE = 56, AND = 57, OR = 58, 
    PLUS = 59, MINUS = 60, MUL = 61, DIV = 62, COMMA = 63, LPAREN = 64, 
    RPAREN = 65, LBRACK = 66, RBRACK = 67, NEWLINE = 68, Identifier = 69, 
    Integer_Literal = 70, Double_Literal = 71, Digit_Sequence = 72, REALEXP = 73, 
    WhiteSpace = 74, SingleLineComment = 75
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

