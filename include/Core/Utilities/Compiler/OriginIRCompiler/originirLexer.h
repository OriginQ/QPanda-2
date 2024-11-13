
// Generated from originir.g4 by ANTLR 4.8

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
    U1_GATE = 28, P_GATE = 29, CNOT_GATE = 30, CZ_GATE = 31, CU_GATE = 32, 
    ISWAP_GATE = 33, SQISWAP_GATE = 34, SWAPZ1_GATE = 35, ISWAPTHETA_GATE = 36, 
    CR_GATE = 37, RXX_GATE = 38, RYY_GATE = 39, RZZ_GATE = 40, RZX_GATE = 41, 
    TOFFOLI_GATE = 42, DAGGER_KEY = 43, ENDDAGGER_KEY = 44, CONTROL_KEY = 45, 
    ENDCONTROL_KEY = 46, QIF_KEY = 47, ELSE_KEY = 48, ENDIF_KEY = 49, QWHILE_KEY = 50, 
    ENDQWHILE_KEY = 51, MEASURE_KEY = 52, RESET_KEY = 53, ASSIGN = 54, GT = 55, 
    LT = 56, NOT = 57, EQ = 58, LEQ = 59, GEQ = 60, NE = 61, AND = 62, OR = 63, 
    PLUS = 64, MINUS = 65, MUL = 66, DIV = 67, COMMA = 68, LPAREN = 69, 
    RPAREN = 70, LBRACK = 71, RBRACK = 72, NEWLINE = 73, Identifier = 74, 
    Integer_Literal = 75, Double_Literal = 76, Digit_Sequence = 77, REALEXP = 78, 
    WhiteSpace = 79, SingleLineComment = 80
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

