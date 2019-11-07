
// Generated from .\qasm.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"




class  qasmLexer : public antlr4::Lexer {
public:
  enum {
    OPENQASM_KEY = 1, INCLUDE_KEY = 2, OPAQUE_KEY = 3, QREG_KEY = 4, CREG_KEY = 5, 
    BARRIER_KEY = 6, IF_KEY = 7, MEASURE_KEY = 8, RESET_KEY = 9, GATE_KEY = 10, 
    PI_KEY = 11, U_GATE_KEY = 12, CX_GATE_KEY = 13, ARROW = 14, EQ = 15, 
    PLUS = 16, MINUS = 17, MUL = 18, DIV = 19, COMMA = 20, SEMI = 21, LPAREN = 22, 
    RPAREN = 23, LBRACKET = 24, RBRACKET = 25, LBRACE = 26, RBRACE = 27, 
    DQM = 28, IDENTIFIER = 29, INTEGER = 30, DECIMAL = 31, FILENAME = 32, 
    NL = 33, WS = 34, LC = 35
  };

  qasmLexer(antlr4::CharStream *input);
  ~qasmLexer();

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

