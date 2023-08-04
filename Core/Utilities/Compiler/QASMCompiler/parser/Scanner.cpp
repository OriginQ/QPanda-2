
#include "Core/Utilities/Compiler/QASMCompiler/parser/Scanner.hpp"

#include <cstdint>
#include <istream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace qasm {
char Scanner::readUtf8Codepoint(std::istream* in) {
  char c = 0;
  in->get(c);
  return c;
}

void Scanner::nextCh() {
  if (!is->eof()) {
    col++;
    ch = readUtf8Codepoint(is);
  } else {
    ch = 0;
  }
  if (ch == '\n') {
    col = 0;
    line++;
  }
}

char Scanner::peek() const {
  if (!is->eof()) {
    return static_cast<char>(is->peek());
  }
  return 0;
}

std::optional<Token> Scanner::consumeWhitespaceAndComments() {
  while (isSpace(ch)) {
    nextCh();
  }
  if (ch == '/' && peek() == '/') {
    Token t(line, col);
    // consume until newline
    std::stringstream content;
    while (ch != '\n' && ch != 0) {
      content << ch;
      nextCh();
    }
    if (ch == '\n') {
      nextCh();
    }

    static const auto INITIAL_LAYOUT_REGEX = std::regex("i (\\d+ )*(\\d+)");
    static const auto OUTPUT_PERMUTATION_REGEX = std::regex("o (\\d+ )*(\\d+)");

    const auto str = content.str();
    if (std::regex_search(str, INITIAL_LAYOUT_REGEX)) {
      t.kind = Token::Kind::InitialLayout;
    } else if (std::regex_search(str, OUTPUT_PERMUTATION_REGEX)) {
      t.kind = Token::Kind::OutputPermutation;
    } else {
      return consumeWhitespaceAndComments();
    }

    t.str = content.str();
    t.endCol = col;
    t.endLine = line;
    return t;
  }
  if (ch == '/' && peek() == '*') {
    // consume /*
    nextCh();
    nextCh();
    while (ch != 0 && (ch != '*' || peek() != '/')) {
      nextCh();
    }
    // consume */
    expect('*');
    expect('/');

    // tail calls should be optimized away
    return consumeWhitespaceAndComments();
  }

  return {};
}

Token Scanner::consumeName() {
  Token t(line, col);
  std::stringstream name;
  while (isFirstIdChar(ch) || isNum(ch)) {
    name << ch;
    nextCh();
  }

  t.str = name.str();
  if (keywords.find(t.str) != keywords.end()) {
    t.kind = keywords[t.str];
  } else {
    t.kind = Token::Kind::Identifier;
  }

  t.endCol = col;
  t.endLine = line;
  return t;
}

bool Scanner::isValidDigit(const uint8_t base, const char c) {
  if (base == 2) {
    return c == '0' || c == '1';
  }
  if (base == 8) {
    return c >= '0' && c <= '7';
  }
  if (base == 10) {
    return isNum(c);
  }
  if (base == 16) {
    return isHex(c);
  }
  return false;
}

std::string Scanner::consumeNumberLiteral(const uint8_t base) {
  std::stringstream ss;
  while (isValidDigit(base, ch) || ch == '_') {
    if (ch != '_') {
      ss << ch;
    }
    nextCh();
  }

  return ss.str();
}

uint64_t Scanner::parseIntegerLiteral(const std::string& str,
                                      const uint8_t base) {
  uint64_t val = 0;
  for (const auto c : str) {
    if (isNum(c)) {
      val *= base;
      val += static_cast<uint64_t>(c) - '0';
    } else {
      val *= base;
      val += static_cast<uint64_t>(c) - 'a' + 10;
    }
  }
  return val;
}

Token Scanner::consumeNumberLiteral() {
  Token t(line, col);
  uint8_t base = 10;

  if (ch == '0') {
    switch (peek()) {
    case 'b':
    case 'B':
      base = 2;
      nextCh();
      nextCh();
      break;
    case 'o':
      base = 8;
      nextCh();
      nextCh();
      break;
    case 'x':
    case 'X':
      base = 16;
      nextCh();
      nextCh();
      break;
    default:
      break;
    }
  }
  bool negative = false;
  if (ch == '-') {
    if (base != 10) {
      error("Negative numbers are only allowed in base 10");
    }
    negative = true;
    nextCh();
  }

  const auto valBeforeDecimalSeparator = consumeNumberLiteral(base);

  if (ch == '.' || ch == 'e' || ch == 'E') {
    if (base != 10) {
      error("Float literals are only allowed in base 10");
    }

    std::stringstream ss{};
    ss << valBeforeDecimalSeparator;

    if (ch == '.') {
      ss << ch;
      nextCh();
      const auto valAfterDecimalSeparator = consumeNumberLiteral(base);
      ss << valAfterDecimalSeparator;
    }

    if (ch == 'e' || ch == 'E') {
      ss << ch;
      nextCh();
      if (ch == '+' || ch == '-') {
        ss << ch;
        nextCh();
      }
      const auto valAfterExponent = consumeNumberLiteral(base);
      ss << valAfterExponent;
    }

    try {
      t.valReal = std::stod(ss.str());
    } catch (std::invalid_argument&) {
      error("Unable to parse float literal");
    }

    t.kind = Token::Kind::FloatLiteral;
    if (negative) {
      t.valReal *= -1;
    }

    return t;
  }

  t.val = static_cast<int64_t>(
      parseIntegerLiteral(valBeforeDecimalSeparator, base));
  t.kind = Token::Kind::IntegerLiteral;
  if (negative) {
    t.val *= -1;
    t.isSigned = true;
  }

  t.endCol = col;
  t.endLine = line;

  return t;
}

Token Scanner::consumeHardwareQubit() {
  Token t(line, col);

  expect('$');

  t.kind = Token::Kind::HardwareQubit;
  t.val = 0;
  while (isNum(ch)) {
    t.val *= 10;
    t.val += static_cast<int64_t>(ch - '0');
    nextCh();
  }

  t.endCol = col;
  t.endLine = line;

  return t;
}

Token Scanner::consumeString() {
  Token t(line, col);
  t.kind = Token::Kind::StringLiteral;

  if (ch != '"' && ch != '\'') {
    error("expected `\"` or `'`");
    t.kind = Token::Kind::None;
    return t;
  }
  const auto delim = ch;
  nextCh();

  std::stringstream content;
  while (ch != delim) {
    content << ch;
    nextCh();
  }

  t.str = content.str();

  expect(delim);

  t.endCol = col;
  t.endLine = line;

  return t;
}

Scanner::Scanner(std::istream* in) : is(in) {
  keywords["OPENQASM"] = Token::Kind::OpenQasm;
  keywords["include"] = Token::Kind::Include;
  keywords["defcalgrammar"] = Token::Kind::DefCalGrammar;
  keywords["def"] = Token::Kind::Def;
  keywords["cal"] = Token::Kind::Cal;
  keywords["defcal"] = Token::Kind::DefCal;
  keywords["gate"] = Token::Kind::Gate;
  keywords["opaque"] = Token::Kind::Opaque;
  keywords["extern"] = Token::Kind::Extern;
  keywords["box"] = Token::Kind::Box;
  keywords["let"] = Token::Kind::Let;
  keywords["break"] = Token::Kind::Break;
  keywords["continue"] = Token::Kind::Continue;
  keywords["if"] = Token::Kind::If;
  keywords["else"] = Token::Kind::Else;
  keywords["end"] = Token::Kind::End;
  keywords["return"] = Token::Kind::Return;
  keywords["for"] = Token::Kind::For;
  keywords["while"] = Token::Kind::While;
  keywords["in"] = Token::Kind::In;
  keywords["pragma"] = Token::Kind::Pragma;
  keywords["input"] = Token::Kind::Input;
  keywords["output"] = Token::Kind::Output;
  keywords["const"] = Token::Kind::Const;
  keywords["readonly"] = Token::Kind::ReadOnly;
  keywords["mutable"] = Token::Kind::Mutable;
  keywords["qreg"] = Token::Kind::Qreg;
  keywords["qubit"] = Token::Kind::QBit;
  keywords["creg"] = Token::Kind::CReg;
  keywords["bool"] = Token::Kind::Bool;
  keywords["bit"] = Token::Kind::Bit;
  keywords["int"] = Token::Kind::Int;
  keywords["uint"] = Token::Kind::Uint;
  keywords["float"] = Token::Kind::Float;
  keywords["angle"] = Token::Kind::Angle;
  keywords["complex"] = Token::Kind::Complex;
  keywords["array"] = Token::Kind::Array;
  keywords["void"] = Token::Kind::Void;
  keywords["duration"] = Token::Kind::Duration;
  keywords["stretch"] = Token::Kind::Stretch;
  keywords["gphase"] = Token::Kind::Gphase;
  keywords["inv"] = Token::Kind::Inv;
  keywords["pow"] = Token::Kind::Pow;
  keywords["ctrl"] = Token::Kind::Ctrl;
  keywords["negctrl"] = Token::Kind::NegCtrl;
  keywords["#dim"] = Token::Kind::Dim;
  keywords["durationof"] = Token::Kind::DurationOf;
  keywords["delay"] = Token::Kind::Delay;
  keywords["reset"] = Token::Kind::Reset;
  keywords["measure"] = Token::Kind::Measure;
  keywords["barrier"] = Token::Kind::Barrier;
  keywords["true"] = Token::Kind::True;
  keywords["false"] = Token::Kind::False;
  keywords["im"] = Token::Kind::Imag;
  keywords["dt"] = Token::Kind::TimeUnitDt;
  keywords["ns"] = Token::Kind::TimeUnitNs;
  keywords["us"] = Token::Kind::TimeUnitUs;
  keywords["mys"] = Token::Kind::TimeUnitMys;
  keywords["ms"] = Token::Kind::TimeUnitMs;
  keywords["s"] = Token::Kind::S;
  keywords["sin"] = Token::Kind::Sin;
  keywords["cos"] = Token::Kind::Cos;
  keywords["tan"] = Token::Kind::Tan;
  keywords["exp"] = Token::Kind::Exp;
  keywords["ln"] = Token::Kind::Ln;
  keywords["sqrt"] = Token::Kind::Sqrt;

  nextCh();
}

Token Scanner::next() {
  if (const auto commentToken = consumeWhitespaceAndComments()) {
    return *commentToken;
  }

  if (isFirstIdChar(ch)) {
    return consumeName();
  }
  if (isNum(ch) || (ch == '.' && isNum(peek())) ||
      (ch == '-' && isNum(peek()))) {
    return consumeNumberLiteral();
  }
  if (ch == '$') {
    return consumeHardwareQubit();
  }

  if (ch == '"' || ch == '\'') {
    return consumeString();
  }

  Token t(line, col);
  switch (ch) {
  case 0:
    t.kind = Token::Kind::Eof;
    // Here we return as we don't want to call nextCh after EOF.
    // We also don't set length, as the eof token has no length.
    return t;
  case '[':
    t.kind = Token::Kind::LBracket;
    break;
  case ']':
    t.kind = Token::Kind::RBracket;
    break;
  case '{':
    t.kind = Token::Kind::LBrace;
    break;
  case '}':
    t.kind = Token::Kind::RBrace;
    break;
  case '(':
    t.kind = Token::Kind::LParen;
    break;
  case ')':
    t.kind = Token::Kind::RParen;
    break;
  case ':':
    t.kind = Token::Kind::Colon;
    break;
  case ';':
    t.kind = Token::Kind::Semicolon;
    break;
  case '.':
    t.kind = Token::Kind::Dot;
    break;
  case ',':
    t.kind = Token::Kind::Comma;
    break;
  case '-':
    switch (peek()) {
    case '>':
      nextCh();
      t.kind = Token::Kind::Arrow;
      break;
    case '=':
      nextCh();
      t.kind = Token::Kind::MinusEquals;
      break;
    default:
      t.kind = Token::Kind::Minus;
      break;
    }
    break;
  case '+':
    switch (peek()) {
    case '=':
      nextCh();
      t.kind = Token::Kind::PlusEquals;
      break;
    case '+':
      nextCh();
      t.kind = Token::Kind::DoublePlus;
      break;
    default:
      t.kind = Token::Kind::Plus;
      break;
    }
    break;
  case '*':
    switch (peek()) {
    case '=':
      nextCh();
      t.kind = Token::Kind::AsteriskEquals;
      break;
    case '*':
      nextCh();
      if (peek() == '=') {
        nextCh();
        t.kind = Token::Kind::DoubleAsteriskEquals;
      } else {
        t.kind = Token::Kind::DoubleAsterisk;
      }
      break;
    default:
      t.kind = Token::Kind::Asterisk;
      break;
    }
    break;
  case '/':
    if (peek() == '=') {
      nextCh();
      t.kind = Token::Kind::SlashEquals;
    } else {
      t.kind = Token::Kind::Slash;
    }
    break;
  case '%':
    if (peek() == '=') {
      nextCh();
      t.kind = Token::Kind::PercentEquals;
    } else {
      t.kind = Token::Kind::Percent;
    }
    break;
  case '|':
    switch (peek()) {
    case '=':
      nextCh();
      t.kind = Token::Kind::PipeEquals;
      break;
    case '|':
      nextCh();
      t.kind = Token::Kind::DoublePipe;
      break;
    default:
      t.kind = Token::Kind::Pipe;
      break;
    }
    break;
  case '&':
    switch (peek()) {
    case '=':
      nextCh();
      t.kind = Token::Kind::AmpersandEquals;
      break;
    case '&':
      nextCh();
      t.kind = Token::Kind::DoubleAmpersand;
      break;
    default:
      t.kind = Token::Kind::Ampersand;
      break;
    }
    break;
  case '^':
    if (peek() == '=') {
      nextCh();
      t.kind = Token::Kind::CaretEquals;
    } else {
      t.kind = Token::Kind::Caret;
    }
    break;
  case '~':
    if (peek() == '=') {
      nextCh();
      t.kind = Token::Kind::TildeEquals;
    } else {
      t.kind = Token::Kind::Tilde;
    }
    break;
  case '!':
    if (peek() == '=') {
      nextCh();
      t.kind = Token::Kind::NotEquals;
    } else {
      t.kind = Token::Kind::ExclamationPoint;
    }
    break;
  case '<':
    switch (peek()) {
    case '=':
      nextCh();
      t.kind = Token::Kind::LessThanEquals;
      break;
    case '<':
      nextCh();
      if (peek() == '=') {
        nextCh();
        t.kind = Token::Kind::LeftShitEquals;
      } else {
        t.kind = Token::Kind::LeftShift;
      }
      break;
    default:
      t.kind = Token::Kind::LessThan;
      break;
    }
    break;
  case '>':
    switch (peek()) {
    case '=':
      nextCh();
      t.kind = Token::Kind::GreaterThanEquals;
      break;
    case '>':
      nextCh();
      if (peek() == '=') {
        nextCh();
        t.kind = Token::Kind::RightShiftEquals;
      } else {
        t.kind = Token::Kind::RightShift;
      }
      break;
    default:
      t.kind = Token::Kind::GreaterThan;
      break;
    }
    break;
  case '=':
    if (peek() == '=') {
      nextCh();
      t.kind = Token::Kind::DoubleEquals;
    } else {
      t.kind = Token::Kind::Equals;
    }
    break;
  case '@':
    t.kind = Token::Kind::At;
    break;
  default: {
    error("Unknown character '" + std::to_string(ch) + "'");
    t.kind = Token::Kind::None;
    nextCh();
    break;
  }
  }

  nextCh();

  t.endCol = col;
  t.endLine = line;
  return t;
}
} // namespace qasm
