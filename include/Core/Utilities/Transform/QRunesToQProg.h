/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QRunesToQProg.h
Author: Yulei
Updated in 2019/04/09 14:47

Classes for QRunesToQProg.

*/

/*! \file QRunesToQProg.h */
#ifndef  _QRUNESTOQPROG_H_
#define  _QRUNESTOQPROG_H_
#include "Core/QPanda.h"
#include <functional>
QPANDA_BEGIN
/**
* @namespace QPanda
* @namespace QGATE_SPACE
*/

/**
* @enum QRunesKeyWords
* @brief Qrunes keywords type
*/
enum QRunesKeyWords {
    DAGGER = 24,/**<  QRunes Dagger start  */
    ENDAGGER,/**<  QRunes Dagger end  */
    CONTROL,/**<  QRunes Dagger start  */
    ENCONTROL,/**<  QRunes control end  */
    QIF,/**<  QRunes QIf start  */
    ELSE,/**<  QRunes else  */
    ENDQIF,/**<  QRunes QIf end  */
    QWHILE,/**<  QRunes QWhile start  */
    ENDQWHILE,/**<  QRunes QWhile end  */
    MEASURE,/**<  QRunes measure node  */
};
/**
* @class QRunesToQProg
* @ingroup Utilities
* @brief QRunes instruction set To Quantum QProg
*/
class QRunesToQProg {
public:
    QRunesToQProg() = delete;
    QRunesToQProg(std::string);
    ~QRunesToQProg() {};

    /**
    * @brief  QRunes Parser interpreter
    * @param[in]  QProg&  Quantum program reference 
    * @return     void  
    * @exception  qprog_syntax_error   quantum program syntax error
    */
    void qRunesParser(QProg&);

private:
    void qRunesAllocation(std::vector<std::string>&, QProg&);

    int traversalQRunes(std::vector<std::string>::iterator, QNode*);

    int handleSingleGate(QNode*, const std::string&, int);

    int handleDoubleGate(QNode*, const std::string&, int, int);

    int handleAngleGate(QNode*, const std::string&, int, double);

    int handleDoubleAngleGate(QNode*, const std::string&, int, int, double);

    int handleMeasureGate(QNode*, const std::string&, int, int);

    int handleDaggerCircuit(std::vector<std::string>::iterator, QNode*);

    int handleControlCircuit(std::vector<std::string>::iterator, QNode*,
        std::vector<Qubit*>&, std::string &);

    std::vector<std::string> m_QRunes;/**< QRunes instruction sets   */
    std::vector<std::string> m_keyWords;/**< keywords instruction sets   */

    QVec m_all_qubits;/**< Qubit vector   */
    std::vector<ClassicalCondition > m_all_cregs;/**< ClassicalCondition vector   */

    std::map<std::string, std::function<QGate(Qubit *)> > 
        m_singleGateFunc;/**< Single quantumgate function map   */
    std::map<std::string, std::function<QGate(Qubit *, Qubit*)> > 
        m_doubleGateFunc;/**< Double quantumgate function map   */
    std::map<std::string, std::function<QGate(Qubit *,double)> > 
        m_angleGateFunc;/**< Single angle quantumgate function map   */
    std::map<std::string, std::function<QGate(Qubit *, Qubit*, double)> > 
        m_doubleAngleGateFunc;/**< Double angle quantumgate function map   */

    std::string  m_sFilePath;/**< QRunes file path  */
};


/**
* @brief   QRunes instruction set transform to quantum program interface
* @ingroup Utilities
* @param[in]  QProg&   Empty Quantum Prog
* @return    void
* @see
    @code
        const string sQRunesPath("D:\\QRunes");
        init(QuantumMachine_type::CPU);
        auto prog = CreateEmptyQProg();

        qRunesToQProg(sQRunesPath, prog);

        finalize();
    @endcode
* @exception    qprog_syntax_error   quantum program syntax error
* @note
*/
void qRunesToQProg(std::string, QProg&);
QPANDA_END

#endif
