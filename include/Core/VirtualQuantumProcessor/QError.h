/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
/*! \file QError.h */
#ifndef _QERROR_H
#define _QERROR_H
/*
*  @enum	QError
*  @brief   quantum program error type
*  @ingroup   QuantumMachine
*/
enum QError
{
    qErrorNone = 2,    /**< no error   */
    undefineError,    /**< undefined error   */
    qParameterError,    /**<wrong parameter   */
    qubitError,    /**< qubits error not only numbers   */
    loadFileError,    /**< load file failed   */
    initStateError,    /**< init quantum state error   */
    destroyStateError,    /**< destroy state error   */
    setComputeUnitError,    /**< set compute unit error   */
    runProgramError,    /**< quantum program running time error   */
    getResultError,    /**< get result error   */
    getQStateError    /**< get quantum state error    */
};

#endif
