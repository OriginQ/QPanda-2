/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

GraphDijkstra.h
Author: Wangjing
Created in 2018-8-31

Classes for get the shortes path of graph

*/
#ifndef METADATAVALIDITYFUNCTIONVECTOR_H
#define METADATAVALIDITYFUNCTIONVECTOR_H

#include "Core/Utilities/QPandaNamespace.h"
#include <iostream>
#include <string>
#include <vector>
#include <functional>

QPANDA_BEGIN

/**
 * @brief Single gate transfer type
 */
enum SingleGateTransferType
{
    SINGLE_GATE_INVALID = -1,
    ARBITRARY_ROTATION,
    DOUBLE_CONTINUOUS,
    SINGLE_CONTINUOUS_DISCRETE,
    DOUBLE_DISCRETE
};

/**
 * @brief  Double gate transfer type
 */
enum DoubleGateTransferType
{
    DOUBLE_GATE_INVALID = -1,
    DOUBLE_BIT_GATE
};

/**
 * @brief Judge if the metadata's type is arbitrary rotation
 * @ingroup Utilities
 * @param std::vector<std::string>&  the gates is judged
 * @param std::vector<std::string>&  output the valid gates  
 * @return Return the style of metadata validity
*/
int arbitraryRotationMetadataValidity(std::vector<std::string>&gates, 
                                      std::vector<std::string>&valid_gates);

/**
 * @brief Judge if the metadata's type is double continuous
 * @ingroup Utilities
 * @param std::vector<std::string>&  the gates is judged
 * @param std::vector<std::string>&  output the valid gates
 * @return Return the style of metadata validity
*/
int doubleContinuousMetadataValidity(std::vector<std::string>&gates,
                                     std::vector<std::string>&valid_gates);

/**
 * @brief Judge if the metadata's type is single continuous and discrete
 * @ingroup Utilities
 * @param std::vector<std::string>&  the gates is judged
 * @param std::vector<std::string>&  output the valid gates
 * @return Return the style of metadata validity
*/
int singleContinuousAndDiscreteMetadataValidity(std::vector<std::string>&gates, 
                                                std::vector<std::string>&valid_gates);


/**
 * @brief Judge if the metadata's type is double discrete
 * @ingroup Utilities
 * @param std::vector<std::string>&  the gates is judged
 * @param std::vector<std::string>&  output the valid gates
 * @return Return the style of metadata validity
*/
int doubleDiscreteMetadataValidity(std::vector<std::string>&gates,
                                   std::vector<std::string>&valid_gates);

/**
 * @brief Judge double gate type
 * @ingroup Utilities
 * @param std::vector<std::string>&  the gates is judged
 * @param std::vector<std::string>&  output the valid gates
 * @return Return the style of metadata validity
*/
int doubleGateMetadataValidity(std::vector<std::string>&gates, 
                               std::vector<std::string>&valid_gates);

/**
 * @brief typedef MetadataValidity_cb that add all functions of metadata validity
 */
typedef  std::function<int(std::vector<std::string>&, std::vector<std::string>&)> MetadataValidity_cb;

/**
 * @brief Metadata Validity
 * @ingroup Utilities
 */
class MetadataValidity
{
public:
    void push_back(MetadataValidity_cb func);
    MetadataValidity_cb operator[](int i);
    size_t size();

    virtual ~MetadataValidity();

private:
    std::vector<MetadataValidity_cb> m_metadata_validity_functions;
};

/**
 * @brief Get single gate metadata Validator type
 * @ingroup Utilities
 */
class SingleGateTypeValidator
{
public:
    SingleGateTypeValidator();
    static int GateType(std::vector<std::string>&gates,
                        std::vector<std::string>&valid_gates);
    virtual ~SingleGateTypeValidator();

    MetadataValidity m_metadata_validity_functions;
private:

};

/**
 * @brief Get double gate metadata Validator type
 * @ingroup Utilities
 */
class DoubleGateTypeValidator
{
public:
    DoubleGateTypeValidator();
    static int GateType(std::vector<std::string>&gates, std::vector<std::string>&valid_gates);
    virtual ~DoubleGateTypeValidator();

    MetadataValidity m_metadata_validity_functions;
private:

};

/* new interface */

/**
* @brief  Verify the validity of single quantum gates
* @ingroup Utilities
* @param[in]  std::vector<std::string>&   gates vertor
* @param[out]  std::vector<std::string>&   output the valid gates
* @return  int		single quantum gate type
*/
int validateSingleQGateType(std::vector<std::string> &gates, std::vector<std::string> &valid_gates);


/**
* @brief  Verify the validity of double quantum gates
* @ingroup Utilitie
* @param[in]  std::vector<std::string>&   the gates is judged
* @param[out]  std::vector<std::string>&   output the valid gates
* @return  int		double quantum gate type
*/
int validateDoubleQGateType(std::vector<std::string> &gates, std::vector<std::string> &valid_gates);

QPANDA_END
#endif // METADATAVALIDITYFUNCTIONVECTOR_H
