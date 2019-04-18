#ifndef _PRIVATE_UTILS_H_
#define _PRIVATE_UTILS_H_

#include <stdint.h> /* uint64_t */

uint64_t bp__compute_hashl(uint64_t key);
uint64_t myhtonll(uint64_t value);
uint64_t myntohll(uint64_t value);

#endif /* _PRIVATE_UTILS_H_ */
