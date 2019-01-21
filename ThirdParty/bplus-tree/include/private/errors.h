#ifndef _PRIVATE_ERRORS_H_
#define _PRIVATE_ERRORS_H_

#define BP_OK              0

#define BP_EFILE           0x101
#define BP_EFILEREAD_OOB   0x102
#define BP_EFILEREAD       0x103
#define BP_EFILEWRITE      0x104
#define BP_EFILEFLUSH      0x105
#define BP_EFILERENAME     0x106
#define BP_ECOMPACT_EXISTS 0x107

#define BP_ECOMP           0x201
#define BP_EDECOMP         0x202

#define BP_EALLOC          0x301
#define BP_EMUTEX          0x302
#define BP_ERWLOCK         0x303

#define BP_ENOTFOUND       0x401
#define BP_ESPLITPAGE      0x402
#define BP_EEMPTYPAGE      0x403
#define BP_EUPDATECONFLICT 0x404
#define BP_EREMOVECONFLICT 0x405

#endif /* _PRIVATE_ERRORS_H_ */
