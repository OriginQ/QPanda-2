#include "bplus.h"
#include "private/values.h"
#include "private/writer.h"
#include "private/utils.h"

#include <stdlib.h> /* malloc, free */
#include <string.h> /* memcpy */


int bp__value_load(bp_db_t *t,
                   const uint64_t offset,
                   const uint64_t length,
                   bp_value_t *value)
{
    int ret;
    char* buff;
    uint64_t buff_len = length;

    /* read data from disk first */
    ret = bp__writer_read((bp__writer_t*) t,
                          kCompressed,
                          offset,
                          &buff_len,
                          (void **) &buff);
    if (ret != BP_OK) return ret;

    value->value = (char *)malloc(buff_len - 16);
    if (value->value == NULL) {
        free(buff);
        return BP_EALLOC;
    }

    /* first 16 bytes are representing previous value */
    value->_prev_offset = myntohll(*(uint64_t *) (buff));
    value->_prev_length = myntohll(*(uint64_t *) (buff + 8));

    /* copy the rest into result buffer */
    memcpy(value->value, buff + 16, buff_len - 16);
    value->length = buff_len - 16;

    free(buff);

    return BP_OK;
}


int bp__value_save(bp_db_t *t,
                   const bp_value_t *value,
                   const bp__kv_t *previous,
                   uint64_t *offset,
                   uint64_t *length)
{
    int ret;
    char* buff;

    buff = (char*)malloc(value->length + 16);
    if (buff == NULL) return BP_EALLOC;

    /* insert offset, length of previous value */
    if (previous != NULL) {
        *(uint64_t *) (buff) = myhtonll(previous->offset);
        *(uint64_t *) (buff + 8) = myhtonll(previous->length);
    } else {
        *(uint64_t *) (buff) = 0;
        *(uint64_t *) (buff + 8) = 0;
    }

    /* insert current value itself */
    memcpy(buff + 16, value->value, value->length);

    *length = value->length + 16;
    ret = bp__writer_write((bp__writer_t *) t,
                           kCompressed,
                           buff,
                           offset,
                           length);
    free(buff);

    return ret;
}


int bp__kv_copy(const bp__kv_t *source, bp__kv_t *target, int alloc)
{
    /* copy key fields */
    if (alloc) {
        target->value = (char *)malloc(source->length);
        if (target->value == NULL) return BP_EALLOC;

        memcpy(target->value, source->value, source->length);
        target->allocated = 1;
    } else {
        target->value = source->value;
        target->allocated = source->allocated;
    }

    target->length = source->length;

    /* copy rest */
    target->offset = source->offset;
    target->config = source->config;

    return BP_OK;
}
