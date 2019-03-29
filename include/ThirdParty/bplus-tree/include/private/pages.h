#ifndef _PRIVATE_PAGES_H_
#define _PRIVATE_PAGES_H_

#include "private/tree.h"
#include "private/values.h"

typedef struct bp__page_s bp__page_t;
typedef struct bp__page_search_res_s bp__page_search_res_t;

enum page_type {
    kPage = 0,
    kLeaf = 1
};

enum search_type {
    kNotLoad = 0,
    kLoad = 1
};

int bp__page_create(bp_db_t *t,
                    const enum page_type type,
                    const uint64_t offset,
                    const uint64_t config,
                    bp__page_t **page);
void bp__page_destroy(bp_db_t *t, bp__page_t *page);
int bp__page_clone(bp_db_t *t, bp__page_t *page, bp__page_t **clone);

int bp__page_read(bp_db_t *t, bp__page_t *page);
int bp__page_load(bp_db_t *t,
                  const uint64_t offset,
                  const uint64_t config,
                  bp__page_t **page);
int bp__page_save(bp_db_t *t, bp__page_t *page);

int bp__page_load_value(bp_db_t *t,
                        bp__page_t *page,
                        const uint64_t index,
                        bp_value_t *value);
int bp__page_save_value(bp_db_t *t,
                        bp__page_t *page,
                        const uint64_t index,
                        const int cmp,
                        const bp_key_t *key,
                        const bp_value_t *value,
                        bp_update_cb cb,
                        void *arg);

int bp__page_search(bp_db_t *t,
                    bp__page_t *page,
                    const bp_key_t *key,
                    const enum search_type type,
                    bp__page_search_res_t *result);
int bp__page_get(bp_db_t *t,
                 bp__page_t *page,
                 const bp_key_t *key,
                 bp_value_t *value);
int bp__page_get_range(bp_db_t *t,
                       bp__page_t *page,
                       const bp_key_t *start,
                       const bp_key_t *end,
                       bp_filter_cb filter,
                       bp_range_cb cb,
                       void *arg);
int bp__page_insert(bp_db_t *t,
                    bp__page_t *page,
                    const bp_key_t *key,
                    const bp_value_t *value,
                    bp_update_cb update_cb,
                    void *arg);
int bp__page_bulk_insert(bp_db_t *t,
                         bp__page_t *page,
                         const bp_key_t *limit,
                         uint64_t *count,
                         bp_key_t **keys,
                         bp_value_t **values,
                         bp_update_cb update_cb,
                         void *arg);
int bp__page_remove(bp_db_t *t,
                    bp__page_t *page,
                    const bp_key_t *key,
                    bp_remove_cb remove_cb,
                    void *arg);
int bp__page_copy(bp_db_t *source, bp_db_t *target, bp__page_t *page);

int bp__page_remove_idx(bp_db_t *t, bp__page_t *page, const uint64_t index);
int bp__page_split(bp_db_t *t,
                   bp__page_t *parent,
                   const uint64_t index,
                   bp__page_t *child);
int bp__page_split_head(bp_db_t *t, bp__page_t **page);

void bp__page_shiftr(bp_db_t *t, bp__page_t *page, const uint64_t index);
void bp__page_shiftl(bp_db_t *t, bp__page_t *page, const uint64_t index);

struct bp__page_s {
    enum page_type type;

    uint64_t length;
    uint64_t byte_size;

    uint64_t offset;
    uint64_t config;

    void *buff_;
    int is_head;

    bp__kv_t keys[1];
};

struct bp__page_search_res_s {
    bp__page_t *child;

    uint64_t index;
    int cmp;
};

#endif /* _PRIVATE_PAGES_H_ */
