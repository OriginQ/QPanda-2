#ifndef _BPLUS_H_
#define _BPLUS_H_

#define BP_PADDING 64

#define BP_KEY_FIELDS   \
    uint64_t length;    \
    char *value;

#include <stdint.h> /* uintx_t */
#include "private/errors.h"

typedef struct bp_db_s bp_db_t;

typedef struct bp_key_s bp_key_t;
typedef struct bp_key_s bp_value_t;

typedef int (*bp_compare_cb)(const bp_key_t *a, const bp_key_t *b);
typedef int (*bp_update_cb)(void *arg,
                            const bp_value_t *previous,
                            const bp_value_t *value);
typedef int (*bp_remove_cb)(void *arg,
                            const bp_value_t *value);
typedef void (*bp_range_cb)(void *arg,
                            const bp_key_t *key,
                            const bp_value_t *value);
typedef int (*bp_filter_cb)(void* arg, const bp_key_t *key);

#include "private/tree.h"

/*
 * Open and close database
 */
int bp_open(bp_db_t *tree, const char *filename);
int bp_close(bp_db_t *tree);

/*
 * Get one value by key
 */
int bp_get(bp_db_t *tree, const bp_key_t *key, bp_value_t *value);
int bp_gets(bp_db_t *tree, const char *key, char **value);

/*
 * Get previous value
 */
int bp_get_previous(bp_db_t *tree,
                    const bp_value_t *value,
                    bp_value_t *previous);

/*
 * Set one value by key (without solving conflicts, overwrite)
 */
int bp_set(bp_db_t *tree,
           const bp_key_t *key,
           const bp_value_t *value);
int bp_sets(bp_db_t *tree,
            const char *key,
            const char *value);

/*
 * Update or create value by key (with solving conflicts)
 */
int bp_update(bp_db_t *tree,
              const bp_key_t *key,
              const bp_value_t *value,
              bp_update_cb update_cb,
              void *arg);
int bp_updates(bp_db_t *tree,
               const char *key,
               const char *value,
               bp_update_cb update_cb,
               void *arg);

/*
 * Set multiple values by keys
 */
int bp_bulk_set(bp_db_t *tree,
                const uint64_t count,
                const bp_key_t **keys,
                const bp_value_t **values);
int bp_bulk_sets(bp_db_t *tree,
                 const uint64_t count,
                 const char **keys,
                 const char **values);

/*
 * Update multiple values by keys
 */
int bp_bulk_update(bp_db_t *tree,
                   const uint64_t count,
                   const bp_key_t **keys,
                   const bp_value_t **values,
                   bp_update_cb update_cb,
                   void *arg);
int bp_bulk_updates(bp_db_t *tree,
                    const uint64_t count,
                    const char **keys,
                    const char **values,
                    bp_update_cb update_cb,
                    void *arg);

/*
 * Remove one value by key
 */
int bp_remove(bp_db_t *tree, const bp_key_t *key);
int bp_removes(bp_db_t *tree, const char *key);

/*
 * Remove value by key only if it's equal to specified one
 */
int bp_removev(bp_db_t *tree,
               const bp_key_t *key,
               bp_remove_cb remove_cb,
               void *arg);
int bp_removevs(bp_db_t *tree,
                const char *key,
                bp_remove_cb remove_cb,
                void *arg);

/*
 * Get all values in range
 * Note: value will be automatically freed after invokation of callback
 */
int bp_get_range(bp_db_t *tree,
                 const bp_key_t *start,
                 const bp_key_t *end,
                 bp_range_cb cb,
                 void *arg);
int bp_get_ranges(bp_db_t *tree,
                  const char *start,
                  const char *end,
                  bp_range_cb cb,
                  void *arg);

/*
 * Get values in range (with custom key-filter)
 * Note: value will be automatically freed after invokation of callback
 */
int bp_get_filtered_range(bp_db_t *tree,
                          const bp_key_t *start,
                          const bp_key_t *end,
                          bp_filter_cb filter,
                          bp_range_cb cb,
                          void *arg);
int bp_get_filtered_ranges(bp_db_t *tree,
                           const char *start,
                           const char *end,
                           bp_filter_cb filter,
                           bp_range_cb cb,
                           void *arg);


/*
 * Set compare function to define order of keys in database
 */
void bp_set_compare_cb(bp_db_t *tree, bp_compare_cb cb);

/*
 * Ensure that all data is written to disk
 */
int bp_fsync(bp_db_t *tree);

struct bp_db_s {
    BP_TREE_PRIVATE
};

struct bp_key_s {
    BP_KEY_FIELDS
    BP_KEY_PRIVATE
};


#endif /* _BPLUS_H_ */
