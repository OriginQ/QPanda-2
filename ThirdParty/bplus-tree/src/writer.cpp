#include "bplus.h"
#include "private/writer.h"

#include <stdlib.h> /* malloc, free */
#include <string.h> /* memset */
#include <iostream>
using IOS = std::ios;
int bp__writer_create(bp__writer_t *w, const char *filename)
{
    size_t filename_length;

    /* copy filename + '\0' char */
    filename_length = strlen(filename) + 1;
    w->filename = (char *)malloc(filename_length);
    if (w->filename == NULL) return BP_EALLOC;
    memcpy(w->filename, filename, filename_length);

    w->ifd.open(filename);
    if (!w->ifd)
    {
        w->fd.open(filename,IOS::out);
        w->fd.close();
    }
    else
    {
        w->ifd.seekg(0,IOS::end);
        w->filesize = w->ifd.tellg();
        memset(&w->padding, 0, sizeof(w->padding));
        w->ifd.close();
    }

    return BP_OK;
}

int bp__writer_destroy(bp__writer_t *w)
{
    free(w->filename);
    w->filename = NULL;
    return BP_OK;
}

int bp__writer_fsync(bp__writer_t *w)
{
    return BP_OK;
#ifdef F_FULLFSYNC
    /* OSX support */
    return fcntl(w->fd, F_FULLFSYNC);
#else
  
#endif
}

int bp__writer_read(bp__writer_t *w,
                    const enum comp_type comp,
                    const uint64_t offset,
                    uint64_t *size,
                    void **data)
{
    char *cdata;

    if (w->filesize < offset + *size) return BP_EFILEREAD_OOB;

    /* Ignore empty reads */
    if (*size == 0) {
        *data = NULL;
        return BP_OK;
    }

    cdata = (char *)calloc(*size,'\0');
    if (cdata == NULL) return BP_EALLOC;
    w->ifd.open(w->filename, IOS::in);
    w->ifd.seekg(offset, IOS::beg);
    w->ifd.read(cdata, *size);

    /* no compression for head */
    if (comp == kNotCompressed) {
        *data = cdata;
    } 
    w->ifd.close();
    return BP_OK;
}

int bp__writer_write(bp__writer_t *w,
                     const enum comp_type comp,
                     const void *data,
                     uint64_t *offset,
                     uint64_t *size)
{
    uint32_t padding = sizeof(w->padding) - (w->filesize % sizeof(w->padding));
    w->fd.open(w->filename, IOS::out | IOS::app);
    /* Write padding */
    if (padding != sizeof(w->padding)) {

        w->fd.write(w->padding, padding);
        w->filesize += padding;
    }

    /* Ignore empty writes */
    if (size == NULL || *size == 0) {
        if (offset != NULL) *offset = w->filesize;
        return BP_OK;
    }

    /* head shouldn't be compressed */
    if (comp == kNotCompressed) {
        w->fd.write((char *)data, *size);
    }
    /* change offset */
    *offset = w->filesize;
    w->filesize += *size;
    w->fd.close();
    return BP_OK;
}

int bp__writer_find(bp__writer_t*w,
                    const enum comp_type comp,
                    const uint64_t size,
                    void *data,
                    bp__writer_cb seek,
                    bp__writer_cb miss)
{
    int ret = 0;
    int match = 0;
    uint64_t offset, size_tmp;

    /* Write padding first */
    ret = bp__writer_write(w, kNotCompressed, NULL, NULL, NULL);
    if (ret != BP_OK) return ret;

    offset = w->filesize;
    size_tmp = size;

    /* Start seeking from bottom of file */
    while (offset >= size) {
        ret = bp__writer_read(w, comp, offset - size, &size_tmp, &data);
        if (ret != BP_OK) break;

        /* Break if matched */
        if (seek(w, data) == 0) {
            match = 1;
            break;
        }

        offset -= size;
    }

    /* Not found - invoke miss */
    if (!match)
        ret = miss(w, data);

    return ret;
}
