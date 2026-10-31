/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>

namespace faiss {

/***********************************************************************
 * IDSelectorRange
 ***********************************************************************/

IDSelectorRange::IDSelectorRange(
        idx_t imin_in,
        idx_t imax_in,
        bool assume_sorted_in)
        : imin(imin_in), imax(imax_in), assume_sorted(assume_sorted_in) {}

bool IDSelectorRange::is_member(idx_t id) const {
    return id >= imin && id < imax;
}

void IDSelectorRange::find_sorted_ids_bounds(
        size_t list_size,
        const idx_t* ids,
        size_t* jmin_out,
        size_t* jmax_out) const {
    FAISS_ASSERT(assume_sorted);
    if (list_size == 0 || imax <= ids[0] || imin > ids[list_size - 1]) {
        *jmin_out = *jmax_out = 0;
        return;
    }
    // bisection to find imin
    if (ids[0] >= imin) {
        *jmin_out = 0;
    } else {
        size_t j0 = 0, j1 = list_size;
        while (j1 > j0 + 1) {
            size_t jmed = (j0 + j1) / 2;
            if (ids[jmed] >= imin) {
                j1 = jmed;
            } else {
                j0 = jmed;
            }
        }
        *jmin_out = j1;
    }
    // bisection to find imax
    if (*jmin_out == list_size || ids[*jmin_out] >= imax) {
        *jmax_out = *jmin_out;
    } else {
        size_t j0 = *jmin_out, j1 = list_size;
        while (j1 > j0 + 1) {
            size_t jmed = (j0 + j1) / 2;
            if (ids[jmed] >= imax) {
                j1 = jmed;
            } else {
                j0 = jmed;
            }
        }
        *jmax_out = j1;
    }
}

/***********************************************************************
 * IDSelectorArray
 ***********************************************************************/

IDSelectorArray::IDSelectorArray(size_t n_in, const idx_t* ids_in)
        : n(n_in), ids(ids_in) {}

bool IDSelectorArray::is_member(idx_t id) const {
    for (size_t i = 0; i < n; i++) {
        if (ids[i] == id) {
            return true;
        }
    }
    return false;
}

/***********************************************************************
 * IDSelectorBatch
 ***********************************************************************/

IDSelectorBatch::IDSelectorBatch(size_t n, const idx_t* indices) {
    nbits = 0;
    while (n > (size_t{1} << nbits)) {
        nbits++;
    }
    nbits += 5;
    // for n = 1M, nbits = 25 is optimal, see P56659518

    mask = ((idx_t)1 << nbits) - 1;
    bloom.resize(size_t{1} << (nbits - 3), 0);
    for (size_t i = 0; i < n; i++) {
        idx_t id = indices[i];
        set.insert(id);
        id &= mask;
        bloom[id >> 3] |= 1 << (id & 7);
    }
}

bool IDSelectorBatch::is_member(idx_t i) const {
    idx_t im = i & mask;
    if (!(bloom[im >> 3] & (1 << (im & 7)))) {
        return false;
    }
    return set.count(i);
}

/***********************************************************************
 * IDSelectorBitmap
 ***********************************************************************/

IDSelectorBitmap::IDSelectorBitmap(size_t n_in, const uint8_t* bitmap_in)
        : n(n_in), bitmap(bitmap_in) {}

bool IDSelectorBitmap::is_member(idx_t ii) const {
    uint64_t i = ii;
    if ((i >> 3) >= n) {
        return false;
    }
    return (bitmap[i >> 3] >> (i & 7)) & 1;
}

/***********************************************************************
 * DocIDSelectorBitmap
 ***********************************************************************/

DocIDSelectorBitmap::DocIDSelectorBitmap(
    size_t idmapLen, const uint32_t *idmap,
    size_t bitmapLen, const uint8_t *bitmap)
    : idmapLen(idmapLen), idmap(idmap),
      bitmapLen(bitmapLen), bitmap(bitmap),
      inverted(false) {}

DocIDSelectorBitmap::DocIDSelectorBitmap(
    size_t idmapLen, const uint32_t *idmap,
    size_t bitmapLen, const uint8_t *bitmap,
    bool inverted)
    : idmapLen(idmapLen), idmap(idmap),
      bitmapLen(bitmapLen), bitmap(bitmap),
      inverted(inverted) {}

bool DocIDSelectorBitmap::is_member(idx_t ii) const {
    uint32_t id = ii;
    if (id < idmapLen) {
        uint32_t docid = idmap[id];
        if ((docid >> 3) >= bitmapLen) {
            return false != inverted;
        }
        return (((bitmap[docid >> 3] >> (docid & 7)) & 1) != 0) != inverted;
    }
    return false;
}

/***********************************************************************
 * DocIDSelectorBitlist
 ***********************************************************************/

template<typename T>
bool bsearch(T key, const T *values, size_t num_values) {
    size_t lower = 0;
    size_t upper = num_values;

    while (lower < upper) {
        size_t pos = (lower + upper) >> 1;
        const T value = values[pos];

        if (key == value)
            return true;

        if (key < value)
            upper = pos;
        else
            lower = pos + 1;
    }

    return false;
}

DocIDSelectorBitlist::DocIDSelectorBitlist(
    size_t idmapLen, const uint32_t *idmap,
    size_t positionsLen, const uint32_t *positions)
        : idmapLen(idmapLen), idmap(idmap),
          positionsLen(positionsLen), positions(positions),
          inverted(false) {}

DocIDSelectorBitlist::DocIDSelectorBitlist(
    size_t idmapLen, const uint32_t *idmap,
    size_t positionsLen, const uint32_t *positions,
    bool inverted)
        : idmapLen(idmapLen), idmap(idmap),
          positionsLen(positionsLen), positions(positions),
          inverted(inverted) {}

bool DocIDSelectorBitlist::is_member(idx_t ii) const {
    uint32_t id = ii;
    if (id < idmapLen)
        return bsearch<uint32_t>(idmap[id], positions, positionsLen) != inverted;
    return false;
}

} // namespace faiss
