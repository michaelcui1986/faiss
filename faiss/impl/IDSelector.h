/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_set>
#include <vector>

#include <faiss/MetricType.h>

/** IDSelector is intended to define a subset of vectors to handle (for removal
 * or as subset to search) */

namespace faiss {

/** Encapsulates a set of ids to handle. */
struct IDSelector {
    virtual bool is_member(idx_t id) const = 0;
    virtual ~IDSelector() {}
};

/** ids between [imin, imax) */
struct IDSelectorRange : IDSelector {
    idx_t imin, imax;

    /// Assume that the ids to handle are sorted. In some cases this can speed
    /// up processing
    bool assume_sorted;

    IDSelectorRange(idx_t imin, idx_t imax, bool assume_sorted = false);

    bool is_member(idx_t id) const final;

    /// for sorted ids, find the range of list indices where the valid ids are
    /// stored
    void find_sorted_ids_bounds(
            size_t list_size,
            const idx_t* ids,
            size_t* jmin,
            size_t* jmax) const;

    ~IDSelectorRange() override {}
};

/** Simple array of elements
 *
 * is_member calls are very inefficient, but some operations can use the ids
 * directly.
 */
struct IDSelectorArray : IDSelector {
    size_t n;
    const idx_t* ids;

    /** Construct with an array of ids to process
     *
     * @param n number of ids to store
     * @param ids elements to store. The pointer should remain valid during
     *            IDSelectorArray's lifetime
     */
    IDSelectorArray(size_t n, const idx_t* ids);
    bool is_member(idx_t id) const final;
    ~IDSelectorArray() override {}
};

/** Ids from a set.
 *
 * Repetitions of ids in the indices set passed to the constructor does not hurt
 * performance.
 *
 * The hash function used for the bloom filter and GCC's implementation of
 * unordered_set are just the least significant bits of the id. This works fine
 * for random ids or ids in sequences but will produce many hash collisions if
 * lsb's are always the same
 */
struct IDSelectorBatch : IDSelector {
    std::unordered_set<idx_t> set;

    // Bloom filter to avoid accessing the unordered set if it is unlikely
    // to be true
    std::vector<uint8_t> bloom;
    int nbits;
    idx_t mask;

    /** Construct with an array of ids to process
     *
     * @param n number of ids to store
     * @param ids elements to store. The pointer can be released after
     *            construction
     */
    IDSelectorBatch(size_t n, const idx_t* indices);
    bool is_member(idx_t id) const final;
    ~IDSelectorBatch() override {}
};

/** One bit per element. Constructed with a bitmap, size ceil(n / 8).
 */
struct IDSelectorBitmap : IDSelector {
    size_t n;
    const uint8_t* bitmap;

    /** Construct with a binary mask
     *
     * @param n size of the bitmap array
     * @param bitmap id will be selected iff id / 8 < n and bit number
     *               (i%8) of bitmap[floor(i / 8)] is 1.
     */
    IDSelectorBitmap(size_t n, const uint8_t* bitmap);
    bool is_member(idx_t id) const final;
    ~IDSelectorBitmap() override {}
};

/** One bit per element. Constructed with a bitmap, size ceil(bitmapLen / 8).
 */
struct DocIDSelectorBitmap : IDSelector {
    size_t idmapLen;
    const uint32_t *idmap;
    size_t bitmapLen;
    const uint8_t *bitmap;
    bool inverted;

    /** Construct with a binary mask
     *
     * @param idmapLen size of the idmap array
     * @param idmap docid will be selected iff id < idmapLen.
     * @param bitmapLen size of the bitmap array
     * @param bitmap docid will be selected iff docid / 8 < bitmapLen and
     *        bit number (docid%8) of bitmap[floor(docid / 8)] is 1.
     * @param inverted bit is returned inverted.
     */
    DocIDSelectorBitmap(size_t idmapLen, const uint32_t *idmap,
                        size_t bitmapLen, const uint8_t *bitmap);
    DocIDSelectorBitmap(size_t idmapLen, const uint32_t *idmap,
                        size_t bitmapLen, const uint8_t *bitmap,
                        bool inverted);
    bool is_member(idx_t id) const final;
    ~DocIDSelectorBitmap() override {}
};

/** A list of uint32_t positions of 1-bits sorted in ascending order.
 */
struct DocIDSelectorBitlist : IDSelector {
    size_t idmapLen;
    const uint32_t *idmap;
    size_t positionsLen;
    const uint32_t *positions;
    bool inverted;

    /*
     * @param idmapLen size of the idmap array
     * @param idmap docid will be selected iff id < idmapLen.
     * @param positionsLen size of the positions array
     * @param positions docid will be selected iff docid is found in positions
     *        with std::bsearch.
     * @param inverted bit is returned inverted.
     */
    DocIDSelectorBitlist(size_t idmapLen, const uint32_t *idmap,
                         size_t positionsLen, const uint32_t *positions);
    DocIDSelectorBitlist(size_t idmapLen, const uint32_t *idmap,
                         size_t positionsLen, const uint32_t *positions,
                         bool inverted);
    bool is_member(idx_t id) const final;
    ~DocIDSelectorBitlist() override {}
};

/** reverts the membership test of another selector */
struct IDSelectorNot : IDSelector {
    const IDSelector* sel;
    explicit IDSelectorNot(const IDSelector* sel_) : sel(sel_) {}
    bool is_member(idx_t id) const final {
        return !sel->is_member(id);
    }
    virtual ~IDSelectorNot() {}
};

// IDSelectorField_LT
template <typename T> struct IDSelectorField_LT : IDSelector {
    const size_t doc_size;
    const size_t field_offset;
    const uint8_t *doc_bytes;
    const uint32_t *doc_ids;
    const T value;

    /**
     * @param doc_size, size of each individual doc
     * @param field_offset, the field's offset in its doc
     * @param doc_bytes, the consecutive bytes for all docs
     * @param value, the constant to compare against
     */
    IDSelectorField_LT(size_t doc_size, size_t field_offset,
                       const uint8_t *doc_bytes,
                       const uint32_t *doc_ids,
                       T value) :
        doc_size(doc_size), field_offset(field_offset),
        doc_bytes(doc_bytes), doc_ids(doc_ids),
        value(value) {}
    bool is_member(idx_t id) const final {
        return *reinterpret_cast<const T *>(doc_bytes + doc_ids[id]*doc_size + field_offset) < value;
    }
    ~IDSelectorField_LT() override {}
};

// IDSelectorField_LTE
template <typename T> struct IDSelectorField_LTE : IDSelector {
    const size_t doc_size;
    const size_t field_offset;
    const uint8_t *doc_bytes;
    const uint32_t *doc_ids;
    const T value;

    /**
     * @param doc_size, size of each individual doc
     * @param field_offset, the field's offset in its doc
     * @param doc_bytes, the consecutive bytes for all docs
     * @param value, the constant to compare against
     */
    IDSelectorField_LTE(size_t doc_size, size_t field_offset,
                        const uint8_t *doc_bytes,
                        const uint32_t *doc_ids,
                        T value) :
        doc_size(doc_size), field_offset(field_offset),
        doc_bytes(doc_bytes), doc_ids(doc_ids),
        value(value) {}
    bool is_member(idx_t id) const final {
        return *reinterpret_cast<const T *>(doc_bytes + doc_ids[id]*doc_size + field_offset) <= value;
    }
    ~IDSelectorField_LTE() override {}
};

// IDSelectorField_GT
template <typename T> struct IDSelectorField_GT : IDSelector {
    const size_t doc_size;
    const size_t field_offset;
    const uint8_t *doc_bytes;
    const uint32_t *doc_ids;
    const T value;

    /**
     * @param doc_size, size of each individual doc
     * @param field_offset, the field's offset in its doc
     * @param doc_bytes, the consecutive bytes for all docs
     * @param value, the constant to compare against
     */
    IDSelectorField_GT(size_t doc_size, size_t field_offset,
                       const uint8_t *doc_bytes,
                       const uint32_t *doc_ids,
                       T value) :
        doc_size(doc_size), field_offset(field_offset),
        doc_bytes(doc_bytes), doc_ids(doc_ids),
        value(value) {}
    bool is_member(idx_t id) const final {
        return *reinterpret_cast<const T *>(doc_bytes + doc_ids[id]*doc_size + field_offset) > value;
    }
    ~IDSelectorField_GT() override {}
};

// IDSelectorField_GTE
template <typename T> struct IDSelectorField_GTE : IDSelector {
    const size_t doc_size;
    const size_t field_offset;
    const uint8_t *doc_bytes;
    const uint32_t *doc_ids;
    const T value;

    /**
     * @param doc_size, size of each individual doc
     * @param field_offset, the field's offset in its doc
     * @param doc_bytes, the consecutive bytes for all docs
     * @param value, the constant to compare against
     */
    IDSelectorField_GTE(size_t doc_size, size_t field_offset,
                        const uint8_t *doc_bytes,
                        const uint32_t *doc_ids,
                        T value) :
        doc_size(doc_size), field_offset(field_offset),
        doc_bytes(doc_bytes), doc_ids(doc_ids),
        value(value) {}
    bool is_member(idx_t id) const final {
        return *reinterpret_cast<const T *>(doc_bytes + doc_ids[id]*doc_size + field_offset) >= value;
    }
    ~IDSelectorField_GTE() override {}
};

// IDSelectorField_EQ
template <typename T> struct IDSelectorField_EQ : IDSelector {
    const size_t doc_size;
    const size_t field_offset;
    const uint8_t *doc_bytes;
    const uint32_t *doc_ids;
    const T value;

    /**
     * @param doc_size, size of each individual doc
     * @param field_offset, the field's offset in its doc
     * @param doc_bytes, the consecutive bytes for all docs
     * @param value, the constant to compare against
     */
    IDSelectorField_EQ(size_t doc_size, size_t field_offset,
                       const uint8_t *doc_bytes,
                       const uint32_t *doc_ids,
                       T value) :
        doc_size(doc_size), field_offset(field_offset),
        doc_bytes(doc_bytes), doc_ids(doc_ids),
        value(value) {}
    bool is_member(idx_t id) const final {
        return *reinterpret_cast<const T *>(doc_bytes + doc_ids[id]*doc_size + field_offset) == value;
    }
    ~IDSelectorField_EQ() override {}
};

// IDSelectorField_NE
template <typename T> struct IDSelectorField_NE : IDSelector {
    const size_t doc_size;
    const size_t field_offset;
    const uint8_t *doc_bytes;
    const uint32_t *doc_ids;
    const T value;

    /**
     * @param doc_size, size of each individual doc
     * @param field_offset, the field's offset in its doc
     * @param doc_bytes, the consecutive bytes for all docs
     * @param value, the constant to compare against
     */
    IDSelectorField_NE(size_t doc_size, size_t field_offset,
                       const uint8_t *doc_bytes,
                       const uint32_t *doc_ids,
                       T value) :
        doc_size(doc_size), field_offset(field_offset),
        doc_bytes(doc_bytes), doc_ids(doc_ids),
        value(value) {}
    bool is_member(idx_t id) const final {
        return *reinterpret_cast<const T *>(doc_bytes + doc_ids[id]*doc_size + field_offset) != value;
    }
    ~IDSelectorField_NE() override {}
};

// IDSelectorField_BITAND
template <typename T> struct IDSelectorField_BITAND : IDSelector {
    const size_t doc_size;
    const size_t field_offset;
    const uint8_t *doc_bytes;
    const uint32_t *doc_ids;
    const T value;

    /**
     * @param doc_size, size of each individual doc
     * @param field_offset, the field's offset in its doc
     * @param doc_bytes, the consecutive bytes for all docs
     * @param value, the constant to compare against
     */
    IDSelectorField_BITAND(size_t doc_size, size_t field_offset,
                           const uint8_t *doc_bytes,
                           const uint32_t *doc_ids,
                           T value) :
        doc_size(doc_size), field_offset(field_offset),
        doc_bytes(doc_bytes), doc_ids(doc_ids),
        value(value) {}
    bool is_member(idx_t id) const final {
        return (*reinterpret_cast<const T *>(doc_bytes + doc_ids[id]*doc_size + field_offset) & value) != 0;
    }
    ~IDSelectorField_BITAND() override {}
};

/// selects all entries (useful for benchmarking)
struct IDSelectorAll : IDSelector {
    bool is_member(idx_t /* id */) const final {
        return true;
    }
    virtual ~IDSelectorAll() {}
};

/// selects no entries (useful for corner cases)
struct IDSelectorNone : IDSelector {
    bool is_member(idx_t /* id */) const final {
        return false;
    }
    virtual ~IDSelectorNone() {}
};

/// does an AND operation on the two given IDSelector's is_membership
/// results.
struct IDSelectorAnd : IDSelector {
    const IDSelector* lhs;
    const IDSelector* rhs;
    IDSelectorAnd(const IDSelector* lhs_, const IDSelector* rhs_)
            : lhs(lhs_), rhs(rhs_) {}
    bool is_member(idx_t id) const final {
        return lhs->is_member(id) && rhs->is_member(id);
    }
    virtual ~IDSelectorAnd() {}
};

/// does an OR operation on the two given IDSelector's is_membership
/// results.
struct IDSelectorOr : IDSelector {
    const IDSelector* lhs;
    const IDSelector* rhs;
    IDSelectorOr(const IDSelector* lhs_, const IDSelector* rhs_)
            : lhs(lhs_), rhs(rhs_) {}
    bool is_member(idx_t id) const final {
        return lhs->is_member(id) || rhs->is_member(id);
    }
    virtual ~IDSelectorOr() {}
};

/// does an XOR operation on the two given IDSelector's is_membership
/// results.
struct IDSelectorXOr : IDSelector {
    const IDSelector* lhs;
    const IDSelector* rhs;
    IDSelectorXOr(const IDSelector* lhs_, const IDSelector* rhs_)
            : lhs(lhs_), rhs(rhs_) {}
    bool is_member(idx_t id) const final {
        return lhs->is_member(id) ^ rhs->is_member(id);
    }
    virtual ~IDSelectorXOr() {}
};

} // namespace faiss
