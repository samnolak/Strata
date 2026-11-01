// include/strata/program/park_admit.hpp - the conversation cache's side of a park's admission (generate.cpp park_source),
// as functions a CPU test can call in the engine's order (tests/program/idle_park_test.cpp).
//
// An admission's park (park_slot before an overwrite, park_current) makes room up front: make_room evicts the oldest
// entries until the estimate fits, and put() later drops the copies the image supersedes.  An idle park (STRATA_IDLE_PARK)
// must leave the cache exactly as it was unless it stores its image, and never evicts: make_room does not know what put()
// will drop, so it would evict another conversation (and may clear the retained K/V or set guard_blocked) before a park
// that is then stopped or refused.  So an idle park only asks, read-only, whether the image fits once put() has dropped
// what it supersedes - before the capture with the estimate and again with the captured image's true size - and the one
// change it makes is put() itself.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "strata/core/conversation_cache.hpp"

namespace strata::program::park_admit {

enum class Room {
    ok,             ///< go on with the capture
    no_room_idle,   ///< an idle park: it does not fit without an eviction (nothing was changed)
    no_room,        ///< an admission's park: make_room could not free enough (it may have evicted)
};

/// Before the capture.  `idle`: read-only (can_fit_superseding); otherwise make_room(incoming, held) as before.
inline Room before_capture(core::ConversationCache& cache, size_t incoming, size_t held, bool idle,
                           const std::vector<int32_t>& ids, const std::vector<core::ConversationImageKey>& images,
                           const std::vector<core::ConversationCheckpoint>& checkpoints, bool cvec) {
    if (idle) return cache.can_fit_superseding(incoming, held, ids, images, checkpoints, cvec) ? Room::ok : Room::no_room_idle;
    return cache.make_room(incoming, held) ? Room::ok : Room::no_room;
}

/// Before put(), with the captured image's true size: an idle park's image still fits without an eviction (an
/// admission's park: always true - put() makes its own room, as before).  Read-only.
inline bool before_put(const core::ConversationCache& cache, const core::SavedConversation& image, size_t held, bool idle) {
    return !idle ||
           cache.can_fit_superseding(image.bytes(), held, image.live.ids, image.live.imgs, image.checkpoints, image.cvec);
}

}  // namespace strata::program::park_admit
