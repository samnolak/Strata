// include/strata/core/capture_abort.hpp - STRATA_IDLE_PARK: a conversation capture (conversation_snapshot_save) that the
// caller can stop part-way.  Header-only and free of CUDA so a CPU test can check it (tests/program/idle_park_test.cpp).
//
// An idle park copies a slot's K/V into host memory layer by layer (~0.4 s at 100K tokens, ~1 s at 300K).  A request that
// arrives meanwhile would wait for the whole copy before its prompt starts.  While a CaptureAbortScope lives on a thread,
// a capture on that thread asks it between its checkpoint copies and its K/V layers whether to stop; when it says so the
// capture ends, publishes nothing (it builds into a new object) and fails with kCaptureAborted as its error - the caller
// treats that as a skipped park.  No scope on the thread (every other caller): never asked, nothing changes.
#pragma once

#include <string>

namespace strata::core {

inline constexpr const char* kCaptureAborted = "capture aborted (a request came in)";

/// The current thread's question: `fn(ctx)` true = stop now.  Null: none (the capture runs to its end).
struct CaptureAbortHook {
    bool (*fn)(void*) = nullptr;
    void* ctx = nullptr;
};
inline CaptureAbortHook& capture_abort_hook() {
    thread_local CaptureAbortHook h;
    return h;
}

/// Installs `fn(ctx)` as this thread's question for its lifetime (the previous one comes back after: scopes nest).
class CaptureAbortScope {
public:
    CaptureAbortScope(bool (*fn)(void*), void* ctx) : prev_(capture_abort_hook()) { capture_abort_hook() = {fn, ctx}; }
    ~CaptureAbortScope() { capture_abort_hook() = prev_; }
    CaptureAbortScope(const CaptureAbortScope&) = delete;
    CaptureAbortScope& operator=(const CaptureAbortScope&) = delete;

private:
    CaptureAbortHook prev_;
};

/// True when this thread's question says stop (false with none).
inline bool capture_abort_requested() {
    const CaptureAbortHook& h = capture_abort_hook();
    return h.fn != nullptr && h.fn(h.ctx);
}

/// A capture's check point: false (with `error` = kCaptureAborted) when it must stop here.
inline bool capture_continue(std::string& error) {
    if (!capture_abort_requested()) return true;
    error = kCaptureAborted;
    return false;
}

/// Whether a capture's error is this abort (a skipped park, not a failed one).
inline bool capture_aborted(const std::string& error) { return error == kCaptureAborted; }

}  // namespace strata::core
