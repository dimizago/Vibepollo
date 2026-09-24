#include "src/platform/windows/display_helper_v2/diagnostics.h"

#include <mutex>
#include <utility>

namespace display_helper::v2::diagnostics {
  namespace {
    // Function-local statics: set_sink() runs from another translation unit's static
    // initializer (diagnostics_boost.cpp), before namespace-scope objects here are
    // guaranteed to be constructed.
    struct sink_state_t {
      std::mutex mutex;
      Sink sink;
    };

    sink_state_t &sink_state() {
      static sink_state_t state;
      return state;
    }
  }  // namespace

  void set_sink(Sink next_sink) {
    auto &state = sink_state();
    std::lock_guard lock {state.mutex};
    state.sink = std::move(next_sink);
  }

  void emit(Level level, std::string message) {
    auto &state = sink_state();
    Sink active_sink;
    {
      std::lock_guard lock {state.mutex};
      active_sink = state.sink;
    }
    if (active_sink) {
      active_sink(level, message);
    }
  }
}  // namespace display_helper::v2::diagnostics
