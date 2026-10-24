// What a process holds on one GPU at once, for the up-front memory checks
// of the Metal and CUDA backends (checkMetalInferenceMemory,
// checkCudaInferenceMemory). Self-play's engine is one hub with two slots;
// an arena wave one hub with one slot per network; with pipeline = on
// both at once, so a run's check adds the groups up.
//
// Mark Castelluccio, with assistance from Claude (Anthropic).
#pragma once

namespace azgo {

// `hubs` hubs of `slotsPerHub` slots, each pass of at most inferenceBatch
// positions.
struct GpuHubs {
    int hubs = 1;
    int slotsPerHub = 2;
    int inferenceBatch = 256;
};

} // namespace azgo
