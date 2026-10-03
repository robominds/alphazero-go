// A checkpoint is everything needed to rebuild a network exactly: its shape
// (NetConfig) and every tensor, including batch norm's running statistics,
// which inference depends on but the optimizer never touches. Each tensor is
// stored with its name and size, so loading can check it is putting each
// array where it came from and report precisely what does not match.
#pragma once
#include <string>
#include "azgo/network.hpp"

namespace azgo {

// Format: magic "AZGO" (u32 0x4F475A41), version u32, NetConfig as 5 x i32
// (boardSize, history, blocks, filters, valueHidden), in version 2 followed by
// the ownership flag (i32, 0 or 1), then u32 tensor count, then per tensor:
// name (str), u64 size, floats. Params in params() order, then buffers in
// buffers() order. Saved atomically. A network without an ownership head is
// written as version 1, byte for byte what older builds wrote (and can still
// read); one with the head as version 2. Version 1 loads as ownership = false.
void saveNetwork(Network& net, const std::string& path);

// Throws std::runtime_error naming the problem: missing file, bad magic,
// unsupported version, tensor name/size mismatch, truncation.
Network loadNetwork(const std::string& path);
NetConfig readCheckpointConfig(const std::string& path);

} // namespace azgo
