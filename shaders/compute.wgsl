// shaders/compute.wgsl

// -----------------------------------------------------------------------------
// BINDING 0: The Search Pattern
// We pass the search string as an array of 32-bit integers (one char per u32).
// This avoids unpacking the pattern inside the loop, saving GPU cycles, and
// avoids WGSL's strict 16-byte uniform array alignment rules.
// -----------------------------------------------------------------------------
@group(0) @binding(0) 
var<storage, read> pattern: array<u32>;

// -----------------------------------------------------------------------------
// BINDING 1: The Raw File Data Chunk (Max 256MB)
// WGSL does not support 8-bit byte arrays (u8). The memory-mapped file data
// is interpreted as an array of 32-bit blocks (4 characters per block).
// -----------------------------------------------------------------------------
@group(0) @binding(1) 
var<storage, read> data: array<u32>;

// -----------------------------------------------------------------------------
// BINDING 2: The Output Matches
// Index 0 is a lock-free atomic counter. The remainder of the array stores 
// the exact starting byte indices where a string match was confirmed.
// -----------------------------------------------------------------------------
struct Results {
    count: atomic<u32>,
    indices: array<u32>,
}

@group(0) @binding(2) 
var<storage, read_write> results: Results;

// Helper function to extract a single 8-bit character from a 32-bit block.
// This assumes little-endian architecture (standard for x86/ARM and WebGPU).
fn get_byte_at(byte_index: u32) -> u32 {
    let word_index = byte_index / 4u;
    let byte_offset = (byte_index % 4u) * 8u;
    
    let word = data[word_index];
    
    // Shift the requested byte to the far right, then mask out the rest
    return (word >> byte_offset) & 0xFFu;
}

const MAX_WORKGROUPS_X: u32 = 65535u;
const WORKGROUP_SIZE: u32 = 256u;

// The compute kernel. We assign 256 threads per workgroup.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Each thread gets a unique global ID mapped to a byte index (supporting 2D dispatch).
    let start_byte_index = global_id.x + global_id.y * (MAX_WORKGROUPS_X * WORKGROUP_SIZE);
    
    let pattern_length = arrayLength(&pattern);
    let total_bytes = arrayLength(&data) * 4u;

    // Boundary Check: If this thread's starting byte plus the length of the 
    // pattern exceeds our chunk size, it cannot possibly be a match.
    // This also prevents out-of-bounds memory reads.
    if (start_byte_index + pattern_length > total_bytes) {
        return;
    }

    // Evaluate the sliding window
    var is_match = true;
    for (var i = 0u; i < pattern_length; i = i + 1u) {
        let data_byte = get_byte_at(start_byte_index + i);
        let pattern_byte = pattern[i];

        if (data_byte != pattern_byte) {
            is_match = false;
            break; // Fast fail: abort loop on first mismatched character
        }
    }

    // Lock-free Write: If we found a full match, use an atomic addition to 
    // reserve a slot in the output buffer without blocking other threads.
    if (is_match) {
        // atomicAdd returns the ORIGINAL value before the addition
        let match_index = atomicAdd(&results.count, 1u);
        
        let max_results = arrayLength(&results.indices);
        
        // Prevent buffer overflow if we find more matches than we allocated space for
        if (match_index < max_results) {
            results.indices[match_index] = start_byte_index;
        }
    }
}
