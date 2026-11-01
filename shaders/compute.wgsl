// -----------------------------------------------------------------------------
// Bindings remain exactly the same
// -----------------------------------------------------------------------------
@group(0) @binding(0) var<storage, read> pattern: array<u32>;
@group(0) @binding(1) var<storage, read> data: array<u32>;

struct Results {
    count: atomic<u32>,
    indices: array<u32>,
}
@group(0) @binding(2) var<storage, read_write> results: Results;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let start_byte_index = global_id.x;
    let pattern_length = arrayLength(&pattern);
    let total_bytes = arrayLength(&data) * 4u;

    if (start_byte_index + pattern_length > total_bytes) {
        return;
    }

    var is_match = true;
    
    // Register Cache to prevent redundant global VRAM reads
    var current_word_index = 0xFFFFFFFFu; 
    var current_word = 0u;

    for (var i = 0u; i < pattern_length; i = i + 1u) {
        let target_byte = start_byte_index + i;
        
        // BITWISE OPTIMIZATION: 
        // target_byte >> 2u is identical to target_byte / 4, but much faster.
        let word_index = target_byte >> 2u; 
        
        // MEMORY OPTIMIZATION: Only hit global VRAM if we crossed into a new u32 block
        if (word_index != current_word_index) {
            current_word = data[word_index];
            current_word_index = word_index;
        }

        // BITWISE OPTIMIZATION:
        // (target_byte & 3u) is identical to target_byte % 4.
        // << 3u is identical to * 8.
        let byte_offset = (target_byte & 3u) << 3u;
        let data_byte = (current_word >> byte_offset) & 0xFFu;
        
        let pattern_byte = pattern[i];

        if (data_byte != pattern_byte) {
            is_match = false;
            break; 
        }
    }

    if (is_match) {
        let match_index = atomicAdd(&results.count, 1u);
        let max_results = arrayLength(&results.indices);
        
        if (match_index < max_results) {
            results.indices[match_index] = start_byte_index;
        }
    }
}