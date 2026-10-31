//! Line reconstruction and match formatting.
//!
//! Takes raw byte offsets returned by the GPU compute pass, jumps into
//! the memory-mapped slice, scans backward and forward for newline (`\n`)
//! boundaries, and extracts complete lines as UTF-8 strings.

use std::borrow::Cow;

/// Finds the `(start, end)` byte range of the line containing `byte_offset`.
///
/// Scans backward for the preceding `\n` (or file start) and forward
/// for the succeeding `\n` (or file end). Trailing `\r` (CRLF) is stripped.
pub fn find_line_range(data: &[u8], byte_offset: usize) -> (usize, usize) {
    if data.is_empty() {
        return (0, 0);
    }

    let offset = byte_offset.min(data.len() - 1);

    // 1. Scan backward for '\n' or start of file
    let line_start = match data[..offset].iter().rposition(|&b| b == b'\n') {
        Some(pos) => pos + 1,
        None => 0,
    };

    // 2. Scan forward for '\n' or end of file
    let mut line_end = match data[offset..].iter().position(|&b| b == b'\n') {
        Some(pos) => offset + pos,
        None => data.len(),
    };

    // Strip trailing carriage return '\r' if present (CRLF support)
    if line_end > line_start && data[line_end - 1] == b'\r' {
        line_end -= 1;
    }

    (line_start, line_end)
}

/// Extracts the complete line containing `byte_offset` as a UTF-8 string slice.
///
/// Invalid UTF-8 sequences are replaced with the replacement character (``).
#[allow(dead_code)]
pub fn extract_line(data: &[u8], byte_offset: usize) -> Cow<'_, str> {
    let (start, end) = find_line_range(data, byte_offset);
    String::from_utf8_lossy(&data[start..end])
}

/// Extracts all unique matching lines from `data` corresponding to `match_indices`.
///
/// Matches occurring on the same line are automatically deduplicated.
pub fn extract_matching_lines<'a>(
    data: &'a [u8],
    match_indices: &[usize],
) -> Vec<(usize, Cow<'a, str>)> {
    let mut lines = Vec::new();
    let mut last_line_start = None;

    for &match_idx in match_indices {
        if match_idx >= data.len() {
            continue;
        }

        let (line_start, line_end) = find_line_range(data, match_idx);

        // Deduplicate consecutive matches residing on the same line
        if last_line_start == Some(line_start) {
            continue;
        }
        last_line_start = Some(line_start);

        let line = String::from_utf8_lossy(&data[line_start..line_end]);
        lines.push((line_start, line));
    }

    lines
}

/// Formats and prints all matching lines to standard output.
///
/// Deduplicates lines containing multiple matches so each matching line
/// is output exactly once.
pub fn print_matches(data: &[u8], match_indices: &[usize]) {
    for (_offset, line) in extract_matching_lines(data, match_indices) {
        println!("{}", line);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_line_range_single_line() {
        let text = b"hello world";
        assert_eq!(find_line_range(text, 0), (0, 11));
        assert_eq!(find_line_range(text, 6), (0, 11));
        assert_eq!(find_line_range(text, 10), (0, 11));
    }

    #[test]
    fn test_find_line_range_multiline() {
        let text = b"line 1\nline 2 with match\nline 3\n";
        // "line 2 with match" starts at index 7 and ends at index 24 ('\n' is at 24)
        let (start, end) = find_line_range(text, 14); // inside "with"
        assert_eq!(start, 7);
        assert_eq!(end, 24);
        assert_eq!(&text[start..end], b"line 2 with match");
    }

    #[test]
    fn test_find_line_range_crlf() {
        let text = b"first line\r\nsecond line with CRLF\r\nthird line\r\n";
        let (start, end) = find_line_range(text, 15);
        assert_eq!(start, 12);
        assert_eq!(end, 33);
        assert_eq!(&text[start..end], b"second line with CRLF");
    }

    #[test]
    fn test_extract_matching_lines_deduplication() {
        let text = b"match1 and match2 on line 1\nno match here\nmatch3 on line 3\n";
        // match1 at 0, match2 at 11, match3 at 42
        let indices = vec![0, 11, 42];
        let lines = extract_matching_lines(text, &indices);

        assert_eq!(lines.len(), 2, "Expected 2 distinct lines");
        assert_eq!(lines[0].1, "match1 and match2 on line 1");
        assert_eq!(lines[1].1, "match3 on line 3");
    }

    #[test]
    fn test_empty_data() {
        assert_eq!(find_line_range(b"", 0), (0, 0));
        assert_eq!(extract_line(b"", 0), "");
        assert!(extract_matching_lines(b"", &[0]).is_empty());
    }
}
