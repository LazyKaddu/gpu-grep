//! Integration tests for Result Formatter (src/scanner/format.rs).

use gpu_grep::scanner::format::{extract_line, extract_matching_lines, find_line_range};

#[test]
fn test_find_line_range_multiline_log() {
    let log_data = b"2026-10-31 10:00:00 [INFO] System initialized\n\
2026-10-31 10:00:05 [ERROR] Connection timeout to database\n\
2026-10-31 10:00:10 [INFO] Retry initiated\n";

    // "ERROR" starts at offset 66
    let error_offset = log_data
        .windows(7)
        .position(|w| w == b"[ERROR]")
        .expect("Pattern [ERROR] must exist");

    let (start, end) = find_line_range(log_data, error_offset);
    let line = &log_data[start..end];

    assert_eq!(
        line,
        b"2026-10-31 10:00:05 [ERROR] Connection timeout to database"
    );
}

#[test]
fn test_extract_line_single_and_multiple() {
    let data = b"alpha\nbeta gamma\ndelta\n";
    let gamma_offset = data.windows(5).position(|w| w == b"gamma").unwrap();

    let line = extract_line(data, gamma_offset);
    assert_eq!(line, "beta gamma");
}

#[test]
fn test_extract_matching_lines_multiple_matches_same_line() {
    let data = b"first line\nERROR: fatal ERROR occurred!\nthird line\n";
    // Find both "ERROR" occurrences on the second line
    let first_error = 11;
    let second_error = 24;

    let matches = extract_matching_lines(data, &[first_error, second_error]);
    assert_eq!(matches.len(), 1, "Should only produce 1 deduplicated line");
    assert_eq!(matches[0].1, "ERROR: fatal ERROR occurred!");
}
