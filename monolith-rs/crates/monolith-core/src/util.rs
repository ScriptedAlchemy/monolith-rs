//! Utilities mirroring a small, testable subset of Python `monolith/core/util.py`.
//!
//! The Python module is TPU + GCS heavy. For Rust parity we port:
//! - `get_bucket_name_and_relavite_path` (string parsing only)
//! - `parse_example_number_meta_file`
//! - `calculate_shard_skip_file_number`
//! - `range_dateset` logic as a pure path filter

use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::error::{MonolithError, Result};

const GS_PREFIX: &str = "gs://";
const CORE_NUMBER_PER_HOST: usize = 8;

const DATE_FORMAT_LEN: usize = 8;
const MIN_DATE: &str = "00000000";
const MAX_DATE: &str = "99999999";

/// Given a GCS path like `gs://bucket/path/to/blob`, returns `(bucket, relative_path)`.
///
/// Python name spelling is preserved: `get_bucket_name_and_relavite_path`.
pub fn get_bucket_name_and_relavite_path(gs_file_path: &str) -> Result<(String, String)> {
    // Python:
    // assert gs_file_path.find(_GS_PREFIX) != -1, "File name: {}".format(gs_file_path)
    if !gs_file_path.contains(GS_PREFIX) {
        return Err(MonolithError::PyAssertionError {
            message: format!("File name: {}", gs_file_path),
        });
    }

    let bucket_name_start = GS_PREFIX.len();
    let bucket_name_end = gs_file_path[bucket_name_start..]
        .find('/')
        .map(|i| i + bucket_name_start)
        .ok_or_else(|| MonolithError::PyAssertionError {
            message: format!("File name: {}", gs_file_path),
        })?;
    let bucket = &gs_file_path[bucket_name_start..bucket_name_end];
    let rel = &gs_file_path[bucket_name_end + 1..];
    Ok((bucket.to_string(), rel.to_string()))
}

/// Parses a meta file containing lines of `file_name,count` and returns them in order.
///
/// Lines without `,` are ignored. The file names must be strictly increasing (dictionary order).
pub fn parse_example_number_meta_file(meta_file: &str) -> Result<Vec<(String, i64)>> {
    let f = File::open(meta_file).map_err(|e| MonolithError::InternalError {
        message: format!("Failed to open {}: {}", meta_file, e),
    })?;
    let reader = BufReader::new(f);

    let mut out = Vec::new();
    let mut previous_file_name = String::new();
    for line in reader.lines() {
        let line = line.map_err(|e| MonolithError::InternalError {
            message: format!("Failed to read {}: {}", meta_file, e),
        })?;
        if !line.contains(',') {
            continue;
        }
        let mut split = line.splitn(2, ',');
        let file_name = split.next().unwrap_or_default().to_string();
        let count_str = split.next().unwrap_or_default().trim();

        if !previous_file_name.is_empty() && !(previous_file_name < file_name) {
            return Err(MonolithError::PyAssertionError {
                message: format!(
                    "File name must be in dictionary ascending order. Previous file name: {}, current file file name: {}",
                    previous_file_name, file_name
                ),
            });
        }
        previous_file_name = file_name.clone();

        let count: i64 = count_str.parse().map_err(|_| MonolithError::PyValueError {
            message: format!("Invalid example count: {}", count_str),
        })?;
        out.push((file_name, count));
    }
    Ok(out)
}

/// Calculates, for each shard (host), how many files have been fully processed at a checkpoint.
pub fn calculate_shard_skip_file_number(
    file_example_number: &[i64],
    shard_num: usize,
    completed_steps_number: i64,
    batch_size_per_core: i64,
) -> Vec<i64> {
    let processed_example_number_per_host =
        batch_size_per_core * completed_steps_number * (CORE_NUMBER_PER_HOST as i64);
    let mut shard_index: usize = 0;

    let mut shard_skip_file_number = vec![0_i64; shard_num];
    let mut shard_accumulated_example_count = vec![0_i64; shard_num];

    for &example_number in file_example_number {
        if example_number + shard_accumulated_example_count[shard_index]
            <= processed_example_number_per_host
        {
            shard_accumulated_example_count[shard_index] += example_number;
            shard_skip_file_number[shard_index] += 1;
        }
        shard_index = (shard_index + 1) % shard_num;
    }

    shard_skip_file_number
}

/// Filters dataset paths by date range, matching `util.range_dateset()` semantics.
///
/// `root_path` is a prefix; the date is parsed from the next 8 chars after `root_path.len()`.
/// If `start_date`/`end_date` are `None`, Python defaults to `"00000000"`/`"99999999"`.
///
/// Note: The Python version operates on TF string tensors and will error at runtime if the
/// substring cannot be parsed to an integer. Here we conservatively drop any paths that don't
/// parse cleanly, which is sufficient to mirror the existing Python tests.
pub fn range_dateset<'a, I>(
    dataset: I,
    root_path: &str,
    start_date: Option<&str>,
    end_date: Option<&str>,
) -> Vec<String>
where
    I: IntoIterator<Item = &'a str>,
{
    let start = start_date
        .unwrap_or(MIN_DATE)
        .parse::<i32>()
        .unwrap_or(i32::MIN);
    let end = end_date
        .unwrap_or(MAX_DATE)
        .parse::<i32>()
        .unwrap_or(i32::MAX);
    let prefix_len = root_path.len();

    dataset
        .into_iter()
        .filter_map(|p| {
            if p.len() < prefix_len + DATE_FORMAT_LEN {
                return None;
            }
            let date_str = &p[prefix_len..prefix_len + DATE_FORMAT_LEN];
            let date: i32 = date_str.parse().ok()?;
            if date >= start && date <= end {
                Some(p.to_string())
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_dataset_single() {
        let root_path = "gs://test_folder/unzipped_tf_records_corrected_repartitioned/";
        let input = vec![
            "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200501/00/part",
            "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200502/00/part",
            "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200503/00/part",
        ];
        let out = range_dateset(
            input.iter().copied(),
            root_path,
            Some("20200502"),
            Some("20200502"),
        );
        assert_eq!(
            out,
            vec!["gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200502/00/part"]
        );
    }

    #[test]
    fn test_range_dataset_multiple() {
        let root_path = "gs://test_folder/unzipped_tf_records_corrected_repartitioned/";
        let input = vec![
            "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200501/00/part",
            "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200502/00/part",
            "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200503/00/part",
            "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200503/01/part",
            "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200504/01/part",
        ];
        let out = range_dateset(
            input.iter().copied(),
            root_path,
            Some("20200502"),
            Some("20200503"),
        );
        assert_eq!(
            out,
            vec![
                "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200502/00/part",
                "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200503/00/part",
                "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200503/01/part",
            ]
        );
    }

    #[test]
    fn test_range_dataset_out_of_boundary() {
        let root_path = "gs://test_folder/unzipped_tf_records_corrected_repartitioned/";
        let input = vec![
            "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200501/00/part",
            "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200502/00/part",
        ];
        let out = range_dateset(
            input.iter().copied(),
            root_path,
            Some("20200401"),
            Some("20200505"),
        );
        assert_eq!(
            out,
            vec![
                "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200501/00/part",
                "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200502/00/part",
            ]
        );
    }

    #[test]
    fn test_range_dataset_no_start_date() {
        let root_path = "gs://test_folder/unzipped_tf_records_corrected_repartitioned/";
        let input = vec![
            "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200501/00/part",
            "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200502/00/part",
        ];
        let out = range_dateset(input.iter().copied(), root_path, None, Some("20200505"));
        assert_eq!(
            out,
            vec![
                "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200501/00/part",
                "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200502/00/part",
            ]
        );
    }

    #[test]
    fn test_range_dataset_no_end_date() {
        let root_path = "gs://test_folder/unzipped_tf_records_corrected_repartitioned/";
        let input = vec![
            "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200501/00/part",
            "gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200502/00/part",
        ];
        let out = range_dateset(input.iter().copied(), root_path, Some("20200502"), None);
        assert_eq!(
            out,
            vec!["gs://test_folder/unzipped_tf_records_corrected_repartitioned/20200502/00/part"]
        );
    }

    #[test]
    fn test_get_bucket_name_and_relavite_path() {
        let (bucket, rel) = get_bucket_name_and_relavite_path("gs://my-bucket/a/b/c.txt").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(rel, "a/b/c.txt");

        let err = get_bucket_name_and_relavite_path("/not-gs/path").unwrap_err();
        assert_eq!(err.to_string(), "File name: /not-gs/path");
    }
}
