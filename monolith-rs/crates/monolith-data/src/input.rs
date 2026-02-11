//! Python `monolith.native_training.input` parity helpers.
//!
//! The upstream Python module provides helpers for generating TensorFlow
//! `tf.train.Example` payloads for simple FFM-style examples.

use monolith_proto::tensorflow_core;
use prost::Message;
use rand::Rng;
use std::collections::HashMap;

/// Python `slot_to_key(slot)` parity.
pub fn slot_to_key(slot: usize) -> String {
    format!("feature_{slot}")
}

fn int64_feature(values: Vec<i64>) -> tensorflow_core::Feature {
    tensorflow_core::Feature {
        kind: Some(tensorflow_core::feature::Kind::Int64List(
            tensorflow_core::Int64List { value: values },
        )),
    }
}

fn float_feature(values: Vec<f32>) -> tensorflow_core::Feature {
    tensorflow_core::Feature {
        kind: Some(tensorflow_core::feature::Kind::FloatList(
            tensorflow_core::FloatList { value: values },
        )),
    }
}

/// Generate an FFM-style `tf.train.Example` (serialized).
///
/// Mirrors `monolith/native_training/input.py::generate_ffm_example`:
/// - Label is always 0 (Python uses `np.random.randint(low=0, high=1)`).
/// - For each slot i, pick `num_ids` in `[1, length]` and sample ids in
///   `[max_vocab * i, max_vocab * i + vocab_size)`.
pub fn generate_ffm_example<R: Rng + ?Sized>(
    rng: &mut R,
    vocab_sizes: &[usize],
    length: usize,
) -> Vec<u8> {
    let max_vocab = vocab_sizes.iter().copied().max().unwrap_or(0) as i64;
    let mut feature: HashMap<String, tensorflow_core::Feature> = HashMap::new();

    // Python always generates 0 here; keep parity.
    feature.insert("label".to_string(), float_feature(vec![0.0]));

    for (i, vocab_size) in vocab_sizes.iter().copied().enumerate() {
        let num_ids = rng.gen_range(1..=length.max(1)) as usize;
        let start = max_vocab * (i as i64);
        let end = start + vocab_size as i64;

        let mut ids = Vec::with_capacity(num_ids);
        for _ in 0..num_ids {
            // If vocab_size==0, Python would crash; keep the behavior by panicking.
            ids.push(rng.gen_range(start..end));
        }

        feature.insert(slot_to_key(i), int64_feature(ids));
    }

    let example = tensorflow_core::Example {
        features: Some(tensorflow_core::Features { feature }),
    };
    example.encode_to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;
    use rand::SeedableRng;

    #[test]
    fn test_slot_to_key() {
        assert_eq!(slot_to_key(0), "feature_0");
        assert_eq!(slot_to_key(123), "feature_123");
    }

    #[test]
    fn test_generate_ffm_example_roundtrip() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let bytes = generate_ffm_example(&mut rng, &[3, 5], 5);

        let ex = tensorflow_core::Example::decode(bytes.as_slice()).expect("decode tf Example");
        let feats = ex.features.expect("features");

        // Label feature is present and fixed to 0.
        let label = feats.feature.get("label").expect("label feature");
        match &label.kind {
            Some(tensorflow_core::feature::Kind::FloatList(l)) => assert_eq!(l.value, vec![0.0]),
            other => panic!("unexpected label feature kind: {other:?}"),
        }

        // Slot keys are present.
        assert!(feats.feature.contains_key("feature_0"));
        assert!(feats.feature.contains_key("feature_1"));
    }
}
