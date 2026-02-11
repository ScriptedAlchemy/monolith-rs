//! TF-free parity for Python `monolith.native_training.nested_tensors`.
//!
//! The Python module flattens nested structures and preserves non-tensor leaves.
//! It also supports 1D `RaggedTensor` by flattening to `(values, row_splits)`.
//!
//! In Rust we model this as a small tree plus a `NestedTensors` helper that can
//! extract a flat list of tensors and then rebuild the nested structure.

use std::collections::BTreeMap;

/// A minimal 1D ragged tensor representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RaggedI64 {
    pub values: Vec<i64>,
    pub row_splits: Vec<i64>,
}

impl RaggedI64 {
    pub fn from_row_splits(values: Vec<i64>, row_splits: Vec<i64>) -> Self {
        Self { values, row_splits }
    }
}

/// A nested structure for `NestedTensors`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NestedI64 {
    Tensor(i64),
    Ragged(RaggedI64),
    Bool(bool),
    Int(i64),
    Str(String),
    Null,
    Seq(Vec<NestedI64>),
    Map(BTreeMap<String, NestedI64>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Tensor(usize),
    Ragged(usize), // index of ragged tensor in `raggeds`
    Bool(usize),
    Int(usize),
    Str(usize),
    Null,
    Seq(Vec<Token>),
    Map(BTreeMap<String, Token>),
}

/// Flattens nested structures into a list of tensor-like leaves and rebuilds results.
#[derive(Debug, Clone)]
pub struct NestedTensors {
    template: Token,
    tensors: Vec<i64>,
    raggeds: Vec<RaggedI64>,
    bools: Vec<bool>,
    ints: Vec<i64>,
    strs: Vec<String>,
}

impl NestedTensors {
    pub fn new(nested: NestedI64) -> Self {
        let mut tensors = Vec::new();
        let mut raggeds = Vec::new();
        let mut bools = Vec::new();
        let mut ints = Vec::new();
        let mut strs = Vec::new();
        let template = Self::tokenize(&nested, &mut tensors, &mut raggeds, &mut bools, &mut ints, &mut strs);
        Self {
            template,
            tensors,
            raggeds,
            bools,
            ints,
            strs,
        }
    }

    fn tokenize(
        nested: &NestedI64,
        tensors: &mut Vec<i64>,
        raggeds: &mut Vec<RaggedI64>,
        bools: &mut Vec<bool>,
        ints: &mut Vec<i64>,
        strs: &mut Vec<String>,
    ) -> Token {
        match nested {
            NestedI64::Tensor(v) => {
                let idx = tensors.len();
                tensors.push(*v);
                Token::Tensor(idx)
            }
            NestedI64::Ragged(r) => {
                let idx = raggeds.len();
                raggeds.push(r.clone());
                Token::Ragged(idx)
            }
            NestedI64::Bool(v) => {
                let idx = bools.len();
                bools.push(*v);
                Token::Bool(idx)
            }
            NestedI64::Int(v) => {
                let idx = ints.len();
                ints.push(*v);
                Token::Int(idx)
            }
            NestedI64::Str(s) => {
                let idx = strs.len();
                strs.push(s.clone());
                Token::Str(idx)
            }
            NestedI64::Null => Token::Null,
            NestedI64::Seq(v) => Token::Seq(
                v.iter()
                    .map(|x| Self::tokenize(x, tensors, raggeds, bools, ints, strs))
                    .collect(),
            ),
            NestedI64::Map(m) => Token::Map(
                m.iter()
                    .map(|(k, v)| {
                        (
                            k.clone(),
                            Self::tokenize(v, tensors, raggeds, bools, ints, strs),
                        )
                    })
                    .collect(),
            ),
        }
    }

    /// Returns flattened tensors. For ragged leaves we append `(values, row_splits)`.
    pub fn get_tensors(&self) -> Vec<i64> {
        let mut out = self.tensors.clone();
        for r in &self.raggeds {
            out.extend_from_slice(&r.values);
            out.extend_from_slice(&r.row_splits);
        }
        out
    }

    /// Rebuild the nested structure from flattened tensors.
    ///
    /// The input layout matches `get_tensors()`: first all scalar tensors in traversal order,
    /// then for each ragged tensor: its values then its row_splits.
    pub fn get_nested_result(&self, flat: &[i64]) -> NestedI64 {
        let tensor_count = self.tensors.len();
        let mut tensor_vals = flat[..tensor_count].to_vec();
        let mut cursor = tensor_count;

        let mut raggeds = Vec::with_capacity(self.raggeds.len());
        for r in &self.raggeds {
            let values_len = r.values.len();
            let splits_len = r.row_splits.len();
            let values = flat[cursor..cursor + values_len].to_vec();
            cursor += values_len;
            let row_splits = flat[cursor..cursor + splits_len].to_vec();
            cursor += splits_len;
            raggeds.push(RaggedI64 { values, row_splits });
        }

        Self::rebuild(
            &self.template,
            &mut tensor_vals,
            &raggeds,
            &self.bools,
            &self.ints,
            &self.strs,
        )
    }

    fn rebuild(
        t: &Token,
        tensors: &mut [i64],
        raggeds: &[RaggedI64],
        bools: &[bool],
        ints: &[i64],
        strs: &[String],
    ) -> NestedI64 {
        match t {
            Token::Tensor(i) => NestedI64::Tensor(tensors[*i]),
            Token::Ragged(i) => NestedI64::Ragged(raggeds[*i].clone()),
            Token::Bool(i) => NestedI64::Bool(bools[*i]),
            Token::Int(i) => NestedI64::Int(ints[*i]),
            Token::Str(i) => NestedI64::Str(strs[*i].clone()),
            Token::Null => NestedI64::Null,
            Token::Seq(v) => NestedI64::Seq(v.iter().map(|x| Self::rebuild(x, tensors, raggeds, bools, ints, strs)).collect()),
            Token::Map(m) => NestedI64::Map(
                m.iter()
                    .map(|(k, v)| (k.clone(), Self::rebuild(v, tensors, raggeds, bools, ints, strs)))
                    .collect(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_tensors_basic() {
        let mut m = BTreeMap::new();
        m.insert("a".to_string(), NestedI64::Tensor(1));
        m.insert(
            "b".to_string(),
            NestedI64::Seq(vec![NestedI64::Tensor(1), NestedI64::Tensor(1)]),
        );
        let n = NestedTensors::new(NestedI64::Map(m));
        let tensors = n.get_tensors();
        let replaced: Vec<i64> = tensors.into_iter().map(|_| 0).collect();
        let result = n.get_nested_result(&replaced);

        let mut exp = BTreeMap::new();
        exp.insert("a".to_string(), NestedI64::Tensor(0));
        exp.insert("b".to_string(), NestedI64::Seq(vec![NestedI64::Tensor(0), NestedI64::Tensor(0)]));
        assert_eq!(result, NestedI64::Map(exp));
    }

    #[test]
    fn nested_tensors_constant_roundtrip() {
        let mut m = BTreeMap::new();
        m.insert("a".to_string(), NestedI64::Map(BTreeMap::from([("b".to_string(), NestedI64::Int(2))])));
        let n = NestedTensors::new(NestedI64::Map(m.clone()));
        assert!(n.get_tensors().is_empty());
        let out = n.get_nested_result(&[]);
        assert_eq!(out, NestedI64::Map(m));
    }

    #[test]
    fn nested_tensors_ragged_roundtrip() {
        let r = RaggedI64::from_row_splits(vec![1, 2, 3], vec![0, 0, 1, 3]);
        let n = NestedTensors::new(NestedI64::Ragged(r.clone()));
        let flat = n.get_tensors();
        let out = n.get_nested_result(&flat);
        assert_eq!(out, NestedI64::Ragged(r));
    }
}

