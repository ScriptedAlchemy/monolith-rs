// Copyright 2022 ByteDance and/or its affiliates.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Stock Prediction Example for Monolith-RS.
//!
//! This module wires the implementation split across the sibling files
//! (app, model, data, indicators, etc.) and exposes the `run` entrypoint.

pub mod app;
pub mod backtest;
pub mod config;
pub mod data;
pub mod indicators;
pub mod instances;
pub mod model;
pub mod report;
pub mod trainer;

#[cfg(test)]
pub mod tests;

pub use app::run;

