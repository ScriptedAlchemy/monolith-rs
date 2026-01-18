// Thin binary entrypoint to keep the example maintainable.
// The full implementation lives in `src/bin/stock_prediction/mod.rs`.

#[path = "stock_prediction/mod.rs"]
mod stock_prediction;

fn main() {
    stock_prediction::run();
}
