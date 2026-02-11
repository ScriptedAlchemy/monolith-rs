use monolith_training::native_training::learning_rate_functions::{
    LearningRateFunction, PolynomialDecay,
};

#[test]
fn polynomial_decay_basic_matches_python_test() {
    // Mirrors monolith/native_training/learning_rate_functions_test.py::test_basic
    let lr_fn = PolynomialDecay::new(0.01, 10).with_end_learning_rate(0.11);

    // Python increments global_step by 1 before first call.
    let lr1 = lr_fn.value(1);
    assert!((lr1 - 0.02).abs() < 1e-6, "got {lr1}");

    let lr2 = lr_fn.value(2);
    assert!((lr2 - 0.03).abs() < 1e-6, "got {lr2}");

    let lr_fn2 = PolynomialDecay::new(0.01, 10).with_end_learning_rate(0.11);
    assert_eq!(lr_fn.to_string(), lr_fn2.to_string());
}

#[test]
fn dense_optimizer_adagrad_update_matches_python_test() {
    // Mirrors monolith/native_training/learning_rate_functions_test.py::test_dense_optimizer
    // In the Python test the global step is never incremented, so the schedule always returns
    // `initial_learning_rate` and Adagrad uses a constant LR=3.0 for all 3 updates.
    let lr_fn = PolynomialDecay::new(3.0, 10).with_end_learning_rate(11.0);

    let mut var0 = vec![1.0_f32, 2.0_f32];
    let mut var1 = vec![3.0_f32, 4.0_f32];
    let grads0 = vec![0.1_f32, 0.1_f32];
    let grads1 = vec![0.01_f32, 0.01_f32];

    let lr = lr_fn.value(0);
    let mut acc0 = vec![0.1_f32; var0.len()];
    let mut acc1 = vec![0.1_f32; var1.len()];

    for _ in 0..3 {
        for i in 0..var0.len() {
            acc0[i] += grads0[i] * grads0[i];
            var0[i] -= lr * grads0[i] / acc0[i].sqrt();
        }
        for i in 0..var1.len() {
            acc1[i] += grads1[i] * grads1[i];
            var1[i] -= lr * grads1[i] / acc1[i].sqrt();
        }
    }

    assert!((var0[0] - (-1.6026099)).abs() < 1e-5, "{var0:?}");
    assert!((var0[1] - (-0.6026099)).abs() < 1e-5, "{var0:?}");
    assert!((var1[0] - 2.7156792).abs() < 1e-5, "{var1:?}");
    assert!((var1[1] - 3.7156792).abs() < 1e-5, "{var1:?}");
}
