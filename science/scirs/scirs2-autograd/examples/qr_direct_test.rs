use ag::tensor_ops::*;
use scirs2_autograd as ag;
use scirs2_core::ndarray::array;

#[allow(dead_code)]
fn main() {
    ag::run(|ctx| {
        let a_data = array![[3.0f32, 4.0], [0.0, 1.0]];
        println!("Input matrix A:\n{:?}", a_data);

        // Create tensor directly instead of using placeholder
        let a = convert_to_tensor(a_data.clone(), ctx);

        // Test QR decomposition
        let (q, r) = qr(a);

        // Evaluate
        let q_result = q.eval(ctx).expect("Operation failed");
        let r_result = r.eval(ctx).expect("Operation failed");

        println!("Q result:\n{:?}", q_result);
        println!("R result:\n{:?}", r_result);

        // Verify Q is orthogonal
        let q_2d = q_result
            .clone()
            .into_dimensionality::<ag::ndarray::Ix2>()
            .expect("Operation failed");
        let r_2d = r_result
            .into_dimensionality::<ag::ndarray::Ix2>()
            .expect("Operation failed");

        let qt_q = q_2d.t().dot(&q_2d);
        println!("Q^T * Q:\n{:?}", qt_q);

        // Verify A = Q * R
        let reconstructed = q_2d.dot(&r_2d);
        println!("Q * R:\n{:?}", reconstructed);

        println!("Original A:\n{:?}", a_data);
    });
}
