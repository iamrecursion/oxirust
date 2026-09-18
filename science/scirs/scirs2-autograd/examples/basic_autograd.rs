use ag::tensor_ops as T;
use scirs2_autograd as ag;

#[allow(dead_code)]
fn main() {
    // Example to compute partial derivatives of z = 2x^2 + 3y + 1
    ag::run(|ctx: &mut ag::Context<_>| {
        // Define placeholders
        let x = ctx.placeholder("x", &[]);
        let y = ctx.placeholder("y", &[]);

        // Define the computation
        let z = 2.0 * x * x + 3.0 * y + 1.0;

        // Calculate partial derivative dz/dy
        let gy = &T::grad(&[z], &[y])[0];
        println!("dz/dy: {:?}", gy.eval(ctx)); // Should be 3.0

        // Calculate partial derivative dz/dx with x = 2.0
        let gx = &T::grad(&[z], &[x])[0];
        let feed = ag::ndarray::arr0(2.0);
        println!(
            "dz/dx at x=2.0: {:?}",
            ctx.evaluator()
                .push(gx)
                .feed(x, feed.into_dyn().view())
                .run()[0]
        ); // Should be 8.0

        // Calculate second derivative d²z/dx²
        let ggx = &T::grad(&[gx], &[x])[0];
        println!("d²z/dx²: {:?}", ggx.eval(ctx)); // Should be 4.0
    });
}
