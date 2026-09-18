//! Smoke tests for newly-wired orphan modules in oximedia-image-transform.

#[test]
fn test_animation_frame_accessible() {
    let _ = std::any::type_name::<oximedia_image_transform::animation::AnimationFrame>();
}
#[test]
fn test_batch_transform_variant() {
    let _ = std::any::type_name::<oximedia_image_transform::batch_transform::TransformVariant>();
}
#[test]
fn test_compose_matrix_identity() {
    use oximedia_image_transform::compose::TransformMatrix;
    let identity = TransformMatrix([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    assert!((identity.0[0] - 1.0).abs() < 1e-10);
}
#[test]
fn test_inverse_transform_identity() {
    use oximedia_image_transform::compose::TransformMatrix;
    use oximedia_image_transform::inverse::invert_transform;
    let identity = TransformMatrix([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    assert!(invert_transform(&identity).is_some());
}
#[test]
fn test_origin_fetch_response() {
    let _ = std::any::type_name::<oximedia_image_transform::origin_fetch::FetchResponse>();
}
