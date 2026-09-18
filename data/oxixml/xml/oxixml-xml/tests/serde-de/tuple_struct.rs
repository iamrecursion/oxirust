use super::*;

#[derive(Debug, Deserialize, PartialEq)]
struct Tuple(f32, String);

#[test]
fn simple() {
    let data: Tuple = from_str("<root>42</root><root>answer</root>").expect("should be Ok");
    assert_eq!(data, Tuple(42.0, "answer".into()));
}

#[test]
fn excess_attribute() {
    let data: Tuple = from_str(
        // Comment for prevent unnecessary formatting - we use the same style in all tests
        r#"<root excess="attribute">42</root><root>answer</root>"#,
    )
    .expect("should be Ok");
    assert_eq!(data, Tuple(42.0, "answer".into()));
}
