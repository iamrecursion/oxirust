//! Tests for https://github.com/tafia/quick-xml/issues/474.
//!
//! That tests ensures that comments and processed instructions is ignored
//! and can split one logical string in pieces.

use super::*;

#[test]
fn text() {
    let mut de = make_de("text");
    assert_eq!(
        de.next().expect("should be Ok"),
        DeEvent::Text("text".into())
    );
}

#[test]
fn cdata() {
    let mut de = make_de("<![CDATA[cdata]]>");
    assert_eq!(
        de.next().expect("should be Ok"),
        DeEvent::Text("cdata".into())
    );
}

#[test]
fn text_and_cdata() {
    let mut de = make_de("text and <![CDATA[cdata]]>");
    assert_eq!(
        de.next().expect("should be Ok"),
        DeEvent::Text("text and cdata".into())
    );
}

#[test]
fn text_and_empty_cdata() {
    let mut de = make_de("text and <![CDATA[]]>");
    assert_eq!(
        de.next().expect("should be Ok"),
        DeEvent::Text("text and ".into())
    );
}

#[test]
fn cdata_and_text() {
    let mut de = make_de("<![CDATA[cdata]]> and text");
    assert_eq!(
        de.next().expect("should be Ok"),
        DeEvent::Text("cdata and text".into())
    );
}

#[test]
fn empty_cdata_and_text() {
    let mut de = make_de("<![CDATA[]]> and text");
    assert_eq!(
        de.next().expect("should be Ok"),
        DeEvent::Text(" and text".into())
    );
}

#[test]
fn cdata_and_cdata() {
    let mut de = make_de(
        "\
                    <![CDATA[cdata]]]]>\
                    <![CDATA[>cdata]]>\
                ",
    );
    assert_eq!(
        de.next().expect("should be Ok"),
        DeEvent::Text("cdata]]>cdata".into())
    );
}

mod comment_between {
    use super::*;

    #[test]
    fn text() {
        let mut de = make_de(
            "\
                        text \
                        <!--comment 1--><!--comment 2--> \
                        text\
                    ",
        );
        assert_eq!(
            de.next().expect("should be Ok"),
            DeEvent::Text("text  text".into())
        );
    }

    #[test]
    fn cdata() {
        let mut de = make_de(
            "\
                        <![CDATA[cdata]]]]>\
                        <!--comment 1--><!--comment 2-->\
                        <![CDATA[>cdata]]>\
                    ",
        );
        assert_eq!(
            de.next().expect("should be Ok"),
            DeEvent::Text("cdata]]>cdata".into())
        );
    }

    #[test]
    fn text_and_cdata() {
        let mut de = make_de(
            "\
                        text \
                        <!--comment 1--><!--comment 2-->\
                        <![CDATA[ cdata]]>\
                    ",
        );
        assert_eq!(
            de.next().expect("should be Ok"),
            DeEvent::Text("text  cdata".into())
        );
    }

    #[test]
    fn text_and_empty_cdata() {
        let mut de = make_de(
            "\
                        text \
                        <!--comment 1--><!--comment 2-->\
                        <![CDATA[]]>\
                    ",
        );
        assert_eq!(
            de.next().expect("should be Ok"),
            DeEvent::Text("text ".into())
        );
    }

    #[test]
    fn cdata_and_text() {
        let mut de = make_de(
            "\
                        <![CDATA[cdata ]]>\
                        <!--comment 1--><!--comment 2--> \
                        text \
                    ",
        );
        assert_eq!(
            de.next().expect("should be Ok"),
            DeEvent::Text("cdata  text ".into())
        );
    }

    #[test]
    fn empty_cdata_and_text() {
        let mut de = make_de(
            "\
                        <![CDATA[]]>\
                        <!--comment 1--><!--comment 2--> \
                        text \
                    ",
        );
        assert_eq!(
            de.next().expect("should be Ok"),
            DeEvent::Text(" text ".into())
        );
    }

    #[test]
    fn cdata_and_cdata() {
        let mut de = make_de(
            "\
                        <![CDATA[cdata]]]>\
                        <!--comment 1--><!--comment 2-->\
                        <![CDATA[]>cdata]]>\
                    ",
        );
        assert_eq!(
            de.next().expect("should be Ok"),
            DeEvent::Text("cdata]]>cdata".into())
        );
    }
}

mod pi_between {
    use super::*;

    #[test]
    fn text() {
        let mut de = make_de(
            "\
                        text \
                        <?pi 1?><?pi 2?> \
                        text\
                    ",
        );
        assert_eq!(
            de.next().expect("should be Ok"),
            DeEvent::Text("text  text".into())
        );
    }

    #[test]
    fn cdata() {
        let mut de = make_de(
            "\
                        <![CDATA[cdata]]]]>\
                        <?pi 1?><?pi 2?>\
                        <![CDATA[>cdata]]>\
                    ",
        );
        assert_eq!(
            de.next().expect("should be Ok"),
            DeEvent::Text("cdata]]>cdata".into())
        );
    }

    #[test]
    fn text_and_cdata() {
        let mut de = make_de(
            "\
                        text \
                        <?pi 1?><?pi 2?>\
                        <![CDATA[ cdata]]>\
                    ",
        );
        assert_eq!(
            de.next().expect("should be Ok"),
            DeEvent::Text("text  cdata".into())
        );
    }

    #[test]
    fn text_and_empty_cdata() {
        let mut de = make_de(
            "\
                        text \
                        <?pi 1?><?pi 2?>\
                        <![CDATA[]]>\
                    ",
        );
        assert_eq!(
            de.next().expect("should be Ok"),
            DeEvent::Text("text ".into())
        );
    }

    #[test]
    fn cdata_and_text() {
        let mut de = make_de(
            "\
                        <![CDATA[cdata ]]>\
                        <?pi 1?><?pi 2?> \
                        text \
                    ",
        );
        assert_eq!(
            de.next().expect("should be Ok"),
            DeEvent::Text("cdata  text ".into())
        );
    }

    #[test]
    fn empty_cdata_and_text() {
        let mut de = make_de(
            "\
                        <![CDATA[]]>\
                        <?pi 1?><?pi 2?> \
                        text \
                    ",
        );
        assert_eq!(
            de.next().expect("should be Ok"),
            DeEvent::Text(" text ".into())
        );
    }

    #[test]
    fn cdata_and_cdata() {
        let mut de = make_de(
            "\
                        <![CDATA[cdata]]]>\
                        <?pi 1?><?pi 2?>\
                        <![CDATA[]>cdata]]>\
                    ",
        );
        assert_eq!(
            de.next().expect("should be Ok"),
            DeEvent::Text("cdata]]>cdata".into())
        );
    }
}
