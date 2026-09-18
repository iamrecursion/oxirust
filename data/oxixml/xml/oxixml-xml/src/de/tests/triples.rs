//! Tests for https://github.com/tafia/quick-xml/issues/474.
//!
//! This tests ensures that any combination of payload data is processed
//! as expected.

use super::*;

mod start {
    use super::*;

    /// <tag1><tag2>...
    // The same name is intentional
    #[allow(clippy::module_inception)]
    mod start {
        use super::*;

        #[test]
        fn start() {
            let mut de = make_de("<tag1><tag2><tag3>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag1"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag2"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag3"))
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        /// Not matching end tag will result to error
        #[test]
        fn end() {
            let mut de = make_de("<tag1><tag2></tag2>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag1"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag2"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::End(BytesEnd::new("tag2"))
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn text() {
            let mut de = make_de("<tag1><tag2> text ");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag1"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag2"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" text ".into())
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn cdata() {
            let mut de = make_de("<tag1><tag2><![CDATA[ cdata ]]>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag1"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag2"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" cdata ".into())
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn eof() {
            let mut de = make_de("<tag1><tag2>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag1"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag2"))
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }
    }

    /// <tag></tag>...
    mod end {
        use super::*;

        #[test]
        fn start() {
            let mut de = make_de("<tag></tag><tag2>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::End(BytesEnd::new("tag"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag2"))
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn end() {
            let mut de = make_de("<tag></tag></tag2>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::End(BytesEnd::new("tag"))
            );
            match de.next() {
                Err(DeError::InvalidXml(Error::IllFormed(cause))) => {
                    assert_eq!(cause, IllFormedError::UnmatchedEndTag("tag2".into()));
                }
                x => panic!(
                    "Expected `Err(InvalidXml(IllFormed(_)))`, but got `{:?}`",
                    x
                ),
            }
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn text() {
            let mut de = make_de("<tag></tag> text ");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::End(BytesEnd::new("tag"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" text ".into())
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn cdata() {
            let mut de = make_de("<tag></tag><![CDATA[ cdata ]]>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::End(BytesEnd::new("tag"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" cdata ".into())
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn eof() {
            let mut de = make_de("<tag></tag>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::End(BytesEnd::new("tag"))
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }
    }

    /// <tag> text ...
    mod text {
        use super::*;

        #[test]
        fn start() {
            let mut de = make_de("<tag> text <tag2>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" text ".into())
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag2"))
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn end() {
            let mut de = make_de("<tag> text </tag>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" text ".into())
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::End(BytesEnd::new("tag"))
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        // start::text::text has no difference from start::text

        #[test]
        fn cdata() {
            let mut de = make_de("<tag> text <![CDATA[ cdata ]]>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" text  cdata ".into())
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn eof() {
            let mut de = make_de("<tag> text ");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" text ".into())
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }
    }

    /// <tag><![CDATA[ cdata ]]>...
    mod cdata {
        use super::*;

        #[test]
        fn start() {
            let mut de = make_de("<tag><![CDATA[ cdata ]]><tag2>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" cdata ".into())
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag2"))
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn end() {
            let mut de = make_de("<tag><![CDATA[ cdata ]]></tag>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" cdata ".into())
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::End(BytesEnd::new("tag"))
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn text() {
            let mut de = make_de("<tag><![CDATA[ cdata ]]> text ");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" cdata  text ".into())
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn cdata() {
            let mut de = make_de("<tag><![CDATA[ cdata ]]><![CDATA[ cdata2 ]]>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" cdata  cdata2 ".into())
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn eof() {
            let mut de = make_de("<tag><![CDATA[ cdata ]]>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" cdata ".into())
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }
    }
}

/// Start from End event will always generate an error
#[test]
fn end() {
    let mut de = make_de("</tag>");
    match de.next() {
        Err(DeError::InvalidXml(Error::IllFormed(cause))) => {
            assert_eq!(cause, IllFormedError::UnmatchedEndTag("tag".into()));
        }
        x => panic!(
            "Expected `Err(InvalidXml(IllFormed(_)))`, but got `{:?}`",
            x
        ),
    }
    assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
}

mod text {
    use super::*;

    mod start {
        use super::*;

        #[test]
        fn start() {
            let mut de = make_de(" text <tag1><tag2>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" text ".into())
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag1"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag2"))
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        /// Not matching end tag will result in error
        #[test]
        fn end() {
            let mut de = make_de(" text <tag></tag>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" text ".into())
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::End(BytesEnd::new("tag"))
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn text() {
            let mut de = make_de(" text <tag> text2 ");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" text ".into())
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" text2 ".into())
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn cdata() {
            let mut de = make_de(" text <tag><![CDATA[ cdata ]]>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" text ".into())
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" cdata ".into())
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn eof() {
            let mut de = make_de(" text <tag>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" text ".into())
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag"))
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }
    }

    /// End event without corresponding start event will always generate an error
    #[test]
    fn end() {
        let mut de = make_de(" text </tag>");
        assert_eq!(
            de.next().expect("should be Ok"),
            DeEvent::Text(" text ".into())
        );
        match de.next() {
            Err(DeError::InvalidXml(Error::IllFormed(cause))) => {
                assert_eq!(cause, IllFormedError::UnmatchedEndTag("tag".into()));
            }
            x => panic!(
                "Expected `Err(InvalidXml(IllFormed(_)))`, but got `{:?}`",
                x
            ),
        }
        assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
    }

    // text::text::something is equivalent to text::something

    mod cdata {
        use super::*;

        #[test]
        fn start() {
            let mut de = make_de(" text <![CDATA[ cdata ]]><tag>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" text  cdata ".into())
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag"))
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn end() {
            let mut de = make_de(" text <![CDATA[ cdata ]]></tag>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" text  cdata ".into())
            );
            match de.next() {
                Err(DeError::InvalidXml(Error::IllFormed(cause))) => {
                    assert_eq!(cause, IllFormedError::UnmatchedEndTag("tag".into()));
                }
                x => panic!(
                    "Expected `Err(InvalidXml(IllFormed(_)))`, but got `{:?}`",
                    x
                ),
            }
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn text() {
            let mut de = make_de(" text <![CDATA[ cdata ]]> text2 ");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" text  cdata  text2 ".into())
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn cdata() {
            let mut de = make_de(" text <![CDATA[ cdata ]]><![CDATA[ cdata2 ]]>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" text  cdata  cdata2 ".into())
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn eof() {
            let mut de = make_de(" text <![CDATA[ cdata ]]>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" text  cdata ".into())
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }
    }
}

mod cdata {
    use super::*;

    mod start {
        use super::*;

        #[test]
        fn start() {
            let mut de = make_de("<![CDATA[ cdata ]]><tag1><tag2>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" cdata ".into())
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag1"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag2"))
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        /// Not matching end tag will result in error
        #[test]
        fn end() {
            let mut de = make_de("<![CDATA[ cdata ]]><tag></tag>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" cdata ".into())
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::End(BytesEnd::new("tag"))
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn text() {
            let mut de = make_de("<![CDATA[ cdata ]]><tag> text ");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" cdata ".into())
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" text ".into())
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn cdata() {
            let mut de = make_de("<![CDATA[ cdata ]]><tag><![CDATA[ cdata2 ]]>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" cdata ".into())
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag"))
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" cdata2 ".into())
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn eof() {
            let mut de = make_de("<![CDATA[ cdata ]]><tag>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" cdata ".into())
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag"))
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }
    }

    /// End event without corresponding start event will always generate an error
    #[test]
    fn end() {
        let mut de = make_de("<![CDATA[ cdata ]]></tag>");
        assert_eq!(
            de.next().expect("should be Ok"),
            DeEvent::Text(" cdata ".into())
        );
        match de.next() {
            Err(DeError::InvalidXml(Error::IllFormed(cause))) => {
                assert_eq!(cause, IllFormedError::UnmatchedEndTag("tag".into()));
            }
            x => panic!(
                "Expected `Err(InvalidXml(IllFormed(_)))`, but got `{:?}`",
                x
            ),
        }
        assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
    }

    mod text {
        use super::*;

        #[test]
        fn start() {
            let mut de = make_de("<![CDATA[ cdata ]]> text <tag>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" cdata  text ".into())
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag"))
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn end() {
            let mut de = make_de("<![CDATA[ cdata ]]> text </tag>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" cdata  text ".into())
            );
            match de.next() {
                Err(DeError::InvalidXml(Error::IllFormed(cause))) => {
                    assert_eq!(cause, IllFormedError::UnmatchedEndTag("tag".into()));
                }
                x => panic!(
                    "Expected `Err(InvalidXml(IllFormed(_)))`, but got `{:?}`",
                    x
                ),
            }
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        // cdata::text::text is equivalent to cdata::text

        #[test]
        fn cdata() {
            let mut de = make_de("<![CDATA[ cdata ]]> text <![CDATA[ cdata2 ]]>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" cdata  text  cdata2 ".into())
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn eof() {
            let mut de = make_de("<![CDATA[ cdata ]]> text ");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" cdata  text ".into())
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }
    }

    // The same name is intentional
    #[allow(clippy::module_inception)]
    mod cdata {
        use super::*;

        #[test]
        fn start() {
            let mut de = make_de("<![CDATA[ cdata ]]><![CDATA[ cdata2 ]]><tag>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" cdata  cdata2 ".into())
            );
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Start(BytesStart::new("tag"))
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn end() {
            let mut de = make_de("<![CDATA[ cdata ]]><![CDATA[ cdata2 ]]></tag>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" cdata  cdata2 ".into())
            );
            match de.next() {
                Err(DeError::InvalidXml(Error::IllFormed(cause))) => {
                    assert_eq!(cause, IllFormedError::UnmatchedEndTag("tag".into()));
                }
                x => panic!(
                    "Expected `Err(InvalidXml(IllFormed(_)))`, but got `{:?}`",
                    x
                ),
            }
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn text() {
            let mut de = make_de("<![CDATA[ cdata ]]><![CDATA[ cdata2 ]]> text ");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" cdata  cdata2  text ".into())
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn cdata() {
            let mut de = make_de("<![CDATA[ cdata ]]><![CDATA[ cdata2 ]]><![CDATA[ cdata3 ]]>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" cdata  cdata2  cdata3 ".into())
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }

        #[test]
        fn eof() {
            let mut de = make_de("<![CDATA[ cdata ]]><![CDATA[ cdata2 ]]>");
            assert_eq!(
                de.next().expect("should be Ok"),
                DeEvent::Text(" cdata  cdata2 ".into())
            );
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
            assert_eq!(de.next().expect("should be Ok"), DeEvent::Eof);
        }
    }
}
