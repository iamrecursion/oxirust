//! Internal reader test harness.
//!
//! Ported from quick-xml 0.41 `src/reader/mod.rs` (`mod test`, the `check!`
//! macro). The macro is instantiated once for the borrowed `&[u8]` source,
//! once for the buffered `BufRead` source, and — with the `async-tokio`
//! feature — once for the asynchronous `AsyncBufRead` source, so that all
//! `XmlSource` implementations are exercised by exactly the same
//! expectations. The `pretty_assertions`/`arbitrary` dependencies of the
//! upstream version have been dropped, and `.unwrap()` replaced by
//! `.expect(...)` per the COOLJAPAN no-unwrap policy.

/// Checks the internal implementation of the various reader methods
macro_rules! check {
    (
        #[$test:meta]
        $read_event:ident,
        // constructor of the XML source on which internal functions will be called
        $source:path,
        $skip:literal,
        // constructor of the buffer to which read data will stored
        $buf:expr
        $(, $async:ident, $await:ident)?
    ) => {
        mod read_bang_element {
            use super::*;
            use crate::errors::{Error, SyntaxError};
            use crate::reader::{BangType, DtdParser};
            use crate::utils::Bytes;

            /// Checks that reading CDATA content works correctly
            mod cdata {
                use super::*;

                /// Checks that if CDATA startup sequence was matched, but an end
                /// sequence is not found, parsing ends with an error
                #[$test]
                $($async)? fn not_closed() {
                    let buf = $buf;
                    let mut position = 0;
                    let mut input = &b"<![CDATA[other content"[$skip..];
                    //                 ^= 0                  ^= 22

                    match $source(&mut input).read_bang_element(buf, &mut position) $(.$await)? {
                        Err(Error::Syntax(cause)) => assert_eq!(cause, SyntaxError::UnclosedCData),
                        x => panic!("Expected `Err(Syntax(_))`, but got `{:?}`", x),
                    }
                    assert_eq!(position, 22);
                }

                /// Checks that CDATA element without content inside parsed successfully
                #[$test]
                $($async)? fn empty() {
                    let buf = $buf;
                    let mut position = 0;
                    let mut input = &b"<![CDATA[]]>other content"[$skip..];
                    //                ^= 0        ^= 12

                    let (ty, bytes) = $source(&mut input)
                        .read_bang_element(buf, &mut position)
                        $(.$await)?
                        .expect("valid CDATA");
                    assert_eq!(
                        (ty, Bytes(bytes)),
                        (BangType::CData, Bytes(b"<![CDATA[]]>"))
                    );
                    assert_eq!(position, 12);
                }

                /// Checks that CDATA element with content parsed successfully.
                /// Additionally checks that sequences inside CDATA that may look like
                /// a CDATA end sequence do not interrupt CDATA parsing
                #[$test]
                $($async)? fn with_content() {
                    let buf = $buf;
                    let mut position = 0;
                    let mut input = &b"<![CDATA[cdata]] ]>content]]>other content]]>"[$skip..];
                    //                 ^= 0                         ^= 29

                    let (ty, bytes) = $source(&mut input)
                        .read_bang_element(buf, &mut position)
                        $(.$await)?
                        .expect("valid CDATA");
                    assert_eq!(
                        (ty, Bytes(bytes)),
                        (BangType::CData, Bytes(b"<![CDATA[cdata]] ]>content]]>"))
                    );
                    assert_eq!(position, 29);
                }
            }

            /// Checks that reading XML comments works correctly.
            mod comment {
                use super::*;

                #[$test]
                $($async)? fn not_properly_end() {
                    let buf = $buf;
                    let mut position = 0;
                    let mut input = &b"<!->other content"[$skip..];
                    //                 ^= 0             ^= 17

                    match $source(&mut input).read_bang_element(buf, &mut position) $(.$await)? {
                        Err(Error::Syntax(cause)) => {
                            assert_eq!(cause, SyntaxError::UnclosedComment)
                        }
                        x => panic!("Expected `Err(Syntax(_))`, but got `{:?}`", x),
                    }
                    assert_eq!(position, 17);
                }

                #[$test]
                $($async)? fn not_closed1() {
                    let buf = $buf;
                    let mut position = 0;
                    let mut input = &b"<!--other content"[$skip..];
                    //                 ^= 0             ^= 17

                    match $source(&mut input).read_bang_element(buf, &mut position) $(.$await)? {
                        Err(Error::Syntax(cause)) => {
                            assert_eq!(cause, SyntaxError::UnclosedComment)
                        }
                        x => panic!("Expected `Err(Syntax(_))`, but got `{:?}`", x),
                    }
                    assert_eq!(position, 17);
                }

                #[$test]
                $($async)? fn not_closed2() {
                    let buf = $buf;
                    let mut position = 0;
                    let mut input = &b"<!-->other content"[$skip..];
                    //                 ^= 0              ^= 18

                    match $source(&mut input).read_bang_element(buf, &mut position) $(.$await)? {
                        Err(Error::Syntax(cause)) => {
                            assert_eq!(cause, SyntaxError::UnclosedComment)
                        }
                        x => panic!("Expected `Err(Syntax(_))`, but got `{:?}`", x),
                    }
                    assert_eq!(position, 18);
                }

                #[$test]
                $($async)? fn not_closed3() {
                    let buf = $buf;
                    let mut position = 0;
                    let mut input = &b"<!--->other content"[$skip..];
                    //                 ^= 0               ^= 19

                    match $source(&mut input).read_bang_element(buf, &mut position) $(.$await)? {
                        Err(Error::Syntax(cause)) => {
                            assert_eq!(cause, SyntaxError::UnclosedComment)
                        }
                        x => panic!("Expected `Err(Syntax(_))`, but got `{:?}`", x),
                    }
                    assert_eq!(position, 19);
                }

                #[$test]
                $($async)? fn empty() {
                    let buf = $buf;
                    let mut position = 0;
                    let mut input = &b"<!---->other content"[$skip..];
                    //                 ^= 0   ^= 7

                    let (ty, bytes) = $source(&mut input)
                        .read_bang_element(buf, &mut position)
                        $(.$await)?
                        .expect("valid comment");
                    assert_eq!((ty, Bytes(bytes)), (BangType::Comment, Bytes(b"<!---->")));
                    assert_eq!(position, 7);
                }

                #[$test]
                $($async)? fn with_content() {
                    let buf = $buf;
                    let mut position = 0;
                    let mut input = &b"<!--->comment<--->other content"[$skip..];
                    //                 ^= 0              ^= 18

                    let (ty, bytes) = $source(&mut input)
                        .read_bang_element(buf, &mut position)
                        $(.$await)?
                        .expect("valid comment");
                    assert_eq!(
                        (ty, Bytes(bytes)),
                        (BangType::Comment, Bytes(b"<!--->comment<--->"))
                    );
                    assert_eq!(position, 18);
                }
            }

            /// Checks that reading DOCTYPE definition works correctly
            mod doctype {
                use super::*;

                mod uppercase {
                    use super::*;

                    #[$test]
                    $($async)? fn not_properly_start() {
                        let buf = $buf;
                        let mut position = 0;
                        let mut input = &b"<!D other content"[$skip..];
                        //                 ^= 0             ^= 17

                        match $source(&mut input).read_bang_element(buf, &mut position) $(.$await)? {
                            Err(Error::Syntax(cause)) => {
                                assert_eq!(cause, SyntaxError::UnclosedDoctype)
                            }
                            x => panic!("Expected `Err(Syntax(_))`, but got `{:?}`", x),
                        }
                        assert_eq!(position, 17);
                    }

                    #[$test]
                    $($async)? fn without_space() {
                        let buf = $buf;
                        let mut position = 0;
                        let mut input = &b"<!DOCTYPEother content"[$skip..];
                        //                 ^= 0                  ^= 22

                        match $source(&mut input).read_bang_element(buf, &mut position) $(.$await)? {
                            Err(Error::Syntax(cause)) => {
                                assert_eq!(cause, SyntaxError::UnclosedDoctype)
                            }
                            x => panic!("Expected `Err(Syntax(_))`, but got `{:?}`", x),
                        }
                        assert_eq!(position, 22);
                    }

                    #[$test]
                    $($async)? fn empty() {
                        let buf = $buf;
                        let mut position = 0;
                        let mut input = &b"<!DOCTYPE>other content"[$skip..];
                        //                 ^= 0      ^= 10

                        let (ty, bytes) = $source(&mut input)
                            .read_bang_element(buf, &mut position)
                            $(.$await)?
                            .expect("valid doctype");
                        assert_eq!(
                            (ty, Bytes(bytes)),
                            (BangType::DocType(DtdParser::Finished), Bytes(b"<!DOCTYPE>"))
                        );
                        assert_eq!(position, 10);
                    }

                    #[$test]
                    $($async)? fn not_closed() {
                        let buf = $buf;
                        let mut position = 0;
                        let mut input = &b"<!DOCTYPE other content"[$skip..];
                        //                 ^= 0                   ^23

                        match $source(&mut input).read_bang_element(buf, &mut position) $(.$await)? {
                            Err(Error::Syntax(cause)) => {
                                assert_eq!(cause, SyntaxError::UnclosedDoctype)
                            }
                            x => panic!("Expected `Err(Syntax(_))`, but got `{:?}`", x),
                        }
                        assert_eq!(position, 23);
                    }
                }

                mod lowercase {
                    use super::*;

                    #[$test]
                    $($async)? fn not_properly_start() {
                        let buf = $buf;
                        let mut position = 0;
                        let mut input = &b"<!d other content"[$skip..];
                        //                 ^= 0             ^= 17

                        match $source(&mut input).read_bang_element(buf, &mut position) $(.$await)? {
                            Err(Error::Syntax(cause)) => {
                                assert_eq!(cause, SyntaxError::UnclosedDoctype)
                            }
                            x => panic!("Expected `Err(Syntax(_))`, but got `{:?}`", x),
                        }
                        assert_eq!(position, 17);
                    }

                    #[$test]
                    $($async)? fn without_space() {
                        let buf = $buf;
                        let mut position = 0;
                        let mut input = &b"<!doctypeother content"[$skip..];
                        //                 ^= 0                  ^= 22

                        match $source(&mut input).read_bang_element(buf, &mut position) $(.$await)? {
                            Err(Error::Syntax(cause)) => {
                                assert_eq!(cause, SyntaxError::UnclosedDoctype)
                            }
                            x => panic!("Expected `Err(Syntax(_))`, but got `{:?}`", x),
                        }
                        assert_eq!(position, 22);
                    }

                    #[$test]
                    $($async)? fn empty() {
                        let buf = $buf;
                        let mut position = 0;
                        let mut input = &b"<!doctype>other content"[$skip..];
                        //                 ^= 0      ^= 10

                        let (ty, bytes) = $source(&mut input)
                            .read_bang_element(buf, &mut position)
                            $(.$await)?
                            .expect("valid doctype");
                        assert_eq!(
                            (ty, Bytes(bytes)),
                            (BangType::DocType(DtdParser::Finished), Bytes(b"<!doctype>"))
                        );
                        assert_eq!(position, 10);
                    }

                    #[$test]
                    $($async)? fn not_closed() {
                        let buf = $buf;
                        let mut position = 0;
                        let mut input = &b"<!doctype other content"[$skip..];
                        //                 ^= 0                   ^= 23

                        match $source(&mut input).read_bang_element(buf, &mut position) $(.$await)? {
                            Err(Error::Syntax(cause)) => {
                                assert_eq!(cause, SyntaxError::UnclosedDoctype)
                            }
                            x => panic!("Expected `Err(Syntax(_))`, but got `{:?}`", x),
                        }
                        assert_eq!(position, 23);
                    }
                }
            }
        }

        mod read_text {
            use super::*;
            use crate::reader::ReadTextResult;
            use crate::utils::Bytes;

            #[$test]
            $($async)? fn empty() {
                let buf = $buf;
                let mut position = 1;
                let mut input = b"".as_ref();
                //                ^= 1

                match $source(&mut input).read_text(buf, &mut position, false) $(.$await)? {
                    ReadTextResult::UpToEof(bytes) => assert_eq!(Bytes(bytes), Bytes(b"")),
                    x => panic!("Expected `UpToEof(_)`, but got `{:?}`", x),
                }
                assert_eq!(position, 1);
            }

            #[$test]
            $($async)? fn markup() {
                let buf = $buf;
                let mut position = 1;
                let mut input = b"<".as_ref();
                //                 ^= 1

                match $source(&mut input).read_text(buf, &mut position, false) $(.$await)? {
                    ReadTextResult::Markup(b) => assert_eq!(b, $buf),
                    x => panic!("Expected `Markup(_)`, but got `{:?}`", x),
                }
                assert_eq!(position, 1);
            }

            #[$test]
            $($async)? fn ref_() {
                let buf = $buf;
                let mut position = 1;
                let mut input = b"&".as_ref();
                //                ^= 1

                match $source(&mut input).read_text(buf, &mut position, false) $(.$await)? {
                    ReadTextResult::Ref(b) => assert_eq!(b, $buf),
                    x => panic!("Expected `Ref(_)`, but got `{:?}`", x),
                }
                assert_eq!(position, 1);
            }

            #[$test]
            $($async)? fn up_to_markup() {
                let buf = $buf;
                let mut position = 1;
                let mut input = b"a<".as_ref();
                //                  ^= 2

                match $source(&mut input).read_text(buf, &mut position, false) $(.$await)? {
                    ReadTextResult::UpToMarkup(bytes) => assert_eq!(Bytes(bytes), Bytes(b"a")),
                    x => panic!("Expected `UpToMarkup(_)`, but got `{:?}`", x),
                }
                assert_eq!(position, 2);
            }

            #[$test]
            $($async)? fn up_to_ref() {
                let buf = $buf;
                let mut position = 1;
                let mut input = b"a&".as_ref();
                //                 ^= 2

                match $source(&mut input).read_text(buf, &mut position, false) $(.$await)? {
                    ReadTextResult::UpToRef(bytes) => assert_eq!(Bytes(bytes), Bytes(b"a")),
                    x => panic!("Expected `UpToRef(_)`, but got `{:?}`", x),
                }
                assert_eq!(position, 2);
            }

            #[$test]
            $($async)? fn up_to_eof() {
                let buf = $buf;
                let mut position = 1;
                let mut input = b"a".as_ref();
                //                 ^= 2

                match $source(&mut input).read_text(buf, &mut position, false) $(.$await)? {
                    ReadTextResult::UpToEof(bytes) => assert_eq!(Bytes(bytes), Bytes(b"a")),
                    x => panic!("Expected `UpToEof(_)`, but got `{:?}`", x),
                }
                assert_eq!(position, 2);
            }
        }

        mod read_ref {
            use super::*;
            use crate::reader::ReadRefResult;
            use crate::utils::Bytes;

            #[$test]
            $($async)? fn up_to_eof() {
                let buf = $buf;
                let mut position = 1;
                let mut input = b"&".as_ref();
                //                 ^= 2

                match $source(&mut input).read_ref(buf, &mut position) $(.$await)? {
                    ReadRefResult::UpToEof(bytes) => assert_eq!(Bytes(bytes), Bytes(b"&")),
                    x => panic!("Expected `UpToEof(_)`, but got `{:?}`", x),
                }
                assert_eq!(position, 2);
            }

            #[$test]
            $($async)? fn up_to_ref() {
                let buf = $buf;
                let mut position = 1;
                let mut input = b"&&".as_ref();
                //                 ^= 2

                match $source(&mut input).read_ref(buf, &mut position) $(.$await)? {
                    ReadRefResult::UpToRef(bytes) => assert_eq!(Bytes(bytes), Bytes(b"&")),
                    x => panic!("Expected `UpToRef(_)`, but got `{:?}`", x),
                }
                assert_eq!(position, 2);
            }

            #[$test]
            $($async)? fn up_to_markup() {
                let buf = $buf;
                let mut position = 1;
                let mut input = b"&<".as_ref();
                //                 ^= 2

                match $source(&mut input).read_ref(buf, &mut position) $(.$await)? {
                    ReadRefResult::UpToMarkup(bytes) => assert_eq!(Bytes(bytes), Bytes(b"&")),
                    x => panic!("Expected `UpToMarkup(_)`, but got `{:?}`", x),
                }
                assert_eq!(position, 2);
            }

            #[$test]
            $($async)? fn empty_ref() {
                let buf = $buf;
                let mut position = 1;
                let mut input = b"&;".as_ref();
                //                  ^= 3

                match $source(&mut input).read_ref(buf, &mut position) $(.$await)? {
                    ReadRefResult::Ref(bytes) => assert_eq!(Bytes(bytes), Bytes(b"&;")),
                    x => panic!("Expected `Ref(_)`, but got `{:?}`", x),
                }
                assert_eq!(position, 3);
            }

            #[$test]
            $($async)? fn normal() {
                let buf = $buf;
                let mut position = 1;
                let mut input = b"&lt;".as_ref();
                //                    ^= 5

                match $source(&mut input).read_ref(buf, &mut position) $(.$await)? {
                    ReadRefResult::Ref(bytes) => assert_eq!(Bytes(bytes), Bytes(b"&lt;")),
                    x => panic!("Expected `Ref(_)`, but got `{:?}`", x),
                }
                assert_eq!(position, 5);
            }
        }

        mod read_element {
            use super::*;
            use crate::errors::{Error, SyntaxError};
            use crate::parser::ElementParser;
            use crate::utils::Bytes;

            /// Checks that nothing was read from empty buffer.
            /// `<` read in peek_one that is called before read_with, that is why it in the input buffer
            /// peek_one, however, does not increment position for simplicity of the code
            #[$test]
            $($async)? fn empty() {
                let buf = $buf;
                let mut position = 0;
                let mut input = &b"<"[$skip..];
                //                  ^= 1

                match $source(&mut input).read_with(ElementParser::default(), buf, &mut position) $(.$await)? {
                    Err(Error::Syntax(cause)) => assert_eq!(cause, SyntaxError::UnclosedTag),
                    x => panic!("Expected `Err(Syntax(_))`, but got `{:?}`", x),
                }
                assert_eq!(position, 1);
            }

            mod open {
                use super::*;

                #[$test]
                $($async)? fn empty_tag() {
                    let buf = $buf;
                    let mut position = 0;
                    let mut input = &b"<>"[$skip..];
                    //                   ^= 2

                    assert_eq!(
                        Bytes(
                            $source(&mut input)
                                .read_with(ElementParser::default(), buf, &mut position)
                                $(.$await)?
                                .expect("valid tag")
                        ),
                        Bytes(b"<>")
                    );
                    assert_eq!(position, 2);
                }

                #[$test]
                $($async)? fn normal() {
                    let buf = $buf;
                    let mut position = 0;
                    let mut input = &b"<tag>"[$skip..];
                    //                      ^= 5

                    assert_eq!(
                        Bytes(
                            $source(&mut input)
                                .read_with(ElementParser::default(), buf, &mut position)
                                $(.$await)?
                                .expect("valid tag")
                        ),
                        Bytes(b"<tag>")
                    );
                    assert_eq!(position, 5);
                }

                #[$test]
                $($async)? fn with_attributes() {
                    let buf = $buf;
                    let mut position = 0;
                    let mut input = &br#"<tag  attr-1=">"  attr2  =  '>'  3attr>"#[$skip..];
                    //                                                          ^= 39

                    assert_eq!(
                        Bytes(
                            $source(&mut input)
                                .read_with(ElementParser::default(), buf, &mut position)
                                $(.$await)?
                                .expect("valid tag")
                        ),
                        Bytes(br#"<tag  attr-1=">"  attr2  =  '>'  3attr>"#)
                    );
                    assert_eq!(position, 39);
                }
            }

            mod self_closed {
                use super::*;

                #[$test]
                $($async)? fn empty_tag() {
                    let buf = $buf;
                    let mut position = 0;
                    let mut input = &b"</>"[$skip..];
                    //                    ^= 3

                    assert_eq!(
                        Bytes(
                            $source(&mut input)
                                .read_with(ElementParser::default(), buf, &mut position)
                                $(.$await)?
                                .expect("valid tag")
                        ),
                        Bytes(b"</>")
                    );
                    assert_eq!(position, 3);
                }

                #[$test]
                $($async)? fn normal() {
                    let buf = $buf;
                    let mut position = 0;
                    let mut input = &b"<tag/>"[$skip..];
                    //                       ^= 6

                    assert_eq!(
                        Bytes(
                            $source(&mut input)
                                .read_with(ElementParser::default(), buf, &mut position)
                                $(.$await)?
                                .expect("valid tag")
                        ),
                        Bytes(b"<tag/>")
                    );
                    assert_eq!(position, 6);
                }

                #[$test]
                $($async)? fn with_attributes() {
                    let buf = $buf;
                    let mut position = 0;
                    let mut input = &br#"<tag  attr-1="/>"  attr2  =  '/>'  3attr/>"#[$skip..];
                    //                                                             ^= 42

                    assert_eq!(
                        Bytes(
                            $source(&mut input)
                                .read_with(ElementParser::default(), buf, &mut position)
                                $(.$await)?
                                .expect("valid tag")
                        ),
                        Bytes(br#"<tag  attr-1="/>"  attr2  =  '/>'  3attr/>"#)
                    );
                    assert_eq!(position, 42);
                }
            }

            mod close {
                use super::*;

                #[$test]
                $($async)? fn empty_tag() {
                    let buf = $buf;
                    let mut position = 0;
                    let mut input = &b"</ >"[$skip..];
                    //                     ^= 4

                    assert_eq!(
                        Bytes(
                            $source(&mut input)
                                .read_with(ElementParser::default(), buf, &mut position)
                                $(.$await)?
                                .expect("valid tag")
                        ),
                        Bytes(b"</ >")
                    );
                    assert_eq!(position, 4);
                }

                #[$test]
                $($async)? fn normal() {
                    let buf = $buf;
                    let mut position = 0;
                    let mut input = &b"</tag>"[$skip..];
                    //                       ^= 6

                    assert_eq!(
                        Bytes(
                            $source(&mut input)
                                .read_with(ElementParser::default(), buf, &mut position)
                                $(.$await)?
                                .expect("valid tag")
                        ),
                        Bytes(b"</tag>")
                    );
                    assert_eq!(position, 6);
                }

                #[$test]
                $($async)? fn with_attributes() {
                    let buf = $buf;
                    let mut position = 0;
                    let mut input = &br#"</tag  attr-1=">"  attr2  =  '>'  3attr>"#[$skip..];
                    //                                                           ^= 40

                    assert_eq!(
                        Bytes(
                            $source(&mut input)
                                .read_with(ElementParser::default(), buf, &mut position)
                                $(.$await)?
                                .expect("valid tag")
                        ),
                        Bytes(br#"</tag  attr-1=">"  attr2  =  '>'  3attr>"#)
                    );
                    assert_eq!(position, 40);
                }
            }
        }

        /// Ensures, that no empty `Text` events are generated
        mod $read_event {
            use crate::events::{
                BytesCData, BytesDecl, BytesEnd, BytesPI, BytesStart, BytesText, Event,
            };
            use crate::reader::Reader;

            /// When `encoding` feature is enabled, encoding should be detected
            /// from BOM (UTF-8) and BOM should be stripped.
            ///
            /// When `encoding` feature is disabled, UTF-8 is assumed and BOM
            /// character should be stripped for consistency
            #[$test]
            $($async)? fn bom_from_reader() {
                let mut reader = Reader::from_reader("\u{feff}\u{feff}".as_bytes());

                assert_eq!(
                    reader.$read_event($buf) $(.$await)? .expect("first event"),
                    Event::Text(BytesText::from_escaped("\u{feff}"))
                );

                assert_eq!(
                    reader.$read_event($buf) $(.$await)? .expect("second event"),
                    Event::Eof
                );
            }

            /// When parsing from &str, encoding is fixed (UTF-8), so
            /// - when `encoding` feature is disabled, the behavior the
            ///   same as in `bom_from_reader` text
            /// - when `encoding` feature is enabled, the behavior should
            ///   stay consistent, so the first BOM character is stripped
            #[$test]
            $($async)? fn bom_from_str() {
                let mut reader = Reader::from_str("\u{feff}\u{feff}");

                assert_eq!(
                    reader.$read_event($buf) $(.$await)? .expect("first event"),
                    Event::Text(BytesText::from_escaped("\u{feff}"))
                );

                assert_eq!(
                    reader.$read_event($buf) $(.$await)? .expect("second event"),
                    Event::Eof
                );
            }

            #[$test]
            $($async)? fn declaration() {
                let mut reader = Reader::from_str("<?xml ?>");

                assert_eq!(
                    reader.$read_event($buf) $(.$await)? .expect("decl"),
                    Event::Decl(BytesDecl::from_start(BytesStart::from_content("xml ", 3)))
                );
            }

            #[$test]
            $($async)? fn doctype() {
                let mut reader = Reader::from_str("<!DOCTYPE x>");

                assert_eq!(
                    reader.$read_event($buf) $(.$await)? .expect("doctype"),
                    Event::DocType(BytesText::from_escaped("x"))
                );
            }

            #[$test]
            $($async)? fn processing_instruction() {
                let mut reader = Reader::from_str("<?xml-stylesheet '? >\" ?>");

                assert_eq!(
                    reader.$read_event($buf) $(.$await)? .expect("pi"),
                    Event::PI(BytesPI::new("xml-stylesheet '? >\" "))
                );
            }

            /// Lone closing tags are not allowed, so testing it together with start tag
            #[$test]
            $($async)? fn start_and_end() {
                let mut reader = Reader::from_str("<tag></tag>");

                assert_eq!(
                    reader.$read_event($buf) $(.$await)? .expect("start"),
                    Event::Start(BytesStart::new("tag"))
                );

                assert_eq!(
                    reader.$read_event($buf) $(.$await)? .expect("end"),
                    Event::End(BytesEnd::new("tag"))
                );
            }

            #[$test]
            $($async)? fn empty() {
                let mut reader = Reader::from_str("<tag/>");

                assert_eq!(
                    reader.$read_event($buf) $(.$await)? .expect("empty"),
                    Event::Empty(BytesStart::new("tag"))
                );
            }

            #[$test]
            $($async)? fn text() {
                let mut reader = Reader::from_str("text");

                assert_eq!(
                    reader.$read_event($buf) $(.$await)? .expect("text"),
                    Event::Text(BytesText::from_escaped("text"))
                );
            }

            #[$test]
            $($async)? fn cdata() {
                let mut reader = Reader::from_str("<![CDATA[]]>");

                assert_eq!(
                    reader.$read_event($buf) $(.$await)? .expect("cdata"),
                    Event::CData(BytesCData::new(""))
                );
            }

            #[$test]
            $($async)? fn comment() {
                let mut reader = Reader::from_str("<!---->");

                assert_eq!(
                    reader.$read_event($buf) $(.$await)? .expect("comment"),
                    Event::Comment(BytesText::from_escaped(""))
                );
            }

            #[$test]
            $($async)? fn eof() {
                let mut reader = Reader::from_str("");

                assert_eq!(reader.$read_event($buf) $(.$await)? .expect("eof"), Event::Eof);
            }
        }
    };
}

// Export macro for the child modules (buffered_reader, slice_reader, async_tokio).
pub(super) use check;
