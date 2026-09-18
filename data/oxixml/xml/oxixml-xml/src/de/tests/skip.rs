//! Checks that skipping and replaying events with [`Deserializer::skip()`] and
//! [`Deserializer::start_replay()`] works correctly.

use super::*;
use crate::de::DeEvent::*;
use crate::events::BytesEnd;
use std::num::NonZeroUsize;

/// Checks that `peek()` and `read()` behaves correctly after `skip()`
#[test]
fn read_and_peek() {
    let mut de = make_de(
        "\
                <root>\
                    <inner>\
                        text\
                        <inner/>\
                    </inner>\
                    <next/>\
                    <target/>\
                </root>\
                ",
    );

    // Initial conditions - both are empty
    assert_eq!(de.read, vec![]);
    assert_eq!(de.write, vec![]);

    assert_eq!(
        de.next().expect("should be Ok"),
        Start(BytesStart::new("root"))
    );
    assert_eq!(
        de.peek().expect("should be Ok"),
        &Start(BytesStart::new("inner"))
    );

    // Mark that start_replay() should begin replay from this point
    let checkpoint = de.skip_checkpoint();
    assert_eq!(checkpoint, 0);

    // Should skip first <inner> tree
    de.skip().expect("should be Ok");
    assert_eq!(de.read, vec![]);
    assert_eq!(
        de.write,
        vec![
            Start(BytesStart::new("inner")),
            Text("text".into()),
            Start(BytesStart::new("inner")),
            End(BytesEnd::new("inner")),
            End(BytesEnd::new("inner")),
        ]
    );

    // Consume <next/>. Now unconsumed XML looks like:
    //
    //   <inner>
    //     text
    //     <inner/>
    //   </inner>
    //   <target/>
    // </root>
    assert_eq!(
        de.next().expect("should be Ok"),
        Start(BytesStart::new("next"))
    );
    assert_eq!(de.next().expect("should be Ok"), End(BytesEnd::new("next")));

    // We finish writing. Next call to `next()` should start replay that messages:
    //
    //   <inner>
    //     text
    //     <inner/>
    //   </inner>
    //
    // and after that stream that messages:
    //
    //   <target/>
    // </root>
    de.start_replay(checkpoint);
    assert_eq!(
        de.read,
        vec![
            Start(BytesStart::new("inner")),
            Text("text".into()),
            Start(BytesStart::new("inner")),
            End(BytesEnd::new("inner")),
            End(BytesEnd::new("inner")),
        ]
    );
    assert_eq!(de.write, vec![]);
    assert_eq!(
        de.next().expect("should be Ok"),
        Start(BytesStart::new("inner"))
    );

    // Mark that start_replay() should begin replay from this point
    let checkpoint = de.skip_checkpoint();
    assert_eq!(checkpoint, 0);

    // Skip `$text` node and consume <inner/> after it
    de.skip().expect("should be Ok");
    assert_eq!(
        de.read,
        vec![
            Start(BytesStart::new("inner")),
            End(BytesEnd::new("inner")),
            End(BytesEnd::new("inner")),
        ]
    );
    assert_eq!(
        de.write,
        vec![
            // This comment here to keep the same formatting of both arrays
            // otherwise rustfmt suggest one-line it
            Text("text".into()),
        ]
    );

    assert_eq!(
        de.next().expect("should be Ok"),
        Start(BytesStart::new("inner"))
    );
    assert_eq!(
        de.next().expect("should be Ok"),
        End(BytesEnd::new("inner"))
    );

    // We finish writing. Next call to `next()` should start replay messages:
    //
    //     text
    //   </inner>
    //
    // and after that stream that messages:
    //
    //   <target/>
    // </root>
    de.start_replay(checkpoint);
    assert_eq!(
        de.read,
        vec![
            // This comment here to keep the same formatting as others
            // otherwise rustfmt suggest one-line it
            Text("text".into()),
            End(BytesEnd::new("inner")),
        ]
    );
    assert_eq!(de.write, vec![]);
    assert_eq!(de.next().expect("should be Ok"), Text("text".into()));
    assert_eq!(
        de.next().expect("should be Ok"),
        End(BytesEnd::new("inner"))
    );
    assert_eq!(
        de.next().expect("should be Ok"),
        Start(BytesStart::new("target"))
    );
    assert_eq!(
        de.next().expect("should be Ok"),
        End(BytesEnd::new("target"))
    );
    assert_eq!(de.next().expect("should be Ok"), End(BytesEnd::new("root")));
    assert_eq!(de.next().expect("should be Ok"), Eof);
}

/// Checks that `read_to_end()` behaves correctly after `skip()`
#[test]
fn read_to_end() {
    let mut de = make_de(
        "\
                <root>\
                    <skip>\
                        text\
                        <skip/>\
                    </skip>\
                    <target>\
                        <target/>\
                    </target>\
                </root>\
                ",
    );

    // Initial conditions - both are empty
    assert_eq!(de.read, vec![]);
    assert_eq!(de.write, vec![]);

    assert_eq!(
        de.next().expect("should be Ok"),
        Start(BytesStart::new("root"))
    );

    // Mark that start_replay() should begin replay from this point
    let checkpoint = de.skip_checkpoint();
    assert_eq!(checkpoint, 0);

    // Skip the <skip> tree
    de.skip().expect("should be Ok");
    assert_eq!(de.read, vec![]);
    assert_eq!(
        de.write,
        vec![
            Start(BytesStart::new("skip")),
            Text("text".into()),
            Start(BytesStart::new("skip")),
            End(BytesEnd::new("skip")),
            End(BytesEnd::new("skip")),
        ]
    );

    // Drop all events that represents <target> tree. Now unconsumed XML looks like:
    //
    //   <skip>
    //     text
    //     <skip/>
    //   </skip>
    // </root>
    assert_eq!(
        de.next().expect("should be Ok"),
        Start(BytesStart::new("target"))
    );
    de.read_to_end(QName(b"target")).expect("should be Ok");
    assert_eq!(de.read, vec![]);
    assert_eq!(
        de.write,
        vec![
            Start(BytesStart::new("skip")),
            Text("text".into()),
            Start(BytesStart::new("skip")),
            End(BytesEnd::new("skip")),
            End(BytesEnd::new("skip")),
        ]
    );

    // We finish writing. Next call to `next()` should start replay that messages:
    //
    //   <skip>
    //     text
    //     <skip/>
    //   </skip>
    //
    // and after that stream that messages:
    //
    // </root>
    de.start_replay(checkpoint);
    assert_eq!(
        de.read,
        vec![
            Start(BytesStart::new("skip")),
            Text("text".into()),
            Start(BytesStart::new("skip")),
            End(BytesEnd::new("skip")),
            End(BytesEnd::new("skip")),
        ]
    );
    assert_eq!(de.write, vec![]);

    assert_eq!(
        de.next().expect("should be Ok"),
        Start(BytesStart::new("skip"))
    );
    de.read_to_end(QName(b"skip")).expect("should be Ok");

    assert_eq!(de.next().expect("should be Ok"), End(BytesEnd::new("root")));
    assert_eq!(de.next().expect("should be Ok"), Eof);
}

/// Checks that replay replayes only part of events
/// Test for https://github.com/tafia/quick-xml/issues/435
#[test]
fn partial_replay() {
    let mut de = make_de(
        "\
                <root>\
                    <skipped-1/>\
                    <skipped-2/>\
                    <inner>\
                        <skipped-3/>\
                        <skipped-4/>\
                        <target-2/>\
                    </inner>\
                    <target-1/>\
                </root>\
                ",
    );

    // Initial conditions - both are empty
    assert_eq!(de.read, vec![]);
    assert_eq!(de.write, vec![]);

    assert_eq!(
        de.next().expect("should be Ok"),
        Start(BytesStart::new("root"))
    );

    // start_replay() should start replay from this point
    let checkpoint1 = de.skip_checkpoint();
    assert_eq!(checkpoint1, 0);

    // Should skip first and second <skipped-N/> elements
    de.skip().expect("should be Ok"); // skipped-1
    de.skip().expect("should be Ok"); // skipped-2
    assert_eq!(de.read, vec![]);
    assert_eq!(
        de.write,
        vec![
            Start(BytesStart::new("skipped-1")),
            End(BytesEnd::new("skipped-1")),
            Start(BytesStart::new("skipped-2")),
            End(BytesEnd::new("skipped-2")),
        ]
    );

    ////////////////////////////////////////////////////////////////////////////////////////

    assert_eq!(
        de.next().expect("should be Ok"),
        Start(BytesStart::new("inner"))
    );
    assert_eq!(
        de.peek().expect("should be Ok"),
        &Start(BytesStart::new("skipped-3"))
    );
    assert_eq!(
        de.read,
        vec![
            // This comment here to keep the same formatting of both arrays
            // otherwise rustfmt suggest one-line it
            Start(BytesStart::new("skipped-3")),
        ]
    );
    assert_eq!(
        de.write,
        vec![
            Start(BytesStart::new("skipped-1")),
            End(BytesEnd::new("skipped-1")),
            Start(BytesStart::new("skipped-2")),
            End(BytesEnd::new("skipped-2")),
        ]
    );

    // start_replay() should start replay from this point
    let checkpoint2 = de.skip_checkpoint();
    assert_eq!(checkpoint2, 4);

    // Should skip third and forth <skipped-N/> elements
    de.skip().expect("should be Ok"); // skipped-3
    de.skip().expect("should be Ok"); // skipped-4
    assert_eq!(de.read, vec![]);
    assert_eq!(
        de.write,
        vec![
            // checkpoint 1
            Start(BytesStart::new("skipped-1")),
            End(BytesEnd::new("skipped-1")),
            Start(BytesStart::new("skipped-2")),
            End(BytesEnd::new("skipped-2")),
            // checkpoint 2
            Start(BytesStart::new("skipped-3")),
            End(BytesEnd::new("skipped-3")),
            Start(BytesStart::new("skipped-4")),
            End(BytesEnd::new("skipped-4")),
        ]
    );
    assert_eq!(
        de.next().expect("should be Ok"),
        Start(BytesStart::new("target-2"))
    );
    assert_eq!(
        de.next().expect("should be Ok"),
        End(BytesEnd::new("target-2"))
    );
    assert_eq!(
        de.peek().expect("should be Ok"),
        &End(BytesEnd::new("inner"))
    );
    assert_eq!(
        de.read,
        vec![
            // This comment here to keep the same formatting of both arrays
            // otherwise rustfmt suggest one-line it
            End(BytesEnd::new("inner")),
        ]
    );
    assert_eq!(
        de.write,
        vec![
            // checkpoint 1
            Start(BytesStart::new("skipped-1")),
            End(BytesEnd::new("skipped-1")),
            Start(BytesStart::new("skipped-2")),
            End(BytesEnd::new("skipped-2")),
            // checkpoint 2
            Start(BytesStart::new("skipped-3")),
            End(BytesEnd::new("skipped-3")),
            Start(BytesStart::new("skipped-4")),
            End(BytesEnd::new("skipped-4")),
        ]
    );

    // Start replay events from checkpoint 2
    de.start_replay(checkpoint2);
    assert_eq!(
        de.read,
        vec![
            Start(BytesStart::new("skipped-3")),
            End(BytesEnd::new("skipped-3")),
            Start(BytesStart::new("skipped-4")),
            End(BytesEnd::new("skipped-4")),
            End(BytesEnd::new("inner")),
        ]
    );
    assert_eq!(
        de.write,
        vec![
            Start(BytesStart::new("skipped-1")),
            End(BytesEnd::new("skipped-1")),
            Start(BytesStart::new("skipped-2")),
            End(BytesEnd::new("skipped-2")),
        ]
    );

    // Replayed events
    assert_eq!(
        de.next().expect("should be Ok"),
        Start(BytesStart::new("skipped-3"))
    );
    assert_eq!(
        de.next().expect("should be Ok"),
        End(BytesEnd::new("skipped-3"))
    );
    assert_eq!(
        de.next().expect("should be Ok"),
        Start(BytesStart::new("skipped-4"))
    );
    assert_eq!(
        de.next().expect("should be Ok"),
        End(BytesEnd::new("skipped-4"))
    );

    assert_eq!(
        de.next().expect("should be Ok"),
        End(BytesEnd::new("inner"))
    );
    assert_eq!(de.read, vec![]);
    assert_eq!(
        de.write,
        vec![
            Start(BytesStart::new("skipped-1")),
            End(BytesEnd::new("skipped-1")),
            Start(BytesStart::new("skipped-2")),
            End(BytesEnd::new("skipped-2")),
        ]
    );

    ////////////////////////////////////////////////////////////////////////////////////////

    // New events
    assert_eq!(
        de.next().expect("should be Ok"),
        Start(BytesStart::new("target-1"))
    );
    assert_eq!(
        de.next().expect("should be Ok"),
        End(BytesEnd::new("target-1"))
    );

    assert_eq!(de.read, vec![]);
    assert_eq!(
        de.write,
        vec![
            Start(BytesStart::new("skipped-1")),
            End(BytesEnd::new("skipped-1")),
            Start(BytesStart::new("skipped-2")),
            End(BytesEnd::new("skipped-2")),
        ]
    );

    // Start replay events from checkpoint 1
    de.start_replay(checkpoint1);
    assert_eq!(
        de.read,
        vec![
            Start(BytesStart::new("skipped-1")),
            End(BytesEnd::new("skipped-1")),
            Start(BytesStart::new("skipped-2")),
            End(BytesEnd::new("skipped-2")),
        ]
    );
    assert_eq!(de.write, vec![]);

    // Replayed events
    assert_eq!(
        de.next().expect("should be Ok"),
        Start(BytesStart::new("skipped-1"))
    );
    assert_eq!(
        de.next().expect("should be Ok"),
        End(BytesEnd::new("skipped-1"))
    );
    assert_eq!(
        de.next().expect("should be Ok"),
        Start(BytesStart::new("skipped-2"))
    );
    assert_eq!(
        de.next().expect("should be Ok"),
        End(BytesEnd::new("skipped-2"))
    );

    assert_eq!(de.read, vec![]);
    assert_eq!(de.write, vec![]);

    // New events
    assert_eq!(de.next().expect("should be Ok"), End(BytesEnd::new("root")));
    assert_eq!(de.next().expect("should be Ok"), Eof);
}

/// Checks that limiting buffer size works correctly
#[test]
fn limit() {
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    #[allow(unused)]
    struct List {
        item: Vec<()>,
    }

    let mut de = make_de(
        "\
                <any-name>\
                    <item/>\
                    <another-item>\
                        <some-element>with text</some-element>\
                        <yet-another-element/>\
                    </another-item>\
                    <item/>\
                    <item/>\
                </any-name>\
                ",
    );
    de.event_buffer_size(NonZeroUsize::new(3));

    match List::deserialize(&mut de) {
        Err(DeError::TooManyEvents(count)) => assert_eq!(count.get(), 3),
        e => panic!("Expected `Err(TooManyEvents(3))`, but got `{:?}`", e),
    }
}

/// Without handling Eof in `skip` this test failed with memory allocation
#[test]
fn invalid_xml() {
    use crate::de::DeEvent::*;

    let mut de = make_de("<root>");

    // Cache all events
    let checkpoint = de.skip_checkpoint();
    de.skip().expect("should be Ok");
    de.start_replay(checkpoint);
    assert_eq!(de.read, vec![Start(BytesStart::new("root")), Eof]);
}
