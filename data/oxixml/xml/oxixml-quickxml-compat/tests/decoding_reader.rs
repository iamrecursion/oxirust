//! The `quick_xml::encoding::DecodingReader` surface, exercised through the
//! compat alias.

use oxixml_quickxml_compat as quick_xml;
use quick_xml::encoding::DecodingReader;
use quick_xml::events::Event;
use quick_xml::reader::Reader;

#[test]
fn reads_a_utf16_document() -> quick_xml::Result<()> {
    let mut xml = vec![0xFF, 0xFE]; // UTF-16 LE BOM
    for unit in "<?xml version='1.0'?><doc>hi</doc>".encode_utf16() {
        xml.extend_from_slice(&unit.to_le_bytes());
    }

    let mut reader = Reader::from_reader(DecodingReader::new(xml.as_slice()));
    let mut buf = Vec::new();
    let mut text = String::new();
    loop {
        buf.clear();
        match reader.read_event_into(&mut buf)? {
            Event::Eof => break,
            Event::Text(event) => text.push_str(&event.decode()?),
            _ => continue,
        }
    }
    assert_eq!(text, "hi");
    Ok(())
}
