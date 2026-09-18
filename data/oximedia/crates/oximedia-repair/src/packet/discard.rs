//! Corrupt packet removal.
//!
//! This module provides functions to remove or replace corrupt packets.

use super::recover::{Packet, PacketStatus};
use crate::Result;

/// Discard corrupt packets from stream.
pub fn discard_corrupt_packets(packets: &mut Vec<Packet>) -> usize {
    let initial_count = packets.len();
    packets.retain(|p| p.status != PacketStatus::Corrupt);
    initial_count - packets.len()
}

/// Replace corrupt packets with silence/black frames.
pub fn replace_corrupt_packets(packets: &mut [Packet]) -> Result<usize> {
    let mut replaced = 0;

    for packet in packets.iter_mut() {
        if packet.status == PacketStatus::Corrupt {
            // Replace with zeros (silence for audio, black for video)
            let size = packet.data.len();
            packet.data = vec![0u8; size];
            packet.status = PacketStatus::Valid;
            replaced += 1;
        }
    }

    Ok(replaced)
}

/// Remove duplicate packets.
pub fn remove_duplicates(packets: &mut Vec<Packet>) -> usize {
    if packets.is_empty() {
        return 0;
    }

    let initial_count = packets.len();
    let mut i = 1;

    while i < packets.len() {
        if packets[i].sequence == packets[i - 1].sequence {
            packets.remove(i);
        } else {
            i += 1;
        }
    }

    initial_count - packets.len()
}

/// Remove packets with invalid timestamps.
pub fn remove_invalid_timestamps(packets: &mut Vec<Packet>) -> usize {
    let initial_count = packets.len();
    packets.retain(|p| p.timestamp >= 0);
    initial_count - packets.len()
}

/// Renumber packet sequences after discarding.
pub fn renumber_sequences(packets: &mut [Packet]) {
    for (i, packet) in packets.iter_mut().enumerate() {
        packet.sequence = i as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discard_corrupt_packets() {
        let mut packets = vec![
            Packet {
                sequence: 0,
                data: vec![1],
                timestamp: 0,
                status: PacketStatus::Valid,
            },
            Packet {
                sequence: 1,
                data: vec![2],
                timestamp: 100,
                status: PacketStatus::Corrupt,
            },
            Packet {
                sequence: 2,
                data: vec![3],
                timestamp: 200,
                status: PacketStatus::Valid,
            },
        ];

        let discarded = discard_corrupt_packets(&mut packets);
        assert_eq!(discarded, 1);
        assert_eq!(packets.len(), 2);
    }

    #[test]
    fn test_remove_duplicates() {
        let mut packets = vec![
            Packet {
                sequence: 0,
                data: vec![1],
                timestamp: 0,
                status: PacketStatus::Valid,
            },
            Packet {
                sequence: 0,
                data: vec![1],
                timestamp: 0,
                status: PacketStatus::Valid,
            },
            Packet {
                sequence: 1,
                data: vec![2],
                timestamp: 100,
                status: PacketStatus::Valid,
            },
        ];

        let removed = remove_duplicates(&mut packets);
        assert_eq!(removed, 1);
        assert_eq!(packets.len(), 2);
    }

    #[test]
    fn test_renumber_sequences() {
        let mut packets = vec![
            Packet {
                sequence: 10,
                data: vec![1],
                timestamp: 0,
                status: PacketStatus::Valid,
            },
            Packet {
                sequence: 20,
                data: vec![2],
                timestamp: 100,
                status: PacketStatus::Valid,
            },
        ];

        renumber_sequences(&mut packets);
        assert_eq!(packets[0].sequence, 0);
        assert_eq!(packets[1].sequence, 1);
    }
}
