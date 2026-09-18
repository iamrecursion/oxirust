// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Integration test: MPEG-TS multi-program stream.
//!
//! Synthesises a transport stream containing two programs (PMT PIDs 0x0100 and
//! 0x0200) in a single PAT and asserts that `TsDemuxer::feed` surfaces both
//! programs with their elementary streams and stream-type bytes intact.
//!
//! TODO.md item: line 62 — "Test MPEG-TS demuxer with real broadcast streams
//! containing multiple programs".

use oximedia_container::demux::{TsDemuxer, TsStreamInfo};

const TS_PACKET_SIZE: usize = 188;
const PAT_PID: u16 = 0x0000;

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Builds a 188-byte TS packet carrying the supplied PSI section. The
/// `payload_unit_start` flag is set and a pointer field of 0 is prepended.
fn make_psi_packet(pid: u16, section: &[u8]) -> Vec<u8> {
    assert!(
        section.len() <= 183,
        "section too large for single TS packet"
    );
    let mut pkt = vec![0xFF_u8; TS_PACKET_SIZE]; // pad with 0xFF (stuffing)
    pkt[0] = 0x47; // sync
    pkt[1] = 0x40 | ((pid >> 8) as u8 & 0x1F); // PUSI=1, PID hi
    pkt[2] = (pid & 0xFF) as u8;
    pkt[3] = 0x10; // AFC=01 payload only, CC=0
    pkt[4] = 0x00; // pointer field
    pkt[5..5 + section.len()].copy_from_slice(section);
    pkt
}

/// Builds a 188-byte PES-carrying TS packet for an elementary stream.
fn make_pes_packet(pid: u16, cc: u8, with_pts: bool, pts: u64) -> Vec<u8> {
    let mut pkt = vec![0xFF_u8; TS_PACKET_SIZE];
    pkt[0] = 0x47;
    pkt[1] = 0x40 | ((pid >> 8) as u8 & 0x1F); // PUSI=1
    pkt[2] = (pid & 0xFF) as u8;
    pkt[3] = 0x10 | (cc & 0x0F); // AFC=01

    // Minimal PES header at payload start (offset 4).
    pkt[4] = 0x00;
    pkt[5] = 0x00;
    pkt[6] = 0x01; // PES start code
    pkt[7] = 0xE0; // stream_id (video stream 0)
    pkt[8] = 0x00; // PES_packet_length hi
    pkt[9] = 0x00; // PES_packet_length lo (0 = unbounded)
    pkt[10] = 0x80; // marker bits
    pkt[11] = if with_pts { 0x80 } else { 0x00 }; // PTS_DTS_flags = 10
    pkt[12] = if with_pts { 0x05 } else { 0x00 }; // header_data_length

    if with_pts {
        let ts = pts & 0x1_FFFF_FFFF; // 33 bits
        pkt[13] = 0x21 | (((ts >> 30) as u8 & 0x07) << 1);
        pkt[14] = ((ts >> 22) & 0xFF) as u8;
        pkt[15] = 0x01 | (((ts >> 14) as u8) & 0xFE);
        pkt[16] = ((ts >> 7) & 0xFF) as u8;
        pkt[17] = 0x01 | (((ts << 1) as u8) & 0xFE);
    }
    pkt
}

/// Builds a PAT section with two programs.
fn make_pat_two_programs(tsid: u16, program_a: (u16, u16), program_b: (u16, u16)) -> Vec<u8> {
    let mut s = Vec::new();
    s.push(0x00); // table_id
                  // section_length: 5 (tsid+version+section) + 4 * num_programs + 4 (CRC) = 5 + 8 + 4 = 17
    let sl: u16 = 5 + 4 * 2 + 4;
    s.push(0xB0 | ((sl >> 8) as u8 & 0x0F));
    s.push((sl & 0xFF) as u8);
    s.push((tsid >> 8) as u8);
    s.push((tsid & 0xFF) as u8);
    s.push(0xC1); // version=0, current
    s.push(0x00); // section_number
    s.push(0x00); // last_section_number
                  // program A
    s.push((program_a.0 >> 8) as u8);
    s.push((program_a.0 & 0xFF) as u8);
    s.push(0xE0 | ((program_a.1 >> 8) as u8 & 0x1F));
    s.push((program_a.1 & 0xFF) as u8);
    // program B
    s.push((program_b.0 >> 8) as u8);
    s.push((program_b.0 & 0xFF) as u8);
    s.push(0xE0 | ((program_b.1 >> 8) as u8 & 0x1F));
    s.push((program_b.1 & 0xFF) as u8);
    // CRC placeholder
    s.extend_from_slice(&[0u8; 4]);
    s
}

/// Builds a PMT section carrying a single elementary stream.
fn make_pmt_section(program_number: u16, pcr_pid: u16, stream_type: u8, es_pid: u16) -> Vec<u8> {
    let mut s = Vec::new();
    s.push(0x02);
    let sl: u16 = 9 + 5 + 4;
    s.push(0xB0 | ((sl >> 8) as u8 & 0x0F));
    s.push((sl & 0xFF) as u8);
    s.push((program_number >> 8) as u8);
    s.push((program_number & 0xFF) as u8);
    s.push(0xC1);
    s.push(0x00);
    s.push(0x00);
    s.push(0xE0 | ((pcr_pid >> 8) as u8 & 0x1F));
    s.push((pcr_pid & 0xFF) as u8);
    s.push(0xF0); // program_info_length=0
    s.push(0x00);
    s.push(stream_type);
    s.push(0xE0 | ((es_pid >> 8) as u8 & 0x1F));
    s.push((es_pid & 0xFF) as u8);
    s.push(0xF0); // ES_info_length=0
    s.push(0x00);
    s.extend_from_slice(&[0u8; 4]); // CRC placeholder
    s
}

fn build_bitstream() -> (Vec<u8>, TsStreamInfo) {
    let mut data = Vec::new();

    // PAT: two programs (prog 1 → 0x0100, prog 2 → 0x0200)
    let pat_section = make_pat_two_programs(0x0042, (1, 0x0100), (2, 0x0200));
    data.extend_from_slice(&make_psi_packet(PAT_PID, &pat_section));

    // PMT for program 1: AV1 (0x85) at PID 0x0101
    let pmt1 = make_pmt_section(1, 0x0101, 0x85, 0x0101);
    data.extend_from_slice(&make_psi_packet(0x0100, &pmt1));

    // PMT for program 2: VP9 (0x84) at PID 0x0201
    let pmt2 = make_pmt_section(2, 0x0201, 0x84, 0x0201);
    data.extend_from_slice(&make_psi_packet(0x0200, &pmt2));

    // A few elementary stream packets for each program (with PTS).
    for cc in 0..4 {
        data.extend_from_slice(&make_pes_packet(0x0101, cc, true, 90_000 * u64::from(cc)));
        data.extend_from_slice(&make_pes_packet(
            0x0201,
            cc,
            true,
            90_000 * u64::from(cc) + 4500,
        ));
    }

    let mut demuxer = TsDemuxer::new();
    demuxer.feed(&data);
    (data, demuxer.stream_info().clone())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[test]
fn test_two_programs_visible_in_pat() {
    let (_data, info) = build_bitstream();
    let pat = info.pat.as_ref().expect("PAT must be present");
    assert_eq!(pat.transport_stream_id, 0x0042);
    assert_eq!(pat.programs.len(), 2, "PAT must list both programs");

    let mut nums: Vec<u16> = pat.programs.iter().map(|p| p.program_number).collect();
    nums.sort_unstable();
    assert_eq!(nums, vec![1, 2]);

    let mut pmt_pids: Vec<u16> = pat.programs.iter().map(|p| p.pmt_pid).collect();
    pmt_pids.sort_unstable();
    assert_eq!(pmt_pids, vec![0x0100, 0x0200]);
}

#[test]
fn test_two_pmts_resolved() {
    let (_data, info) = build_bitstream();
    assert_eq!(info.pmts.len(), 2, "both PMTs must be parsed");

    let pmt1 = info.pmts.get(&1).expect("PMT for program 1");
    assert_eq!(pmt1.pcr_pid, 0x0101);
    assert_eq!(pmt1.streams.len(), 1);
    assert_eq!(
        pmt1.streams[0].stream_type, 0x85,
        "program 1 elementary stream must be AV1 (0x85)"
    );
    assert_eq!(pmt1.streams[0].elementary_pid, 0x0101);

    let pmt2 = info.pmts.get(&2).expect("PMT for program 2");
    assert_eq!(pmt2.pcr_pid, 0x0201);
    assert_eq!(pmt2.streams.len(), 1);
    assert_eq!(
        pmt2.streams[0].stream_type, 0x84,
        "program 2 elementary stream must be VP9 (0x84)"
    );
    assert_eq!(pmt2.streams[0].elementary_pid, 0x0201);
}

#[test]
fn test_per_pid_stream_type_populated() {
    let (_data, info) = build_bitstream();

    let pid_a = info
        .pids
        .get(&0x0101)
        .expect("PID 0x0101 must have PidInfo");
    assert_eq!(pid_a.stream_type, Some(0x85));
    assert!(
        pid_a.packet_count >= 4,
        "expected at least 4 packets for PID 0x0101"
    );
    assert!(
        pid_a.pts_first.is_some(),
        "PTS for PID 0x0101 must be captured"
    );

    let pid_b = info
        .pids
        .get(&0x0201)
        .expect("PID 0x0201 must have PidInfo");
    assert_eq!(pid_b.stream_type, Some(0x84));
    assert!(pid_b.packet_count >= 4);
    assert!(pid_b.pts_first.is_some());
}

#[test]
fn test_pat_pid_observed_separately() {
    let (_data, info) = build_bitstream();
    // PID 0x0000 (PAT) must be in the PID map but should not have a stream type.
    let pat_pid_info = info.pids.get(&PAT_PID).expect("PAT PID must be tracked");
    assert!(
        pat_pid_info.stream_type.is_none(),
        "PAT PID has no stream type"
    );
    assert!(pat_pid_info.packet_count >= 1);
}
