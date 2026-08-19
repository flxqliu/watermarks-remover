"""Tests for av_meta.py (MP4/MOV, WAV, MP3 AI/C2PA provenance metadata)."""

from __future__ import annotations

import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "service" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import pytest

from av_meta import (
    clean_av,
    detect_av_format,
    inspect_av,
)

# LOCAL PATCH (Circe, 2026-08-19): MP4 cleaning now remuxes real media through
# ffmpeg instead of hand-editing ISOBMFF boxes (see av_meta.py's patch note),
# so the hand-built byte-structure fixtures below (_mp4_with_xmp,
# _mp4_with_udta_tag -- ftyp + moov/uuid + mdat with no real trak/codec data)
# can no longer be meaningfully "cleaned": there's nothing for ffmpeg to
# remux. The MP4 tests that exercise clean_av() use real, tiny ffmpeg-encoded
# fixtures instead (_real_mp4 below); detect_av_format() and inspect_av()
# tests that only walk box structure (never call clean_av) keep the original
# hand-built fixtures, since those are still exactly what they test.
FFMPEG = shutil.which("ffmpeg")
requires_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not installed")


def _real_mp4(dest: Path, *, comment: str | None = None, artist: str | None = None) -> None:
    cmd = [
        FFMPEG,
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1:size=32x32:rate=5",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
    ]
    if comment is not None:
        cmd += ["-metadata", f"comment={comment}"]
    if artist is not None:
        cmd += ["-metadata", f"artist={artist}"]
    cmd += ["-movflags", "+faststart", str(dest)]
    subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)


def _real_mp4_with_xmp(dest: Path, xmp_description: str) -> None:
    """Real, decodable MP4 with a genuine XMP uuid box (exiftool embeds it
    via the same top-level uuid mechanism the C2PA spec uses)."""
    _real_mp4(dest)
    exiftool = shutil.which("exiftool")
    assert exiftool, "exiftool required for this fixture"
    subprocess.run(
        [exiftool, "-overwrite_original", f"-XMP-dc:Description={xmp_description}", str(dest)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _isobmff_box(fourcc: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload) + 8) + fourcc + payload


def _mp4(*top_level_boxes: bytes) -> bytes:
    ftyp = _isobmff_box(b"ftyp", b"isom" + struct.pack(">I", 0) + b"isomiso2mp41")
    return ftyp + b"".join(top_level_boxes)


def _riff_chunk(cid: bytes, payload: bytes) -> bytes:
    pad = b"\x00" if len(payload) & 1 else b""
    return cid + struct.pack("<I", len(payload)) + payload + pad


def _wav(*chunks: bytes) -> bytes:
    body = b"WAVE" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _wav_fmt_chunk() -> bytes:
    return _riff_chunk(b"fmt ", struct.pack("<HHIIHH", 1, 1, 44100, 88200, 2, 16))


def _wav_data_chunk(n: int = 8) -> bytes:
    return _riff_chunk(b"data", b"\x01" * n)


def _wav_list_info(text: bytes) -> bytes:
    isft = _riff_chunk(b"ISFT", text + (b"\x00" if len(text) % 2 == 0 else b""))
    return _riff_chunk(b"LIST", b"INFO" + isft)


def _id3v2_size_bytes(n: int) -> bytes:
    return bytes([(n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F])


def _id3v2_frame(frame_id: bytes, payload: bytes, *, major: int = 3) -> bytes:
    size = _id3v2_size_bytes(len(payload)) if major == 4 else struct.pack(">I", len(payload))
    return frame_id + size + b"\x00\x00" + payload


def _mp3(*frames: bytes, major: int = 3) -> bytes:
    body = b"".join(frames)
    header = b"ID3" + bytes([major, 0, 0]) + _id3v2_size_bytes(len(body))
    audio = bytes([0xFF, 0xFB, 0x90, 0x00]) * 4  # placeholder MPEG frame-sync bytes
    return header + body + audio


# ---------------------------------------------------------------------------
# detect_av_format
# ---------------------------------------------------------------------------


def test_detect_mp4():
    assert detect_av_format(_mp4()) == "mp4"


def test_detect_wav():
    assert detect_av_format(_wav(_wav_fmt_chunk())) == "wav"


def test_detect_mp3_with_id3():
    assert detect_av_format(_mp3(_id3v2_frame(b"TIT2", b"\x00Song"))) == "mp3"


def test_detect_mp3_frame_sync_only():
    data = bytes([0xFF, 0xFB, 0x90, 0x00]) * 10
    assert detect_av_format(data) == "mp3"


def test_detect_unknown():
    assert detect_av_format(b"not a known av container") == "unknown"


# ---------------------------------------------------------------------------
# MP4 / MOV
# ---------------------------------------------------------------------------


@requires_ffmpeg
def test_mp4_xmp_uuid_detected_and_stripped(tmp_path):
    src = tmp_path / "clip.mp4"
    _real_mp4_with_xmp(src, "Generated by AI toolchain")

    report = inspect_av(src)
    assert report.format == "mp4"
    assert report.has_ai_metadata is True
    assert any("uuid" in f.lower() for f in report.findings)

    dest = tmp_path / "clip.cleaned.mp4"
    result = clean_av(src, dest, strip_all_metadata=True)
    assert result["still_has_ai_metadata"] is False
    assert b"Generated by AI" not in dest.read_bytes()
    # New action wording says "...any uuid/jumb/C2PA/XMP box..." -- the
    # ffmpeg remux drops it as a side effect of writing a fresh container,
    # rather than a dedicated "drop uuid box" step, but the word survives.
    assert any("uuid" in a.lower() for a in result["actions"])


@requires_ffmpeg
def test_mp4_moov_udta_generator_tag_detected_and_stripped(tmp_path):
    src = tmp_path / "clip.mp4"
    _real_mp4(src, comment="ElevenLabs AI Voice Generator")
    report = inspect_av(src)
    assert report.has_ai_metadata is False  # ElevenLabs isn't in the flat hint list

    # Use an explicit AI hint that IS in the flat list to prove detection works.
    src2 = tmp_path / "clip2.mp4"
    _real_mp4(src2, comment="Generated by AI")
    report2 = inspect_av(src2)
    assert report2.has_ai_metadata is True
    assert any("udta" in f for f in report2.findings)

    dest = tmp_path / "clip2.cleaned.mp4"
    result = clean_av(src2, dest, strip_all_metadata=True)
    assert b"Generated by AI" not in dest.read_bytes()
    assert result["still_has_ai_metadata"] is False


@requires_ffmpeg
def test_mp4_udta_stripped_by_default_even_without_ai_hint(tmp_path):
    """Default strip_all_metadata=True strips the tag regardless of hint
    match, matching this project's existing default behaviour for image
    metadata."""
    src = tmp_path / "clip.mp4"
    _real_mp4(src, comment="Adobe Premiere Pro 2026")
    dest = tmp_path / "clip.cleaned.mp4"
    result = clean_av(src, dest, strip_all_metadata=True)
    assert b"Adobe Premiere Pro 2026" not in dest.read_bytes()
    assert any("remux via ffmpeg" in a for a in result["actions"])


@requires_ffmpeg
def test_mp4_keep_non_ai_metadata_preserves_unflagged_udta(tmp_path):
    src = tmp_path / "clip.mp4"
    _real_mp4(src, comment="Adobe Premiere Pro 2026")
    dest = tmp_path / "clip.cleaned.mp4"
    result = clean_av(src, dest, strip_all_metadata=False)
    assert b"Adobe Premiere Pro 2026" in dest.read_bytes()
    # Selective mode: ffprobe found no AI-hint-matching key, so drop_keys was
    # empty and nothing got nulled.
    assert not any("cleared AI-hint-matching tags" in a for a in result["actions"])


@requires_ffmpeg
def test_mp4_clean_file_is_idempotent_when_already_clean(tmp_path):
    src = tmp_path / "clean.mp4"
    _real_mp4(src)
    dest = tmp_path / "clean.cleaned.mp4"
    result = clean_av(src, dest, strip_all_metadata=True)
    assert result["still_has_ai_metadata"] is False
    assert result["still_has_c2pa"] is False


# ---------------------------------------------------------------------------
# WAV
# ---------------------------------------------------------------------------


def test_wav_list_info_ai_hint_detected_and_stripped(tmp_path):
    data = _wav(_wav_fmt_chunk(), _wav_list_info(b"Generated by AI"), _wav_data_chunk(8))
    src = tmp_path / "voice.wav"
    src.write_bytes(data)

    report = inspect_av(src)
    assert report.format == "wav"
    assert report.has_ai_metadata is True
    assert any("LIST INFO" in f for f in report.findings)

    dest = tmp_path / "voice.cleaned.wav"
    result = clean_av(src, dest, strip_all_metadata=True)
    cleaned = dest.read_bytes()
    assert b"Generated by AI" not in cleaned
    assert result["still_has_ai_metadata"] is False


def test_wav_audio_data_untouched(tmp_path):
    audio = bytes(range(256)) * 4
    data = _wav(_wav_fmt_chunk(), _wav_list_info(b"Generated by AI"), _riff_chunk(b"data", audio))
    src = tmp_path / "voice.wav"
    src.write_bytes(data)
    dest = tmp_path / "voice.cleaned.wav"
    clean_av(src, dest, strip_all_metadata=True)
    cleaned = dest.read_bytes()
    assert audio in cleaned
    # RIFF size field must match the new (shorter) total.
    riff_size = struct.unpack("<I", cleaned[4:8])[0]
    assert riff_size == len(cleaned) - 8


def test_wav_clean_file_already_clean_no_changes(tmp_path):
    audio = b"\x01\x02\x03\x04" * 4
    data = _wav(_wav_fmt_chunk(), _riff_chunk(b"data", audio))
    src = tmp_path / "clean.wav"
    src.write_bytes(data)
    dest = tmp_path / "clean.cleaned.wav"
    result = clean_av(src, dest, strip_all_metadata=True)
    assert "no WAV metadata chunks removed" in result["actions"][0]
    assert dest.read_bytes() == data


def test_wav_keep_non_ai_metadata_preserves_unflagged_info(tmp_path):
    data = _wav(_wav_fmt_chunk(), _wav_list_info(b"Adobe Audition"), _wav_data_chunk())
    src = tmp_path / "voice.wav"
    src.write_bytes(data)
    dest = tmp_path / "voice.cleaned.wav"
    clean_av(src, dest, strip_all_metadata=False)
    assert b"Adobe Audition" in dest.read_bytes()


# ---------------------------------------------------------------------------
# MP3 (ID3v2)
# ---------------------------------------------------------------------------


def test_mp3_id3v23_ai_hint_detected(tmp_path):
    data = _mp3(
        _id3v2_frame(b"TIT2", b"\x00My Track"),
        _id3v2_frame(b"TSSE", b"\x00Generated by AI"),
    )
    src = tmp_path / "song.mp3"
    src.write_bytes(data)

    report = inspect_av(src)
    assert report.format == "mp3"
    assert report.has_ai_metadata is True
    assert any("TSSE" in f for f in report.findings)


def test_mp3_strip_all_drops_whole_tag(tmp_path):
    data = _mp3(
        _id3v2_frame(b"TIT2", b"\x00My Track"),
        _id3v2_frame(b"TSSE", b"\x00Generated by AI"),
    )
    src = tmp_path / "song.mp3"
    src.write_bytes(data)
    dest = tmp_path / "song.cleaned.mp3"
    result = clean_av(src, dest, strip_all_metadata=True)
    cleaned = dest.read_bytes()
    assert b"My Track" not in cleaned
    assert b"Generated by AI" not in cleaned
    assert result["still_has_ai_metadata"] is False
    # The MPEG frame-sync audio placeholder bytes survive untouched.
    assert cleaned.endswith(bytes([0xFF, 0xFB, 0x90, 0x00]) * 4)


def test_mp3_keep_mode_drops_only_flagged_frame(tmp_path):
    data = _mp3(
        _id3v2_frame(b"TIT2", b"\x00My Track"),
        _id3v2_frame(b"TSSE", b"\x00Generated by AI"),
    )
    src = tmp_path / "song.mp3"
    src.write_bytes(data)
    dest = tmp_path / "song.cleaned.mp3"
    result = clean_av(src, dest, strip_all_metadata=False)
    cleaned = dest.read_bytes()
    assert b"My Track" in cleaned  # legitimate tag survives
    assert b"Generated by AI" not in cleaned  # flagged frame is gone
    assert any("TSSE" in a for a in result["actions"])


def test_mp3_id3v24_syncsafe_frame_size_round_trip(tmp_path):
    data = _mp3(
        _id3v2_frame(b"TIT2", b"\x00My Track", major=4),
        _id3v2_frame(b"TSSE", b"\x00Generated by AI", major=4),
        major=4,
    )
    src = tmp_path / "song.mp3"
    src.write_bytes(data)
    report = inspect_av(src)
    assert report.has_ai_metadata is True

    dest = tmp_path / "song.cleaned.mp3"
    result = clean_av(src, dest, strip_all_metadata=False)
    cleaned = dest.read_bytes()
    assert b"My Track" in cleaned
    assert b"Generated by AI" not in cleaned
    # Re-inspecting the cleaned, rewritten v2.4 tag must still parse correctly.
    after = inspect_av(dest)
    assert after.has_ai_metadata is False
    assert result["still_has_ai_metadata"] is False


def test_mp3_no_id3_tag_clean_noop(tmp_path):
    data = bytes([0xFF, 0xFB, 0x90, 0x00]) * 20
    src = tmp_path / "raw.mp3"
    src.write_bytes(data)
    dest = tmp_path / "raw.cleaned.mp3"
    result = clean_av(src, dest, strip_all_metadata=True)
    assert dest.read_bytes() == data
    assert result["still_has_ai_metadata"] is False


def test_mp3_id3v22_falls_back_to_whole_tag_scan_and_drop(tmp_path):
    # v2.2: 3-byte frame IDs, no per-frame decomposition -- whole-tag handling.
    body = b"TT2\x00\x00\x10\x00Generated by AI"
    header = b"ID3" + bytes([2, 0, 0]) + _id3v2_size_bytes(len(body))
    data = header + body + bytes([0xFF, 0xFB, 0x90, 0x00]) * 4
    src = tmp_path / "old.mp3"
    src.write_bytes(data)

    report = inspect_av(src)
    assert report.has_ai_metadata is True

    dest = tmp_path / "old.cleaned.mp3"
    result = clean_av(src, dest, strip_all_metadata=False)
    cleaned = dest.read_bytes()
    assert b"Generated by AI" not in cleaned
    assert any("ID3v2.2" in a for a in result["actions"])
