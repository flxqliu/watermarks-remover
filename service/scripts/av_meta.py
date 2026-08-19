#!/usr/bin/env python3
"""AI/C2PA provenance metadata for audio and video containers.

Extends the file-cleaners layer (image_meta.py for PNG/JPEG/..., container_meta.py
for SVG/PDF/DOCX/...) to MP4/MOV/M4A/M4V (ISOBMFF), WAV, and MP3. Generative
audio/video tools embed provenance the same way image generators do -- C2PA
manifests and XMP in ISOBMFF boxes, generator tags in RIFF chunks and ID3v2
frames -- so this reuses the existing ISOBMFF box walker from image_meta.py
(the same mechanism already proven for AVIF/HEIC) rather than duplicating it.

Metadata only: waveform/pixel data is never touched, matching every other
cleaner in this project.

Known scope limits (documented, not silently mishandled):
- MP4/MOV: legacy QuickTime files with no top-level `ftyp` box are not
  detected by signature (rare in practice; modern encoders always write one).
- MP3: ID3v2.2 (3-byte frame IDs, pre-iTunes era) tags are detected but not
  decomposed into frames -- stripping falls back to a whole-tag drop, which
  is always safe. ID3v1 (fixed 128-byte trailer at EOF) is not handled.
"""

from __future__ import annotations

import json
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common import (
    classify_finding_confidence,
    safe_arg,
    safe_write_bytes,
    subprocess_preexec_fn,
    which,
)
from image_meta import (
    AI_META_HINTS,
    XMP_UUID,  # noqa: F401 -- re-exported for callers that want the raw constant
    _contains_any,
    _parse_isobmff_boxes,
    inspect_isobmff,
)

AV_EXTS = {".mp4", ".mov", ".m4a", ".m4v", ".wav", ".mp3"}


@dataclass
class AVInspectReport:
    path: str
    format: str  # mp4 | wav | mp3 | unknown
    has_c2pa: bool
    has_ai_metadata: bool
    findings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "format": self.format,
            "has_c2pa": self.has_c2pa,
            "has_ai_metadata": self.has_ai_metadata,
            "findings": self.findings,
            "findings_confidence": [classify_finding_confidence(f) for f in self.findings],
            "notes": self.notes,
        }


def detect_av_format(data: bytes) -> str:
    """Sniff MP4/MOV/M4A/M4V (ISOBMFF), WAV, or MP3 from magic bytes."""
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "mp4"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if len(data) >= 3 and data[:3] == b"ID3":
        return "mp3"
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return "mp3"  # MPEG frame sync with no ID3v2 header (rare but valid)
    return "unknown"


def _classify_c2pa(hits: list[str]) -> bool:
    return any(h.lower() in ("c2pa", "contentcredentials", "jumb", "contentauth") for h in hits)


# ---------------------------------------------------------------------------
# MP4 / MOV / M4A / M4V (ISOBMFF)
# ---------------------------------------------------------------------------
#
# Top-level C2PA (jumb/c2pa box) and XMP (uuid box) DETECTION reuses
# inspect_isobmff() from image_meta.py unchanged -- that is exactly the
# mechanism the C2PA spec defines for ISOBMFF-family containers, already
# proven correct for AVIF/HEIC. moov/udta (QuickTime "user data", where
# generator/tool tags commonly live) is MP4-specific and handled here.
#
# STRIPPING is a different story from images -- see the patch note below.


def _inspect_moov_udta(data: bytes) -> tuple[bool, bool, list[str]]:
    has_c2pa = False
    has_ai = False
    findings: list[str] = []
    for fourcc, payload, _size, _hdr in _parse_isobmff_boxes(data):
        if fourcc != b"moov":
            continue
        for s_fourcc, s_payload, _s_size, _s_hdr in _parse_isobmff_boxes(payload):
            if s_fourcc != b"udta":
                continue
            hits = _contains_any(s_payload, AI_META_HINTS)
            if hits:
                has_ai = True
                if _classify_c2pa(hits):
                    has_c2pa = True
                findings.append(f"MP4 moov/udta box: {', '.join(hits[:8])}")
    return has_c2pa, has_ai, findings


def _inspect_mp4(data: bytes) -> tuple[bool, bool, list[str]]:
    has_c2pa, has_ai, findings = inspect_isobmff(data, fmt="mp4")
    udta_c2pa, udta_ai, udta_findings = _inspect_moov_udta(data)
    return has_c2pa or udta_c2pa, has_ai or udta_ai, findings + udta_findings


# ---------------------------------------------------------------------------
# LOCAL PATCH (Circe, 2026-08-19) -- not upstream yet.
#
# _strip_mp4 used to edit ISOBMFF boxes in place (drop moov/udta, drop
# top-level uuid/jumb via strip_isobmff) without touching the sample offset
# tables (stco/co64) inside moov/.../stbl, which store ABSOLUTE file offsets
# into mdat. Any box dropped before mdat -- and moov comes before mdat in
# every faststart-optimized file, which is how video gets prepared for web/
# social publishing -- shifts mdat by the size of the dropped box, so every
# stco/co64 entry now points N bytes short of the real sample. The file
# decodes as garbage ("OK pulito" on a container ffmpeg reads as
# `Invalid NAL unit size`) while every metadata-only check in this module
# still reports success, because none of them touch pixel/waveform data.
#
# Fixed by remuxing through ffmpeg instead of hand-editing boxes: ffmpeg's
# muxer builds moov (and stco/co64) from scratch from its own internal state,
# so it cannot reproduce an offset-table desync by construction, and it drops
# any top-level box its demuxer doesn't understand (custom uuid/jumb C2PA
# manifests included) as a side effect of writing a fresh container instead
# of copying bytes forward.
#
# Box editing is now used NOWHERE in the MP4 write path, not even as a
# "drop this then remux to fix it" step: that doesn't work. A remux
# demuxes the file it's given -- if a prior box edit already desynced
# stco/co64, the demuxer reads packets at the (now wrong) offsets and the
# remux faithfully copies forward whatever garbage it finds there. The only
# safe order is "ffmpeg decides what to drop, in the same pass that
# rebuilds the offsets", which is exactly what _remux_mp4_ffmpeg does.
#
# Upstream issue/PR pending -- see the note in clean_av()'s docstring for
# the tracking link once filed.
# ---------------------------------------------------------------------------


def _mp4_format_tags(path: Path) -> dict[str, str]:
    """Format-level metadata tags via ffprobe, or {} if unavailable/unparseable.

    ffprobe exposes the same tag dictionary ffmpeg's muxer writes back on
    remux (moov/udta/meta/ilst atoms it recognizes as key=value pairs) --
    querying it is how _strip_mp4 finds out *which specific keys* to null in
    keep_non_ai_metadata mode, without ever touching a box by hand.
    """
    ffprobe = which("ffprobe")
    if not ffprobe:
        return {}
    try:
        r = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format_tags",
                "-of",
                "json",
                safe_arg(str(path)),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            preexec_fn=subprocess_preexec_fn,
        )
    except Exception:  # noqa: BLE001 -- best-effort probe, caller treats {} as "found nothing"
        return {}
    if r.returncode != 0:
        return {}
    try:
        tags = json.loads(r.stdout).get("format", {}).get("tags", {})
    except (json.JSONDecodeError, AttributeError):
        return {}
    return tags if isinstance(tags, dict) else {}


def _mp4_hint_keys(path: Path) -> tuple[str, ...]:
    tags = _mp4_format_tags(path)
    return tuple(
        k for k, v in tags.items() if _contains_any(str(v).encode("utf-8", "replace"), AI_META_HINTS)
    )


def _remux_mp4_ffmpeg(
    data: bytes, *, drop_all_metadata: bool, drop_keys: tuple[str, ...] = ()
) -> tuple[bytes, list[str]]:
    """Rebuild an MP4/MOV/M4A/M4V container via ffmpeg -c copy.

    Recomputes stco/co64 from scratch (see module-level patch note above).
    ffmpeg's muxer also can't reproduce a top-level box it doesn't
    understand, so any uuid/jumb C2PA or XMP manifest is dropped as a side
    effect of writing a fresh container -- on every call, in every mode,
    with no extra code needed here.

    drop_all_metadata=True clears the whole format-tag dictionary in the same
    pass (-map_metadata -1). Otherwise only the keys named in *drop_keys* are
    nulled (-metadata k=); every other tag passes through untouched. This is
    how keep_non_ai_metadata mode drops just the AI-hint-matching keys
    without a separate box-edit step -- see _strip_mp4 for why a box edit
    before a remux doesn't work.
    """
    ffmpeg = which("ffmpeg")
    if not ffmpeg:
        raise ValueError(
            "ffmpeg is required to clean MP4/MOV/M4A/M4V safely: offset tables "
            "(stco/co64) can only be recomputed by a real remux, and hand-editing "
            "ISOBMFF boxes corrupts them whenever moov precedes mdat (faststart "
            "files, i.e. anything prepared for web/social) -- install ffmpeg"
        )

    with tempfile.TemporaryDirectory(prefix="wm-av-") as td:
        tmp_in = Path(td) / "in.mp4"
        tmp_out = Path(td) / "out.mp4"
        tmp_in.write_bytes(data)
        cmd = [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            safe_arg(str(tmp_in)),
            "-map",
            "0",
            "-c",
            "copy",
        ]
        if drop_all_metadata:
            cmd += ["-map_metadata", "-1"]
        else:
            for k in drop_keys:
                cmd += ["-metadata", f"{k}="]
        cmd.append(safe_arg(str(tmp_out)))
        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
                preexec_fn=subprocess_preexec_fn,
            )
        except subprocess.TimeoutExpired as e:
            raise ValueError(f"ffmpeg remux timed out: {e}") from e
        if r.returncode != 0 or not tmp_out.is_file() or tmp_out.stat().st_size == 0:
            detail = "; ".join((r.stderr or "").strip().splitlines()[:5])
            raise ValueError(f"ffmpeg remux failed (rc={r.returncode}): {detail}")
        cleaned = tmp_out.read_bytes()

    if drop_all_metadata:
        note = (
            "remux via ffmpeg (-map 0 -c copy -map_metadata -1): rebuilt "
            "moov/stco/co64 from scratch, dropped every tag and any "
            "uuid/jumb/C2PA/XMP box not carried through by ffmpeg's own "
            "metadata model"
        )
    elif drop_keys:
        note = (
            "remux via ffmpeg (-map 0 -c copy, selective -metadata clear): rebuilt "
            f"moov/stco/co64, cleared AI-hint-matching tags [{', '.join(drop_keys)}], "
            "dropped any uuid/jumb/C2PA/XMP box, kept unflagged tags"
        )
    else:
        note = (
            "remux via ffmpeg (-map 0 -c copy): rebuilt moov/stco/co64, dropped "
            "any uuid/jumb/C2PA/XMP box, no format tag matched an AI hint"
        )
    return cleaned, [note]


def _strip_mp4(data: bytes, *, strip_all_metadata: bool) -> tuple[bytes, list[str]]:
    # Both branches end in exactly one remux from *data* (or, in the residual
    # fallback below, from that remux's own output) -- never from a
    # box-edited intermediate. See the module patch note above for why.
    if strip_all_metadata:
        cleaned, actions = _remux_mp4_ffmpeg(data, drop_all_metadata=True)
    else:
        with tempfile.TemporaryDirectory(prefix="wm-av-probe-") as td:
            probe = Path(td) / "probe.mp4"
            probe.write_bytes(data)
            drop_keys = _mp4_hint_keys(probe)
        cleaned, actions = _remux_mp4_ffmpeg(data, drop_all_metadata=False, drop_keys=drop_keys)

    # Belt-and-suspenders: the remux above already strips every top-level
    # uuid/jumb/C2PA/XMP box unconditionally (ffmpeg's muxer structurally
    # can't reproduce them) and every AI-hint-matching format tag. If a hint
    # still shows up in the result, it's a signal outside both of those
    # mechanisms (e.g. a stream-level tag ffprobe's format_tags doesn't
    # surface); take one more probe-and-clear pass rather than silently
    # shipping the residue.
    has_c2pa, has_ai, findings = _inspect_mp4(cleaned)
    if has_c2pa or has_ai:
        if strip_all_metadata:
            actions.append(
                f"warning: residual AI/C2PA signal survived full strip: {findings}"
            )
        else:
            with tempfile.TemporaryDirectory(prefix="wm-av-probe2-") as td:
                probe = Path(td) / "probe.mp4"
                probe.write_bytes(cleaned)
                more_keys = _mp4_hint_keys(probe)
            if more_keys:
                cleaned, extra_actions = _remux_mp4_ffmpeg(
                    cleaned, drop_all_metadata=False, drop_keys=more_keys
                )
                actions.append(
                    f"residual metadata found after first pass, cleared: {', '.join(more_keys)}"
                )
                actions += extra_actions
            else:
                actions.append(f"warning: residual AI/C2PA signal outside format tags: {findings}")

    return cleaned, actions


# ---------------------------------------------------------------------------
# ID3v2 (shared by MP3 files and WAV's optional `id3 ` chunk)
# ---------------------------------------------------------------------------


def _id3v2_size(data: bytes, offset: int) -> int:
    b0, b1, b2, b3 = data[offset], data[offset + 1], data[offset + 2], data[offset + 3]
    return ((b0 & 0x7F) << 21) | ((b1 & 0x7F) << 14) | ((b2 & 0x7F) << 7) | (b3 & 0x7F)


def _id3v2_size_bytes(n: int) -> bytes:
    return bytes([(n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F])


def _parse_id3v2_frames(data: bytes) -> tuple[int, int, list[tuple[bytes, bytes]]] | None:
    """Parse an ID3v2 tag at the start of *data*.

    Returns (tag_total_size, major_version, frames); frames is a list of
    (frame_id, frame_payload) for v2.3/v2.4 tags (4-byte frame IDs). v2.2
    tags (3-byte frame IDs) are detected but returned with an empty frame
    list -- callers fall back to whole-tag byte-scanning and whole-tag drop.
    """
    if len(data) < 10 or data[:3] != b"ID3":
        return None
    major = data[3]
    tag_size = _id3v2_size(data, 6)
    total = 10 + tag_size
    if total > len(data):
        return None
    if major < 3:
        return total, major, []

    frames: list[tuple[bytes, bytes]] = []
    pos = 10
    while pos + 10 <= total:
        frame_id = data[pos : pos + 4]
        if frame_id == b"\x00\x00\x00\x00":
            break  # padding
        frame_size = (
            _id3v2_size(data, pos + 4)
            if major == 4
            else struct.unpack(">I", data[pos + 4 : pos + 8])[0]
        )
        frame_start = pos + 10
        frame_end = frame_start + frame_size
        if frame_size < 0 or frame_end > total:
            break
        frames.append((frame_id, data[frame_start:frame_end]))
        pos = frame_end
    return total, major, frames


def _inspect_id3v2(data: bytes) -> tuple[bool, bool, list[str]]:
    parsed = _parse_id3v2_frames(data)
    if parsed is None:
        return False, False, []
    total, major, frames = parsed
    findings: list[str] = []
    has_ai = False
    has_c2pa = False

    if not frames:
        hits = _contains_any(data[:total], AI_META_HINTS)
        if hits:
            has_ai = True
            has_c2pa = _classify_c2pa(hits)
            findings.append(f"ID3v2.{major} tag: {', '.join(hits[:8])}")
        return has_c2pa, has_ai, findings

    for frame_id, payload in frames:
        hits = _contains_any(payload, AI_META_HINTS)
        if hits:
            has_ai = True
            if _classify_c2pa(hits):
                has_c2pa = True
            label = frame_id.decode("latin-1", errors="replace")
            findings.append(f"ID3v2 frame {label}: {', '.join(hits[:8])}")
    return has_c2pa, has_ai, findings


def _strip_id3v2(data: bytes, *, strip_all_metadata: bool) -> tuple[bytes, list[str]]:
    parsed = _parse_id3v2_frames(data)
    if parsed is None:
        return data, []
    total, major, frames = parsed
    rest = data[total:]

    if not frames:
        # v2.2 (undecomposed) or an empty v2.3/2.4 tag: only a whole-tag drop
        # is safe here, since frame boundaries were never decoded.
        if not strip_all_metadata and not _contains_any(data[:total], AI_META_HINTS):
            return data, ["no ID3v2 tag removed (no AI/C2PA markers found)"]
        return rest, [f"drop ID3v2.{major} tag ({total} bytes)"]

    if strip_all_metadata:
        return rest, [f"drop ID3v2.{major} tag ({total} bytes)"]

    kept = bytearray()
    actions: list[str] = []
    for frame_id, payload in frames:
        hits = _contains_any(payload, AI_META_HINTS)
        if hits:
            label = frame_id.decode("latin-1", errors="replace")
            actions.append(f"drop ID3v2 frame {label}: {', '.join(hits[:8])}")
            continue
        size_bytes = (
            _id3v2_size_bytes(len(payload)) if major == 4 else struct.pack(">I", len(payload))
        )
        kept.extend(frame_id + size_bytes + b"\x00\x00" + payload)

    if not actions:
        return data, ["no ID3v2 frames removed (already clean or none matched)"]

    header = bytes([ord("I"), ord("D"), ord("3"), major, 0, 0]) + _id3v2_size_bytes(len(kept))
    return header + bytes(kept) + rest, actions


# ---------------------------------------------------------------------------
# WAV (RIFF)
# ---------------------------------------------------------------------------


def _inspect_wav(data: bytes) -> tuple[bool, bool, list[str]]:
    findings: list[str] = []
    has_ai = False
    has_c2pa = False
    pos = 12  # past "RIFF" + size(4) + "WAVE"
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        csize = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        cstart = pos + 8
        cend = cstart + csize
        if cend > len(data):
            break
        payload = data[cstart:cend]
        if cid == b"LIST" and payload[:4] == b"INFO":
            hits = _contains_any(payload, AI_META_HINTS)
            if hits:
                has_ai = True
                if _classify_c2pa(hits):
                    has_c2pa = True
                findings.append(f"WAV LIST INFO chunk: {', '.join(hits[:8])}")
        elif cid in (b"id3 ", b"ID3 "):
            c2pa, ai, sub_findings = _inspect_id3v2(payload)
            if ai:
                has_ai = True
                has_c2pa = has_c2pa or c2pa
                findings.extend(f"WAV id3 chunk / {f}" for f in sub_findings)
        pos = cend + (csize & 1)  # chunks are word-aligned
    return has_c2pa, has_ai, findings


def _strip_wav(data: bytes, *, strip_all_metadata: bool) -> tuple[bytes, list[str]]:
    actions: list[str] = []
    out = bytearray(data[:12])
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        csize = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        cstart = pos + 8
        cend = cstart + csize
        if cend > len(data):
            out.extend(data[pos:])
            pos = len(data)
            break
        payload = data[cstart:cend]
        pad = csize & 1
        chunk_total = data[pos : cend + pad]

        drop = False
        is_info = cid == b"LIST" and payload[:4] == b"INFO"
        is_id3 = cid in (b"id3 ", b"ID3 ")
        if (is_info or is_id3) and (strip_all_metadata or _contains_any(payload, AI_META_HINTS)):
            actions.append(f"drop WAV {'LIST INFO' if is_info else 'id3'} chunk")
            drop = True

        if not drop:
            out.extend(chunk_total)
        pos = cend + pad

    struct.pack_into("<I", out, 4, len(out) - 8)
    if not actions:
        actions.append("no WAV metadata chunks removed (already clean or none matched)")
    return bytes(out), actions


# ---------------------------------------------------------------------------
# Unified inspect / clean
# ---------------------------------------------------------------------------


def inspect_av(path: Path) -> AVInspectReport:
    data = path.read_bytes()
    fmt = detect_av_format(data)
    if fmt == "mp4":
        has_c2pa, has_ai, findings = _inspect_mp4(data)
    elif fmt == "wav":
        has_c2pa, has_ai, findings = _inspect_wav(data)
    elif fmt == "mp3":
        has_c2pa, has_ai, findings = _inspect_id3v2(data)
    else:
        has_c2pa, has_ai, findings = False, False, ["unsupported format (MP4/MOV/M4A/WAV/MP3)"]

    notes: list[str] = []
    if fmt == "unknown":
        notes.append("format not fully inspected; only MP4/MOV/M4A/WAV/MP3 are supported")

    return AVInspectReport(
        path=str(path),
        format=fmt,
        has_c2pa=has_c2pa,
        has_ai_metadata=has_ai,
        findings=findings,
        notes=notes,
    )


# LOCAL PATCH (Circe, 2026-08-19) -- not upstream yet. See the note above
# _remux_mp4_ffmpeg() for the corruption bug this responds to.
#
# Corruption anti-guard: metadata-only checks (has_c2pa/has_ai_metadata) never
# read a single sample byte, so they cannot detect a desynced offset table --
# the moov/udta bug shipped with every metadata check in this module still
# reporting the output clean. The only check that actually proves the
# container is intact is decoding it. This re-decodes every cleaned AV file
# (not just MP4 -- WAV/MP3 get the same guard on the same principle, even
# though their chunk-based stripping isn't offset-based and isn't known to
# have this bug) before it is ever written to `dest`. On failure the scratch
# file is discarded, `dest` is never touched, and clean_av raises instead of
# returning -- which both call sites (server.py's /clean, clean_file.py's
# `wm-clean`) already turn into an error response/exit code, never "OK".
def _decodable(path: Path) -> tuple[bool, str]:
    """(ok, detail) -- can ffmpeg decode *path* at all, re-checked every call.

    No caching: this is called once on the original and once on the cleaned
    scratch file, a few hundred ms each on the short probe below, and the two
    answers must reflect the actual bytes on disk at call time.
    """
    ffmpeg = which("ffmpeg")
    if not ffmpeg:
        return True, "skipped (ffmpeg not installed)"
    try:
        r = subprocess.run(
            [ffmpeg, "-v", "error", "-i", safe_arg(str(path)), "-t", "3", "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            preexec_fn=subprocess_preexec_fn,
        )
    except Exception as e:  # noqa: BLE001 -- any failure here means "not decodable"
        return False, f"errored: {e}"
    if r.returncode != 0:
        detail = "; ".join((r.stderr or "").strip().splitlines()[:3])
        return False, f"rc={r.returncode}: {detail}"
    return True, "decoded cleanly"


def _verify_av_decodable(original: Path, cleaned_path: Path, fmt: str) -> tuple[bool, str]:
    """True unless cleaning turned a decodable file into a non-decodable one.

    Metadata-only checks never read a sample byte, so none of them can catch
    a desynced offset table -- see the moov/udta incident documented above
    _remux_mp4_ffmpeg, where every check in this module still reported the
    output clean while ffmpeg read garbage NAL units from it. Re-decoding is
    the only check that actually proves the container is intact.

    This blocks on a REGRESSION against the *original*, not on the cleaned
    file failing some absolute standalone standard: a source that was already
    malformed or not real media (a hand-built test fixture, a file some other
    tool half-wrote) shouldn't become uncleanable just because ffmpeg was
    always going to reject it -- only cleaning that breaks a file that used
    to work is treated as *our* bug.
    """
    ffmpeg = which("ffmpeg")
    if not ffmpeg:
        # MP4 cleaning already requires ffmpeg to remux at all (see
        # _remux_mp4_ffmpeg), so this branch is only reachable for WAV/MP3,
        # whose chunk/frame-based stripping doesn't touch offset tables and
        # isn't known to have this failure mode -- degrade to "unverified",
        # not "failed", but say so rather than silently skipping the check.
        return True, f"post-clean decode check skipped for {fmt} (ffmpeg not installed)"

    original_ok, _ = _decodable(original)
    if not original_ok:
        return True, "post-clean decode check skipped (original was not decodable either)"

    cleaned_ok, detail = _decodable(cleaned_path)
    if cleaned_ok:
        return True, "post-clean decode check passed (ffmpeg re-decode, 3s, vs. decodable original)"
    return False, f"cleaning turned a decodable {fmt.upper()} into a non-decodable one ({detail})"


def clean_av(path: Path, dest: Path, *, strip_all_metadata: bool = True) -> dict[str, Any]:
    data = path.read_bytes()
    fmt = detect_av_format(data)
    if fmt == "mp4":
        cleaned, actions = _strip_mp4(data, strip_all_metadata=strip_all_metadata)
    elif fmt == "wav":
        cleaned, actions = _strip_wav(data, strip_all_metadata=strip_all_metadata)
    elif fmt == "mp3":
        cleaned, actions = _strip_id3v2(data, strip_all_metadata=strip_all_metadata)
    else:
        raise ValueError(f"unsupported audio/video format for cleaning: {fmt}")

    # Write to a scratch path and verify BEFORE dest ever sees the bytes --
    # see the anti-corruption guard note above.
    dest.parent.mkdir(parents=True, exist_ok=True)
    scratch = dest.with_name(dest.name + ".verify-tmp")
    safe_write_bytes(scratch, cleaned)
    verified, verify_note = _verify_av_decodable(path, scratch, fmt)
    if not verified:
        scratch.unlink(missing_ok=True)
        raise ValueError(
            f"cleaned {fmt.upper()} failed the post-clean integrity check "
            f"({verify_note}) -- refusing to output a corrupted file; original "
            "left untouched"
        )
    actions.append(verify_note)
    scratch.replace(dest)

    after = inspect_av(dest)
    return {
        "input": str(path),
        "output": str(dest),
        "format": fmt,
        "actions": actions,
        "bytes_in": len(data),
        "bytes_out": len(cleaned),
        "still_has_c2pa": after.has_c2pa,
        "still_has_ai_metadata": after.has_ai_metadata,
        "post_findings": after.findings,
    }
