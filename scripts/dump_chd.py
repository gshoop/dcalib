#!/usr/bin/env python3
"""Write extractData ``.chd`` files from the adc2kev cache (plan section 10.3).

The legacy ``Dcalib`` reads one ``.chd`` text file per anode. This script writes
them from the same 1-anode/1-cathode events that ``dcalib legacy`` uses
(:func:`dcalib.events.build_board_events`), so the ROOT binary and the replica
see exactly the same events::

    venv/bin/python scripts/dump_chd.py --boards 1:17,5:22,3:20 --out /path/to/chd
    Dcalib_linux -p /path/to/chd calibration.kev   # writes chd/Dcalib_output/chd.dcc

File name: ``node%dboard%02drena%02dchannel%02d.chd`` (extractData's
``ChannelWriter::filenameFor``), e.g. ``node1board17rena00channel09.chd``.

Content: two lines per event, the anode first, each ``node board rena channel
polarity PHA`` with ``polarity`` 1 (anode) or 0 (cathode) and the raw PHA,
written with ``"%d %d %d %d %d %d\\n"`` as ``exportPhotoEvt`` does, no header,
in CTS order. Every anode with at least one event of the source gets a file,
calibrated or not (Dcalib skips the uncalibrated ones itself).

``Dcalib`` treats a directory argument containing a ``.`` anywhere in its path
as an unrecognised argument, and derives the ``.dcc`` name from the part of the
directory name before ``inputs`` when it contains that word; the script refuses
such output paths. The output directory must not contain ``.chd`` files
already (``--overwrite`` deletes them first), since Dcalib reads every
``.chd`` file in it.

Exit codes: 0 on success, 1 if nothing could be written (no events on the
boards, an I/O error), 2 for bad arguments or a missing cache.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from dcalib.events import build_board_events
from dcalib.options import SOURCE_CS, SOURCE_GE

DEFAULT_CACHE = Path.home() / "adc2kev-test-data/full-system/data_20260911_124617.cache.h5"
DEFAULT_BOARDS = "1:17,5:22,3:20"
SOURCES = {"ge": SOURCE_GE, "cs": SOURCE_CS}


def parse_boards(text: str) -> list[tuple[int, int]]:
    """Parse ``node:board[,node:board...]``.

    Raises:
        argparse.ArgumentTypeError: On a malformed entry.
    """
    boards: list[tuple[int, int]] = []
    for part in text.split(","):
        try:
            node, board = (int(v) for v in part.strip().split(":"))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"expected node:board, got {part!r}") from exc
        boards.append((node, board))
    if not boards:
        raise argparse.ArgumentTypeError("no boards given")
    return boards


def chd_filename(node: int, board: int, rena: int, channel: int) -> str:
    """extractData's per-anode file name."""
    return f"node{node}board{board:02d}rena{rena:02d}channel{channel:02d}.chd"


def format_chd(
    node: int,
    board: int,
    anode_rena: int,
    anode_channel: int,
    anode_pha: Sequence[int],
    cathode_rena: Sequence[int],
    cathode_channel: Sequence[int],
    cathode_pha: Sequence[int],
) -> str:
    """Return one anode's ``.chd`` text (anode line, then cathode line, per event)."""
    lines = []
    for a_pha, c_rena, c_channel, c_pha in zip(
        anode_pha, cathode_rena, cathode_channel, cathode_pha
    ):
        lines.append(f"{node} {board} {anode_rena} {anode_channel} 1 {int(a_pha)}\n")
        lines.append(f"{node} {board} {int(c_rena)} {int(c_channel)} 0 {int(c_pha)}\n")
    return "".join(lines)


def dump_board(
    cache: Path, node: int, board: int, out: Path, source: int, cts_window: int
) -> tuple[int, int]:
    """Write the ``.chd`` files of one board.

    Returns:
        ``(files written, events written)``.
    """
    events = build_board_events(cache, node, board, cts_window)
    events = events.take(events.source == source)
    n_files = 0
    for (rena, channel), rows in events.anode_rows().items():
        text = format_chd(
            node,
            board,
            rena,
            channel,
            events.anode_pha[rows].tolist(),
            events.cathode_rena[rows].tolist(),
            events.cathode_channel[rows].tolist(),
            events.cathode_pha[rows].tolist(),
        )
        (out / chd_filename(node, board, rena, channel)).write_text(text, encoding="ascii")
        n_files += 1
    return n_files, len(events)


def check_out_dir(out: Path, overwrite: bool) -> str | None:
    """Return an error message if Dcalib could not use ``out`` (or it holds .chd files)."""
    resolved = out.resolve()
    if "." in str(resolved):
        return f"Dcalib rejects a directory path containing '.': {resolved}"
    if "inputs" in resolved.name:
        return f"Dcalib names the .dcc from the text before 'inputs': {resolved.name}"
    if out.exists() and not out.is_dir():
        return f"{out} exists and is not a directory"
    stale = sorted(out.glob("*.chd")) if out.is_dir() else []
    if stale and not overwrite:
        return f"{out} already has {len(stale)} .chd files (use --overwrite to replace them)"
    for path in stale:
        path.unlink()
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dump_chd.py", description="Write extractData .chd files from the adc2kev cache."
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="adc2kev cache")
    parser.add_argument(
        "--boards",
        type=parse_boards,
        default=parse_boards(DEFAULT_BOARDS),
        help=f"node:board list (default {DEFAULT_BOARDS})",
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--source", choices=sorted(SOURCES), default="ge", help="default ge")
    parser.add_argument("--cts-window", type=int, default=48, help="clustering window (48)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing .chd files")
    args = parser.parse_args(argv)

    if not args.cache.is_file():
        print(f"error: cache not found: {args.cache}", file=sys.stderr)
        return 2
    error = check_out_dir(args.out, args.overwrite)
    if error is not None:
        print(f"error: {error}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    total_files = total_events = 0
    try:
        for node, board in args.boards:
            n_files, n_events = dump_board(
                args.cache, node, board, args.out, SOURCES[args.source], args.cts_window
            )
            print(f"n{node} b{board}: {n_files} files, {n_events:,} events")
            total_files += n_files
            total_events += n_events
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Wrote {total_files} .chd files ({total_events:,} {args.source} events) to {args.out} "
        f"in {time.perf_counter() - t0:.1f} s"
    )
    return 0 if total_files else 1


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
