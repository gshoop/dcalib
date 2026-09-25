"""Tests of the channel sets, labels and strip positions."""

from __future__ import annotations

import pytest

from dcalib import channels


@pytest.mark.parametrize("board", [15, 16])
def test_counts_and_partition(board: int) -> None:
    anodes = channels.anode_channels(board)
    cathodes = channels.cathode_channels(board)
    assert len(anodes) == channels.ANODES_PER_BOARD == 39
    assert len(cathodes) == channels.CATHODES_PER_BOARD == 8
    assert not set(anodes) & set(cathodes)
    assert all(channels.is_active_channel(r, c) for r, c in anodes + cathodes)
    assert list(anodes) == sorted(anodes)
    lut = channels.cathode_mask_lut(board)
    assert int(lut.sum()) == 8 and all(lut[r, c] for r, c in cathodes)


def test_cathode_rule_matches_legacy() -> None:
    even = {(r, c) for r in (0, 1) for c in range(25, 29)}
    odd = {(0, c) for c in range(4, 8)} | {(1, c) for c in range(7, 11)}
    assert set(channels.cathode_channels(16)) == even
    assert set(channels.cathode_channels(15)) == odd


def test_labels_and_strips() -> None:
    labels = {channels.electrode_label(16, r, c) for r, c in channels.anode_channels(16)}
    assert labels == {f"A{i:02d}" for i in range(1, 40)}
    for board in (15, 16):
        by_strip = channels.anodes_by_strip(board)
        assert sorted(by_strip) == list(channels.anode_channels(board))
        for position, (rena, channel) in enumerate(by_strip, start=1):
            assert channels.strip_position(board, rena, channel) == position
    assert channels.is_cathode(16, 0, 25) and not channels.is_cathode(16, 0, 10)


def test_active_channels_and_nodes() -> None:
    assert channels.is_active_channel(0, 4) and channels.is_active_channel(1, 28)
    assert not channels.is_active_channel(0, 3) and not channels.is_active_channel(1, 6)
    assert not channels.is_active_channel(2, 10)
    assert tuple(range(1, 11)) == channels.ACTIVE_NODES


def test_anode_key() -> None:
    key = channels.AnodeKey(1, 17, 0, 9)
    assert str(key) == "n1 b17 r0 ch9"
    assert sorted([channels.AnodeKey(2, 15, 0, 4), key]) == [key, channels.AnodeKey(2, 15, 0, 4)]
    assert key == (1, 17, 0, 9)
