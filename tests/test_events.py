"""Tests of the 1-anode/1-cathode event building (plan 10.1, synthetic part)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dcalib import events as ev
from tests.synthetic_cache import BoardData, Hits, write_cache

# Board 16 (even): anodes include RENA 0 channel 10 and RENA 1 channel 12,
# cathodes are channels 25-28 of both RENAs.
A1 = (0, 10)
A2 = (1, 12)
C1 = (0, 25)
C2 = (1, 26)


def _events(anode: Hits, cathode: Hits, window: int = 48) -> ev.BoardEvents:
    def polarity(h: Hits) -> ev.PolarityHits:
        return ev.PolarityHits(
            np.asarray(h.rena, np.int8),
            np.asarray(h.channel, np.int8),
            np.asarray(h.pha, np.int16),
            np.asarray(h.cts, np.int64),
            np.asarray(h.source, np.int8),
        )

    return ev.events_from_hits(ev.BoardHits(1, 16, polarity(anode), polarity(cathode)), window)


class TestGreedyClusterIds:
    def test_inclusive_window_and_anchor_rule(self) -> None:
        cts = np.array([0, 48, 49, 97, 98, 98], dtype=np.int64)
        # 48 joins (inclusive); 49 is within 48 of the previous hit but 49 ticks
        # after the anchor, so it anchors cluster 1; 97 = 49 + 48 joins it.
        assert ev.greedy_cluster_ids(cts, 48).tolist() == [0, 0, 1, 1, 2, 2]

    def test_empty_and_zero_window(self) -> None:
        assert ev.greedy_cluster_ids(np.empty(0, np.int64), 48).size == 0
        cts = np.array([5, 5, 6], dtype=np.int64)
        assert ev.greedy_cluster_ids(cts, 0).tolist() == [0, 0, 1]


class TestOneAnodeOneCathode:
    def test_inclusive_edge_at_plus_48(self) -> None:
        anode = Hits().add(*A1, 1000, 100).add(*A1, 1100, 1000)
        cathode = Hits().add(*C1, 700, 148).add(*C1, 710, 1049)
        events = _events(anode, cathode)
        # +48 is in the window; +49 is not (anode-only and cathode-only clusters).
        assert len(events) == 1
        assert events.anode_pha.tolist() == [1000]
        assert events.cathode_pha.tolist() == [700]
        assert events.dcts.tolist() == [48]
        census = events.census[0]
        assert census[1, 1] == 1 and census[1, 0] == 1 and census[0, 1] == 1

    def test_anchor_rule_splits_chained_hits(self) -> None:
        # Cathode at 40 joins the anode's cluster; the anode at 89 is within 48
        # of the cathode but 89 ticks after the anchor, so it starts a new
        # (anode-only) cluster instead of making the first one 2A1C.
        anode = Hits().add(*A1, 1000, 0).add(*A2, 1200, 89)
        cathode = Hits().add(*C1, 600, 40)
        events = _events(anode, cathode)
        assert len(events) == 1
        assert (events.anode_rena[0], events.anode_channel[0]) == A1
        assert events.census[0, 1, 0] == 1  # the anode-only cluster

    def test_multi_hit_and_single_polarity_clusters_dropped(self) -> None:
        anode, cathode = Hits(), Hits()
        # 2A1C at t = 0
        anode.add(*A1, 1000, 0).add(*A2, 1001, 3)
        cathode.add(*C1, 600, 5)
        # 1A2C at t = 1000
        anode.add(*A1, 1002, 1000)
        cathode.add(*C1, 601, 1002).add(*C2, 602, 1010)
        # anode only at t = 2000, cathode only at t = 3000
        anode.add(*A1, 1003, 2000)
        cathode.add(*C1, 603, 3000)
        # 1A1C at t = 4000
        anode.add(*A2, 1004, 4000)
        cathode.add(*C2, 604, 4020)
        events = _events(anode, cathode)
        assert len(events) == 1
        assert events.anode_pha.tolist() == [1004] and events.cathode_pha.tolist() == [604]
        census = events.census[0]
        assert census[2, 1] == 1 and census[1, 2] == 1 and census[1, 1] == 1
        assert census[1, 0] == 1 and census[0, 1] == 1
        assert census.sum() == 5

    def test_sources_clustered_separately(self) -> None:
        # A Ge anode and a Cs cathode 10 ticks apart never pair up.
        anode = Hits().add(*A1, 1000, 100, source=0).add(*A1, 1010, 500, source=1)
        cathode = Hits().add(*C1, 600, 110, source=1).add(*C1, 610, 520, source=1)
        events = _events(anode, cathode)
        assert len(events) == 1
        assert events.source.tolist() == [1]
        assert events.anode_pha.tolist() == [1010]
        assert events.census[0, 1, 0] == 1
        assert events.census[1, 0, 1] == 1 and events.census[1, 1, 1] == 1

    def test_unsorted_input_rows_follow_cts(self) -> None:
        rng = np.random.default_rng(3)
        n = 300
        t = np.cumsum(rng.integers(100, 400, n))
        anode = Hits().extend(0, 10, np.arange(n) + 1000, t, 0)
        cathode = Hits().extend(0, 25, np.arange(n) + 500, t + rng.integers(-48, 49, n), 0)
        perm_a, perm_c = rng.permutation(n), rng.permutation(n)
        shuffled_a = Hits().extend(
            0, 10, np.asarray(anode.pha)[perm_a], np.asarray(anode.cts)[perm_a], 0
        )
        shuffled_c = Hits().extend(
            0, 25, np.asarray(cathode.pha)[perm_c], np.asarray(cathode.cts)[perm_c], 0
        )
        events = _events(shuffled_a, shuffled_c)
        # The cathode may precede its anode (negative dcts); every pair survives.
        assert len(events) == n
        assert np.all(np.diff(events.cts) > 0)
        np.testing.assert_array_equal(events.anode_pha, np.arange(n) + 1000)
        np.testing.assert_array_equal(events.cathode_pha, np.arange(n) + 500)
        assert events.dcts.min() < 0 < events.dcts.max()

    def test_rows_ordered_by_source_then_cts(self) -> None:
        anode = Hits().add(*A1, 1, 900, 1).add(*A1, 2, 100, 0).add(*A2, 3, 50, 1)
        cathode = Hits().add(*C1, 4, 905, 1).add(*C1, 5, 101, 0).add(*C2, 6, 52, 1)
        events = _events(anode, cathode)
        assert events.source.tolist() == [0, 1, 1]
        assert events.cts.tolist() == [100, 50, 900]
        rows = events.anode_rows()
        assert list(rows) == [A1, A2]
        assert rows[A1].tolist() == [0, 2] and rows[A2].tolist() == [1]

    def test_take_and_share(self) -> None:
        anode = Hits().add(*A1, 1, 0).add(*A1, 2, 1000).add(*A2, 3, 2000)
        cathode = Hits().add(*C1, 4, 1).add(*C1, 5, 1001)
        events = _events(anode, cathode)
        assert len(events) == 2
        assert events.one_anode_one_cathode_share(0) == pytest.approx(2 / 3)
        assert np.isnan(events.one_anode_one_cathode_share(1))
        sub = events.take(events.anode_pha == 2)
        assert len(sub) == 1 and sub.census is events.census
        assert events.nbytes > 0


class TestCacheReader:
    def test_build_from_cache_file(self, tmp_path: Path) -> None:
        board = BoardData()
        board.anode.add(*A1, 1000, 100).add(*A2, 1100, 5000, source=1)
        board.cathode.add(*C1, 700, 120).add(*C2, 800, 5010, source=1)
        path = write_cache(tmp_path / "x.cache.h5", {(3, 16): board})
        events = ev.build_board_events(path, 3, 16)
        assert (events.node, events.board, len(events)) == (3, 16, 2)
        assert events.source.tolist() == [0, 1]
        assert events.anode_channel.tolist() == [10, 12]
        assert events.dcts.tolist() == [20, 10]
        assert ev.list_boards(path) == {(3, 16): 4}

    def test_missing_board_and_polarity(self, tmp_path: Path) -> None:
        board = BoardData()
        board.anode.add(*A1, 1000, 100)
        path = write_cache(tmp_path / "x.cache.h5", {(3, 16): board})
        assert len(ev.build_board_events(path, 3, 18)) == 0
        events = ev.build_board_events(path, 3, 16)
        assert len(events) == 0 and events.census[0, 1, 0] == 1
