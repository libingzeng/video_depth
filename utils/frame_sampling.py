#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from collections import namedtuple
from enum import Enum, unique, auto
from typing import Iterable, NamedTuple, Dict, Any, Set
import numpy as np

import pdb

from .frame_range import FrameRange

@unique
class SamplePairsMode(Enum):
    EXHAUSTED = 0
    CONSECUTIVE = auto()
    HIERARCHICAL = auto()
    HIERARCHICAL2 = auto()

    @classmethod
    def name_mode_map(cls):
        return {v.name.lower(): v for v in cls}

    @classmethod
    def names(cls):
        return [v.name.lower() for v in cls]


# param is default to {} while mode is required
class SamplePairsOptions(NamedTuple):
    mode: SamplePairsMode
    params: Dict[str, Any] = {}


Pair = namedtuple("Pair", ["first", "second"])
Pairs_t = Set[Pair]


class SamplePairs:
    @classmethod
    def sample(
        cls,
        opts: Iterable[SamplePairsOptions],
        frame_range: FrameRange,
        two_way=False,
    ) -> Pairs_t:
        num_frames = len(frame_range)

        rel_pairs = set()
        for opt in opts:
            rel_pairs = rel_pairs.union(cls.factory(num_frames, opt, two_way))

        pairs = set()
        for rel_pair in rel_pairs:
            pair = Pair(
                frame_range.index_to_frame[rel_pair[0]],
                frame_range.index_to_frame[rel_pair[1]]
            )
            # Filter out pairs where no end is in depth_frames. Can be optimized
            # when constructing these pairs
            if (pair[0] in frame_range.frames() or pair[1] in frame_range.frames()):
                pairs.add(pair)
        return pairs

    @classmethod
    def sample_2(
        cls,
        opts: Iterable[SamplePairsOptions],
        frame_range: FrameRange,
        two_way=False,
        out_dir=None,
        pos_dist_min=0.03,
    ) -> Pairs_t:
        ##: lbz modified

        meta_path = out_dir + '/metadata_scaled.npz'
        meta = np.load(meta_path)
        extrinsics, intrinsics = meta['extrinsics'], meta['intrinsics']

        num_frames = len(frame_range)

        rel_pairs = set()
        for opt in opts:
            rel_pairs = rel_pairs.union(cls.factory_2(num_frames, opt, two_way, meta, pos_dist_min))

        pairs = set()
        for rel_pair in rel_pairs:
            pair = Pair(
                frame_range.index_to_frame[rel_pair[0]],
                frame_range.index_to_frame[rel_pair[1]]
            )
            # Filter out pairs where no end is in depth_frames. Can be optimized
            # when constructing these pairs
            if (pair[0] in frame_range.frames() or pair[1] in frame_range.frames()):
                pairs.add(pair)
        return pairs

    @classmethod
    def factory(
        cls, num_frames: int, opt: SamplePairsOptions, two_way: bool
    ) -> Pairs_t:
        funcs = {
            SamplePairsMode.EXHAUSTED: cls.sample_exhausted,
            SamplePairsMode.CONSECUTIVE: cls.sample_consecutive,
            SamplePairsMode.HIERARCHICAL: cls.sample_hierarchical,
            SamplePairsMode.HIERARCHICAL2: cls.sample_hierarchical2,
        }

        return funcs[opt.mode](num_frames, two_way, **opt.params)

    @staticmethod
    def sample_hierarchical(
        num_frames: int,
        two_way: bool,
        min_dist=1,
        max_dist=None,
        include_mid_point=False,
    ) -> Pairs_t:
        """
        Args:
            min_dist, max_dist: minimum and maximum distance to the neighbour
        """
        assert min_dist >= 1

        if max_dist is None:
            max_dist = num_frames - 1
        min_level = np.ceil(np.log2(min_dist)).astype(int)
        max_level = np.floor(np.log2(max_dist)).astype(int)

        step_level = (lambda l: max(0, l - 1)) if include_mid_point else (lambda l: l)
        signs = (-1, 1) if two_way else (1,)
        pairs = set()
        for level in range(min_level, max_level + 1):
            dist = 1 << level
            step = 1 << step_level(level)
            for start in range(0, num_frames, step):
                for sign in signs:
                    end = start + sign * dist
                    if end < 0 or end >= num_frames:
                        continue
                    pairs.add(Pair(start, end))
                    # pdb.set_trace()
        return pairs

    @classmethod
    def sample_hierarchical2(
        cls, num_frames: int, two_way: bool, min_dist=1, max_dist=None
    ) -> Pairs_t:
        return cls.sample_hierarchical(
            num_frames,
            two_way,
            min_dist=min_dist,
            max_dist=max_dist,
            include_mid_point=True,
        )

    @classmethod
    def factory_2(
        cls, num_frames: int, opt: SamplePairsOptions, two_way: bool, meta=None, pos_dist_min=0.03,
    ) -> Pairs_t:
        funcs = {
            SamplePairsMode.HIERARCHICAL2: cls.sample_hierarchical2_2,
        }
        # pdb.set_trace()
        return funcs[opt.mode](meta, pos_dist_min, num_frames, two_way, **opt.params)

    @classmethod
    def sample_hierarchical2_2(
        cls, meta, pos_dist_min, num_frames: int, two_way: bool, min_dist=1, max_dist=None
    ) -> Pairs_t:
        return cls.sample_hierarchical_2(
            meta,
            pos_dist_min,
            num_frames,
            two_way,
            min_dist=min_dist,
            max_dist=max_dist,
            include_mid_point=True,
        )

    @staticmethod
    def sample_hierarchical_2(
        meta,
        pos_dist_min,
        num_frames: int,
        two_way: bool,
        min_dist=1,
        max_dist=None,
        include_mid_point=False,
    ) -> Pairs_t:
        """
        Args:
            min_dist, max_dist: minimum and maximum distance to the neighbour
        """
        assert min_dist >= 1

        if max_dist is None:
            max_dist = num_frames - 1
        min_level = np.ceil(np.log2(min_dist)).astype(int)
        max_level = np.floor(np.log2(max_dist)).astype(int)

        step_level = (lambda l: max(0, l - 1)) if include_mid_point else (lambda l: l)
        signs = (-1, 1) if two_way else (1,)
        pairs = set()
        for level in range(min_level, max_level + 1):
            dist1 = 1 << level
            dist2 = 1 << (level + 1)
            # step = 1 << step_level(level)
            for start in range(0, num_frames, 1):
                # if start == 1:
                #     pdb.set_trace()
                for sign in signs:
                    for end in range(start + sign * dist1, start + sign * dist2, int(sign*(dist1 + dist2)/2)):
                        if end < 0 or end >= num_frames:
                            continue
                        else:
                            camera_pos_dist = np.linalg.norm(meta['extrinsics'][start, ...][..., -1:] - meta['extrinsics'][end, ...][..., -1:])
                            if camera_pos_dist > pos_dist_min: # pos_dist_min is 0.03 by default
                                pairs.add(Pair(start, end))
                                pairs.add(Pair(end, start))
                                break
        # pdb.set_trace()
        return pairs

    @classmethod
    def sample_consecutive(cls, num_frames: int, two_way: bool) -> Pairs_t:
        return cls.sample_hierarchical(num_frames, two_way, min_dist=1, max_dist=1)

    @staticmethod
    def sample_exhausted(cls, num_frames: int, two_way: bool) -> Pairs_t:
        second_frame_range = (
            (lambda i, N: range(N)) if two_way else (lambda i, N: range(i + 1, N))
        )

        pairs = set()
        for i in range(num_frames):
            for j in second_frame_range(i, num_frames):
                if i != j:
                    pairs.add(Pair(i, j))
        return pairs

    @classmethod
    def to_one_way(cls, pairs) -> Pairs_t:
        def ordered(pair):
            if pair[0] > pair[1]:
                return Pair(*pair[::-1])
            return Pair(*pair)

        return {ordered(p) for p in pairs}


def to_in_range(pairs, frame_range=None):
    if frame_range is None:
        return pairs

    def in_range(idx):
        return frame_range[0] <= idx and idx < frame_range[1]

    return [pair for pair in pairs if all(in_range(i) for i in pair)]
