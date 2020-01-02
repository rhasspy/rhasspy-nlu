"""Utility methods for rhasspynlu"""
import itertools
import typing


def pairwise(iterable: typing.Iterable[typing.Any]):
    """s -> (s0,s1), (s1,s2), (s2,s3), ..."""
    a, b = itertools.tee(iterable)
    return zip(a, itertools.islice(b, 1, None))
