import pathlib
from excitertools import Iter


def test_open():
    fname = pathlib.Path(__file__)
    x = Iter.open(fname).map(str.strip).filter(None).map(str.upper).collect()
    assert len(x) == 7
    assert x[0] == "IMPORT PATHLIB"
