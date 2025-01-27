from pathlib import Path
import excitertools


def test_fileinput():
    data = Path(__file__).parent / "fileinput_data.txt"
    output = (
        excitertools.fileinput(files=[data])
        .map(str.upper)
        .concat("")
    )
    assert output == "ONE\nTWO\nTHREE\n"
