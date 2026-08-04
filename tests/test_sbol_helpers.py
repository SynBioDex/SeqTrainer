from pathlib import Path

from seqtrainer.data.sbol import get_sequence_from_sbol


def test_get_sequence_from_sbol_fixture():
    fixture = Path("data/sbol_data/sample_design_0.xml")
    sequence = get_sequence_from_sbol(fixture)
    assert isinstance(sequence, str)
    assert len(sequence) > 0
