from data_pipeline.fetch_in_the_wild import _normalise_label


def test_known_in_the_wild_labels_are_mapped_explicitly():
    assert _normalise_label("real") == "bonafide"
    assert _normalise_label("False") == "bonafide"
    assert _normalise_label("spoof") == "spoof"
    assert _normalise_label("true") == "spoof"


def test_unknown_label_is_rejected():
    assert _normalise_label("not-labelled") is None
    assert _normalise_label(None) is None
