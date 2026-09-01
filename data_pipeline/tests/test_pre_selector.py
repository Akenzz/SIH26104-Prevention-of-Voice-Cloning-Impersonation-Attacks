import pandas as pd

from data_pipeline.pipeline.pre_selector import run


def _row(path, label, split_hint, speaker, *, held_out=False):
    return {
        "src_path": path,
        "label": label,
        "speaker_id": speaker,
        "language": "en",
        "generator_id": "none" if label == "bonafide" else "generator-a",
        "source": "asvspoof2019_la",
        "split_hint": split_hint,
        "held_out": held_out,
    }


def test_selector_never_promotes_dev_or_eval_rows_to_train(tmp_path):
    rows = [
        _row("train-bona.wav", "bonafide", "train", "train-speaker"),
        _row("train-spoof.wav", "spoof", "train", "train-spoof-speaker"),
        _row("dev-bona.wav", "bonafide", "dev", "dev-speaker"),
        _row("dev-spoof.wav", "spoof", "dev", "dev-spoof-speaker"),
        _row("eval-bona.wav", "bonafide", "eval", "eval-speaker"),
        _row("eval-spoof.wav", "spoof", "eval", "eval-spoof-speaker"),
        _row("heldout-spoof.wav", "spoof", "train", "heldout-speaker", held_out=True),
    ]
    input_path = tmp_path / "raw_index.csv"
    output_path = tmp_path / "selected.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)

    selected = run(input_path, output_path)
    split_by_path = dict(zip(selected["src_path"], selected["split"]))

    assert split_by_path["dev-bona.wav"] == "dev"
    assert split_by_path["dev-spoof.wav"] == "dev"
    assert split_by_path["eval-bona.wav"] == "eval"
    assert split_by_path["eval-spoof.wav"] == "eval"
    assert split_by_path["heldout-spoof.wav"] == "eval_ood"
    assert not set(selected.loc[selected["split"] == "train", "src_path"]).intersection(
        {"dev-bona.wav", "dev-spoof.wav", "eval-bona.wav", "eval-spoof.wav", "heldout-spoof.wav"}
    )
