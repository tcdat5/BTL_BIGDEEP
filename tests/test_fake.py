import pandas as pd

from common.data import BENIGN, ID_COL, LABEL_COL, TS_COL
from common.fake import FAKE_ATTACKS, make_fake_flows, write_fake_raw_csvs


def test_make_fake_flows_shape_and_labels():
    df = make_fake_flows(1000, seed=1)
    assert len(df) == 1000 and df[ID_COL].is_unique
    assert set(df[LABEL_COL]) == {BENIGN, *FAKE_ATTACKS}
    assert (df[LABEL_COL] == BENIGN).mean() > 0.7
    assert df[TS_COL].str.match(r"\d{2}/\d{2}/2018 \d{2}:\d{2}:\d{2}").all()


def test_make_fake_flows_deterministic():
    pd.testing.assert_frame_equal(make_fake_flows(300, seed=7), make_fake_flows(300, seed=7))


def test_write_fake_raw_csvs_has_dirty_rows(tmp_path):
    files = write_fake_raw_csvs(tmp_path, n_per_file=200)
    assert len(files) == 3
    raw = pd.read_csv(tmp_path / files[2], dtype=str, keep_default_na=False)
    assert {"Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp", "Const Col"} <= set(raw.columns)
    assert (raw["Label"] == "Label").sum() == 1
    assert (raw["Flow Byts/s"] == "Infinity").sum() == 1
