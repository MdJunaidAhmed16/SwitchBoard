"""Attempt 2 machinery: balanced weights, weighted fits, calibration and the fine-tuned v1."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from switchboard.router.attempts import ATTEMPTS, balanced_weights
from switchboard.router.baselines import HeuristicRouter
from switchboard.router.calibrate import apply_temperature, fit_temperature
from switchboard.router.data import RouterData
from switchboard.router.evaluate import _prepare
from switchboard.router.metrics import auroc, ece
from switchboard.router.v1 import V1Router


def _train() -> pd.DataFrame:
    # Benchmark "a" is 90% correct, "b" 20% correct, "c" all correct.
    rows = [("a", True)] * 90 + [("a", False)] * 10 + [("b", True)] * 4 + [("b", False)] * 16
    rows += [("c", True)] * 5
    df = pd.DataFrame(rows, columns=["benchmark", "label"])
    return df.assign(router_text=[f"text {i}" for i in range(len(df))])


# --- Balanced weights ---------------------------------------------------------------------------


def test_every_benchmark_gets_equal_total_weight() -> None:
    df = _train()
    totals = balanced_weights(df).groupby(df["benchmark"]).sum()
    assert np.allclose(totals, totals.iloc[0])


def test_within_a_benchmark_both_classes_get_equal_weight() -> None:
    df = _train()
    w = balanced_weights(df)
    for bench in ("a", "b"):
        part = df["benchmark"] == bench
        assert w[part & df["label"]].sum() == pytest.approx(w[part & ~df["label"]].sum())


def test_benchmark_identity_carries_no_label_information_under_the_weights() -> None:
    df = _train()
    w = balanced_weights(df)
    for bench in ("a", "b"):
        part = df["benchmark"] == bench
        weighted_rate = (w[part] * df.loc[part, "label"]).sum() / w[part].sum()
        assert weighted_rate == pytest.approx(0.5)


def test_weights_average_one_and_single_class_benchmarks_are_handled() -> None:
    w = balanced_weights(_train())
    assert w.mean() == pytest.approx(1.0)
    assert np.isfinite(w).all()


def test_weighted_heuristic_fit_runs() -> None:
    df = _train()
    router = HeuristicRouter()
    router.fit(df.assign(weight=balanced_weights(df).to_numpy()), df)
    assert router.predict_proba(["text 1"]).shape == (1,)


# --- Attempt preparation ------------------------------------------------------------------------


def _data(names: dict[str, str] | None = None) -> RouterData:
    train = _train()
    train["benchmark"] = train["benchmark"].map(names or {"a": "gsm8k", "b": "mmlu", "c": "qasc"})
    val = train.head(5).assign(benchmark="arc_challenge")
    return RouterData(train=train, val=val, test=val.assign(benchmark="math"))


def test_attempt_1_trains_only_on_its_three_benchmarks_without_weights() -> None:
    full = _data({"a": "gsm8k", "b": "mmlu", "c": "mbpp"})
    extra = full.train.head(3).assign(benchmark="qasc")
    data = RouterData(pd.concat([full.train, extra], ignore_index=True), full.val, full.test)
    prepared = _prepare(data, ATTEMPTS[1])
    assert set(prepared.train["benchmark"]) == {"gsm8k", "mmlu", "mbpp"}
    assert "weight" not in prepared.train.columns


def test_attempt_1_refuses_to_run_without_its_benchmarks() -> None:
    with pytest.raises(ValueError, match="needs labels"):
        _prepare(_data(), ATTEMPTS[1])


def test_attempt_2_uses_every_train_benchmark_with_weights() -> None:
    data = _prepare(_data(), ATTEMPTS[2])
    assert set(data.train["benchmark"]) == {"gsm8k", "mmlu", "qasc"}
    assert "weight" in data.train.columns


# --- Temperature scaling -------------------------------------------------------------------------


def _overconfident(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    true_logit = rng.normal(0, 1, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-true_logit))).astype(int)
    return true_logit * 4.0, y  # logits four times too sharp


def test_temperature_fixes_overconfidence_without_changing_ranking() -> None:
    logits, y = _overconfident()
    t = fit_temperature(logits, y)
    assert 3.0 < t < 5.0
    before, after = apply_temperature(logits, 1.0), apply_temperature(logits, t)
    assert ece(y, after) < ece(y, before)
    assert auroc(y, after) == pytest.approx(auroc(y, before))


def test_temperature_below_one_for_underconfident_logits() -> None:
    logits, y = _overconfident()
    assert fit_temperature(logits / 16.0, y) < 1.0


# --- v1 on a tiny local BERT (no download) --------------------------------------------------------


WORDS = ["easy", "hard", "question", "about", "maths", "code", "the", "a"]


def _tiny_backbone(tmp_path: Path) -> Any:
    from transformers import BertConfig, BertModel, BertTokenizerFast

    # An in-memory vocabulary: with transformers 5, a vocab_file path here silently maps every
    # word to [UNK], which makes every input identical and any model unable to learn.
    vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS])}

    def load() -> tuple[Any, Any]:
        tokenizer = BertTokenizerFast(vocab=vocab)
        config = BertConfig(vocab_size=5 + len(WORDS), hidden_size=32, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=64,
                            max_position_embeddings=64)  # fmt: skip
        return tokenizer, BertModel(config)

    return load


def _toy(n: int = 32) -> pd.DataFrame:
    texts = [f"{'easy' if i % 2 else 'hard'} question about {'maths' if i % 3 else 'code'}"
             for i in range(n)]  # fmt: skip
    return pd.DataFrame({"router_text": texts, "label": [bool(i % 2) for i in range(n)]})


def test_tiny_tokenizer_distinguishes_words(tmp_path: Path) -> None:
    tokenizer, _ = _tiny_backbone(tmp_path)()
    ids = tokenizer(["easy question", "hard question"])["input_ids"]
    assert ids[0] != ids[1]
    assert 1 not in ids[0]  # no [UNK]


def test_v1_can_overfit_one_batch(tmp_path: Path) -> None:
    # 13-testing-and-reports: the model must drive loss near zero on 32 examples.
    batch = _toy(32)
    router = V1Router(_tiny_backbone(tmp_path), seed=0, max_epochs=60, batch_size=32,
                      lr_encoder=1e-3, lr_head=1e-2, warmup_fraction=0.05, patience=100,
                      device="cpu")  # fmt: skip
    router.fit(batch, batch)
    assert router.history[-1]["train_loss"] < 0.1
    assert auroc(batch["label"], router.predict_proba(list(batch["router_text"]))) == 1.0


def test_v1_temperature_is_fitted_on_validation_only(tmp_path: Path) -> None:
    train, val = _toy(32), _toy(16)
    router = V1Router(_tiny_backbone(tmp_path), seed=0, max_epochs=3, batch_size=16,
                      lr_encoder=1e-3, lr_head=1e-2, device="cpu")  # fmt: skip
    router.fit(train, val)
    expected = fit_temperature(router.logits(list(val["router_text"])), val["label"].astype(int))
    assert router.temperature == pytest.approx(expected)
    assert router.best_epoch is not None


def test_v1_is_deterministic_for_a_seed(tmp_path: Path) -> None:
    def run() -> np.ndarray:
        r = V1Router(_tiny_backbone(tmp_path), seed=3, max_epochs=2, batch_size=8, device="cpu")
        r.fit(_toy(16), _toy(16))
        return r.predict_proba(list(_toy(8)["router_text"]))

    assert np.allclose(run(), run())
