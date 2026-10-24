import numpy as np
import pytest

from hand_gesture_recognition.pipeline.feature_extraction.extraction_strategies.frequency_domain import FrequencyDomainFeatures


def test_output_shape_per_channel():
    X = np.random.default_rng(0).normal(0, 1, size=(5, 40, 8)).astype(np.float32)
    out = FrequencyDomainFeatures(sampling_rate=200.0).transform(X)
    assert out.shape == (5, 4 * 8)  # 4 features x 8 channels
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_flat_window_no_nan():
    X = np.zeros((3, 40, 4), dtype=np.float32)  # all-zero -> total power 0
    out = FrequencyDomainFeatures(sampling_rate=200.0).transform(X)
    assert out.shape == (3, 16)
    assert np.isfinite(out).all()
    # total power must be 0 for a flat (DC-removed) window
    assert (out[:, :4] == 0.0).all()


def test_peak_frequency_of_pure_tone():
    fs = 200.0
    n = 100
    t = np.arange(n) / fs
    tone = np.sin(2 * np.pi * 20.0 * t).astype(np.float32)
    X = tone[None, :, None]  # (1, n, 1)
    out = FrequencyDomainFeatures(sampling_rate=fs).transform(X)
    total_power, peak_freq, mean_freq, median_freq = out[0]
    assert total_power > 0
    assert peak_freq == pytest.approx(20.0, abs=2.0)
    assert mean_freq == pytest.approx(20.0, abs=4.0)
    assert median_freq == pytest.approx(20.0, abs=4.0)


def test_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        FrequencyDomainFeatures(sampling_rate=200.0).transform(np.zeros((4, 8)))


def test_rejects_too_few_samples():
    with pytest.raises(ValueError):
        FrequencyDomainFeatures(sampling_rate=200.0).transform(np.zeros((2, 1, 3)))


def test_rejects_nonpositive_sampling_rate():
    with pytest.raises(ValueError):
        FrequencyDomainFeatures(sampling_rate=0.0).transform(np.zeros((2, 8, 3)))
