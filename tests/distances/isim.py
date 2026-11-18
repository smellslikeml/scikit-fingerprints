from collections.abc import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_array

from skfp.distances import (
    isim,
    rand_binary_similarity,
    russell_binary_similarity,
    tanimoto_binary_similarity,
)
from skfp.fingerprints import ECFPFingerprint

SIMILARITIES = ["tanimoto", "russell", "sokal_michener"]


def _get_random_binary_matrices() -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    return [
        rng.integers(0, 2, size=(n_samples, n_features)).astype(np.uint8)
        for n_samples, n_features in [(3, 5), (5, 32), (10, 64), (20, 128)]
    ]


def _mean_pairwise_similarity(
    X: np.ndarray, pairwise_similarity_func: Callable
) -> float:
    """Brute-force mean of pairwise similarities over all N(N-1)/2 pairs."""
    n_samples = len(X)
    sims = [
        pairwise_similarity_func(X[i], X[j])
        for i in range(n_samples)
        for j in range(i + 1, n_samples)
    ]
    return float(np.mean(sims))


def _brute_force_tanimoto_ratio_of_sums(X: np.ndarray) -> float:
    """Brute-force iSIM Tanimoto: ratio of summed "both 1" to summed mismatches."""
    n_samples = len(X)
    a_total = 0
    dis_total = 0
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            a_total += np.sum(np.logical_and(X[i], X[j]))
            dis_total += np.sum(np.logical_xor(X[i], X[j]))

    return float(a_total / (a_total + dis_total))


def _reference_calculate_isim(X: np.ndarray, n_ary: str) -> float:
    """Re-implementation of calculate_isim from the reference iSIM package."""
    c_total = np.sum(X, axis=0)
    n_objects = len(X)
    a = np.sum(c_total * (c_total - 1) / 2)

    if n_ary == "RR":
        p = n_objects * (n_objects - 1) * len(c_total) / 2
        return a / p
    elif n_ary == "JT":
        total_dis = np.sum((n_objects - c_total) * c_total)
        return a / (a + total_dis)
    else:  # SM
        off_coincidences = n_objects - c_total
        d = np.sum(off_coincidences * (off_coincidences - 1) / 2)
        p = n_objects * (n_objects - 1) * len(c_total) / 2
        return (a + d) / p


@pytest.mark.parametrize("X", _get_random_binary_matrices())
def test_isim_russell_and_sokal_michener_are_exact_mean_pairwise(X):
    # RR and SM divide by the constant n_features per pair, so iSIM is exactly
    # the mean over all N(N-1)/2 pairwise similarities
    assert_allclose(
        isim(X, "russell"), _mean_pairwise_similarity(X, russell_binary_similarity)
    )
    # Sokal-Michener (simple matching) is the pairwise Rand similarity
    assert_allclose(
        isim(X, "sokal_michener"), _mean_pairwise_similarity(X, rand_binary_similarity)
    )


@pytest.mark.parametrize("X", _get_random_binary_matrices())
def test_isim_tanimoto_is_ratio_of_sums(X):
    # iSIM Tanimoto aggregates bit counts over all pairs; it is NOT the mean
    # of the pairwise Tanimoto similarities
    assert_allclose(isim(X, "tanimoto"), _brute_force_tanimoto_ratio_of_sums(X))


def test_isim_tanimoto_differs_from_mean_of_pairwise_tanimoto():
    # mean of pairwise Tanimoto values (4/9) differs from the aggregate ratio
    # of sums (3/7); the two must not be confused
    X = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1]])

    mean_pairwise = _mean_pairwise_similarity(X, tanimoto_binary_similarity)
    assert_allclose(mean_pairwise, 4 / 9)
    assert_allclose(isim(X, "tanimoto"), 3 / 7)
    assert not np.isclose(isim(X, "tanimoto"), mean_pairwise)


@pytest.mark.parametrize(
    "similarity, n_ary",
    [("tanimoto", "JT"), ("russell", "RR"), ("sokal_michener", "SM")],
)
@pytest.mark.parametrize("X", _get_random_binary_matrices())
def test_isim_matches_reference_implementation(X, similarity, n_ary):
    assert_allclose(isim(X, similarity), _reference_calculate_isim(X, n_ary))


@pytest.mark.parametrize("similarity", SIMILARITIES)
@pytest.mark.parametrize("X", _get_random_binary_matrices())
def test_isim_dense_list_and_sparse_parity(X, similarity):
    X_sparse = csr_array(X)
    X_list = X.tolist()
    assert_allclose(isim(X, similarity), isim(X_sparse, similarity))
    assert_allclose(isim(X, similarity), isim(X_list, similarity))


@pytest.mark.parametrize(
    "similarity, pairwise_func",
    [
        ("tanimoto", tanimoto_binary_similarity),
        ("russell", russell_binary_similarity),
        ("sokal_michener", rand_binary_similarity),
    ],
)
def test_isim_two_samples_reduces_to_pairwise(similarity, pairwise_func):
    # with N=2 there is a single pair, so iSIM equals the pairwise index
    X = np.array([[1, 0, 1, 1, 0], [1, 1, 0, 1, 0]])
    assert_allclose(isim(X, similarity), pairwise_func(X[0], X[1]))

    X = np.array([[1, 0, 1, 1, 0], [0, 1, 1, 0, 1]])
    assert_allclose(isim(X, similarity), pairwise_func(X[0], X[1]))


def test_isim_all_zero_set():
    X = np.zeros((4, 8))

    # no shared "on" bits, but also no mismatches; all-zero fingerprints are
    # identical, so Tanimoto similarity is 1 by the library convention
    assert_allclose(isim(X, "tanimoto"), 1.0)
    # Russell similarity counts shared "on" bits, of which there are none
    assert_allclose(isim(X, "russell"), 0.0)
    assert_allclose(isim(X, "sokal_michener"), 1.0)


@pytest.mark.parametrize("similarity", SIMILARITIES)
def test_isim_single_sample_is_nan(similarity):
    # a single fingerprint leaves no pairs to average over
    X = np.array([[1, 0, 1]])
    assert np.isnan(isim(X, similarity))

    X = csr_array([[0, 0, 0]])
    assert np.isnan(isim(X, similarity))


def test_isim_single_column():
    # N=3 with a single bit: one pair shares the "on" bit, two pairs mismatch
    X = np.array([[1], [1], [0]])
    assert_allclose(isim(X, "tanimoto"), 1 / 3)
    assert_allclose(isim(X, "russell"), 1 / 3)
    assert_allclose(isim(X, "sokal_michener"), 1 / 3)

    X = np.array([[0], [1]])
    assert_allclose(isim(X, "tanimoto"), 0.0)
    assert_allclose(isim(X, "russell"), 0.0)
    assert_allclose(isim(X, "sokal_michener"), 0.0)


def test_isim_aliases():
    X = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1]])
    assert_allclose(isim(X, "JT"), isim(X, "tanimoto"))
    assert_allclose(isim(X, "RR"), isim(X, "russell"))
    assert_allclose(isim(X, "SM"), isim(X, "sokal_michener"))


def test_isim_invalid_similarity():
    X = np.array([[1, 0, 1], [1, 1, 0]])
    with pytest.raises(ValueError, match="similarity"):
        isim(X, "dice")
    with pytest.raises(ValueError, match="similarity"):
        isim(X, 42)


def test_isim_ecfp_fingerprints(mols_list):
    fps = ECFPFingerprint().transform(mols_list[:10])

    assert_allclose(isim(fps, "tanimoto"), _brute_force_tanimoto_ratio_of_sums(fps))
    assert_allclose(
        isim(fps, "russell"), _mean_pairwise_similarity(fps, russell_binary_similarity)
    )
    assert_allclose(
        isim(fps, "sokal_michener"),
        _mean_pairwise_similarity(fps, rand_binary_similarity),
    )
    assert_allclose(isim(fps), isim(csr_array(fps)))
