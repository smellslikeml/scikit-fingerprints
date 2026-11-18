import numpy as np
from scipy.sparse import csr_array
from sklearn.utils._param_validation import StrOptions, validate_params

from skfp.distances.utils import array_to_csr


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "similarity": [
            StrOptions({"tanimoto", "JT", "russell", "RR", "sokal_michener", "SM"})
        ],
    },
    prefer_skip_nested_validation=True,
)
def isim(
    X: list | np.ndarray | csr_array,
    similarity: str = "tanimoto",
) -> float:
    r"""
    Instant Similarity (iSIM) for a set of binary fingerprints.

    Computes the iSIM index [1]_ [2]_ for a whole set of binary fingerprints, i.e.
    the average similarity between all pairs of rows, using only the column sums.
    This avoids the O(N^2) pairwise calculations, scaling as O(N * n_features)
    with the number of fingerprints.

    For ``N`` fingerprints with ``m`` bits each, let :math:`c_j` be the number of
    fingerprints with bit :math:`j` set. From the column sums we get:

    .. math::

        a = \sum_j \frac{c_j (c_j - 1)}{2}, \qquad
        d = \sum_j \frac{(N - c_j) (N - c_j - 1)}{2}

        dis = \sum_j (N - c_j) c_j, \qquad
        p = \frac{N (N - 1) m}{2}

    where

    - :math:`a` - bit pairs that are both 1, summed over all fingerprint pairs
    - :math:`d` - bit pairs that are both 0, summed over all fingerprint pairs
    - :math:`dis` - mismatched bits, summed over all fingerprint pairs
    - :math:`p` - total number of compared bit pairs

    The supported similarities are:

    .. math::

        iSIM_{RR} = \frac{a}{p}, \qquad
        iSIM_{SM} = \frac{a + d}{p}, \qquad
        iSIM_{JT} = \frac{a}{a + dis}

    The calculated similarity falls within the range :math:`[0, 1]`.

    Notes
    -----
    The Russell ("RR") and Sokal-Michener ("SM") values are exactly equal to the
    average of the corresponding pairwise similarities over all
    :math:`N(N-1)/2` pairs of fingerprints, see :py:func:`russell_binary_similarity`
    and :py:func:`rand_binary_similarity`.

    The Tanimoto ("JT") value is a ratio of the aggregated bit counts, and not the
    average of the pairwise Tanimoto similarities. It approximates the median
    pairwise Tanimoto similarity of the set [1]_.

    Sets with fewer than 2 fingerprints result in NaN, as there are no pairs to
    compare. For the Tanimoto similarity, all-zero sets result in a similarity
    of 1, following the convention of :py:func:`tanimoto_binary_similarity`.

    Parameters
    ----------
    X : {ndarray, CSR sparse array}
        Binary input array or sparse matrix, with fingerprints as rows, of shape
        :math:`N \times m`.

    similarity : {"tanimoto", "JT", "russell", "RR", "sokal_michener", "SM"}, default="tanimoto"
        Similarity index to use: Tanimoto (Jaccard), Russell-Rao, or Sokal-Michener
        (simple matching), with uppercase aliases "JT", "RR", and "SM".

    Returns
    -------
    similarity : float
        iSIM index of the whole set of fingerprints for the chosen similarity.

    References
    ----------
    .. [1] `López-Pérez, K., Kim, T. D., Miranda-Quintana, R. A.
        "iSIM: instant similarity"
        Digital Discovery, 2024, 3, 1160-1171 pp.
        <https://doi.org/10.1039/D4DD00041B>`_

    .. [2] `mqcomplab/iSIM repository
        <https://github.com/mqcomplab/iSIM>`_

    Examples
    --------
    >>> from skfp.distances import isim
    >>> import numpy as np
    >>> X = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
    >>> sim = isim(X, similarity="tanimoto")
    >>> sim
    0.25
    >>> sim = isim(X, similarity="russell")
    >>> sim
    0.2222222222222222
    >>> sim = isim(X, similarity="sokal_michener")
    >>> sim
    0.3333333333333333

    >>> from scipy.sparse import csr_array
    >>> X = csr_array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
    >>> sim = isim(X, similarity="tanimoto")
    >>> sim
    0.25
    """
    X = array_to_csr(X)
    n_samples, n_features = X.shape

    if n_samples < 2:
        # with a single fingerprint there are no pairs to compare
        return float("nan")

    # column sums: number of fingerprints with each bit set
    c = np.asarray(X.sum(axis=0), dtype=np.float64).ravel()

    # (dis)agreements between all fingerprint pairs, summed over bits
    a = np.sum(c * (c - 1) / 2)  # bit pairs where both are 1
    d = np.sum((n_samples - c) * (n_samples - c - 1) / 2)  # bit pairs, both 0
    dis = np.sum((n_samples - c) * c)  # bit pairs where exactly one is 1
    p = n_samples * (n_samples - 1) * n_features / 2  # all compared bit pairs

    if similarity in ("tanimoto", "JT"):
        # all-zero sets have no "on" bits and no mismatches; following the
        # convention of tanimoto_binary_similarity, they are fully similar
        denominator = a + dis
        sim = a / denominator if denominator != 0 else 1
    elif similarity in ("russell", "RR"):
        sim = a / p
    else:  # sokal_michener / SM
        sim = (a + d) / p

    return float(sim)
