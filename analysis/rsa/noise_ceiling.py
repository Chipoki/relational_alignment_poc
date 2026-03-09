"""analysis/rsa/noise_ceiling.py – Compute the inter-subject RSA noise ceiling."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .rdm import RDM


class NoiseCeiling:
    """
    Estimates the noise ceiling for RSA correlations across subjects.

    The noise ceiling provides the upper bound for model performance:
    it represents how well you could predict any one subject's RDM
    using the average of the remaining subjects' RDMs.

    Upper bound: correlate each subject's RDM with the mean of ALL subjects.
    Lower bound: correlate each subject's RDM with the mean of OTHERS only.
    """

    def compute(self, rdms: list[RDM]) -> dict[str, float]:
        """
        Compute upper and lower noise ceiling bounds.

        Parameters
        ----------
        rdms : list of RDMs from N subjects (same ROI and state)

        Returns
        -------
        dict with keys "upper" and "lower" containing mean Spearman ρ values
        """
        vecs = np.stack([r.upper_triangle() for r in rdms], axis=0)  # (N, k)
        n = len(rdms)

        upper_corrs, lower_corrs = [], []
        mean_all = vecs.mean(axis=0)

        for i in range(n):
            rho_upper, _ = spearmanr(vecs[i], mean_all)
            upper_corrs.append(rho_upper)

            mean_others = (vecs.sum(axis=0) - vecs[i]) / (n - 1)
            rho_lower, _ = spearmanr(vecs[i], mean_others)
            lower_corrs.append(rho_lower)

        return {
            "upper": float(np.mean(upper_corrs)),
            "lower": float(np.mean(lower_corrs)),
        }
