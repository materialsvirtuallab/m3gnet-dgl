import unittest

import numpy as np
import torch

from megnet.utils.math import (
    SPHERICAL_BESSEL_ROOTS,
    Gaussian,
    SphericalBesselFunction,
    SphericalHarmonicsFunction,
    combine_sbf_shf,
    spherical_bessel_roots,
    spherical_bessel_smooth,
)


class TestMath(unittest.TestCase):
    def test_spherical_bessel_roots(self):
        roots = spherical_bessel_roots(max_l=1, max_n=5)
        roots2 = SPHERICAL_BESSEL_ROOTS
        self.assertTrue(np.allclose(roots2[0, :5], roots.ravel()))

    def test_gaussian(self):
        centers = np.linspace(1.0, 5.0, 10)
        width = 0.5
        gf = Gaussian(centers=centers, width=width)
        r = np.linspace(1.0, 4.0, 10)
        self.assertTupleEqual(gf(r).numpy().shape, (10, 10))

    def test_spherical_bessel_harmonics_function(self):
        r = torch.empty(10).normal_()
        sbf = SphericalBesselFunction(max_l=3, cutoff=5.0, max_n=3, smooth=False)
        res = sbf(r)
        res2 = sbf.rbf_j0(r, cutoff=5.0, max_n=3)
        self.assertTrue(np.allclose(res[:, :3].numpy().ravel(), res2.numpy().ravel()))

        self.assertTrue(res.numpy().shape == (10, 9))

        sbf2 = SphericalBesselFunction(max_l=3, cutoff=5.0, max_n=3, smooth=True)

        res2 = sbf2(r)
        self.assertTupleEqual(tuple(res2.shape), (10, 3))

        shf = SphericalHarmonicsFunction(max_l=3, use_phi=True)
        res_shf = shf(costheta=np.linspace(-1, 1, 10), phi=np.linspace(0, 2 * np.pi, 10))

        self.assertTrue(res_shf.numpy().shape == (10, 9))
        combined = combine_sbf_shf(res, res_shf, max_n=3, max_l=3, use_phi=True)

        self.assertTrue(combined.shape == (10, 27))

        res_shf2 = SphericalHarmonicsFunction(max_l=3, use_phi=False)(
            costheta=np.linspace(-1, 1, 10), phi=np.linspace(0, 2 * np.pi, 10)
        )
        combined = combine_sbf_shf(res, res_shf2, max_n=3, max_l=3, use_phi=False)

        self.assertTrue(combined.shape == (10, 9))
        rdf = spherical_bessel_smooth(r, cutoff=5.0, max_n=3)
        self.assertTrue(rdf.numpy().shape == (10, 3))


if __name__ == "__main__":
    unittest.main()
