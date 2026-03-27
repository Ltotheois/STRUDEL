# -*- coding: utf-8 -*-
#!/usr/bin/env python3

# Author: Luis Bonah
# Description : Tests for mol-strudel library

import unittest
import mol_strudel as strudel
import numpy as np


m_H = 1.0078250321
m_D = 2.01410175
m_C12 = 12.0000000000
m_C13 = 13.0033548378


# Structure
def fzmat(params):
    r1, r2, r3, r4, r5, a1, a2, a3, a4, a5 = params
    zmat = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0],
            [2, 1, 0, 1, 90, 0],
            [3, 2, 1, r1, a1, 90],
            [3, 2, 4, r1, a1, 180],
            [4, 3, 5, r2, a2, 0],
            [5, 3, 4, r2, a2, 0],
            [4, 3, 2, r3, a3, 180],
            [5, 3, 2, r3, a3, 180],
            [6, 4, 8, r4, a4, 0],
            [7, 5, 9, r4, a4, 0],
            [3, 2, 1, r5, a5, 0],
            [3, 2, 12, r5, a5, 180],
        ]
    )
    return zmat


masses = np.array(
    [
        [0, 0, m_C12, m_C12, m_C12, m_C12, m_C12, m_H, m_H, m_H, m_H, m_H, m_H],
        [0, 0, m_C13, m_C12, m_C12, m_C12, m_C12, m_H, m_H, m_H, m_H, m_H, m_H],
        [0, 0, m_C12, m_C13, m_C12, m_C12, m_C12, m_H, m_H, m_H, m_H, m_H, m_H],
        [0, 0, m_C12, m_C12, m_C12, m_C13, m_C12, m_H, m_H, m_H, m_H, m_H, m_H],
        [0, 0, m_C12, m_C12, m_C12, m_C12, m_C12, m_H, m_H, m_H, m_H, m_D, m_H],
        [0, 0, m_C12, m_C12, m_C12, m_C12, m_C12, m_D, m_H, m_H, m_H, m_H, m_H],
        [0, 0, m_C12, m_C12, m_C12, m_C12, m_C12, m_H, m_H, m_D, m_H, m_H, m_H],
        [0, 0, m_C12, m_C12, m_C12, m_C12, m_C12, m_D, m_D, m_D, m_D, m_D, m_H],
        [0, 0, m_C12, m_C12, m_C12, m_C12, m_C12, m_D, m_D, m_D, m_D, m_D, m_D],
    ]
)

B_e_SE = np.array(
    [
        [8489.1547821, 8292.10179645, 4305.6652761],
        [8292.09989565, 8280.2792131, 4251.27110857],
        [8482.79164378, 8105.05257055, 4253.08233339],
        [8408.38058222, 8172.55843117, 4252.6445067],
        [8195.61210694, 7918.39856564, 4178.35274526],
        [8477.17488123, 7650.51651789, 4123.17196563],
        [8371.82888223, 7735.10825254, 4122.22236194],
        [7057.54861499, 6730.36305402, 3555.25523105],
        [6656.95026379, 6653.76484463, 3469.28772252],
    ]
)

I_e_SE = strudel.h / (8 * np.pi**2 * (B_e_SE * 1e6))


p0 = [
    1.51510954,
    1.36129576,
    1.0872902,
    1.08797532,
    1.10219658,
    51.52642623,
    109.36844732,
    124.00818436,
    126.2353022,
    126.57661571,
]


strfit_popt = [
    1.499064,
    1.346353,
    1.078645,
    1.079071,
    1.094413,
    51.541323,
    109.330246,
    124.013202,
    126.081669,
    126.71045,
]
strfit_perr = [
    0.000337,
    0.00035,
    0.00024,
    0.000215,
    0.000165,
    0.015407,
    0.023845,
    0.107397,
    0.060174,
    0.011515,
]

strfit_xyz_coords = np.array(
    [
        [0.000000, 0.232701, -1.000000],
        [0.000000, 0.232701, 0.000000],
        [0.000000, 1.232701, 0.000000],
        [-1.173853, 0.300358, 0.000000],
        [1.173853, 0.300358, 0.000000],
        [-0.732671, -0.971659, 0.000000],
        [0.732671, -0.971659, 0.000000],
        [-2.202415, 0.625218, 0.000000],
        [2.202415, 0.625218, 0.000000],
        [-1.348355, -1.857846, 0.000000],
        [1.348355, -1.857846, 0.000000],
        [0.000000, 1.886910, -0.877355],
        [0.000000, 1.886910, 0.877355],
    ]
)


class TestPropargylChloride(unittest.TestCase):
    def test_params_and_structure(self):
        # Check if optimizing the structure via the moments of inertia yields the same results
        # as STRFIT by Zbigniew Kisiel
        # Z. Kisiel, J. Mol. Spectrosc. 218, 58-67 (2003)

        popt, pcov, perr = strudel.fit_moments_of_inertia(
            fzmat, masses, I_e_SE, p0=p0, sigmas=None
        )

        self.assertTrue(np.allclose(popt, strfit_popt))
        self.assertTrue(np.allclose(perr, strfit_perr, rtol=1e-2))

        # Check if the the the conversion from internal to xyz coordinates gives the same results
        # as STRFIT by Zbigniew Kisiel
        # Z. Kisiel, J. Mol. Spectrosc. 218, 58-67 (2003)

        ms = masses[0]
        zmat = fzmat(popt)
        coords = strudel.internal_to_cartesian(zmat)
        coords = strudel.transform_to_principal_axes(coords, ms)

        for column_strudel, column_strfit in zip(coords.T, strfit_xyz_coords.T):
            self.assertTrue(
                np.allclose(column_strudel, column_strfit)
                or np.allclose(column_strudel, -column_strfit)
            )
