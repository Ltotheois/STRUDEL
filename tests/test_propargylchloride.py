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
m_Cl35 = 34.96885268
m_Cl37 = 36.96590259


# Structure
def fzmat(params):
    r_CH_A, r_CC_T, r_CC_S, r_CCl, r_CH_2, a_HCC, a_CCC, a_CCCl, a_ClCH, d_CCClH = (
        params
    )
    zmat = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, r_CH_A, 0, 0],
            [2, 1, 0, 1, 90, 0],
            [2, 3, 1, r_CC_T, a_HCC, 180],
            [4, 2, 3, 1, 90, 0],
            [4, 5, 2, r_CC_S, a_CCC, 180],
            [6, 4, 5, r_CCl, a_CCCl, 180],
            [6, 7, 4, r_CH_2, a_ClCH, d_CCClH],
            [6, 7, 4, r_CH_2, a_ClCH, -d_CCClH],
        ]
    )
    return zmat


# Masses of Istotopologues


masses = np.array(
    [
        [m_H, m_C12, 0, m_C12, 0, m_C12, m_Cl35, m_H, m_H],
        [m_D, m_C12, 0, m_C12, 0, m_C12, m_Cl35, m_H, m_H],
        [m_H, m_C12, 0, m_C12, 0, m_C12, m_Cl37, m_H, m_H],
        [m_D, m_C12, 0, m_C12, 0, m_C12, m_Cl37, m_H, m_H],
        [m_H, m_C12, 0, m_C12, 0, m_C13, m_Cl35, m_H, m_H],
        [m_H, m_C12, 0, m_C12, 0, m_C13, m_Cl37, m_H, m_H],
        [m_H, m_C12, 0, m_C13, 0, m_C12, m_Cl35, m_H, m_H],
        [m_H, m_C12, 0, m_C13, 0, m_C12, m_Cl37, m_H, m_H],
        [m_H, m_C13, 0, m_C12, 0, m_C12, m_Cl35, m_H, m_H],
        [m_H, m_C13, 0, m_C12, 0, m_C12, m_Cl37, m_H, m_H],
    ]
)


# Rotational Parameters
B_0_exp = np.array(
    [
        [24299.739404, 3079.762495, 2777.65244],
        [23397.636723, 2890.312631, 2611.556823],
        [24147.307989, 3013.784157, 2721.929399],
        [23238.887481, 2828.830778, 2559.327354],
        [23465.113821, 3080.022163, 2766.525818],
        [23310.529001, 3013.966014, 2711.016153],
        [24272.241064, 3048.650844, 2751.964431],
        [24119.956454, 2982.295123, 2695.875144],
        [24101.738526, 2979.670648, 2693.485206],
        [23947.670529, 2915.162456, 2638.790717],
    ]
)


alpha_QCC = np.array(
    [
        [114.9820879, 7.4759909, 10.7019012],
        [95.7796453, 6.4698322, 9.3984037],
        [113.5229352, 7.3526254, 10.4489849],
        [94.3300681, 6.3749714, 9.1839139],
        [107.9798532, 7.195414, 10.4935982],
        [106.532603, 7.0785797, 10.2445339],
        [114.4201338, 7.4643048, 10.615563],
        [112.9717285, 7.3380333, 10.3608491],
        [114.0019731, 7.202077, 10.2734707],
        [112.5372614, 7.0818831, 10.0283921],
    ]
)

sigmas = np.array(
    [
        [0.000230, 0.000033, 0.000032],
        [0.054316, 0.004242, 0.004011],
        [0.000380, 0.000036, 0.000036],
        [0.203992, 0.013291, 0.012110],
        [0.014902, 0.002095, 0.001858],
        [0.016068, 0.003551, 0.004317],
        [0.014402, 0.001453, 0.001835],
        [0.021238, 0.002602, 0.002907],
        [0.007242, 0.001064, 0.001030],
        [0.009461, 0.001602, 0.001610],
    ]
)

# Original values from Hirota (do no show any improvement over refit values)
# B_0_exp[1] = (23396.70, 2890.33, 2611.55)
# B_0_exp[3] = (23238.18, 2828.85, 2559.31)
# sigmas[1] = (0.5, 0.2, 0.2)
# sigmas[3] = (0.5, 0.2, 0.2)


B_e_SE = B_0_exp + alpha_QCC
I_e_SE = strudel.h / (8 * np.pi**2 * (B_e_SE * 1e6))
sigmas_I = (strudel.h / (8 * np.pi**2 * (B_e_SE * 1e6))) * sigmas / B_e_SE

# Initial Parameters from QCC
r_CH_A = 1.062015414210830
r_CC_T = 1.204283288462626
r_CC_S = 1.452887442305081
r_CCl = 1.788740700888962
r_CH_2 = 1.085555016351510
a_HCC = 90.290562298590032
a_CCC = 89.290866457341892
a_CCCl = 111.764280062208471
a_ClCH = 106.913272031224523
d_CCClH = 121.558873649541113

initial_params_qcc = (
    r_CH_A,
    r_CC_T,
    r_CC_S,
    r_CCl,
    r_CH_2,
    a_HCC,
    a_CCC,
    a_CCCl,
    a_ClCH,
    d_CCClH,
)
initial_params_rounded = [1.06, 1.20, 1.45, 1.79, 1.09, 90, 89, 112, 107, 122]
initial_params_rough = [1, 1, 1, 1, 1, 90, 90, 90, 90, 90]


strfit_xyz_coords = np.array(
    [
        [-3.288947, -0.933176, 0.000000],
        [-2.358376, -0.421399, 0.000000],
        [-2.840268, 0.454831, -0.000000],
        [-1.300217, 0.153576, -0.000000],
        [-1.777659, 1.032240, -0.000000],
        [-0.032301, 0.862993, 0.000000],
        [1.357652, -0.262895, 0.000000],
        [0.064437, 1.484166, -0.884995],
        [0.064437, 1.484166, 0.884995],
    ]
)


class TestPropargylChloride(unittest.TestCase):

    # Check if the the the conversion from internal to xyz coordinates gives the same results
    # as STRFIT by Zbigniew Kisiel
    # Z. Kisiel, J. Mol. Spectrosc. 218, 58-67 (2003)
    def test_structure(self):
        initial_params = initial_params_qcc
        ms = masses[0]

        zmat = fzmat(initial_params)
        coords = strudel.internal_to_cartesian(zmat)
        coords = strudel.transform_to_principal_axes(coords, ms)

        for column_strudel, column_strfit in zip(coords.T, strfit_xyz_coords.T):
            self.assertTrue(
                np.allclose(column_strudel, column_strfit)
                or np.allclose(column_strudel, -column_strfit)
            )

    # Check that fitting the appropriately weighted moments of inertia and rotational constants
    # results in the same structure
    def test_weighting(self):
        p0 = initial_params_qcc

        popt_I, pcov_I, perr_I = strudel.fit_moments_of_inertia(
            fzmat, masses, I_e_SE, p0=p0, sigmas=sigmas_I
        )
        popt_B, pcov_B, perr_B = strudel.fit_rotational_constants(
            fzmat, masses, B_e_SE, p0=p0, sigmas=sigmas
        )

        self.assertTrue(np.allclose(popt_I, popt_B))
        self.assertTrue(np.allclose(perr_I, perr_B, rtol=1e-2))

    def test_masking_constants(self):
        constants_mask = np.full((10, 3), True)
        constants_mask[1, :] = constants_mask[3, :] = False
        p0 = initial_params_qcc

        popt, pcov, perr = strudel.fit_rotational_constants(
            fzmat, masses, B_e_SE, p0=p0, sigmas=sigmas, constants_mask=constants_mask
        )

        popt_lit = [
            1.07602329,
            1.20599051,
            1.45093789,
            1.78778322,
            1.08027226,
            90.43826162,
            89.45225462,
            111.70794512,
            107.49403574,
            120.86964862,
        ]
        perr_lit = [
            4.40528270e-04,
            7.18659921e-05,
            9.21914970e-05,
            6.73691313e-05,
            1.84321677e-04,
            2.52099116e-02,
            8.64300763e-03,
            3.25236835e-03,
            1.94369004e-02,
            2.63147343e-02,
        ]

        self.assertTrue(np.allclose(popt, popt_lit))
        self.assertTrue(np.allclose(perr, perr_lit, rtol=1e-2))

        popt, pcov, perr = strudel.fit_moments_of_inertia(
            fzmat, masses, I_e_SE, p0=p0, sigmas=sigmas_I, constants_mask=constants_mask
        )

        self.assertTrue(np.allclose(popt, popt_lit))
        self.assertTrue(np.allclose(perr, perr_lit, rtol=1e-2))

    def test_printing_results(self):
        constants_mask_all = np.full((10, 3), True)
        constanst_mask_exc = np.full((10, 3), True)
        constanst_mask_exc[1, :] = constanst_mask_exc[3, :] = False

        constants_masks = [constants_mask_all, constanst_mask_exc]

        for constants_mask in constants_masks:
            I_e_SE = strudel.h / (8 * np.pi**2 * B_e_SE * 1e6)
            sigmas_I = (strudel.h / (8 * np.pi**2 * B_e_SE * 1e6)) * sigmas / B_e_SE
            p0 = initial_params_qcc

            popt, pcov, perr = strudel.fit_moments_of_inertia(
                fzmat, masses, I_e_SE, p0=p0, sigmas=None, constants_mask=constants_mask
            )
            _ = strudel.summarize_results(
                fzmat,
                masses,
                popt,
                perr,
                B_e_SE,
                moments_of_inertia=True,
                sigmas=None,
                constants_mask=constants_mask,
            )

            popt, pcov, perr = strudel.fit_moments_of_inertia(
                fzmat, masses, I_e_SE, p0=p0, sigmas=sigmas_I
            )
            _ = strudel.summarize_results(
                fzmat,
                masses,
                popt,
                perr,
                B_e_SE,
                moments_of_inertia=True,
                sigmas=sigmas_I,
                constants_mask=constants_mask,
            )

            popt, pcov, perr = strudel.fit_rotational_constants(
                fzmat, masses, B_e_SE, p0=p0, sigmas=None
            )
            _ = strudel.summarize_results(
                fzmat,
                masses,
                popt,
                perr,
                B_e_SE,
                sigmas=None,
                constants_mask=constants_mask,
            )

            popt, pcov, perr = strudel.fit_rotational_constants(
                fzmat, masses, B_e_SE, p0=p0, sigmas=sigmas
            )
            _ = strudel.summarize_results(
                fzmat,
                masses,
                popt,
                perr,
                B_e_SE,
                sigmas=sigmas,
                constants_mask=constants_mask,
            )
