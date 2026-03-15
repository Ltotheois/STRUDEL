# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Example: Structure Determination for Propargyl Chloride

This script demonstrates how to exclude rotational constants from the fit.
Here, the two deuterated isotopologues are excluded due to the much lower
quality of their rotational constants for the imaginary case where no
uncertainties were available.
Obviously, the proper way would be to weight the rotational constants accordingly.

The option to quickly exclude constants can help with exploring
the influence of different isotopologues on the structure.

Data is taken from
Bonah et al., unpublished (2026)
and lines were refitted from
E. Hirota & Y. Morino, Bulletin of the Chemical Society of Japan 34, 341–48 (1961)
for the deuterated isotopologues.

Run with:
    python examples/ExcludedConstants.py
"""


import mol_strudel as str
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
B_e_SE = np.array([
 [24414.7214919, 3087.2384859, 2788.3543412],
 [23493.4163683, 2896.7824632, 2620.9552267],
 [24260.8309242, 3021.1367824, 2732.3783839],
 [23333.2175491, 2835.2057494, 2568.5112679],
 [23573.0936742, 3087.217577 , 2777.0194162],
 [23417.061604 , 3021.0445937, 2721.2606869],
 [24386.6611978, 3056.1151488, 2762.579994 ],
 [24232.9281825, 2989.6331563, 2706.2359931],
 [24215.7404991, 2986.872725 , 2703.7586767],
 [24060.2077904, 2922.2443391, 2648.8191091],
])


# Initial Parameters from QCC
r_CH_A = 1.062015414210830
r_CC_T = 1.204283288462626
r_CC_S = 1.452887442305081
r_CCl =  1.788740700888962
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


if __name__ == '__main__':
    # The two deuterated isotopologues are excluded
    # They are the second and fourth isotopologues
    constants_mask = np.full((10, 3), True)
    constants_mask[1, :] = constants_mask[3, :] = False

    popt, pcov, perr = str.fit_rotational_constants(fzmat, masses, B_e_SE, p0=initial_params_qcc, constants_mask=constants_mask)
    output = str.summarize_results(fzmat, masses, popt, perr, B_e_SE, constants_mask=constants_mask)
    print(output['report_params'])
    print()
    print(output['report_stats'])
    print()
    print(output['report_coords'])
