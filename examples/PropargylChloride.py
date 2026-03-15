# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Example: Structure Determination for Propargyl Chloride

This script demonstrates how to determine the structure of a molecule
using mol_strudel from rotational constants or moments of inertia
of several isotopologues.

The example uses propargyl chloride (C3H3Cl) data and compares the performances
of fitting the moments of inertia and rotational constants. Each case is presented
for weighted and unweighted data.

The results show that the difference in fitting the moments of inertia or the rotational
constants is only due to the different weighting. When weighting the fits appropriately,
the results only differ insignficantly.
Furthermore, the structures resulting from the weighted fits differ substantially from
the unweighted structures.
This is the main reason for the creation of this python library.

Data is taken from
Bonah et al., unpublished (2026)
and lines were refitted from
E. Hirota & Y. Morino, Bulletin of the Chemical Society of Japan 34, 341–48 (1961)
for the deuterated isotopologues.

Run with:
    python examples/PropargylChloride.py
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

B_e_SE = B_0_exp + alpha_QCC


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


if __name__ == '__main__':
    I_e_SE = (str.h / (8 * np.pi**2 * B_e_SE * 1e6))
    sigmas_I = (str.h / (8 * np.pi**2 * B_e_SE * 1e6)) * sigmas / B_e_SE
    p0 = initial_params_qcc

    outputs = []
    popt, pcov, perr = str.fit_moments_of_inertia(fzmat, masses, I_e_SE, p0=p0, sigmas=None)
    outputs.append(str.summarize_results(fzmat, masses, popt, perr, B_e_SE, moments_of_inertia=True, sigmas=None))

    popt, pcov, perr = str.fit_moments_of_inertia(fzmat, masses, I_e_SE, p0=p0, sigmas=sigmas_I)
    outputs.append(str.summarize_results(fzmat, masses, popt, perr, B_e_SE, moments_of_inertia=True, sigmas=sigmas_I))

    popt, pcov, perr = str.fit_rotational_constants(fzmat, masses, B_e_SE, p0=p0, sigmas=None)
    outputs.append(str.summarize_results(fzmat, masses, popt, perr, B_e_SE, sigmas=None))

    popt, pcov, perr = str.fit_rotational_constants(fzmat, masses, B_e_SE, p0=p0, sigmas=sigmas)
    outputs.append(str.summarize_results(fzmat, masses, popt, perr, B_e_SE, sigmas=sigmas))

    print()
    print('|' + ' Inertia (unw.)    | Inertia (wei.)    | Rot Const (unw.)  | Rot Const (wei.)  |' )
    print('|' + (('-' * 19) + '|') * len(outputs))
    for i in range(len(p0)):
        values = [output['params'][i] for output in outputs]
        print('| ' + ' | '.join([f'{val[0]:8.4f} ± {val[1]:6.4f}' for val in values]) + ' |')
    print()
