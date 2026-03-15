# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Example: Structure Determination for Cyclpentadiene

This script demonstrates an unweighted structure fit to the
rotational constants and the moments of inertia of cyclopentadiene (c-C5H6).

Data is taken from
Bonah et al., J. Mol. Spectrosc. 408, 111967 (2025)
and from
Damiani et al., Chem. Phys. Lett. 37, 265-269 (1976)
for the deuterated isotopologues.

Run with:
    python examples/Cyclopentadiene.py
"""


import mol_strudel as str
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


if __name__ == '__main__':
    print_kwargs = {'print_stats': True, 'print_resid': False, 'print_struct': False, 'print_params': True}

    print('Fit to the rotational constants\n')
    popt, pcov, perr = str.fit_rotational_constants(fzmat, masses, B_e_SE, p0=p0)
    _ = str.print_results(fzmat, masses, popt, perr, B_e_SE, **print_kwargs)

    I_e_SE = str.h / (8 * np.pi**2 * (B_e_SE * 1e6))
    print('\n\n\nFit to the moments of inertia\n')
    popt, pcov, perr = str.fit_moments_of_inertia(fzmat, masses, I_e_SE, p0=p0)
    _ = str.print_results(fzmat, masses, popt, perr, I_e_SE, moments_of_inertia=True, **print_kwargs)