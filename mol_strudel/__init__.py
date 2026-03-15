# -*- coding: utf-8 -*-
#!/usr/bin/env python3

# Author: Luis Bonah
# Description : Structure determination of molecules via internal ccoordinates

import numpy as np
from scipy.optimize import curve_fit
import scipy.constants as const

h = const.h
m0 = const.m_u


def internal_to_cartesian(zmat):
    xyz_array = []

    for i, (i_radius, i_angle, i_dihedral, radius, angle, dihedral) in enumerate(zmat):
        if i == 0:
            xyz_array.append(np.array((0, 0, 0)))
        elif i == 1:
            xyz_array.append(np.array((radius, 0, 0)))
        elif i == 2:
            a = xyz_array[int(i_radius) - 1]
            b = xyz_array[int(i_angle) - 1]
            direction = 1 if (b - a)[0] >= 0 else -1

            theta = np.radians(angle)
            x = radius * np.cos(theta) * direction + a[0]
            y = radius * np.sin(theta) * direction + a[1]

            xyz_array.append(np.array((x, y, 0)))
        else:
            theta = np.radians(angle)
            phi = np.radians(dihedral)

            x = radius * np.cos(theta)
            y = radius * np.cos(phi) * np.sin(theta)
            z = radius * np.sin(phi) * np.sin(theta)

            a = xyz_array[int(i_radius) - 1]
            b = xyz_array[int(i_angle) - 1]
            c = xyz_array[int(i_dihedral) - 1]

            ab = a - b
            ab = ab / np.linalg.norm(ab)

            bc = b - c
            nv = np.cross(bc, ab)
            nv = nv / np.linalg.norm(nv)
            ncbc = np.cross(nv, ab)

            new_x = a[0] - ab[0] * x + ncbc[0] * y + nv[0] * z
            new_y = a[1] - ab[1] * x + ncbc[1] * y + nv[1] * z
            new_z = a[2] - ab[2] * x + ncbc[2] * y + nv[2] * z
            xyz_array.append(np.array((new_x, new_y, new_z)))

    return np.array(xyz_array)


def diagonalize_I_tensor(coords, masses):
    tensor = np.zeros((3, 3))
    for c, m in zip(coords, masses):
        x, y, z = c
        tensor[0, 0] += m * (y**2 + z**2)
        tensor[1, 1] += m * (x**2 + z**2)
        tensor[2, 2] += m * (x**2 + y**2)
        tensor[0, 1] -= m * x * y
        tensor[0, 2] -= m * x * z
        tensor[1, 2] -= m * y * z
        tensor[1, 0] -= m * x * y
        tensor[2, 0] -= m * x * z
        tensor[2, 1] -= m * y * z

    tensor *= 1e-20 * m0
    eigvals, eigvecs = np.linalg.eigh(tensor)

    return (eigvals, eigvecs)


def moments_of_inertia_to_rotational_constants(eigvals):
    return h / (8 * np.pi**2 * eigvals) / 1e6


def calculate_moments_of_inertia(
    create_zmat, masses_array, *params, constants_mask=None
):
    zmat = create_zmat(params)
    coords = internal_to_cartesian(zmat)
    masses_array = np.array(masses_array)

    moments_of_inertia = []
    for masses in masses_array:
        # Move molecule to center of mass
        center_of_mass = np.sum(coords * masses[:, np.newaxis], axis=0) / np.sum(masses)
        coords -= center_of_mass

        # Diagonalize I tensor
        eigvals, eigvecs = diagonalize_I_tensor(coords, masses)
        moments_of_inertia.extend(eigvals)

    moments_of_inertia = np.array(moments_of_inertia)
    if constants_mask is not None:
        moments_of_inertia = moments_of_inertia[constants_mask.flatten()]

    return moments_of_inertia


def calculate_rotational_constants(
    create_zmat, masses_array, *params, constants_mask=None
):
    moments_of_inertia = calculate_moments_of_inertia(
        create_zmat, masses_array, *params, constants_mask=constants_mask
    )
    return moments_of_inertia_to_rotational_constants(moments_of_inertia)


def fit_rotational_constants(
    create_zmat, masses_array, Bs, sigmas=None, constants_mask=None, **curve_fit_kwargs
):
    if sigmas is not None:
        sigmas = sigmas.flatten()

    Bs = Bs.flatten()
    if constants_mask is not None:
        constants_mask = constants_mask.flatten()
        Bs = Bs[constants_mask]
        if sigmas is not None:
            sigmas = sigmas[constants_mask]

    fit_function = lambda *args, **kwargs: calculate_rotational_constants(
        create_zmat, *args, **kwargs, constants_mask=constants_mask
    )
    popt, pcov = curve_fit(
        fit_function, masses_array, Bs, sigma=sigmas, **curve_fit_kwargs
    )
    perr = np.sqrt(np.diag(pcov))
    return popt, pcov, perr


def fit_moments_of_inertia(
    create_zmat, masses_array, Is, sigmas=None, constants_mask=None, **curve_fit_kwargs
):
    if sigmas is not None:
        sigmas = sigmas.flatten()

    Is = Is.flatten()
    if constants_mask is not None:
        constants_mask = constants_mask.flatten()
        Is = Is[constants_mask]
        if sigmas is not None:
            sigmas = sigmas[constants_mask]

    fit_function = lambda *args, **kwargs: calculate_moments_of_inertia(
        create_zmat, *args, **kwargs, constants_mask=constants_mask
    )
    popt, pcov = curve_fit(
        fit_function, masses_array, Is, sigma=sigmas, **curve_fit_kwargs
    )
    perr = np.sqrt(np.diag(pcov))
    return popt, pcov, perr


def print_results(
    create_zmat,
    masses_array,
    popt,
    perr,
    ys,
    constants_mask=None,
    moments_of_inertia=False,
    sigmas=None,
    print_params=True,
    print_stats=True,
    print_resid=True,
    print_struct=True,
):
    if moments_of_inertia:
        fit_function = lambda *args, **kwargs: calculate_moments_of_inertia(
            *args, **kwargs, constants_mask=constants_mask
        )
    else:
        fit_function = lambda *args, **kwargs: calculate_rotational_constants(
            *args, **kwargs, constants_mask=constants_mask
        )

    ys = ys.flatten()
    if sigmas is not None:
        sigmas = sigmas.flatten()

    if constants_mask is not None:
        constants_mask = constants_mask.flatten()
        ys = ys[constants_mask]
        if sigmas is not None:
            sigmas = sigmas[constants_mask]

    # Calculate Residuals
    ys_fit = fit_function(create_zmat, masses_array, *popt)
    residuals = ys - ys_fit

    if moments_of_inertia:
        residuals *= 1e20 / const.u
        if sigmas is not None:
            sigmas *= 1e20 / const.u

    # Parameters
    if print_params:
        print("Parameters:")
        for val, err in zip(popt, perr):
            print(f"{val:11.6f} ± {err:11.6f}")

    # Statistics
    N_degf = len(ys) - len(popt)
    DEVIATION = np.sqrt(np.sum(residuals**2) / N_degf)
    RMS = np.sqrt(np.mean(residuals**2))

    if sigmas is not None:
        WRMS = np.sqrt(np.mean((residuals / sigmas) ** 2))
    else:
        WRMS = None

    if print_stats:
        if moments_of_inertia:
            print()
            print(
                f"Deviation of Fit: {DEVIATION:12.8f} [uA²]  (Sqrt( Sum( (Io-c)**2 )/Degf)"
            )
            print(f"RMS of Fit:       {RMS:12.8f} [uA²]  (Sqrt( Mean( (Io-c)**2 ))")
            print(f"Degf:           {N_degf:5.0f}")
        else:
            print()
            print(
                f"Deviation of Fit: {DEVIATION*1000:5.0f} kHz  (Sqrt( Sum( (Bo-c)**2 )/Degf)"
            )
            print(f"RMS of Fit:       {RMS*1000:5.0f} kHz  (Sqrt( Mean( (Bo-c)**2 ))")
            print(f"Degf:             {N_degf:5.0f}")

        if sigmas is not None:
            print(
                f"WRMS of Fit:      {WRMS:5.2f}    (Sqrt( Mean( (Bo-c)**2 / sigma**2 ))"
            )

    # Residuals
    if print_resid:
        print()
        if moments_of_inertia:
            print("Residuals [uA²]:")
            print(residuals)
        else:
            print("Residuals [kHz]:")
            print(residuals * 1000)

    # Print XYZ Structure
    masses = np.array(masses_array[0])
    zmat = create_zmat(popt)
    coords = internal_to_cartesian(zmat)

    center_of_mass = np.sum(coords * masses[:, np.newaxis], axis=0) / np.sum(masses)
    coords -= center_of_mass
    eigvals, eigvecs = diagonalize_I_tensor(coords, masses)
    coords = coords @ eigvecs

    if print_struct:
        print()
        print("Cartesian Coordinates [A]:")
        for x, y, z in coords:
            print(f"{x:13.6f} {y:13.6f} {z:13.6f}")

    return (N_degf, DEVIATION, RMS, WRMS, residuals, coords)
