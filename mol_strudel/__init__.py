# -*- coding: utf-8 -*-
#!/usr/bin/env python3

# Author: Luis Bonah
# Description : Structure determination of molecules via internal ccoordinates

import numpy as np
from scipy.optimize import curve_fit
import scipy.constants as const

h = const.h
m0 = const.m_u


representations = {
    "Ir": ("z", "x", "y"),
    "IIr": ("y", "z", "x"),
    "IIIr": ("x", "y", "z"),
    "Il": ("z", "y", "x"),
    "IIl": ("x", "z", "y"),
    "IIIl": ("y", "x", "z"),
}


# See Gordy and Cook, "Microwave Molecular Spectra"
# Equations 8.101, 8.102, 8.103 as well as 8.114, 8.115, 8.116
def correct_rotational_constants_for_centrifugal_distortion(
    Bs, Ds, reduction="A", representation="Ir"
):
    correction = {}
    if reduction == "A":
        Delta_J, Delta_JK, Delta_K, delta_J, delta_K = Ds

        correction["x"] = 2 * Delta_J + Delta_JK - 2 * delta_J - 2 * delta_K
        correction["y"] = 2 * Delta_J + Delta_JK + 2 * delta_J + 2 * delta_K
        correction["z"] = 2 * Delta_J

    elif reduction == "S":
        D_J, D_JK, D_K, d_1, d_2 = Ds

        correction["x"] = 2 * D_J + D_JK + 2 * d_1 + 4 * d_2
        correction["y"] = 2 * D_J + D_JK - 2 * d_1 + 4 * d_2
        correction["z"] = 2 * D_J + 6 * d_2

    else:
        raise ValueError('Reduction has to be either "A" or "S".')

    if representation not in representations.keys():
        raise ValueError(
            f"Representation has to be any of the following values: {representations.keys()}"
        )
    index_order = representations[representation]
    correction = {
        index_abc: correction[index_xyz]
        for index_abc, index_xyz in zip("abc", index_order)
    }

    A, B, C = Bs

    A_c = A + correction["a"]
    B_c = B + correction["b"]
    C_c = C + correction["c"]
    return (A_c, B_c, C_c)


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


def transform_to_principal_axes(coords, masses):
    center_of_mass = np.sum(coords * masses[:, np.newaxis], axis=0) / np.sum(masses)
    coords -= center_of_mass
    eigvals, eigvecs = diagonalize_I_tensor(coords, masses)
    coords = coords @ eigvecs
    return coords


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

    def fit_function(*args, **kwargs):
        return calculate_rotational_constants(
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

    def fit_function(*args, **kwargs):
        return calculate_moments_of_inertia(
            create_zmat, *args, **kwargs, constants_mask=constants_mask
        )

    popt, pcov = curve_fit(
        fit_function, masses_array, Is, sigma=sigmas, **curve_fit_kwargs
    )
    perr = np.sqrt(np.diag(pcov))
    return popt, pcov, perr


def summarize_results(
    create_zmat,
    masses_array,
    popt,
    perr,
    ys,
    constants_mask=None,
    moments_of_inertia=False,
    sigmas=None,
):
    if moments_of_inertia:

        def fit_function(*args, **kwargs):
            return calculate_moments_of_inertia(
                *args, **kwargs, constants_mask=constants_mask
            )
    else:

        def fit_function(*args, **kwargs):
            return calculate_rotational_constants(
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
    report_params = ["Parameters:"]
    params = []
    for val, err in zip(popt, perr):
        params.append((val, err))
        report_params.append(f"{val:11.6f} ± {err:11.6f}")
    report_params = "\n".join(report_params)

    # Statistics
    stats = {}
    stats["Degf"] = N_degf = len(ys) - len(popt)
    stats["Deviation"] = deviation = np.sqrt(np.sum(residuals**2) / N_degf)
    stats["RMS"] = rms = np.sqrt(np.mean(residuals**2))

    if sigmas is not None:
        stats["wrms"] = wrms = np.sqrt(np.mean((residuals / sigmas) ** 2))
    else:
        wrms = None

    report_stats = ["Statistics:"]
    if moments_of_inertia:
        report_stats.append(
            f"Deviation of Fit: {deviation:12.8f} [uA²]  (Sqrt( Sum( (Io-c)**2 )/Degf)"
        )
        report_stats.append(
            f"RMS of Fit:       {rms:12.8f} [uA²]  (Sqrt( Mean( (Io-c)**2 ))"
        )
        report_stats.append(f"Degf:           {N_degf:5.0f}")
    else:
        report_stats.append(
            f"Deviation of Fit: {deviation * 1000:5.0f} kHz  (Sqrt( Sum( (Bo-c)**2 )/Degf)"
        )
        report_stats.append(
            f"RMS of Fit:       {rms * 1000:5.0f} kHz  (Sqrt( Mean( (Bo-c)**2 ))"
        )
        report_stats.append(f"Degf:             {N_degf:5.0f}")

    if sigmas is not None:
        report_stats.append(
            f"WRMS of Fit:      {wrms:5.2f}    (Sqrt( Mean( (Bo-c)**2 / sigma**2 ))"
        )
    report_stats = "\n".join(report_stats)

    # Residuals
    report_residuals = []
    if moments_of_inertia:
        report_residuals.append("Residuals [uA²]:")
        report_residuals.append(str(residuals))
    else:
        report_residuals.append("Residuals [kHz]:")
        report_residuals.append(str(residuals * 1000))
    report_residuals = "\n".join(report_residuals)

    # XYZ Coordinates
    masses = np.array(masses_array[0])
    zmat = create_zmat(popt)
    coords = internal_to_cartesian(zmat)
    coords = transform_to_principal_axes(coords, masses)

    report_coords = ["Cartesian Coordinates [A]:"]
    for x, y, z in coords:
        report_coords.append(f"{x:13.6f} {y:13.6f} {z:13.6f}")
    report_coords = "\n".join(report_coords)

    output = {
        "params": params,
        "stats": stats,
        "residuals": residuals,
        "coords": coords,
        "report_params": report_params,
        "report_stats": report_stats,
        "report_residuals": report_residuals,
        "report_coords": report_coords,
    }
    return output
