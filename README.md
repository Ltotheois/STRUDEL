# STRucture DEtermination Libary

STRUDEL is a python library for determining the structure of molecules from their rotational constants of moments of inertia.
The main improvement over other codes is that the moments of inertia/rotational constants can be weighted according to their uncertainties.

The structure is defined via internal coordinates in the Z-matrix formalism.

Examples for different molecules are provided in the *examples* folder.

## Example Usage

An examplary workflow is explained for propargyl chloride (C3H3Cl).
First, the libraray is imported and the masses for the different isotopes are defined for easier handling
```python
import mol_strudel as strudel
import numpy as np


m_H = 1.0078250321
m_D = 2.01410175
m_C12 = 12.0000000000
m_C13 = 13.0033548378
m_Cl35 = 34.96885268
m_Cl37 = 36.96590259
```

Then, the function creating the internal coordinates matrix is defined
```python
def fzmat(params):
    r_CH_A, r_CC_T, r_CC_S, r_CCl, r_CH_2, a_HCC, a_CCC, a_CCCl, a_ClCH, d_CCClH = params
    zmat = np.array([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, r_CH_A, 0, 0],
        [2, 1, 0, 1, 90, 0],
        [2, 3, 1, r_CC_T, a_HCC, 180],
        [4, 2, 3, 1, 90, 0],
        [4, 5, 2, r_CC_S, a_CCC, 180],
        [6, 4, 5, r_CCl, a_CCCl, 180],
        [6, 7, 4, r_CH_2, a_ClCH, d_CCClH],
        [6, 7, 4, r_CH_2, a_ClCH, -d_CCClH],
    ])
    return zmat
```
The function *fzmat* creates the z-matrix.
Each row describes the position of one atom relative to other atoms.

The first three numbers in each row are the indices of the reference atoms $a$, $b$, and $c$.
The last three values are the radius $R$, angle $\Theta$, and dihedral angle $\Phi$.

The position of the new atom $d$ is defined via
- its bond length to the atom $a$
- its bond angle with atom $b$
- its dihedral angle atom to $c$

This means that the atom in the first row is just in the origin.
Please note, that indexing starts at 1 and hence the reference atoms for the first atom are all zero.
The second atom is defined relative to the first atom and has a distance of *r_CH_A*.
The third atom is then at a distance of 1 Angstrom from the second atom and at an angle of 90 Degree relative to the first atom.

Also note, taht the third and fifth atom are dummy atoms which are just used to define the structure but are not actually real atoms.

Afterwards, the masses of the isotopologues are defined
```python
masses = np.array([
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
])
```
Each row is an isotopologue and the order of atoms is according to their definition in the z-matrix.
The dummy atoms have a mass of zero.

Next, the rotational constants of the isotopologues are defined in the same order
```python
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

# Be - B0 values
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

# Uncertainties of B_0_exp
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
```
Here, we have defined both the $B_0$ values and the $B_e - B_0$ values (from quantum chemical calculations) as we want to determine a semi-experimental equilibrium structure $r_e^\text{SE}$.
In addition, the uncertainties of the rotational constants are defined as *sigmas*.

The last piece of data we need are the initial parameters

```python
p0 = [1.06, 1.20, 1.45, 1.79, 1.09, 90, 89, 112, 107, 122]
```

With all the data prepared, the structure can be fitted by running
```python
popt, pcov, perr = strudel.fit_rotational_constants(fzmat, masses, B_e_SE, p0=p0, sigmas=None)
```
Afterwards, the results can be displayed by running
```python
output = strudel.summarize_results(fzmat, masses, popt, perr, B_e_SE, sigmas=sigmas)
print(output['report_params'])
print()
print(output['report_stats'])
print()
print(output['report_residuals'])
print()
print(output['report_coords'])
```

The complete code is found in the *examples* folder.