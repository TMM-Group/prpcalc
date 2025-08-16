#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pymatgen.core import Structure, Lattice

# --- 用户设置 ---
POSCAR_IN = "POSCAR" #你的优化好的、无应变的POSCAR文件名
# 施加一系列的正应变和压应变
STRAIN_MAGNITUDES = [-0.0050, -0.0025, 0.0025, 0.0050]
# -----------------

def create_strain_tensor(voigt_strain):
    """
    根据 Voigt 表示的应变向量，创建应变张量矩阵。
    """
    e1, e2, e3, e4, e5, e6 = voigt_strain
    return np.array([
        [e1,    e6/2,  e5/2],
        [e6/2,  e2,    e4/2],
        [e5/2,  e4/2,  e3]
    ])

def generate_strained_structures():
    try:
        structure_pristine = Structure.from_file(POSCAR_IN)
    except FileNotFoundError:
        print(f"Error: The input file '{POSCAR_IN}' was not found.")
        return

    # 保存原始结构
    structure_pristine.to(filename="POSCAR_0", fmt="poscar")
    print("The original structure has been saved to POSCAR_0.")

    original_lattice_matrix = structure_pristine.lattice.matrix

    # --- 施加单轴应变 ---
    print("\nGenerating uniaxial strained structures (ε₁)...")
    for strain_val in STRAIN_MAGNITUDES:
        voigt_strain = [strain_val, 0, 0, 0, 0, 0]
        strain_tensor = create_strain_tensor(voigt_strain)
        deformation_gradient = np.eye(3) + strain_tensor
        new_lattice_matrix = np.dot(original_lattice_matrix, deformation_gradient)
        new_lattice = Lattice(new_lattice_matrix)
        strained_structure = Structure(
            new_lattice,
            structure_pristine.species,
            structure_pristine.frac_coords
        )
        fname = f"POSCAR_eps1_{strain_val:.4f}"
        strained_structure.to(filename=fname, fmt="poscar")
        print(f"  Generated: {fname}")

    # --- 施加剪切应变 ---
    print("\nGenerating shear strained structures (ε₄)...")
    for strain_val in STRAIN_MAGNITUDES:
        voigt_strain = [0, 0, 0, strain_val, 0, 0]
        strain_tensor = create_strain_tensor(voigt_strain)
        deformation_gradient = np.eye(3) + strain_tensor
        new_lattice_matrix = np.dot(original_lattice_matrix, deformation_gradient)
        new_lattice = Lattice(new_lattice_matrix)
        strained_structure = Structure(
            new_lattice,
            structure_pristine.species,
            structure_pristine.frac_coords
        )
        fname = f"POSCAR_eps4_{strain_val:.4f}"
        strained_structure.to(filename=fname, fmt="poscar")
        print(f"  Generated: {fname}")

    print("\nStrained structure files have been successfully generated.")

if __name__ == "__main__":
    generate_strained_structures()