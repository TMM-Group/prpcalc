#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pymatgen.io.vasp import Vasprun
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.core import Spin
from scipy.constants import k as k_B_J_K, e as elementary_charge
from scipy.optimize import brentq
import argparse
from pathlib import Path

# --- 常量 ---
K_B_EV_K = k_B_J_K / elementary_charge  # 8.617333262e-5 eV/K

def fermi_dirac_dist(energy, e_fermi, kT):
    """计算费米-狄拉克分布函数 f(E)"""
    # 添加数值稳定性处理，防止 exp 参数过大或过小
    arg = (energy - e_fermi) / kT
    return 1.0 / (1.0 + np.exp(np.clip(arg, -100, 100)))

def calculate_n_p(e_fermi, energies_abs, density_per_cm3, vbm_abs, cbm_abs, kT):
    """
    对于给定的费米能级，使用费米-狄拉克分布计算电子和空穴浓度。
    """
    # --- 电子浓度 n ---
    cb_mask = energies_abs > cbm_abs
    integrand_n = density_per_cm3[cb_mask] * fermi_dirac_dist(energies_abs[cb_mask], e_fermi, kT)
    # 使用 np.trapezoid 替代已弃用的 np.trapz
    n = np.trapezoid(integrand_n, energies_abs[cb_mask])

    # --- 空穴浓度 p ---
    # p = integral(g(E) * (1 - f(E)))
    vb_mask = energies_abs < vbm_abs
    integrand_p = density_per_cm3[vb_mask] * (1.0 - fermi_dirac_dist(energies_abs[vb_mask], e_fermi, kT))
    p = np.trapezoid(integrand_p, energies_abs[vb_mask])
    
    return n, p

def get_intrinsic_properties_fd():
    """
    从VASP计算结果中，通过求解电中性方程来严谨地计算本征半导体性质。
    使用完整的费米-狄拉克分布，并可选地应用剪刀算符修正。

    Returns:
        dict: 包含计算结果的字典，如果材料是金属则返回None。
        np.ndarray: 包含能量和DOS数据的数组 (N, 2)。
    """
    # ===用户输入部分===
    vasprun_path = Path(input("Please enter the path to the vasprun.xml file (default is vasprun.xml):") or "vasprun.xml")
    poscar_path = Path(input("Please enter the file path of POSCAR or CONTCAR (default is POSCAR):") or "POSCAR")
    try:
        temperature = float(input("Please enter the temperature (K) (default 300):") or 300.0)
    except ValueError:
        temperature = 300.0
    try:
        exp_gap_str = input("Please enter the experimental band gap Eg (eV):")
        experimental_gap = float(exp_gap_str) if exp_gap_str.strip() else None
    except ValueError:
        experimental_gap = None

    # ===核心计算部分===

    # 1. 读取 VASP 输出
    if not vasprun_path.exists():
        raise FileNotFoundError(f"vasprun.xml not found at: {vasprun_path}")
    vr = Vasprun(vasprun_path, parse_dos=True)
    dos = vr.tdos
    efermi_0k = dos.efermi # VASP计算的0K费米能级

    # 2. 自动确定能带边
    try:
        Eg, cbm_abs, vbm_abs = dos.get_interpolated_gap()
        if Eg <= 1e-4:
            print("Warning: Band gap is zero or negligible. Material is likely a metal.")
            return None, None
    except ValueError:
        print("Warning: Could not determine band gap from DOS. Material may be a metal.")
        return None, None

    # 3. 准备能量和DOS数据
    # 创建能量数组的副本，以防修改影响其他部分
    energies_abs = np.array(dos.energies)
    densities = dos.densities
    
    if isinstance(densities, dict):
        total_density = np.array(densities.get(Spin.up, 0)) + np.array(densities.get(Spin.down, 0))
    else:
        total_density = np.array(densities)

    # 4. 体积归一化
    try:
        struct = Structure.from_file(poscar_path)
    except FileNotFoundError:
        struct = vr.final_structure
        print(f"Warning: '{poscar_path}' not found. Using final structure from vasprun.xml.")
        
    volume_cm3 = struct.lattice.volume * 1e-24
    density_per_cm3 = total_density / volume_cm3
    
    # --- [新增] 剪刀算符修正 (Scissor Operator) ---
    if experimental_gap is not None and experimental_gap > 0:
        print("\nApplying Scissor Operator...")
        print(f"  - Calculated Gap (Eg_calc): {Eg:.4f} eV")
        print(f"  - Experimental Gap (Eg_exp): {experimental_gap:.4f} eV")
        
        correction = Eg - experimental_gap
        print(f"  - Shifting conduction band by: {-correction:.4f} eV")

        # 对能量数组进行修正
        cb_mask_original = energies_abs > cbm_abs
        energies_abs[cb_mask_original] -= correction
        
        # 更新修正后的CBM和Eg
        cbm_abs -= correction
        Eg = experimental_gap  # Eg现在是实验带隙

    dos_data_for_output = np.vstack([energies_abs - efermi_0k, density_per_cm3]).T
    
    # --- 核心部分: 求解本征费米能级 ---
    kT = K_B_EV_K * temperature
    
    def charge_neutrality_eq(e_fermi, energies, dens, vbm, cbm, temp_kT):
        n, p = calculate_n_p(e_fermi, energies, dens, vbm, cbm, temp_kT)
        return n - p

    try:
        e_fermi_intrinsic = brentq(
            charge_neutrality_eq, 
            a=vbm_abs,
            b=cbm_abs,
            args=(energies_abs, density_per_cm3, vbm_abs, cbm_abs, kT)
        )
    except ValueError:
        print("\nError: Could not find intrinsic Fermi level. The root might not be bracketed.")
        print("This can happen with very large band gaps or noisy DOS data.")
        return None, None

    # 5. 计算本征载流子浓度
    ni_n, ni_p = calculate_n_p(e_fermi_intrinsic, energies_abs, density_per_cm3, vbm_abs, cbm_abs, kT)
    ni = (ni_n + ni_p) / 2.0

    results = {
        "ni_cm-3": ni,
        "Eg_eV": Eg,
        "Efermi_0K_eV": efermi_0k,
        "VBM_abs_eV": vbm_abs,
        "CBM_abs_eV": cbm_abs,
        "E_intrinsic_fermi_abs_eV": e_fermi_intrinsic,
        "E_intrinsic_fermi_rel_0K_eV": e_fermi_intrinsic - efermi_0k,
        "Temperature_K": temperature,
    }

    try:
        if results:
            print("\n--- Intrinsic Semiconductor Properties (Fermi-Dirac Method) ---")
            print(f"Temperature: {results['Temperature_K']:.1f} K")
            print("-" * 60)
            print(f"Band Gap (Eg):    {results['Eg_eV']:.4f} eV")
            print(f"VBM (absolute):   {results['VBM_abs_eV']:.4f} eV")
            print(f"CBM (absolute):   {results['CBM_abs_eV']:.4f} eV")
            print(f"Fermi Level (0K): {results['Efermi_0K_eV']:.4f} eV")
            print("-" * 60)
            print(f"Intrinsic Fermi Level (at T):          {results['E_intrinsic_fermi_abs_eV']:.4f} eV")
            print(f"  (Relative to 0K Efermi):             {results['E_intrinsic_fermi_rel_0K_eV']:+.4f} eV")
            print(f"Intrinsic Carrier Concentration (ni):  {results['ni_cm-3']:.3e} cm^-3")
            print("-" * 60)

            # 直接写入文件
            header = "# Energy (eV relative to 0K E_fermi)   DOS (states/eV/cm^3)\n"
            if experimental_gap is not None:
                header = "# Energy (eV relative to 0K E_fermi, Scissor-Corrected)   DOS (states/eV/cm^3)\n"
            np.savetxt("tdos_normalized.dat", dos_data_for_output, fmt="%.8f    %.8e", header=header, comments='')
            print("Normalized DOS data written to tdos_normalized.dat")

    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
  get_intrinsic_properties_fd()
