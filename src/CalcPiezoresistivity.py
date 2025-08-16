#!/usr/bin/env python
# -*- coding: utf-8 -*-

# b_calculate_piezoresistivity.py (Simplified Version)
# -----------------------------------------------------------------
#
# 这个脚本用于从 `btp2 dope` 命令生成的 *.condtens 文件中计算
# 立方晶系的压阻相关系数。
#
# 新增功能:
# - [FINAL] 采用健壮的方法解析 VASP OUTCAR，精确处理其
#   标题、分隔行和数据块的位置，并正确处理非标准的
#   弹性矩阵行列顺序。
# - 采用标准的矩阵运算 (π = m * s) 代替展开式来计算压阻系数，
#   代码更具通用性和物理意义。
#
# -----------------------------------------------------------------

import numpy as np
from scipy.stats import linregress
import sys

# =================================================================
# --- 用户设置 ---
# =================================================================

FNAME_PREFIX_0 = 'case0'
FNAME_PREFIX_1 = 'case1'
FNAME_PREFIX_4 = 'case4'

STRAIN_MAGNITUDES_EPS1 = [-0.0050, -0.0025, 0.0025, 0.0050]
STRAIN_MAGNITUDES_EPS4 = [-0.0050, -0.0025, 0.0025, 0.0050]

OUTCAR_FILENAME = 'OUTCAR'

# =================================================================
# --- 函数定义 ---
# =================================================================

def get_elastic_and_compliance_matrices(filename='OUTCAR'):
    """
    从 OUTCAR 文件中解析完整的6x6弹性矩阵 C_ij (GPa)，并正确处理
    VASP 的非标准行列顺序。然后通过求逆计算柔顺矩阵 s_ij (GPa⁻¹)。
    返回: C_matrix_gpa (GPa), s_matrix_gpa_inv (GPa⁻¹)
    """
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        anchor_index = -1
        for i, line in enumerate(lines):
            if 'TOTAL ELASTIC MODULI (kBar)' in line:
                anchor_index = i
                break
        
        if anchor_index == -1:
            print(f"Error: 'TOTAL ELASTIC MODULI' not found in '{filename}'.")
            return None, None
            
        idx_map = {'XX': 1, 'YY': 2, 'ZZ': 3, 'XY': 6, 'YZ': 4, 'ZX': 5}
        
        col_labels = lines[anchor_index + 1].split()[1:]
        data_lines = lines[anchor_index + 3 : anchor_index + 9]
        row_labels = [line.split()[0] for line in data_lines]
        
        c_matrix_vasp_order_kbar = np.array(
            [[float(x) for x in line.split()[1:]] for line in data_lines]
        )
        
        if not all(label in idx_map for label in row_labels + col_labels):
            print(f"Error: Elastic constant labels in OUTCAR are unrecognized. Rows: {row_labels}, Columns: {col_labels}")
            return None, None
            
        c_matrix_standard_kbar = np.zeros((6, 6))

        for i, row_label in enumerate(row_labels):
            for j, col_label in enumerate(col_labels):
                std_i = idx_map[row_label] - 1
                std_j = idx_map[col_label] - 1
                c_matrix_standard_kbar[std_i, std_j] = c_matrix_vasp_order_kbar[i, j]
                
        c_matrix_gpa = c_matrix_standard_kbar / 10.0
        s_matrix_gpa_inv = np.linalg.inv(c_matrix_gpa)

    except FileNotFoundError:
        print(f"Error: The OUTCAR file '{filename}' was not found.")
        return None, None
    except (IndexError, ValueError) as e:
        print(f"Error: Failed to parse the OUTCAR file '{filename}': {e}")
        return None, None
    except np.linalg.LinAlgError:
        print(f"Error: The elastic matrix read from '{filename}' is singular and cannot be inverted.")
        return None, None
        
    return c_matrix_gpa, s_matrix_gpa_inv

def get_sigma_from_condtens(filename):
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.strip().startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 12:
                    continue
                sigma_values = [float(v) for v in parts[3:12]]
                return np.array(sigma_values).reshape(3, 3)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except (ValueError, IndexError) as e:
        print(f"Error: Failed to parse the file '{filename}': {e}")
    print(f"Warning: No valid data lines found in the file '{filename}'.")
    return None

def write_fit_data(filename, strains, delta_rhos):
    # 此函数不变
    try:
        with open(filename, 'w') as f:
            f.write(f"# {'Strain':<18} {'Delta_rho/rho_ref':<20}\n")
            for strain, delta_rho in zip(strains, delta_rhos):
                f.write(f"{strain:<18.8f} {delta_rho:<20.8e}\n")
        print(f"    -> The fitted data has been successfully written to '{filename}'.")
    except IOError as e:
        print(f"    Warning: Failed to write to the file '{filename}': {e}")

def calculate_piezoresistivity():
    print("=" * 50)
    print("      Calculation of piezoresistive coefficient (directly read the data after doping)")
    print("=" * 50)
    
    # --- 1. 读取弹性常数和计算柔顺系数 ---
    print(f"--> Reading the elastic matrix from the '{OUTCAR_FILENAME}' file and calculating the compliance matrix...")
    C_matrix_gpa, s_matrix_gpa_inv = get_elastic_and_compliance_matrices(OUTCAR_FILENAME)
    if C_matrix_gpa is None:
        sys.exit(1)
        
    C11, C12, C44 = C_matrix_gpa[0, 0], C_matrix_gpa[0, 1], C_matrix_gpa[3, 3]
    s11_gpa, s12_gpa, s44_gpa = s_matrix_gpa_inv[0, 0], s_matrix_gpa_inv[0, 1], s_matrix_gpa_inv[3, 3]
    
    print(f"    The read elastic constants (GPa): C11={C11:.2f}, C12={C12:.2f}, C44={C44:.2f}")
    print(f"    The calculated compliance matrix (GPa⁻¹): s11={s11_gpa:.4e}, s12={s12_gpa:.4e}, s44={s44_gpa:.4e}")
    print("-" * 50)

    # --- 2. 获取参考态 (无应变) 的电阻率 ---
    print("--> Processing the reference structure (unstrained)...")
    fname_0 = f"{FNAME_PREFIX_0}.condtens"
    sigma0_div_tau = get_sigma_from_condtens(fname_0)
    if sigma0_div_tau is None: sys.exit(1)
    rho0_mul_tau = np.linalg.inv(sigma0_div_tau)
    rho_ref = rho0_mul_tau[0, 0]
    print(f"    The calculated reference resistivity ρ₀*τ = {rho_ref:.4e}")

    # --- 3. 收集并拟合 ε₁ 应变的数据 (求解 m₁₁, m₁₂) ---
    print("\n--> Processing the uniaxial strain (ε₁) data...")
    delta_rho1_list, delta_rho2_list, valid_strains_eps1 = [], [], []
    for strain in STRAIN_MAGNITUDES_EPS1:
        fname = f"{FNAME_PREFIX_1}_{strain:.4f}.condtens"
        sigma_strained_div_tau = get_sigma_from_condtens(fname)
        if sigma_strained_div_tau is None: continue
        rho_strained_mul_tau = np.linalg.inv(sigma_strained_div_tau)
        delta_rho1_list.append((rho_strained_mul_tau[0, 0] - rho_ref) / rho_ref)
        delta_rho2_list.append((rho_strained_mul_tau[1, 1] - rho_ref) / rho_ref)
        valid_strains_eps1.append(strain)
    write_fit_data('fit_data_m11.txt', valid_strains_eps1, delta_rho1_list)
    write_fit_data('fit_data_m12.txt', valid_strains_eps1, delta_rho2_list)
    m11 = linregress(valid_strains_eps1, delta_rho1_list).slope
    m12 = linregress(valid_strains_eps1, delta_rho2_list).slope
    print(f"    Fit m₁₁: slope = {m11:.4f}, R-squared = {linregress(valid_strains_eps1, delta_rho1_list).rvalue**2:.6f}")
    print(f"    Fit m₁₂: slope = {m12:.4f}, R-squared = {linregress(valid_strains_eps1, delta_rho2_list).rvalue**2:.6f}")

    # --- 4. 收集并拟合 ε₄ 应变的数据 (求解 m₄₄) ---
    print("\n--> Processing the shear strain (ε₄) data...")
    delta_rho4_voigt_list, valid_strains_eps4 = [], []
    for strain in STRAIN_MAGNITUDES_EPS4:
        fname = f"{FNAME_PREFIX_4}_{strain:.4f}.condtens"
        sigma_strained_div_tau = get_sigma_from_condtens(fname)
        if sigma_strained_div_tau is None: continue
        rho_strained_mul_tau = np.linalg.inv(sigma_strained_div_tau)
        delta_rho4_voigt = (rho_strained_mul_tau[1, 2] + rho_strained_mul_tau[2, 1]) / rho_ref
        delta_rho4_voigt_list.append(delta_rho4_voigt)
        valid_strains_eps4.append(strain)
    write_fit_data('fit_data_m44.txt', valid_strains_eps4, delta_rho4_voigt_list)
    m44 = linregress(valid_strains_eps4, delta_rho4_voigt_list).slope
    print(f"    Fit m₄₄: slope = {m44:.4f}, R-squared = {linregress(valid_strains_eps4, delta_rho4_voigt_list).rvalue**2:.6f}")

    # --- 5. 计算最终物理量 (π 和 GF)，采用标准矩阵运算 ---
    print("\n--> The piezoresistive coefficient is being calculated through the matrix operation π = m * s....")
    # 5.1 根据立方晶系对称性，构造弹阻(m)矩阵
    m_matrix = np.zeros((6, 6))
    m_matrix[0, 0] = m_matrix[1, 1] = m_matrix[2, 2] = m11
    m_matrix[0, 1] = m_matrix[0, 2] = m_matrix[1, 0] = m_matrix[1, 2] = m_matrix[2, 0] = m_matrix[2, 1] = m12
    m_matrix[3, 3] = m_matrix[4, 4] = m_matrix[5, 5] = m44
    
    # 5.2 执行矩阵乘法: π(GPa⁻¹) = m(无量纲) * s(GPa⁻¹)
    pi_matrix_gpa = m_matrix @ s_matrix_gpa_inv
    
    # 5.3 从结果矩阵中提取所需分量
    pi11_gpa = pi_matrix_gpa[0, 0]
    pi12_gpa = pi_matrix_gpa[0, 1]
    pi44_gpa = pi_matrix_gpa[3, 3]
    
    # 5.4 计算应变灵敏度因子 GF = π/s (无量纲)
    gf_l = pi11_gpa / s11_gpa
    gf_t = pi12_gpa / s11_gpa
    
    # --- 6. 打印最终结果 ---
    print("\n" + "="*22 + " Final result " + "="*22)
    print("Elastoresistance Coeff. (m):")
    print(f"  m₁₁ = {m11:.4f}")
    print(f"  m₁₂ = {m12:.4f}")
    print(f"  m₄₄ = {m44:.4f}")

    print("\nPiezoresistive Coeff. (π) [x 10⁻¹¹ Pa⁻¹]:")
    print(f"  π₁₁ = {pi11_gpa * 100:.4f}")
    print(f"  π₁₂ = {pi12_gpa * 100:.4f}")
    print(f"  π₄₄ = {pi44_gpa * 100:.4f}")

    print("\nGauge Factor (GF) for [100] direction:")
    print(f"  GF_longitudinal = {gf_l:.4f}")
    print(f"  GF_transverse   = {gf_t:.4f}")
    print("=" * 58)

if __name__ == "__main__":
    calculate_piezoresistivity()