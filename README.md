This **PiezoResistive Property Calculator (PRPCalc)** computes key parameters of piezoresistive properties and comprises two main modules.

**Carrier concentration module**: Based on first-principles density-of-states (DOS) data, it solves for the intrinsic carrier concentration and the corresponding Fermi level.

**Piezoresistive performance module**: It generates tensile/compressive or shear strained structures and, in conjunction with first-principles calculations (i.e., [VASP](https://www.vasp.at/)) and Boltzmann transport theory (i.e., [BoltzTraP2](https://gitlab.com/sousaw/BoltzTraP2), computes the elastoresistance coefficient, piezoresistive coefficient, and strain sensitivity factor (gauge factor).

The program is intended for materials with **cubic crystal symmetry**, and the underlying methodology can be extended to more complex crystal systems.
