Based on: Elisa Marchetto, Sebastian Flassbeck, Andrew Mao, and Jakob Assländer, Contrast-Optimized Basis Functions for Self-Navigated Motion Correction in 3D quantitative MRI, In press, Magnetic Resonance in Medicine.
Preprint: https://arxiv.org/abs/2412.19552 

This code uses the generalized eigendecomposition to enhance the contrast-to-noise ratio between fingerprints for two tissue types. 
This method effectively rotates the SVD subspace, creating a contrast-optimized basis that enhances contrast in the first and last coefficient images.

This code is implemented in Julia v1.11.3.

Steps:
1. Generate sets of fingerprints for parenchyma and CSF using sim.jl
2. Derive the contrast-optimized subspace using the generalized eigendecomposition using generate_contrastOpt_basis.jl

For more information, please feel free to contact:
Elisa Marchetto elisa.marchetto.93@gmail.com
Sebastian Flassbeck sebastian.flassbeck@nyulangone.org
Jakob Asslaender jakob.asslaender@nyulangone.org
