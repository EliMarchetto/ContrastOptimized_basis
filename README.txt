Based on: 
1. Sebastian Flassbeck, Elisa Marchetto, Andrew Mao, and Jakob Assländer, Contrast-Optimized Basis Functions for Self-Navigated Motion Correction in 3D quantitative MRI, ISMRM 2024, Abstract #0394. 
2. Elisa Marchetto, Sebastian Flassbeck, Andrew Mao, and Jakob Assländer, Contrast-Optimized Basis Functions for Self-Navigated Motion Correction in 3D quantitative MRI, 2024 ISMRM Workshop on Motion Correction in MR.

This code uses the generalized eigendecomposition to enhance the contrast-to-noise ratio between fingerprints for two tissue types. 
This method effectively rotates the SVD subspace, creating a contrast-optimized basis that enhances contrast in the first and last coefficient images.

This code is implemented in Julia v1.10.

Steps:
1. Generate sets of fingerprints for parenchyma and CSF using sim.jl
2. Derive the contrast-optimized subspace using the generalized eigendecomposition using generate_contrastOpt_basis.jl

Elisa Marchetto
elisa.marchetto@nyulangone.org