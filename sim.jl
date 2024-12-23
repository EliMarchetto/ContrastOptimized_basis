cd(Base.source_dir())
using Pkg
Pkg.activate(".")
using LinearAlgebra
using MAT
using Random
using MRIgeneralizedBloch
using ProgressBars

##
B0 = 2.89

## determine how many jobs are run for each type of tissue
# batchsize = 2^11 # 2048
batchsize = 500 # use a smaller batch for test purposes

# Fingerprints to simulate:
distribution = :parenchyma
# distribution = :CSF

## simulation parameters
T2smin = 5e-6
T2smax = 25e-6
B1min = 0.5
B1max = 1.3
ω1_max = 2e3π

idx_grad = [2, 3, 4, 5, 6, 7] # which gradients should be orthogonalized


## load control
control = matread("FA_pattern_3T_v0p10p9.mat")
nSeq = size(control["alpha"], 2)

α = [control["alpha"][:, i] for i = 1:nSeq]
TRF = [control["TRF"][:, i] for i = 1:nSeq]
TR = control["TR"]
TRF_max = maximum(maximum.(TRF))

R2slT = precompute_R2sl(T2s_min=T2smin, T2s_max=T2smax, B1_max=B1max, TRF_max=TRF_max, ω1_max=ω1_max)
grad_list = [grad_m0s(), grad_R1f(), grad_R2f(), grad_Rx(), grad_R1s(), grad_T2s(), grad_ω0(), grad_B1()]


## set limits: mean, std, min, max
if distribution == :parenchyma
    m0s_limits = (0.15, 0.15, 0.0, 0.35)
    R1f_limits = (0.45, 0.20, 0.1, 1.00)
    R2f_limits = (12.5, 4.00, 1.0, 20.0)
    Rex_limits = (10.0, 5.00, 2.0, 30.0)
    R1s_limits = (3.00, 1.50, 0.5, 6.00)
    T2s_limits = (14e-6, 4e-6, T2smin, T2smax)
    ω0_limits = (0, π / TR / 1.643, -Inf, Inf) # 10% of samples are outside of first passband
    B1_limits = (0.9, 0.3, B1min, B1max)
elseif distribution == :CSF
    m0s_limits = (1e-3, 0.00, 0.00, Inf)
    R1f_limits = (0.25, 0.05, 0.15, 1.0)
    R2f_limits = (1.00, 5.00, 0.20, 2.0)
    Rex_limits = (10.0, 0.00, 0.00, Inf)
    R1s_limits = (3.00, 0.00, 0.00, Inf)
    T2s_limits = (12e-6, 0.0, 0.00, Inf)
    ω0_limits = (0, π / TR / 1.643, -Inf, Inf) # 10% of samples are outside of first passband
    B1_limits = (0.9, 0.3, B1min, B1max)
elseif distribution == :fat
    m0s_limits = (1e-3, 0.0, 0.0, Inf)
    R1f_limits = (2.50, 0.5, 1.5, 10.0)
    R2f_limits = (10.0, 2.5, 5.0, 20.0)
    Rex_limits = (10.0, 0.0, 0.0, Inf)
    R1s_limits = (3.00, 0.0, 0.0, Inf)
    T2s_limits = (12e-6, 0.0, 0.0, Inf)
    ω0_limits = (-3.5e-6 * 267.52218744e6 * B0, 2π * 65, -Inf, Inf) # FWS is -3.5ppm * gamma * B0 (T)
    B1_limits = (0.75, 0.125, B1min, B1max) # lower B1 expected for skull fat
elseif distribution == :broad
    m0s_limits = (0.15, 0.15, 0.0, 0.35)
    R1f_limits = (0.75, 0.45, 0.1, 2.00)
    R2f_limits = (24.0, 10.5, 3.0, 45.0)
    Rex_limits = (30.0, 10.0, 5.0, 50.0)
    R1s_limits = (3.00, 2.0, 0.5, 6.00)
    T2s_limits = (12e-6, 5e-6, T2smin, T2smax)
    ω0_limits = (0, π / TR / 1.643, -Inf, Inf) # 10% of samples are outside of first passband
    B1_limits = (0.9, 0.3, B1min, B1max)
end


## function to select random parameters
function rand_parameter(limits, rng; T=Float64)
    _mean, _std, _min, _max = limits
    p = Inf
    while p <= _min || p >= _max # equal sign avoids Inf being valid
        p = _mean + _std * randn(rng, T)
    end
    return p
end


## ######################################################################################
# perform simulations
#########################################################################################
@info "Simulating fingerprints"
flush(stderr)

s = zeros(ComplexF32, length(α[1]), length(α), length(grad_list) + 1, batchsize)
p = zeros(length(grad_list), batchsize) #8 parameters in split R1 model excluding M0

rng = MersenneTwister()
iter = ProgressBar(1:batchsize)
for i in iter
    m0s = rand_parameter(m0s_limits, rng)
    R1f = rand_parameter(R1f_limits, rng)
    R2f = rand_parameter(R2f_limits, rng)
    Rex = rand_parameter(Rex_limits, rng)
    R1s = rand_parameter(R1s_limits, rng)
    T2s = rand_parameter(T2s_limits, rng)
    ω0  = rand_parameter(ω0_limits, rng)
    B1  = rand_parameter(B1_limits, rng)

    p[:, i] = [m0s, R1f, R2f, Rex, R1s, T2s, ω0, B1]

    for j ∈ eachindex(α)
        si = calculatesignal_linearapprox(α[j], TRF[j], TR, ω0, B1, m0s, R1f, R2f, Rex, R1s, T2s, R2slT; grad_list=grad_list)
        s[:, j, :, i] .= reshape(si, (length(α[j]), length(grad_list) + 1))
    end
    flush(stderr)
end

s = reshape(s, :, length(grad_list) + 1, batchsize)

## save files
file = matopen("fingerprints_data_$(distribution).mat", "w")
write(file, "s", s)
write(file, "p", p)
close(file)



## ######################################################################################
# compute orothogonal gradients
#########################################################################################
@info "Calculating ograds"
flush(stderr)

si = zeros(ComplexF32, size(s, 1), length(idx_grad) + 1, size(s, 3))
s_temp = zeros(ComplexF32, size(si, 1))
si[:, 1, :] .= s[:, 1, :] #copy signal
@time for i ∈ axes(s, 3) #for batchsize
    for j ∈ eachindex(idx_grad) #for each interested gradient
        # swap interested gradient to last row
        s_temp .= s[:, end, i] #copy the last row
        s[:, end, i] .= s[:, idx_grad[j], i] #write the interested gradient into last row
        s[:, idx_grad[j], i] .= s_temp #shift last row into new spot

        #QR factorization
        Q, R = qr(@view(s[:, :, i])) #QR factorization on this sample
        si[:, j+1, i] .= Q[:, size(R, 2)] #save the unit length orthogonalized gradient

        # undo the swap
        s_temp .= s[:, end, i] #copy the interested gradient
        s[:, end, i] .= s[:, idx_grad[j], i] #write last gradient back into last row
        s[:, idx_grad[j], i] .= s_temp #write interested gradient back
    end
end

file = matopen("fingerprints_data_$(distribution).mat", "w")
write(file, "s", si)
close(file)

exit()