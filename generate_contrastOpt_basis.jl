cd(Base.source_dir())

using Pkg
Pkg.activate()
Pkg.instantiate()

using LinearAlgebra
using MAT
using Statistics
using Random
using MRIgeneralizedBloch
using ProgressBars

# Load simulated fingerprints for parenchyma and CSF generated using sim.jl
s1 = matread("fingerprints_data_parenchyma.mat")["s"]
s2 = matread("fingerprints_data_CSF.mat")["s"]

iSeq = 1 # select a sequence (1 to 6)
Nspokes = 1142
xa_t = transpose(s1[1+Nspokes*(iSeq-1):Nspokes+Nspokes*(iSeq-1),1,:])
xb_t = transpose(s2[1+Nspokes*(iSeq-1):Nspokes+Nspokes*(iSeq-1),1,:])

##
usv = svd([xa_t; xb_t])

Nc = 3
b = usv.V[:,1:Nc]

xa = xa_t * b
xb = xb_t * b

##
A = xa' * xa
B = xb' * xb

##
F = eigen(A, B)
U = b * F.vectors[:,end:-1:1]

U_temp = cat(dims=2,U[:,1],U[:,end],U[:,2:end-1])

U_orth = copy(U_temp)
U_orth[:,1] ./= norm(U_orth[:,1])

function ProjUV(U,V)
    return U'*V/(U'U) * U
end

for i = 2:size(U,2)
    for j = i-1:-1:1
        U_orth[:,i] -= ProjUV(U_orth[:,j],U_temp[:,i])
    end
    U_orth[:,i] ./= norm(U_orth[:,i])
end

pa = mean(xa_t*U_orth[:,1])

U = sign(real.(pa)) * U_orth
