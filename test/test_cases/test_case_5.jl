using Distributions
using SparseArrays
using LinearAlgebra

# Matrix dimensions
m, n = 1000, 1000
density = 0.1 # Fraction of nonzero elements

A = sprand(m, n, density)
qr_A = qr(A)
Q = qr_A.Q

eigenvals = [ 1/sqrt(i) for i=1:n ]
Λ = Diagonal(eigenvals)

# Modify diagonal to control eigenvalues
A = Q*Λ*Q'

matvecA = create_matvecA(A)

function f(x)::Real
    if x > 0
        return log(1+x) + x
    else
        return -1e6
    end
end

exactfA = Q*Diagonal(f.(eigenvals))*Q'