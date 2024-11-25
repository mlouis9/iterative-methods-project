using FastExpm  # For computing matrix exponentials as a benchmark

# Matrix dimensions
m, n = 100, 100
density = 0.1 # Fraction of nonzero elements

A = sprand(m, n, density)
qr_A = qr(A)
Q = qr_A.Q

max_eigenval = 1e6
eigenvals = LinRange(1, max_eigenval, n)
Λ = Diagonal(eigenvals)

# Modify diagonal to control eigenvalues
A = Q*Λ*Q'

function matvecA(x::Vector)::Vector
    return A*x
end

function f(x::Real)::Real
    if x > 0
        return x^(1/3)
    else
        return -(-x)^(1/3)
    end
end

exactfA = Q*Diagonal(eigenvals.^(1/3))*Q'