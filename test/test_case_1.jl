using FastExpm  # For computing matrix exponentials as a benchmark

# Matrix dimensions
m, n = 100, 100
density = 0.1 # Fraction of nonzero elements

A = sprand(m, n, density)
A = 0.5*(A + A') + n*I

function matvecA(x::Vector)::Vector
    return A*x
end

function f(x::Real)::Real
    return exp(x)
end

exactfA = fastExpm(A);