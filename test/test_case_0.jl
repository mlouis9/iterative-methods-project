using FastExpm  # For computing matrix exponentials as a benchmark

# Matrix dimensions
m, n = 10, 10
density = 0.1 # Fraction of nonzero elements

A = Matrix(Diagonal(collect(1:n)))

function matvecA(x::Vector)::Vector
    return A*x
end

function f(x::Real)::Real
    return exp(x)
end

exactfA = fastExpm(A);