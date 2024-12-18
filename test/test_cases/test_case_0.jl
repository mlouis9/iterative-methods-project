using FastExpm  # For computing matrix exponentials as a benchmark
using MyProject.Utils

# Matrix dimensions
m, n = 10, 10
density = 0.1 # Fraction of nonzero elements

A = Matrix(Diagonal(collect(1:n)))

matvecA = create_matvecA(A)

function f(x::Real)::Real
    return exp(x)
end

exactfA = fastExpm(A);