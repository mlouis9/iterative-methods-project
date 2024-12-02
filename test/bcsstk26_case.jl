using SparseArrays
using LinearAlgebra
using Arpack
using MatrixMarket

matrix_file = "bcsstk26.mtx"
A = mmread(matrix_file)
n = size(A, 1)

function matvecA(x::AbstractVector)::AbstractVector
    return A*x
end

function f(x)::Real
    if x > 0
        return log(1+x) + x
    else
        return -1e6
    end
end

eigenvals = eigvals(Matrix(A))
exact = sum(f.(eigenvals))