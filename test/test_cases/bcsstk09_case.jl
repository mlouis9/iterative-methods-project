using SparseArrays
using LinearAlgebra
using Arpack
using MatrixMarket
using MyProject.Utils

matrix_file = "test_cases/bcsstk09.mtx"
A = mmread(matrix_file)
n = size(A, 1)

matvecA = create_matvecA(A)

function f(x)::Real
    if x > 0
        return log(1+x) + x
    else
        return -1e6
    end
end

eigenvals = eigvals(Matrix(A))
exact = sum(f.(eigenvals))