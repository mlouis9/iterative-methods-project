module StochasticLanczos

using LinearAlgebra
using Primes

# Define lanczos_step! function
function lanczos_step!(A::Function, v1::Vector)
    # Placeholder implementation for lanczos_step! (complete as needed)
end

"""
    lanczos(A::Function, q::Vector, kmax::Int)::Matrix

Performs kmax steps of the Lanczos iteration on a matrix A, that is implicitly accessible only through matrix-vector
multiplications.
"""
function lanczos(A::Function, q::Vector, kmax::Int)::Matrix
    T = zeros(kmax, kmax)
    r = copy(q)
    β = norm(r)
    q1 = zeros(length(q))

    for k = 1:kmax
        q0 = copy(q1)
        q1 = r/β
        r = A(q1)
        α = dot(q1, r)
        T[k, k] = α
        if k >1
            T[k-1, k] = β
            T[k, k-1] = β
        end
        r = r - α*q1 - β*q0
        β = norm(r)
    end
    return T
end


function enforce_signs(Q, R)
    for i in 1:min(size(R)...)
        if R[i, i] < 0
            Q[:, i] .= -Q[:, i]
            R[i, :] .= -R[i, :]
        end
    end

    return Q, R
end


"""
    divisors(n::Int)

Utility function for finding all of the positive divisors of an integer `n`, this is useful for finding all of the allowable block sizes in the block Lanczos algorithm
"""
function divisors(n::Int)
    d = Int64[1]
    for (p,e) in factor(n)
        t = Int64[]
        r = 1

        for i in 1:e
            r *= p
            for u in d
                push!(t, u*r)
            end
        end

        append!(d, t)
    end

    return sort(d)
end


"""
    block_lanczos(A::Function, Q::Matrix, kmax::Int)::Matrix

Performs kmax steps of the block Lanczos iteration on a matrix A, that is implicitly accessible only through matrix-vector
multiplications.
"""
function block_lanczos(matvecA::Function, V::Matrix, kmax::Int; 
    reorthogonalize=true, reorthogonalization_fraction=0.1)

    @assert kmax ≥ 1 "kmax must be at least 1"
    n, b = size(V)

    # Pre-allocate a large array for all Q-blocks: Qbig is n × ((kmax+1)*b)
    Qbig = zeros(Float64, n, (kmax+1)*b)
    # Views to access Q-block k: Qblock(k) = Qbig[:, (k-1)*b+1 : k*b]
    Qblock(k) = @view Qbig[:, (k-1)*b+1 : k*b]

    # Pre-allocate arrays for A and B blocks
    A = Vector{Matrix{Float64}}(undef, kmax)
    B = Vector{Matrix{Float64}}(undef, kmax)

    # Pre-allocate workspace arrays
    Rmat = zeros(Float64, n, b)  # Residual block
    W = zeros(Float64, b, b)     # For corrections and projections

    # Initial QR decomposition of V
    QR = qr(V)
    Q_init, R_out = enforce_signs(Matrix(QR.Q), QR.R)
    Qblock(1) .= Q_init  # Place Q[1] into Qbig
    A[1] = Qblock(1)' * matvecA(Qblock(1))

    # Compute initial residual: R = A(Q1) - Q1*A[1]
    # Use matvecA(Qblock(1)) once and store to reduce calls
    mul!(Rmat, Qblock(1), A[1], 1.0, 0.0)   # Rmat = Q1*A[1]
    # R = A(Q1) - Rmat
    @views begin
        Rtemp = matvecA(Qblock(1)) # Temporary to hold A*Q1
        Rmat .= Rtemp .- Rmat
    end

    # Second QR
    QR = qr(Rmat)
    Q2, B1 = enforce_signs(Matrix(QR.Q), QR.R)
    Qblock(2) .= Q2
    B[1] = B1

    # Main Lanczos loop
    for k in 2:kmax
        # A[k] = Q[k]' * A(Q[k])
        # Precompute A(Q[k]) and store it to avoid multiple calls
        Qk = Qblock(k)
        Qkm1 = Qblock(k-1)
        A[k] = Qk' * matvecA(Qk)

        # Compute residual:
        # R = A(Q[k]) - Q[k]*A[k] - Q[k-1]*B[k-1]^T
        @views begin
            Rtemp = matvecA(Qk)
            # Rmat = Rtemp
            Rmat .= Rtemp
            # Rmat -= Qk*A[k]
            mul!(Rmat, Qk, A[k], -1.0, 1.0)
            # Rmat -= Q[k-1]*B[k-1]'
            mul!(Rmat, Qkm1, B[k-1]', -1.0, 1.0)
        end

        if reorthogonalize
            num_to_reorth = max(1, round(Int, reorthogonalization_fraction * k))
            indices = sort(rand(1:k, num_to_reorth))
            # Instead of multiple small corrections, we can do them one by one.
            # Another optimization might be to combine them into a single large projection if possible.
            for i in indices
                Qi = Qblock(i)
                # W = Qi' * Rmat (b×b = (b×n)*(n×b))
                mul!(W, Qi', Rmat, 1.0, 0.0)
                # Rmat = Rmat - Qi * W
                mul!(Rmat, Qi, W, -1.0, 1.0)
            end
        end

        # QR decomposition to get Q[k+1], B[k]
        QR = qr(Rmat)
        Qnew, Bk = enforce_signs(Matrix(QR.Q), QR.R)
        Qblock(k+1) .= Qnew
        B[k] = Bk
    end

    # Construct the tridiagonal block matrix T
    T = zeros(kmax*b, kmax*b)
    @views for k = 1:kmax
        T[(k-1)*b+1:k*b, (k-1)*b+1:k*b] .= A[k]
        if k > 1
            T[(k-1)*b+1:k*b, (k-2)*b+1:(k-1)*b] .= B[k-1]
            T[(k-2)*b+1:(k-1)*b, (k-1)*b+1:k*b] .= B[k-1]'
        end
    end

    return T, R_out
end





# Define adaptive stochastic Lanczos quadrature function
function stochastic_lanczos_quadrature_adaptive(f::Function, A::Function, ω::Vector, tol=eps())
    # Placeholder implementation for adaptive quadrature (complete as needed)
end

"""
    stochastic_lanczos_quadrature_fixed(f::Function, A::Function, ω::Vector, k::Int)

Uses k steps of the Stochastic Lanczos Quadrature to approximate the quadratic form ``\\omega^\\top f(A)\\omega``
"""
function stochastic_lanczos_quadrature(f::Function, A::Function, ω::Vector, k::Int)::Real
    T_k = lanczos(A, ω/norm(ω), k) # Note the initial vector has to be normalized
    E = eigen(T_k)
    U = E.vectors
    Θ = E.values

    return sum([ U[1, i]^2 * f(Θ[i]) for i = 1:k ])*norm(ω)^2
end

"""
    block_stochastic_lanczos_quadrature(f::Function, A::Function, Ω::Matrix, k::Int, b::Int, method::String="block")::Real

Uses k steps of the Stochastic Lanczos Quadrature to approximate the trace of the block quadratic form ``\\Omega^\\top f(A)\\Omega``. This can
either be done using the block Lanczos iteration on the matrix Ω, or using the unblocked iteration on the columns of Ω.
"""
function block_stochastic_lanczos_quadrature(f::Function, A::Function, Ω::Matrix, k::Int, method::String="block";
    reorthogonalization_fraction::Real=0.1)::Real
    if method == "single"
        total = 0
        for ω in eachcol(Ω)
            T_k = lanczos(A, ω/norm(ω), k) # Note the initial vector has to be normalized
            E = eigen(T_k)
            U = E.vectors
            Θ = E.values

            total += sum([ U[1, i]^2 * f(Θ[i]) for i = 1:k ])*norm(ω)^2

        end
    elseif method == "block"
        b = size(Ω, 2) # Block size
        T_k, R = block_lanczos(A, Ω, k; reorthogonalization_fraction=reorthogonalization_fraction)
        E = eigen(T_k)
        U = E.vectors
        Θ = E.values
        total = sum([ norm(R'*U[1:b, i])^2 * f(Θ[i]) for i = 1:k*b ])
    end

    return total 
end

# Export the main functions
export stochastic_lanczos_quadrature, block_stochastic_lanczos_quadrature, lanczos, block_lanczos, divisors, enforce_signs

end
