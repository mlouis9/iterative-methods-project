module Estimators

using ..StochasticLanczos
using Statistics
using LinearAlgebra

"""
    function hutchinson_estimator(f::Function, A::Function, n::Int, s::Int, b::Int, k::Int)::Real

Use a Hutchinson estimator with s samples to estimate the functional trace ``\\mathrm{tr}(f(\\mathbf{A}))``, where the s samples are 
blocked into blocks of size b and the block stochastic Lanczos quadrature is used to estimate the block functional trace for each of 
the blocks. Note b <= n. k is the number of iterations of block stochastic Lanczos to perform per block, choose it sufficiently large 
so that the estimate always converges.

# Arguments
- `f::Function`: The function applied to A
- `A::Function`: A function that applies A to a set of vectors X
- `n::Int`: The dimension of A (assumed to be ``\\mathbf{A}\\in \\mathbb{R}^{n\\times n}``)
- `s::Int`: The number of samples to use for the Hutchinson estimator
- `b::Int`: The block size (note ``b\\leq n`` and ``b`` must be a divisor of ``n`` otherwise an error will be thrown)
- `k::Int`: The number of steps of block stochastic Lanczos to perform
- 'Ω_provided::AbstractMatrix`: If a determinstic estimate is desired (i.e. one that can be compared with different `k`'s), then you may specify your own
    Ω matrix.

Returns the functional trace estimate
"""
function hutchinson_estimator(f::Function, A::Function, n::Int, s::Int, b::Int, k::Int; 
    Ω_provided::Union{Nothing, AbstractMatrix}=nothing, reorthogonalization_fraction::Real=0.1)::Real
    # Draw s random vectors of length n from a Rademacher distribution unless a set is already provided (primarily for benchmarking)
    Ω = Ω_provided === nothing ? rand([-1, 1], n, s) : Ω_provided

    # Do some input checks on the block size b
    if b > s
        error("The block size b=$b must be less than s=$s")
    elseif mod(s, b) != 0
        error("The block size b=$b must evenly divide the number of samples s=$s")
    end

    # Now partition the columns of Ω into blocks of size b and estimate the functional trace using the block stochastic Lanczos quadrature
    functional_trace_estimate = 0
    for i=1:s÷b
        block = Ω[ :, (i-1)*b + 1 : i*b ]
        functional_trace_estimate += block_stochastic_lanczos_quadrature(f, A, block, k, "block"; 
            reorthogonalization_fraction=reorthogonalization_fraction)
    end
    functional_trace_estimate /= s # Divide by s to complete the Hutchinson trace estimation

    return functional_trace_estimate
end

"""
    hutch_pp_estimator(f::Function, A::Function, n::Int, s::Int, b::Int, k::Int)

Applies the Hutch++ variance-reduced stochastic trace estimator to estimate the functional trace of A, where A is only accessible by
matrix-vector products.
"""
function hutch_pp_estimator(f::Function, A::Function, n::Int, s::Int, b::Int, k::Int;
    Ω_provided::Union{Nothing, AbstractMatrix}=nothing, reorthogonalization_fraction::Real=0.1)
    # Draw s random vectors
    Ω = Ω_provided === nothing ? rand([-1, 1], n, s) : Ω_provided

    # Input checks
    if mod(s, 2) != 0
        error("The number of samples must be divisible by 2, but you provided s=$(s)")
    end
    if b > s÷2
        error("The block size b=$b must be less than s/2=$(s/2)")
    elseif mod(s÷2, b) != 0
        error("The block size b=$b must evenly divide the remaining samples s/2=$(s/2)")
    end

    # --------------------------------
    # Form the low rank approximation
    # --------------------------------
    Y = A(Ω)
    QR = qr(Y)
    Q = Matrix(QR.Q)

    # Compute A_tilde = Q^T * A * Q
    A_tilde = Q' * A(Q)

    # Compute tr(f(A_tilde))
    E = eigen(A_tilde)
    Λ = E.values
    functional_trace_estimate = sum( f.(Λ) )

    # ---------------------------------------
    # Now estimate the trace of the residual
    # ---------------------------------------
    Ω_prime = ( I - Q*Q' ) * Ω[:, 1:s÷2]
    functional_trace_estimate += hutchinson_estimator(f, A, n, s÷2, b, k; Ω_provided=Ω_prime, 
        reorthogonalization_fraction=reorthogonalization_fraction)

    return functional_trace_estimate
end



"""
    estimator_statistics(estimator::Function, args...; num_samples::Int=50)::Tuple{Real, Real}

Estimate the mean and variance of a stochastic trace estimator

# Arguments
- `estimator::Function`: The stochastic trace estimator function (e.g. `hutchinson_estimator`) whose mean and variance you'd like to estimate
- `args`: The arguments used for the stochastic trace estimator
- `num_samples::Int`: The number of samples to use to estimate the estimator statistics

Returns the estimated sample mean and variance
"""
function estimator_statistics(estimator::Function, args...; num_samples::Int=50)::Tuple{Real, Real}
    # Collect num_samples samples of the estimator
    samples = [estimator(args...) for _ in 1:num_samples]

    # Compute sample mean and variance
    sample_mean = mean(samples)
    sample_variance = var(samples)

    return sample_mean, sample_variance
end

export hutchinson_estimator, estimator_statistics, hutch_pp_estimator

end