module Estimators

using ..StochasticLanczos
using Statistics

"""
    function hutchinson_estimator(f::Function, A::Function, n::Int, s::Int, b::Int, k::Int)::Real

Use a Hutchinson estimator with s samples to estimate the functional trace ``\\mathrm{tr}(f(\\mathbf{A}))``, where the s samples are 
blocked into blocks of size b and the block stochastic Lanczos quadrature is used to estimate the block functional trace for each of 
the blocks. Note b <= n. k is the number of iterations of block stochastic Lanczos to perform per block, choose it sufficiently large 
so that the estimate always converges.

# Arguments
- `f::Function`: The function applied to A
- `A::Function`: The function that gives matrix-vector products with A
- `n::Int`: The dimension of A (assumed to be ``\\mathbf{A}\\in \\mathbb{R}^{n\\times n}``)
- `s::Int`: The number of samples to use for the Hutchinson estimator
- `b::Int`: The block size (note ``b\\leq n`` and ``b`` must be a divisor of ``n`` otherwise an error will be thrown)
- `k::Int`: The number of steps of block stochastic Lanczos to perform
- 'Ω_provided::AbstractMatrix`: If a determinstic estimate is desired (i.e. one that can be compared with different `k`'s), then you may specify your own
    Ω matrix.

Returns the functional trace estimate
"""
function hutchinson_estimator(f::Function, A::Function, n::Int, s::Int, b::Int, k::Int; Ω_provided::Union{Nothing, AbstractMatrix}=nothing)::Real
    # Draw s random vectors of length n from a Rademacher distribution unless a set is already provided (primarily for benchmarking)
    Ω = Ω_provided === nothing ? [rand([-1, 1]) for _ in 1:n, _ in 1:s] : Ω_provided

    # Do some input checks on the block size b
    if b > n
        error("The block size b=$b must be less than s=$s")
    elseif mod(s, b) != 0
        error("The block size b=$b must evenly divide the number of samples s=$s")
    end

    # Now partition the columns of Ω into blocks of size b and estimate the functional trace using the block stochastic Lanczos quadrature
    functional_trace_estimate = 0
    for i=1:s÷b
        block = Ω[ :, (i-1)*b + 1 : i*b ]
        functional_trace_estimate += block_stochastic_lanczos_quadrature(f, A, block, k, "block")
    end
    functional_trace_estimate /= s # Divide by s to complete the Hutchinson trace estimation

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

export hutchinson_estimator, estimator_statistics

end