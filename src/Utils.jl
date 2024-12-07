module Utils

using Base.Threads
using Plots
using ..Estimators

"""
Used for computing convergence of Lanczos iteration to Hutch estimator for a fixed set of samples as a function of block size b
"""
function compute_block_estimates(ks::AbstractVector, bs::AbstractVector, hutchinson_estimator::Function)
    block_estimates = zeros(length(bs), length(ks))

    # Now, compute all of the estimates
    Threads.@threads for idx in CartesianIndices(block_estimates)
        b_index, k_index = Tuple(idx)
        b = bs[b_index]
        k = ks[k_index]
        block_estimates[b_index, k_index] = hutchinson_estimator(b, k)
    end

    return block_estimates
end

"""
Used for plotting convergence of Lanczos iteration to Hutch estimator for a fixed set of samples as a function of block size b.

Note the residual is with respect to the converged Hutch estimate for a given sample size (not necessarily the converged trace estimate)
"""
function plot_block_estimates(block_estimates, ks, bs, converged_estimate; k_max=ks[end], logscale=true)
    max_k_index = findfirst(>=(k_max), ks)

    plot_scale = logscale ? :log10 : :identity
    p = plot()
    for (b_index, b) in enumerate(bs)
        plot!(p, ks[1:max_k_index], abs.(block_estimates[b_index, 1:max_k_index] .- converged_estimate), label="b=$b", xlabel="Lanczos Iterations (k)", 
        ylabel="Residual", gridlinewidth=2, yscale=plot_scale)
    end

    return p
end

"""
Used for plotting convergence of Lanczos iteration to Hutch estimator for a fixed set of samples as a function of block size b.

Note the residual is with respect to the converged Hutch estimate for a given sample size (not necessarily the converged trace estimate)
"""
function plot_block_variance_estimates(block_estimates, block_stddev_of_variance_estimates, ks, bs; k_max=ks[end], logscale=false)
    max_k_index = findfirst(>=(k_max), ks)

    plot_scale = logscale ? :log10 : :identity

    palette = cgrad(:rainbow, length(bs)).colors
    p = plot()
    for (b_index, b) in enumerate(bs)
        x = ks[1:max_k_index]
        y = block_estimates[b_index, 1:max_k_index]
        σ = block_stddev_of_variance_estimates[b_index, 1:max_k_index]
        color = palette[b_index]
        plot!(p, x, y, label="b=$b", xlabel="Lanczos Iterations (k)", ylabel="Estimator Variance", gridlinewidth=2, yscale=plot_scale,
            color=color)
        plot!(p, x, y .+ σ,  fill_between=(y .- σ), color=color, alpha=0.1, label="")
        plot!(p, x, y .+ 2σ, fill_between=(y .- 2σ), color=color, alpha=0.05, label="")
    end

    return p
end

"""
    create_matvecA(A::Union{Function, AbstractArray})::Function

This is a utility function that creates functions that apply the matrix A to a given set of vectors. The main difference is whether
A is provided explicitly as a matrix or as a function that returns matvecs (if A is not obtainable explicitly), wherein different methods
are dispatched to form block matrix vector products with A. Most importantly, if A is given explicitly, we can take advantage of BLAS level
3 when forming block matrix vector products with A.
"""
function create_matvecA(A::Union{Function, AbstractArray})::Function
    if typeof(A) <: Function
        return X::AbstractArray -> hcat([A(xcol) for xcol in eachcol(X)]...)
    else # A is given explicitly
        return X::AbstractArray -> A * X
    end
end

export compute_block_estimates, plot_block_estimates, plot_block_variance_estimates, create_matvecA

end