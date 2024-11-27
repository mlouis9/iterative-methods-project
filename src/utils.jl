module Utils

using Base.Threads
using Plots

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

    plot_scale = logscale ? :log10 : :ln
    p = plot()
    for (b_index, b) in enumerate(bs)
        plot!(p, ks[1:max_k_index], abs.(block_estimates[b_index, 1:max_k_index] .- converged_estimate), label="b=$b", xlabel="Lanczos Iterations (k)", 
        ylabel="Residual", gridlinewidth=2, yscale=plot_scale)
    end

    return p
end

export compute_block_estimates, plot_block_estimates

end