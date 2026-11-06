# Hansen (1992), with the 1996 erratum: the standardized likelihood ratio test.

# The grids of p and q of section 4.1 (Grids 1 to 3) and 4.3 (Grids 4 and 5).
const GRIDS = (0.20:0.20:0.80, 0.15:0.15:0.90, 0.12:0.11:0.89, 0.10:0.075:0.925, 0.10:0.05:0.90)

# The series the functions take: a vector of growth rates or returns as it is, or the growth rates of a
# column of levels of a Tables.jl table (see `growth`).
series(data::AbstractVector, column) = Float64.(data)
series(data, column) = growth(data, column)

# The grid of p and q: Grid 1 to 5 of the article when an integer, else the probabilities given.
pqgrid(grid::Integer) = GRIDS[grid]
pqgrid(grid) = grid

# The default grid of mud, the difference between the regimes: Hansen's values (0.1, ..., 2 for a switch
# in the mean or the intercept, 0.2, ..., 2 for Models A and B) times the standard deviation of the series.
differences(x, model) = (model in (:modelA, :modelB) ? (0.2:0.2:2.0) : (0.1:0.1:2.0)) .* std(x)

function checkorder(order, model = :intercept)
	order >= 1 || throw(ArgumentError("the AR order p must be at least 1"))
	model in (:modelA, :modelB) && order < 2 && throw(ArgumentError("Models A and B switch phi2: the AR order p must be at least 2"))
	model in (:hamilton, :intercept, :modelA, :modelB) || throw(ArgumentError("unknown model $model"))
end

# The sample of an AR(order): the dependent variable and its lags.
function sample(x, order)
	n = length(x)
	y = x[order+1:n]
	lags = reduce(hcat, [x[order+1-j:n-j] for j in 1:order])
	return y, lags
end

lognormal(e, σ) = -0.5 * log(2π) - log(σ) - 0.5 * (e / σ)^2

# The log likelihood of each observation of Hamilton's model (11), x(t) = mu + mud s(t) + u(t),
# phi(L)u(t) = e(t), of AR order r = size(lags, 2), with θ = (mu, phi1, ..., phir, sigma, mud, p, q): the
# filter runs over the 2^(r + 1) states of (s(t), ..., s(t-r)) (bit k of a state is s(t-k)), started from
# the stationary distribution of (s(t-1), ..., s(t-r)).
function hamilton_loglik(θ, y, lags)
	order = size(lags, 2)
	μ, φ, σ, μd, p, q = θ[1], θ[2:order+1], abs(θ[order+2]), θ[order+3], θ[order+4], θ[order+5]
	T = eltype(θ)
	transition(now, before) = before == 1 ? (now == 1 ? p : 1 - p) : (now == 1 ? 1 - q : q)
	ergodic = (1 - q) / (2 - p - q)
	states = 1 << order
	ξ = zeros(T, states)  # (s(t-1), ..., s(t-r)): bit k - 1 is s(t-k)
	for m in 0:states-1
		s = [(m >> k) & 1 for k in 0:order-1]
		ξ[m+1] = (s[order] == 1 ? ergodic : 1 - ergodic) * prod((transition(s[k], s[k+1]) for k in 1:order-1); init = one(T))
	end
	ll = zeros(T, length(y))
	joint = zeros(T, 2states)
	for t in eachindex(y)
		total = zero(T)
		for j in 0:2states-1
			m = j >> 1
			e = y[t] - μ - μd * (j & 1)
			for k in 1:order
				e -= φ[k] * (lags[t, k] - μ - μd * ((j >> k) & 1))
			end
			joint[j+1] = ξ[m+1] * transition(j & 1, m & 1) * exp(lognormal(e, σ))
			total += joint[j+1]
		end
		ll[t] = log(total)
		for m in 0:states-1
			ξ[m+1] = (joint[m+1] + joint[m+1+states]) / total
		end
	end
	return ll
end

# The log likelihood of each observation of the switching parameters model,
# x(t) = X(t)'(b + s(t) bd) + sigma(s(t)) e(t), X(t) = (1, x(t-1), ..., x(t-r)), sigma(0) = |sigma|,
# sigma(1) = |sigma| + |sigmad|, with a first-order Markov chain s(t) started from its stationary
# distribution.
function switching_loglik(b, σ, bd, σd, p, q, y, X)
	T = promote_type(eltype(b), eltype(bd), typeof(σ), typeof(σd), typeof(p), typeof(q))
	σ0 = abs(σ)
	σ1 = σ0 + abs(σd)
	ξ = (1 - q) / (2 - p - q)  # P(s(t-1) = 1 | the past)
	ll = zeros(T, length(y))
	for t in eachindex(y)
		x = view(X, t, :)
		e0 = y[t] - dot(x, b)
		e1 = e0 - dot(x, bd)
		one = (1 - q) + (p - (1 - q)) * ξ
		f1 = one * exp(lognormal(e1, σ1))
		total = f1 + (1 - one) * exp(lognormal(e0, σ0))
		ξ = f1 / total
		ll[t] = log(total)
	end
	return ll
end

logistic(u) = 1 / (1 + exp(-u))
logit(p) = log(p / (1 - p))

# Maximizes f over x by BFGS with forward-mode derivatives, from x0: the maximizer and the maximum.
function hill_climb(f, x0)
	objective(x) = -f(x)
	gradient!(G, x) = ForwardDiff.gradient!(G, objective, x)
	result = optimize(objective, gradient!, float.(x0), BFGS(),
		Optim.Options(g_tol = 1e-9, f_reltol = 1e-14, iterations = 5000))
	return Optim.minimizer(result), -Optim.minimum(result)
end

# White's heteroscedasticity-consistent covariance of the maximum likelihood estimates θ of the
# log likelihoods loglik(θ), one for each observation: H⁻¹ (Σ s(t) s(t)') H⁻¹.
function white(loglik, θ)
	scores = ForwardDiff.jacobian(loglik, θ)
	hessian = ForwardDiff.hessian(θ -> sum(loglik(θ)), θ)
	h = inv(hessian)
	return Symmetric(h * (scores' * scores) * h)
end

# Maximum likelihood with p and q (the last entries of θ, `probabilities` of them) estimated on the
# logit scale: the estimates, their White covariance and the log likelihood.
function mle(loglik, starts, probabilities)
	k = length(first(starts))
	free(u) = [u[1:k-probabilities]; logistic.(u[k-probabilities+1:k])]
	best = nothing
	for start in starts
		u0 = [start[1:k-probabilities]; logit.(start[k-probabilities+1:k])]
		u, value = hill_climb(u -> sum(loglik(free(u))), u0)
		(best === nothing || value > best[2]) && (best = (u, value))
	end
	θ = free(best[1])
	return θ, white(loglik, θ), sum(loglik(θ))
end

table(θ, V, ll) = [θ sqrt.(diag(V)); ll NaN]

# The OLS estimates of the AR(order), intercept first, and the maximum likelihood estimate of sigma.
function ar(x, order)
	y, lags = sample(x, order)
	X = [ones(length(y)) lags]
	b = X \ y
	return b, sqrt(mean(abs2, y - X * b))
end

"""
    ar_estimates(data; column = :gnp, p = 4)

The maximum likelihood estimates of the Gaussian AR(p), phi(L)x(t) = mu + e(t), of a series: `data` is
a vector of growth rates or returns, or a Tables.jl table whose `column` holds levels (see
[`growth`](@ref)). A (p + 3) × 2 matrix: mu (the intercept), phi1 to phip and sigma in its rows, their
estimates and White's heteroscedasticity-consistent standard errors in its columns, then the log
likelihood (NaN for its standard error). Table I of Hansen (1992) is `ar_estimates(gnp82)`, US real GNP
growth, 1952:2 to 1984:4.
"""
function ar_estimates(data; column = :gnp, p = 4)
	checkorder(p)
	x = series(data, column)
	y, lags = sample(x, p)
	X = [ones(length(y)) lags]
	loglik(θ) = lognormal.(y - X * θ[1:p+1], θ[p+2])
	b, σ = ar(x, p)
	θ = [b; σ]
	return table(θ, white(loglik, θ), sum(loglik(θ)))
end

"""
    hamilton_estimates(data; column = :gnp, p = 4, mud = nothing)

The maximum likelihood estimates of Hamilton's Markov switching model (11), x(t) = mu + mud s(t) + u(t),
phi(L)u(t) = e(t) of order p, with P = P(s(t) = 1 | s(t-1) = 1) and Q = P(s(t) = 0 | s(t-1) = 0), of a
series: `data` is a vector of growth rates or returns, or a Tables.jl table whose `column` holds levels
(see [`growth`](@ref)). The maximization starts, as in Hansen's program, from the point of Grid 3 of P
and Q and of the grid `mud` of the difference between the regimes (by default 0.1, ..., 2 times the
standard deviation of the series) with the largest likelihood ratio. A (p + 6) × 2 matrix: mu, mud,
phi1 to phip, sigma, P and Q in its rows, their estimates and White's standard errors in its columns,
then the log likelihood (NaN for its standard error). Table II of Hansen (1992) is
`hamilton_estimates(gnp82; mud = 0.1:0.1:2.0)`.
"""
function hamilton_estimates(data; column = :gnp, p = 4, mud = nothing)
	checkorder(p, :hamilton)
	x = series(data, column)
	y, lags = sample(x, p)
	start = best_grid_point(x, :hamilton, GRIDS[3], (something(mud, differences(x, :hamilton)),), p)
	loglik(θ) = hamilton_loglik(θ, y, lags)
	θ, V, ll = mle(loglik, [start], 2)
	order = [1; p + 3; 2:p+2; p + 4; p + 5]
	return table(θ[order], V[order, order], ll)
end

"""
    intercept_estimates(data; column = :gnp, p = 4, mud = nothing)

The maximum likelihood estimates of the Markov switching intercept model (12),
phi(L)x(t) = mu + mud s(t) + e(t) of order p, of a series: `data` is a vector of growth rates or returns,
or a Tables.jl table whose `column` holds levels (see [`growth`](@ref)). The maximization starts, as in
Hansen's program, from the point of Grid 3 of P and Q and of the grid `mud` (by default 0.1, ..., 2 times
the standard deviation of the series) with the largest likelihood ratio. A (p + 6) × 2 matrix: mu, mud,
phi1 to phip, sigma, P and Q in its rows, their estimates and White's standard errors in its columns,
then the log likelihood (NaN for its standard error). Table V of Hansen (1992) is
`intercept_estimates(gnp82; mud = 0.1:0.1:2.0)`.
"""
function intercept_estimates(data; column = :gnp, p = 4, mud = nothing)
	checkorder(p)
	x = series(data, column)
	y, lags = sample(x, p)
	X = [ones(length(y)) lags]
	start = best_grid_point(x, :intercept, GRIDS[3], (something(mud, differences(x, :intercept)),), p)
	loglik(θ) = switching_loglik(θ[1:p+1], θ[p+2], [θ[p+3]; zeros(p)], 0.0, θ[p+4], θ[p+5], y, X)
	θ, V, ll = mle(loglik, [start], 2)
	order = [1; p + 3; 2:p+2; p + 4; p + 5]
	return table(θ[order], V[order, order], ll)
end

# The grids of the switching parameters of a model: mud, then phi2d for Models A and B.
switchgrids(x, model, mud, phi2d) = let m = [something(mud, differences(x, model));]
	model in (:modelA, :modelB) ? (m, [phi2d;]) : (m,)
end

# The log likelihood of each observation under the alternative, at the switching parameters α,
# transition probabilities p and q, and θ, the parameters concentrated out: (mu, phi1, ..., phir, sigma)
# with mu the mean for Hamilton's model, the intercept for the others.
function alternative(model, θ, α, p, q, y, lags, X)
	model == :hamilton && return hamilton_loglik([θ; α[1]; p; q], y, lags)
	order = size(lags, 2)
	bd = zeros(eltype(α), order + 1)
	bd[1] = α[1]
	length(α) > 1 && (bd[3] = α[2])
	return switching_loglik(θ[1:order+1], θ[order+2], bd, 0.0, p, q, y, X)
end

struct GridSearch
	d::Matrix{Float64}       # l1(t) - l0(t), one column for each grid point
	points::Vector{Vector{Float64}}  # (switching parameters..., p, q) of each grid point
	thetas::Vector{Vector{Float64}}  # the parameters concentrated out at each grid point
end

const SEARCHES = Dict{Any, GridSearch}()

# The grid search of section 3.3: for each (p, q) of the grid `pq` of probabilities (q = 1 - p for
# Model A), the likelihood is maximized over θ at each value of the switching parameters, on their grids
# `values`, starting from the null estimates at the first value of the last switching parameter, then
# from the previous result.
function grid_search(x, model, pq, values, order)
	get!(SEARCHES, (model, collect(pq), values, order, x)) do
		y, lags = sample(x, order)
		X = [ones(length(y)) lags]
		b, σ = ar(x, order)
		null = model == :hamilton ? [b[1] / (1 - sum(b[2:end])); b[2:end]; σ] : [b; σ]
		l0 = lognormal.(y - X * b, σ)
		pairs = model == :modelA ? [(p, 1 - p) for p in pq] : [(p, q) for p in pq for q in pq]
		outer = length(values) == 1 ? [Float64[]] : [[v] for v in values[1]]
		d, points, thetas = Vector{Float64}[], Vector{Float64}[], Vector{Float64}[]
		for (p, q) in pairs, o in outer
			start = null
			for v in values[end]
				α = [o; v]
				θ, _ = hill_climb(θ -> sum(alternative(model, θ, α, p, q, y, lags, X)), start)
				push!(d, alternative(model, θ, α, p, q, y, lags, X) - l0)
				push!(points, [α; p; q])
				push!(thetas, θ)
				start = θ
			end
		end
		GridSearch(reduce(hcat, d), points, thetas)
	end
end

# The grid point with the largest (unstandardized) likelihood ratio, as a starting value for the
# maximum likelihood estimates: (θ..., switching parameters..., p, q).
function best_grid_point(x, model, pq, values, order)
	search = grid_search(x, model, pq, values, order)
	i = argmax(vec(sum(search.d; dims = 1)))
	return [search.thetas[i]; search.points[i]]
end

# The standardized likelihood ratios LR(α)/V(α)^(1/2) of each grid point, and the normalized
# likelihood differences q(t, α)/V(α)^(1/2), V(α) the sum of squares of the demeaned q(t, α).
function standardized(search)
	centered = search.d .- mean(search.d; dims = 1)
	scale = sqrt.(sum(abs2, centered; dims = 1))
	return vec(sum(search.d; dims = 1) ./ scale), centered ./ scale
end

# The unconstrained switching parameters model of Table VIII, θ = (mu, phi1, ..., phi4, sigma, mud,
# phi1d, ..., phi4d, sigmad, p, q), estimated from the grid point of Model B (Grid 3, the grid `mud`)
# with the largest likelihood ratio.
function unconstrained(x, mud)
	y, lags = sample(x, 4)
	X = [ones(length(y)) lags]
	s = best_grid_point(x, :modelB, GRIDS[3], switchgrids(x, :modelB, mud, -1.0:0.2:0.8), 4)  # (mu, phi1, ..., phi4, sigma, mud, phi2d, p, q)
	start = [s[1:6]; s[7]; 0.0; s[8]; 0.0; 0.0; 0.01; s[9]; s[10]]
	loglik(θ) = switching_loglik(θ[1:5], θ[6], θ[7:11], θ[12], θ[13], θ[14], y, X)
	return mle(loglik, [start], 2)
end

"""
    switching_estimates(data; column = :gnp, mud = nothing)

Table VIII of Hansen (1992), `switching_estimates(gnp82; mud = 0.2:0.2:2.0)`: the maximum likelihood
estimates of the Markov switching parameters model, in which the intercept, the AR(4) coefficients and
the error standard deviation all switch between the states (a `d` ends the name of the difference
between state 1 and state 0; sigma(1) = sigma + |sigmad|), of a series: `data` is a vector of growth
rates or returns, or a Tables.jl table whose `column` holds levels (see [`growth`](@ref)); `mud` is the
grid of the starting search (by default 0.2, ..., 2 times the standard deviation of the series). Unconstrained,
and constrained by phi3 = phi4 = phi1d = phi3d = phi4d = sigmad = 0 and q = 1 - p. The constrained
model, an AR(2), is estimated on the AR(2) sample, 1951:4 to 1984:4 (n = 133): that is the sample of
the published constrained estimates and likelihood (on the AR(4) sample the likelihood is -174.893, not
-176.990). The maximizations start from the grid points of Model B and Model A (Grid 3) with the
largest likelihood ratio. A 15 × 4
matrix: mu, mud, phi1 to phi4, phi1d to phi4d, sigma, sigmad, p and q in its rows, then the log
likelihood; the unconstrained estimates and White's standard errors, then the constrained ones, in
its columns (NaN where a parameter is constrained).
"""
function switching_estimates(data; column = :gnp, mud = nothing)
	x = series(data, column)
	θ, V, ll = unconstrained(x, mud)
	order = [1, 7, 2, 3, 4, 5, 8, 9, 10, 11, 6, 12, 13, 14]
	free = table(θ[order], V[order, order], ll)
	# constrained, (mu, phi1, phi2, sigma, mud, phi2d, p): an AR(2), on its own sample, 1951:4 to 1984:4
	s = best_grid_point(x, :modelA, GRIDS[3], switchgrids(x, :modelA, mud, -1.0:0.2:0.8), 4)  # (mu, phi1, ..., phi4, sigma, mud, phi2d, p, q)
	y2 = x[3:end]
	X2 = [ones(length(y2)) x[2:end-1] x[1:end-2]]
	loglik(c) = switching_loglik(c[1:3], c[4], [c[5]; 0; c[6]], 0.0, c[7], 1 - c[7], y2, X2)
	c, W, lc = mle(loglik, [[s[1:3]; s[6:8]; s[9]]], 1)
	constrained = fill(NaN, 15, 2)
	for (row, i) in zip([1, 2, 3, 4, 8, 11, 13], [1, 5, 2, 3, 6, 4, 7])
		constrained[row, :] = [c[i], sqrt(W[i, i])]
	end
	constrained[15, 1] = lc
	return [free constrained]
end

"""
    wald_tests(data; column = :gnp, mud = nothing)

Table IX of Hansen (1992), `wald_tests(gnp82; mud = 0.2:0.2:2.0)`: Wald tests computed from the
unconstrained estimates of [`switching_estimates`](@ref) and their White covariance, `data`, `column`
and `mud` as there: of the switching intercept model (phi1d = phi2d = phi3d = phi4d = sigmad = 0), and of
the simple switching parameters model (phi3 = phi4 = phi1d = phi3d = phi4d = sigmad = 0 and
p + q = 1). A 2 × 3 matrix: the tests in its rows, the statistic, its degrees of freedom and its
chi-squared p-value in its columns.
"""
function wald_tests(data; column = :gnp, mud = nothing)
	θ, V, _ = unconstrained(series(data, column), mud)
	unit(i) = [j == i ? 1.0 : 0.0 for j in 1:14]'
	intercept = reduce(vcat, unit.([8, 9, 10, 11, 12]))
	simple = [reduce(vcat, unit.([4, 5, 8, 10, 11, 12])); unit(13) + unit(14)]
	rows = map([(intercept, zeros(5)), (simple, [zeros(6); 1])]) do (R, r)
		e = R * θ - r
		w = e' * ((R * V * R') \ e)
		[w size(R, 1) ccdf(Chisq(size(R, 1)), w)]
	end
	return reduce(vcat, rows)
end

const MODELS ="`:hamilton` (Hamilton's model (11), the mean switches), `:intercept` (the switching intercept model (12)), `:modelA` or `:modelB` (the intercept and phi2 switch, with q = 1 - p or q free)"

# The standardized likelihood ratios of each grid of p and q in `grids`.
function searches(data, model, column, p, grids, mud, phi2d)
	checkorder(p, model)
	x = series(data, column)
	values = switchgrids(x, model, mud, phi2d)
	return [standardized(grid_search(x, model, pqgrid(g), values, p)) for g in grids]
end

"""
    standardized_lr(data, model; column = :gnp, p = 4, grids = 1:3, mud = nothing, phi2d = -1.0:0.2:0.8)

The standardized likelihood ratio statistic LR* of Hansen (1992) of the null of one regime, an AR(p),
against two regimes that switch by a Markov chain, as in `model`, one of $MODELS: the supremum over the
grid of LR(α)/V(α)^(1/2), α the switching parameters and the transition probabilities. `data` is a
vector of growth rates or returns, or a Tables.jl table whose `column` holds levels (see
[`growth`](@ref)). The grids: `mud`, the difference between the regimes (by default 0.1, ..., 2 times the
standard deviation of the series, 0.2, ..., 2 times for Models A and B); `phi2d`, the difference in
phi2 (Models A and B); `grids`, those of p and q, each an integer for Grid 1 to 5 of the article or a
range of probabilities such as `0.1:0.1:0.9`. A column: one row for each grid in `grids`. Tables III,
VI and X of the erratum are `standardized_lr(gnp82, :hamilton; mud = 0.1:0.1:2.0)`, with `:intercept`
and `grids = 1:5`, and with `:modelA` or `:modelB`, `grids = 1:4` and `mud = 0.2:0.2:2.0`.
"""
function standardized_lr(data, model::Symbol; column = :gnp, p = 4, grids = 1:3, mud = nothing, phi2d = -1.0:0.2:0.8)
	return reshape([maximum(first(s)) for s in searches(data, model, column, p, grids, mud, phi2d)], :, 1)
end

"""
    lr_pvalues(data, model; column = :gnp, p = 4, grids = 1:3, mud = nothing, phi2d = -1.0:0.2:0.8,
               bandwidths = 0:4, reps = 1000, seed = 1992)

The p-values of the standardized LR statistic of [`standardized_lr`](@ref) (the same `data`, `model`,
`column`, `p` and grids), by the erratum's method (3'): `reps` draws of the supremum over the grid of
sum(k = 0..M) sum(i) q(i, α) u(i + k) / (sqrt(1 + M) V(α)^(1/2)), u standard normal (from a StableRNG
with `seed`, the same draws for every grid point and every set of `bandwidths`), for each Bartlett
bandwidth M in `bandwidths`. The distribution of LR* is only bounded, so the p-values are conservative.
A matrix: one row for each grid in `grids`, one column for each M, as in Tables III, VI and X of the
erratum.
"""
function lr_pvalues(data, model::Symbol; column = :gnp, p = 4, grids = 1:3, mud = nothing, phi2d = -1.0:0.2:0.8,
		bandwidths = 0:4, reps = 1000, seed = 1992)
	found = searches(data, model, column, p, grids, mud, phi2d)
	result = zeros(length(grids), length(bandwidths))
	for (r, (statistics, Q)) in enumerate(found)
		n = size(Q, 1)
		# drawn observation by observation, so that u(1), ..., u(n + M) do not depend on the bandwidths asked for
		u = permutedims(randn(StableRNG(seed), reps, n + maximum(bandwidths)))
		sums = zeros(size(Q, 2), reps)
		for k in 0:maximum(bandwidths)
			sums .+= Q' * u[1+k:n+k, :]
			c = findfirst(==(k), bandwidths)
			c === nothing && continue
			draws = vec(maximum(sums; dims = 1)) ./ sqrt(1 + k)
			result[r, c] = mean(draws .> maximum(statistics))
		end
	end
	return result
end
