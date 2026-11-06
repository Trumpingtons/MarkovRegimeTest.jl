# Hansen (1992), with the 1996 erratum: the standardized likelihood ratio test.

const LAGS = 4

# The grids of p and q of section 4.1 (Grids 1 to 3) and 4.3 (Grids 4 and 5).
const GRIDS = (0.20:0.20:0.80, 0.15:0.15:0.90, 0.12:0.11:0.89, 0.10:0.075:0.925, 0.10:0.05:0.90)

# The sample of an AR(4): the dependent variable, 1952Q2 to 1984Q4 (n = 131), and its four lags.
function sample(x)
	n = length(x)
	y = x[LAGS+1:n]
	lags = reduce(hcat, [x[LAGS+1-j:n-j] for j in 1:LAGS])
	return y, lags
end

lognormal(e, σ) = -0.5 * log(2π) - log(σ) - 0.5 * (e / σ)^2

# The log likelihood of each observation of Hamilton's model (11), x(t) = mu + mud s(t) + u(t),
# phi(L)u(t) = e(t), with θ = (mu, phi1, ..., phi4, sigma, mud, p, q): the filter runs over the 32
# states of (s(t), ..., s(t-4)) (bit k of a state is s(t-k)), started from the stationary
# distribution of (s(t-1), ..., s(t-4)).
function hamilton_loglik(θ, y, lags)
	μ, φ, σ, μd, p, q = θ[1], θ[2:5], abs(θ[6]), θ[7], θ[8], θ[9]
	T = eltype(θ)
	transition(now, before) = before == 1 ? (now == 1 ? p : 1 - p) : (now == 1 ? 1 - q : q)
	ergodic = (1 - q) / (2 - p - q)
	ξ = zeros(T, 16)  # (s(t-1), ..., s(t-4)): bit k - 1 is s(t-k)
	for m in 0:15
		s = [(m >> k) & 1 for k in 0:3]
		ξ[m+1] = (s[4] == 1 ? ergodic : 1 - ergodic) * prod(transition(s[k], s[k+1]) for k in 1:3)
	end
	ll = zeros(T, length(y))
	joint = zeros(T, 32)
	for t in eachindex(y)
		total = zero(T)
		for j in 0:31
			m = j >> 1
			e = y[t] - μ - μd * (j & 1)
			for k in 1:LAGS
				e -= φ[k] * (lags[t, k] - μ - μd * ((j >> k) & 1))
			end
			joint[j+1] = ξ[m+1] * transition(j & 1, m & 1) * exp(lognormal(e, σ))
			total += joint[j+1]
		end
		ll[t] = log(total)
		for m in 0:15
			ξ[m+1] = (joint[m+1] + joint[m+17]) / total
		end
	end
	return ll
end

# The log likelihood of each observation of the switching parameters model,
# x(t) = X(t)'(b + s(t) bd) + sigma(s(t)) e(t), X(t) = (1, x(t-1), ..., x(t-4)), sigma(0) = |sigma|,
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

# The OLS estimates of the AR(4), intercept first, and the maximum likelihood estimate of sigma.
function ar(x)
	y, lags = sample(x)
	X = [ones(length(y)) lags]
	b = X \ y
	return b, sqrt(mean(abs2, y - X * b))
end

"""
    ar_estimates(gnp)

Table I: the maximum likelihood estimates of the Gaussian AR(4), phi(L)x(t) = mu + e(t), of the growth
rate of US real GNP (1952:2 to 1984:4), from a Tables.jl table of GNP levels (see [`growth`](@ref)).
A 7 × 2 matrix: mu (the intercept), phi1 to phi4 and sigma in its rows, their estimates and White's
heteroscedasticity-consistent standard errors in its columns, then the log likelihood (NaN for its
standard error).
"""
function ar_estimates(gnp)
	x = growth(gnp)
	y, lags = sample(x)
	X = [ones(length(y)) lags]
	loglik(θ) = lognormal.(y - X * θ[1:5], θ[6])
	b, σ = ar(x)
	θ = [b; σ]
	return table(θ, white(loglik, θ), sum(loglik(θ)))
end

"""
    hamilton_estimates(gnp)

Table II: the maximum likelihood estimates of Hamilton's Markov switching model (11),
x(t) = mu + mud s(t) + u(t), phi(L)u(t) = e(t), with p = P(s(t) = 1 | s(t-1) = 1) and
q = P(s(t) = 0 | s(t-1) = 0), from a Tables.jl table of GNP levels (see [`growth`](@ref)). The
maximization starts, as in Hansen's program, from the grid point of Grid 3 with the largest likelihood
ratio. A 10 × 2 matrix: mu, mud, phi1 to phi4, sigma, p and q in its rows, their estimates and White's
standard errors in its columns, then the log likelihood (NaN for its standard error).
"""
function hamilton_estimates(gnp)
	x = growth(gnp)
	y, lags = sample(x)
	start = best_grid_point(x, :hamilton, 3)
	loglik(θ) = hamilton_loglik(θ, y, lags)
	θ, V, ll = mle(loglik, [start], 2)
	order = [1, 7, 2, 3, 4, 5, 6, 8, 9]
	return table(θ[order], V[order, order], ll)
end

"""
    intercept_estimates(gnp)

Table V: the maximum likelihood estimates of the Markov switching intercept model (12),
phi(L)x(t) = mu + mud s(t) + e(t), from a Tables.jl table of GNP levels (see [`growth`](@ref)). The
maximization starts, as in Hansen's program, from the grid point of Grid 3 with the largest likelihood
ratio. A 10 × 2 matrix: mu, mud, phi1 to phi4, sigma, p and q in its rows, their estimates and White's
standard errors in its columns, then the log likelihood (NaN for its standard error).
"""
function intercept_estimates(gnp)
	x = growth(gnp)
	y, lags = sample(x)
	X = [ones(length(y)) lags]
	start = best_grid_point(x, :intercept, 3)
	loglik(θ) = switching_loglik(θ[1:5], θ[6], [θ[7]; zeros(4)], 0.0, θ[8], θ[9], y, X)
	θ, V, ll = mle(loglik, [start], 2)
	order = [1, 7, 2, 3, 4, 5, 6, 8, 9]
	return table(θ[order], V[order, order], ll)
end

# The models of the standardized LR tests: the values of the switching parameters on their grids
# (mud, then phi2d for Models A and B) and whether q = 1 - p.
const SWITCHING = Dict(
	:hamilton => ([0.1:0.1:2.0;],),
	:intercept => ([0.1:0.1:2.0;],),
	:modelA => ([0.2:0.2:2.0;], [-1.0:0.2:0.8;]),
	:modelB => ([0.2:0.2:2.0;], [-1.0:0.2:0.8;]),
)

# The log likelihood of each observation under the alternative, at the switching parameters α,
# transition probabilities p and q, and θ, the parameters concentrated out: (mu, phi1, ..., phi4, sigma)
# with mu the mean for Hamilton's model, the intercept for the others.
function alternative(model, θ, α, p, q, y, lags, X)
	model == :hamilton && return hamilton_loglik([θ; α[1]; p; q], y, lags)
	bd = zeros(eltype(α), LAGS + 1)
	bd[1] = α[1]
	length(α) > 1 && (bd[3] = α[2])
	return switching_loglik(θ[1:5], θ[6], bd, 0.0, p, q, y, X)
end

struct GridSearch
	d::Matrix{Float64}       # l1(t) - l0(t), one column for each grid point
	points::Vector{Vector{Float64}}  # (switching parameters..., p, q) of each grid point
	thetas::Vector{Vector{Float64}}  # the parameters concentrated out at each grid point
end

const SEARCHES = Dict{Any, GridSearch}()

# The grid search of section 3.3: for each (p, q) of Grid `grid` (q = 1 - p for Model A), the
# likelihood is maximized over θ at each value of the switching parameters, starting from the null
# estimates at the first value of the last switching parameter, then from the previous result.
function grid_search(x, model, grid)
	get!(SEARCHES, (model, grid, x)) do
		y, lags = sample(x)
		X = [ones(length(y)) lags]
		b, σ = ar(x)
		null = model == :hamilton ? [b[1] / (1 - sum(b[2:5])); b[2:5]; σ] : [b; σ]
		l0 = lognormal.(y - X * b, σ)
		values = SWITCHING[model]
		pairs = model == :modelA ? [(p, 1 - p) for p in GRIDS[grid]] : [(p, q) for p in GRIDS[grid] for q in GRIDS[grid]]
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
function best_grid_point(x, model, grid)
	search = grid_search(x, model, grid)
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
# phi1d, ..., phi4d, sigmad, p, q), estimated from the grid point of Model B (Grid 3) with the largest
# likelihood ratio.
function unconstrained(x)
	y, lags = sample(x)
	X = [ones(length(y)) lags]
	s = best_grid_point(x, :modelB, 3)  # (mu, phi1, ..., phi4, sigma, mud, phi2d, p, q)
	start = [s[1:6]; s[7]; 0.0; s[8]; 0.0; 0.0; 0.01; s[9]; s[10]]
	loglik(θ) = switching_loglik(θ[1:5], θ[6], θ[7:11], θ[12], θ[13], θ[14], y, X)
	return mle(loglik, [start], 2)
end

"""
    switching_estimates(gnp)

Table VIII: the maximum likelihood estimates of the Markov switching parameters model of US real GNP,
in which the intercept, the AR(4) coefficients and the error standard deviation all switch between
the states (a `d` ends the name of the difference between state 1 and state 0;
sigma(1) = sigma + |sigmad|), from a Tables.jl table of GNP levels (see [`growth`](@ref)): unconstrained,
and constrained by phi3 = phi4 = phi1d = phi3d = phi4d = sigmad = 0 and q = 1 - p. The constrained
model, an AR(2), is estimated on the AR(2) sample, 1951:4 to 1984:4 (n = 133): that is the sample of
the published constrained estimates and likelihood (on the AR(4) sample the likelihood is -174.893, not
-176.990). The maximizations start from the grid points of Model B and Model A (Grid 3) with the
largest likelihood ratio. A 15 × 4
matrix: mu, mud, phi1 to phi4, phi1d to phi4d, sigma, sigmad, p and q in its rows, then the log
likelihood; the unconstrained estimates and White's standard errors, then the constrained ones, in
its columns (NaN where a parameter is constrained).
"""
function switching_estimates(gnp)
	x = growth(gnp)
	y, lags = sample(x)
	X = [ones(length(y)) lags]
	θ, V, ll = unconstrained(x)
	order = [1, 7, 2, 3, 4, 5, 8, 9, 10, 11, 6, 12, 13, 14]
	free = table(θ[order], V[order, order], ll)
	# constrained, (mu, phi1, phi2, sigma, mud, phi2d, p): an AR(2), on its own sample, 1951:4 to 1984:4
	s = best_grid_point(x, :modelA, 3)  # (mu, phi1, ..., phi4, sigma, mud, phi2d, p, q)
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
    wald_tests(gnp)

Table IX: Wald tests computed from the unconstrained estimates of Table VIII (see
[`switching_estimates`](@ref)) and their White covariance, from a Tables.jl table of GNP levels (see
[`growth`](@ref)): of the switching intercept model (phi1d = phi2d = phi3d = phi4d = sigmad = 0), and of
the simple switching parameters model (phi3 = phi4 = phi1d = phi3d = phi4d = sigmad = 0 and
p + q = 1). A 2 × 3 matrix: the tests in its rows, the statistic, its degrees of freedom and its
chi-squared p-value in its columns.
"""
function wald_tests(gnp)
	θ, V, _ = unconstrained(growth(gnp))
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

const MODELS ="`:hamilton` (Hamilton's model (11), Table III), `:intercept` (the switching intercept model (12), Table VI), `:modelA` or `:modelB` (mu and phi2 switch, with q = 1 - p or q free, Table X)"

"""
    standardized_lr(gnp, model; grids)

The standardized likelihood ratio statistic LR* of the null of an AR(4) against `model`, one of
$MODELS, the supremum over the grid of LR(α)/V(α)^(1/2), with α the switching parameters (mud on
0.1, ..., 2; for Models A and B mud on 0.2, ..., 2 and phi2d on -1, ..., 0.8) and the transition
probabilities on Grid 1 to 5 of p and q, from a Tables.jl table of GNP levels (see [`growth`](@ref)).
A column: one row for each grid in `grids` (default `1:3`), as in Tables III, VI and X of the erratum.
"""
function standardized_lr(gnp, model::Symbol; grids = 1:3)
	x = growth(gnp)
	return reshape([maximum(first(standardized(grid_search(x, model, g)))) for g in grids], :, 1)
end

"""
    lr_pvalues(gnp, model; grids = 1:3, bandwidths = 0:4, reps = 1000, seed = 1992)

The asymptotic p-values of the standardized LR statistic of [`standardized_lr`](@ref), by the
erratum's method (3'): `reps` draws of the supremum over the grid of
sum(k = 0..M) sum(i) q(i, α) u(i + k) / (sqrt(1 + M) V(α)^(1/2)), u standard normal (from a
StableRNG with `seed`, the same draws for every grid point and every set of `bandwidths`), for each Bartlett bandwidth M in
`bandwidths`. `model` is one of $MODELS. A matrix: one row for each grid in `grids`, one column for each
M, as in Tables III, VI and X of the erratum.
"""
function lr_pvalues(gnp, model::Symbol; grids = 1:3, bandwidths = 0:4, reps = 1000, seed = 1992)
	x = growth(gnp)
	result = zeros(length(grids), length(bandwidths))
	for (r, g) in enumerate(grids)
		statistics, Q = standardized(grid_search(x, model, g))
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
