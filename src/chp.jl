# Carrasco, Hu and Ploberger (2014): the optimal tests supTS and expTS.

"The seed of the authors' GAUSS programs, used here with a StableRNG (GAUSS's draws cannot be reproduced)."
const SEED = 2938438

"""
    rhogrid(bound; step = 0.01)

The grid of rho of the authors' programs: from -bound, adding `step` while rho <= bound + step, so 141
points for bound 0.7 and 197 for 0.98, the middle one 5e-16 rather than 0.
"""
function rhogrid(bound; step = 0.01)
	grid = Float64[]
	rho = -bound
	while rho <= bound + step
		push!(grid, rho)
		rho += step
	end
	return grid
end

"""
    nullfit(x, p)

The Gaussian AR(p) of a series `x` under the null of constant parameters, conditional on its first `p`
values: least squares of x(t) on 1 and its `p` lags, which is the maximum likelihood estimate. A named
tuple: `t` (the length of `x`), `n` (the observations used, t - p), `y`, `lags`, `b` (intercept and
AR coefficients), `u` (residuals), `sigma` (sqrt(u'u / n)), `phi` and `mu` (the mean).
"""
function nullfit(x::AbstractVector, p::Integer)
	t = length(x)
	n = t - p
	y = x[p+1:t]
	lags = reshape(reduce(hcat, [x[p+1-j:t-j] for j in 1:p]; init = zeros(n, 0)), n, p)
	X = [ones(n) lags]
	b = X \ y
	u = y - X * b
	phi = b[2:end]
	return (; t, n, y, lags, b, u, sigma = sqrt(dot(u, u) / n), phi, mu = b[1] / (1 - sum(phi)))
end

# The scores (n x k) and second derivatives (n x k x k) of the log-likelihood of each observation under
# the null, with respect to theta = (mu, phi1, ..., phip, sigma^2): supplement, section A.3.
function derivatives(fit)
	(; n, u, lags, phi, mu, sigma) = fit
	p = length(phi)
	k = p + 2
	s2 = sigma^2
	a = 1 - sum(phi)
	d = lags .- mu
	L = zeros(n, k)
	L[:, 1] = u .* a ./ s2
	L[:, 2:1+p] = u .* d ./ s2
	L[:, k] = u .^ 2 ./ (2 * s2^2) .- 1 / (2 * s2)
	H = zeros(n, k, k)
	for t in 1:n
		H[t, 1, 1] = -a^2 / s2
		H[t, 1, k] = H[t, k, 1] = -u[t] * a / s2^2
		H[t, k, k] = 1 / (2 * s2^2) - u[t]^2 / s2^3
		for i in 1:p
			H[t, 1, 1+i] = H[t, 1+i, 1] = -u[t] / s2 - a * d[t, i] / s2
			H[t, 1+i, k] = H[t, k, 1+i] = -u[t] * d[t, i] / s2^2
			for j in 1:p
				H[t, 1+i, 1+j] = -d[t, i] * d[t, j] / s2
			end
		end
	end
	return L, H
end

# The positions in theta of the switching parameters: :mean, :phi1, ..., :variance.
function positions(switching, p)
	return map(collect(switching)) do name
		name === :mean && return 1
		name === :variance && return p + 2
		m = match(r"^phi(?<lag>\d+)$", string(name))
		m !== nothing && 1 <= parse(Int, m[:lag]) <= p && return 1 + parse(Int, m[:lag])
		throw(ArgumentError("unknown switching parameter $name: use :mean, :variance or :phi1 to :phi$p"))
	end
end

# The directions h of the switching parameters, one in each column: (1) for one parameter; else `count`
# draws of independent normals, each divided by its squared length as in the authors' programs.
function directions(k, count, seed)
	k == 1 && return ones(1, 1)
	h = permutedims(randn(StableRNG(seed), count, k))
	return h ./ sum(abs2, h; dims = 1)
end

# Psi of (3.5) with tau = sqrt(e'e) (p. 779): sqrt(2 pi) exp((x - 1)^2 / 2) Phi(x - 1), written with erfcx.
psi(x) = sqrt(pi / 2) * erfcx(-(x - 1) / sqrt(2))

# supTS and expTS of a null fit: the maximum and the mean, over the grid of rho and the directions, of the
# terms of (3.4) and (3.5), as computed in the authors' ha_mean.g and ha_mean_var.g.
function statistics(fit, S, D, grid, T)
	L, H = derivatives(fit)
	n = fit.n
	k = length(S)
	pairs = [(a, b) for a in 1:k for b in a:k]
	Ls = L[:, S]
	A = [(H[t, S[a], S[b]] + Ls[t, a] * Ls[t, b]) / 2 for t in 1:n, (a, b) in pairs]
	W = [D[a, j] * D[b, j] * (a == b ? 1 : 2) for (a, b) in pairs, j in axes(D, 2)]
	F = qr(L)
	xs = zeros(n, k)
	V = similar(A)
	supts = -Inf
	total = 0.0
	for rho in grid
		for t in 2:n, a in 1:k
			xs[t, a] = rho * (xs[t-1, a] + Ls[t-1, a])
		end
		for (j, (a, b)) in enumerate(pairs), t in 1:n
			V[t, j] = A[t, j] + (Ls[t, a] * xs[t, b] + Ls[t, b] * xs[t, a]) / 2
		end
		E = V - L * (F \ V)
		sums = vec(sum(V; dims = 1))
		G = (transpose(E) * E) ./ n
		for w in eachcol(W)
			esqe = dot(w, G, w)
			if esqe < 1e-5
				supts = max(supts, 0.0)
				total += 1.0
			else
				tspe = dot(w, sums) / sqrt(T) / sqrt(esqe)
				supts = max(supts, max(tspe, 0.0)^2 / 2)
				total += psi(tspe)
			end
		end
	end
	return supts, total / (length(grid) * size(W, 2))
end

"""
    chptest(x; p = 4, switching = [:mean], bound = 0.7, grid = rhogrid(bound), count = 100, seed = $SEED)

Carrasco, Hu and Ploberger's supTS (3.4) and expTS (3.5)-(3.6) tests of the null of a Gaussian AR(p) with
constant parameters against parameters that switch with a Markov chain, from the series `x` (for GNP,
the growth rates of [`growth`](@ref)). `switching` names the parameters that switch: `:mean`,
`:variance` (of the errors) and `:phi1` to `:phip`. With more than one, the direction h of the switch is
drawn `count` times with a StableRNG and `seed`. The test needs only the AR(p) fit under the null.
A named tuple `(supTS, expTS)`; [`chpbootstrap`](@ref) gives their p-values.
"""
function chptest(x::AbstractVector; p = 4, switching = [:mean], bound = 0.7, grid = rhogrid(bound), count = 100, seed = SEED)
	S = positions(switching, p)
	supts, expts = statistics(nullfit(x, p), S, directions(length(S), count, seed), grid, length(x))
	return (; supTS = supts, expTS = expts)
end

"""
    chpbootstrap(x; p = 4, switching = [:mean], bound = 0.7, grid = rhogrid(bound), count = 100,
                 seed = $SEED, reps = 3000, bootseed = seed + 1)

The parametric bootstrap of section 3 for [`chptest`](@ref): `reps` series simulated from the AR(p) fitted
under the null, its first `p` values drawn from the sample's mean and standard deviation as in the
authors' programs, each refitted and tested with the same directions h. A named tuple: the statistics
`supTS` and `expTS`, their p-values `supP` and `expP` (the share of bootstrap statistics above them), and
the bootstrap statistics `supB` and `expB`.
"""
function chpbootstrap(x::AbstractVector; p = 4, switching = [:mean], bound = 0.7, grid = rhogrid(bound), count = 100,
		seed = SEED, reps = 3000, bootseed = seed + 1)
	S = positions(switching, p)
	D = directions(length(S), count, seed)
	t = length(x)
	fit = nullfit(x, p)
	supts, expts = statistics(fit, S, D, grid, t)
	rng = StableRNG(bootseed)
	m, s = mean(fit.y), std(fit.y)
	supb = zeros(reps)
	expb = zeros(reps)
	for r in 1:reps
		ys = zeros(t)
		ys[1:p] = m .+ s .* randn(rng, p)
		for i in p+1:t
			ys[i] = fit.b[1] + sum(fit.b[1+j] * ys[i-j] for j in 1:p; init = 0.0) + fit.sigma * randn(rng)
		end
		supb[r], expb[r] = statistics(nullfit(ys, p), S, D, grid, t)
	end
	return (; supTS = supts, expTS = expts, supP = mean(supts .< supb), expP = mean(expts .< expb), supB = supb, expB = expb)
end

# The value at a percentile of sorted draws, as the authors' programs take it: the floor(reps q)-th.
percentile(sorted, q) = sorted[floor(Int, length(sorted) * q)]

const PERCENTILES = [0.99, 0.95, 0.90, 0.80, 0.70, 0.50, 0.10, 0.05, 0.01]

"""
    asymptoticvalues(bound; reps = 100000, terms = 500, seed = $SEED)

The asymptotic critical values of supTS and expTS for the model without regressors (6.1), from (A.2) to
(A.4) of the supplement: `reps` draws of K(rho) = sign(rho) sqrt(1 - rho^2) (Z0 + sum of rho^r Zr, r = 1
to `terms`) on [`rhogrid`](@ref)`(bound)`. A 9 x 2 matrix, supTS and expTS in its columns, the
percentiles 99, 95, 90, 80, 70, 50, 10, 5 and 1 in its rows, as in Table A-I of the supplement.
"""
function asymptoticvalues(bound; reps = 100000, terms = 500, seed = SEED)
	grid = rhogrid(bound)
	rng = StableRNG(seed)
	z = zeros(terms + 1)
	sups = zeros(reps)
	exps = zeros(reps)
	for i in 1:reps
		randn!(rng, z)
		best = -Inf
		total = 0.0
		for rho in grid
			su = z[terms+1]
			for r in terms:-1:1
				su = z[r] + rho * su
			end
			su *= sign(rho) * sqrt(1 - rho^2)
			best = max(best, max(su, 0.0)^2 / 2)
			total += psi(su)
		end
		sups[i] = best
		exps[i] = total / length(grid)
	end
	sort!(sups)
	sort!(exps)
	return [percentile(sorted, q) for q in PERCENTILES, sorted in (sups, exps)]
end

"""
    empiricalvalues(bound; reps = 10000, t = 500, seed = $SEED)

The empirical critical values of supTS and expTS for the model without regressors (6.1): [`chptest`](@ref)
with `p = 0` and a switching mean on `reps` samples of `t` standard normals, on [`rhogrid`](@ref)`(bound)`.
A 9 x 2 matrix laid out as [`asymptoticvalues`](@ref).
"""
function empiricalvalues(bound; reps = 10000, t = 500, seed = SEED)
	grid = rhogrid(bound)
	rng = StableRNG(seed)
	D = ones(1, 1)
	values = [statistics(nullfit(randn(rng, t), 0), [1], D, grid, t) for _ in 1:reps]
	sups = sort(first.(values))
	exps = sort(last.(values))
	return [percentile(sorted, q) for q in PERCENTILES, sorted in (sups, exps)]
end

# Minus the log-likelihood of each observation of the two-state Markov switching AR(p) with switching mean
# and variance, as in the authors' ha_alternative.PRG (from Hansen's code): theta = (mean in state 0, phi,
# sd in state 0, difference in mean, sd in state 1, p, q), Hamilton's filter over the states of the last
# p periods, started from the stationary probability (1 - q) / (2 - p - q) of state 1. The standard
# deviations enter as their absolute values, so that the optimiser may cross 0.
function mslik(theta, y, lags)
	n, p = size(lags)
	T = eltype(theta)
	c = sqrt(2 * pi)
	alpha0 = theta[1]
	phi = theta[2:1+p]
	sig0 = theta[2+p]
	alpha1 = theta[3+p]
	sig1 = theta[4+p]
	P = theta[5+p]
	Q = theta[6+p]
	rho = (1 - Q) / (2 - P - Q)
	b8 = alpha0 * (1 - sum(phi))
	half = 2^(p - 1)
	ss = [Float64(((r - 1) >> (p - col)) & 1) for r in 1:2^p, col in 1:p]
	sphi0 = ss[1:half, :] * phi * alpha1
	sphi1 = ss[half+1:end, :] * phi * alpha1
	pp = T[1-rho, rho]
	pxa = T[Q, 1-P]
	pxb = 1 .- pxa
	for it in 2:p
		nr = ones(2^(it - 2))
		pp = [pp .* kron(pxa, nr); pp .* kron(pxb, nr)]
	end
	p0 = pp[1:half]
	p1 = pp[half+1:end]
	f = zeros(T, n)
	for t in 1:n
		z = y[t] - b8 - dot(view(lags, t, :), phi)
		z0 = z .+ sphi0
		z1 = z .+ sphi1
		q00 = @. exp(-z0^2 / (2 * sig0^2)) / (abs(sig0) * c) * p0 * Q
		q01 = @. exp(-z1^2 / (2 * sig0^2)) / (abs(sig0) * c) * p1 * (1 - P)
		q10 = @. exp(-(z0 - alpha1)^2 / (2 * sig1^2)) / (abs(sig1) * c) * p0 * (1 - Q)
		q11 = @. exp(-(z1 - alpha1)^2 / (2 * sig1^2)) / (abs(sig1) * c) * p1 * P
		if p > 1
			s = 1:2:half
			p0 = [q00[s] + q00[s.+1]; q01[s] + q01[s.+1]]
			p1 = [q10[s] + q10[s.+1]; q11[s] + q11[s.+1]]
		else
			p0 = q00 + q01
			p1 = q10 + q11
		end
		ff = sum(p0) + sum(p1)
		p0 = p0 ./ ff
		p1 = p1 ./ ff
		f[t] = ff
	end
	return -log.(f)
end

# theta with p and q as logits, as the optimiser sees them.
probabilities(theta) = [theta[1:end-2]; @. exp(theta[end-1:end]) / (1 + exp(theta[end-1:end]))]

"""
    msfit(x; p = 4, start = [-0.4, 0.01, -0.06, -0.25, -0.21, 0.77, 1.5, 0.77, 2.2, 1.1])

The two-state Markov switching AR(p) model (6.2), with switching mean and variance, fitted by maximum
likelihood to the series `x` as in the authors' ha_alternative.PRG: BFGS from `start` (the mean in state
0, the AR coefficients, the standard deviation in state 0, the difference in mean between the states,
the standard deviation in state 1, and p and q as logits), with sandwich standard errors (the inverse
Hessian around the outer product of the scores) and the delta method for p and q. A named tuple:
`estimates`, `se` (both in that order, p and q as probabilities) and `L`, minus the log-likelihood.
"""
function msfit(x::AbstractVector; p = 4, start = [-0.4, 0.01, -0.06, -0.25, -0.21, 0.77, 1.5, 0.77, 2.2, 1.1])
	fit = nullfit(x, p)
	objective(theta) = sum(mslik(probabilities(theta), fit.y, fit.lags))
	gradient!(G, theta) = ForwardDiff.gradient!(G, objective, theta)
	result = optimize(objective, gradient!, start, BFGS(), Optim.Options(g_tol = 1e-10, iterations = 10000))
	theta = Optim.minimizer(result)
	g = ForwardDiff.jacobian(th -> mslik(probabilities(th), fit.y, fit.lags), theta)
	h = pinv(ForwardDiff.hessian(objective, theta))
	v = h * (transpose(g) * g) * h
	estimates = probabilities(theta)
	estimates[[2 + p, 4 + p]] = abs.(estimates[[2 + p, 4 + p]])
	k = length(theta)
	vt = estimates[k-1:k] .* (1 .- estimates[k-1:k])
	v[k-1:k, k-1:k] .*= vt * transpose(vt)
	return (; estimates, se = sqrt.(diag(v)), L = Optim.minimum(result))
end

# The QML sandwich standard errors of the AR(p) under the null in (mu, phi, sigma), as GAUSS's CML with
# _cml_CovPar = 3 in the authors' ha_null.g.
function arse(fit)
	loglik(theta) = begin
		p = length(fit.phi)
		e = fit.y .- fit.lags * theta[2:1+p] .- (1 - sum(theta[2:1+p])) * theta[1]
		@. -log(2 * pi) / 2 - log(theta[p+2]^2) / 2 - e^2 / (2 * theta[p+2]^2)
	end
	theta = [fit.mu; fit.phi; fit.sigma]
	g = ForwardDiff.jacobian(loglik, theta)
	h = inv(ForwardDiff.hessian(th -> sum(loglik(th)), theta))
	return sqrt.(diag(h * (transpose(g) * g) * h))
end

"""
    tableA2(gnp82, gnpc96)

Table A-II of the supplement: the Gaussian AR(4) of US real GNP growth under the null, by maximum
likelihood, with QML sandwich standard errors, 1952Q2 to 1984Q4 (`gnp82`) and 1952Q2 to 2010Q4
(`gnpc96`), Tables.jl tables of levels. A 7 x 4 matrix: mu, phi1 to phi4, sigma and T in its rows; the
estimates and standard errors of each sample in its columns.
"""
function tableA2(gnp82, gnpc96)
	columns = map((gnp82, gnpc96)) do table
		fit = nullfit(growth(table), 4)
		[[fit.mu; fit.phi; fit.sigma; fit.n] [arse(fit); NaN]]
	end
	return reduce(hcat, columns)
end

"""
    table3mean(gnp82, gnpc96)

Table III, switch in mean: supTS and expTS of [`chptest`](@ref) for Hamilton's AR(4) of US real GNP
growth, 1952Q2 to 1984Q4 (`gnp82`) and 1952Q2 to 2010Q4 (`gnpc96`). A 2 x 2 matrix: supTS and expTS in
its rows, the samples in its columns.
"""
table3mean(gnp82, gnpc96) = reduce(hcat, [collect(chptest(growth(table))) for table in (gnp82, gnpc96)])

"""
    table3meanvariance(gnp82, gnpc96)

Table III, switch in mean and variance, model (6.2): supTS and expTS of [`chptest`](@ref) with the mean
and the variance switching, 100 directions h. Laid out as [`table3mean`](@ref).
"""
table3meanvariance(gnp82, gnpc96) =
	reduce(hcat, [collect(chptest(growth(table); switching = [:mean, :variance])) for table in (gnp82, gnpc96)])

"""
    table3pvalues(gnp82, gnpc96; reps = 3000)

Table III, the bootstrap p-values of [`chpbootstrap`](@ref) with `reps` samples. A 4 x 2 matrix: supTS
and expTS of the switch in mean, then of the switch in mean and variance, in its rows; the samples in its
columns.
"""
function table3pvalues(gnp82, gnpc96; reps = 3000)
	return reduce(hcat, map((gnp82, gnpc96)) do table
		x = growth(table)
		level = chpbootstrap(x; reps)
		both = chpbootstrap(x; switching = [:mean, :variance], reps)
		[level.supP, level.expP, both.supP, both.expP]
	end)
end

"""
    tableA1()

Table A-I of the supplement: the asymptotic ([`asymptoticvalues`](@ref)) and empirical
([`empiricalvalues`](@ref)) critical values of supTS and expTS for the model without regressors, rho in
[-0.7, 0.7] then in [-0.98, 0.98]. An 18 x 4 matrix: the percentiles in its rows; supTS asymptotic, supTS
empirical, expTS asymptotic and expTS empirical in its columns.
"""
function tableA1()
	return reduce(vcat, map((0.7, 0.98)) do bound
		a = asymptoticvalues(bound)
		e = empiricalvalues(bound)
		[a[:, 1] e[:, 1] a[:, 2] e[:, 2]]
	end)
end

"""
    tableA3(gnpc96)

Table A-III of the supplement: the two-state Markov switching AR(4) with switching mean and variance of
[`msfit`](@ref), 1952Q2 to 2010Q4. An 11 x 2 matrix: mu (state 0), mu1 (the difference in mean), phi1 to
phi4, sigma (state 0), sigma1 (state 1), p, q and L (minus the log-likelihood) in its rows; the estimates
and standard errors in its columns.
"""
function tableA3(gnpc96)
	fit = msfit(growth(gnpc96))
	order = [1, 7, 2, 3, 4, 5, 6, 8, 9, 10]
	return [fit.estimates[order] fit.se[order]; fit.L NaN]
end
