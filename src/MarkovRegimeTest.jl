module MarkovRegimeTest

using Distributions, ForwardDiff, LinearAlgebra, Optim, Random, SpecialFunctions, StableRNGs, Statistics, Tables

# Hansen (1992), with the 1996 erratum
export ar_estimates, hamilton_estimates, intercept_estimates, switching_estimates, wald_tests, standardized_lr,
	lr_pvalues
# Carrasco, Hu and Ploberger (2014)
export rhogrid, nullfit, chptest, chpbootstrap, asymptoticvalues, empiricalvalues, msfit, table3mean,
	table3meanvariance, table3pvalues, tableA1, tableA2, tableA3
export growth

"""
    growth(gnp)

The quarterly growth rates of US real GNP in per cent, 100 times the first difference of the log of the
levels, from a Tables.jl table with a column `gnp` of quarterly levels in time order (the datasets
`gnp82`, 1951Q1 to 1984Q4, and `gnpc96`, 1951Q1 to 2010Q4, of PollisDatasets): a vector one shorter
than the table, from its second quarter on.
"""
growth(gnp) = 100 .* diff(log.(Float64.(Tables.getcolumn(Tables.columns(gnp), :gnp))))

include("hansen.jl")
include("chp.jl")

end
