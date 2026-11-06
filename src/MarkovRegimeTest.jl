module MarkovRegimeTest

using Distributions, ForwardDiff, LinearAlgebra, Optim, Random, SpecialFunctions, StableRNGs, Statistics, Tables

# Hansen (1992), with the 1996 erratum
export ar_estimates, hamilton_estimates, intercept_estimates, switching_estimates, wald_tests, standardized_lr,
	lr_pvalues
# Hamilton (1989): the regimes, their probabilities and their dates
export meanvariance_estimates, filtered_probabilities, smoothed_probabilities, regime_spells, turning_points
# Carrasco, Hu and Ploberger (2014)
export rhogrid, nullfit, chptest, chpbootstrap, asymptoticvalues, empiricalvalues, msfit, table3mean,
	table3meanvariance, table3pvalues, tableA1, tableA2, tableA3
export growth

"""
    growth(table, column = :gnp)

The growth rates in per cent, 100 times the first difference of the log, of the levels in `column` of a
Tables.jl table, in time order: a vector one shorter than the table. With the default column, the
quarterly growth of US real GNP from the datasets `gnp82` (1951Q1 to 1984Q4) and `gnpc96` (1951Q1 to
2010Q4) of PollisDatasets; with a column of prices, returns in per cent. Every function of Hansen's test
takes such a table and its `column`, or the vector of growth rates or returns itself.
"""
growth(table, column = :gnp) = 100 .* diff(log.(Float64.(Tables.getcolumn(Tables.columns(table), Symbol(column)))))

include("hansen.jl")
include("chp.jl")

end
