# MarkovRegimeTest.jl

Tests of one regime against two, and the dating of the regimes, in Markov switching models of GNP.

Two tests of a one-regime autoregression against a two-regime Markov switching model, on US real GNP. Hansen's standardized likelihood ratio test, with the 1996 erratum's Bartlett-kernel p-values: the AR(4), Hamilton's Markov switching model, the switching intercept model and the switching parameters model, the Wald tests between them, and the standardized LR statistics on grids of the parameters not identified under the null. Carrasco, Hu and Ploberger's optimal tests supTS and expTS, which need only the estimates under the null, with parametric bootstrap p-values: for a switch in the mean and for a switch in the mean and the variance, the critical values of the model without regressors, and the Markov switching AR(4) with switching mean and variance. Hamilton's filter and full-sample smoother for the probabilities of the two regimes at each period, and the spells of the regimes with their first and last periods, as Hamilton dated the recessions of US GNP.

Articles:

- Hansen, B. E. (1992). The likelihood ratio test under nonstandard conditions: Testing the Markov switching model of GNP. Journal of Applied Econometrics 7(S1), S61-S82. Erratum: Journal of Applied Econometrics 11(2), 195-198 (1996). [doi:10.1002/jae.3950070506](https://doi.org/10.1002/jae.3950070506)
- Carrasco, M., Hu, L. and Ploberger, W. (2014). Optimal test for Markov switching parameters. Econometrica 82(2), 765-784. [doi:10.3982/ECTA8609](https://doi.org/10.3982/ECTA8609)
- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica 57(2), 357-384. [doi:10.2307/1912559](https://doi.org/10.2307/1912559)

## Usage

```julia
using MarkovRegimeTest, CSV, Downloads, Tables
data = "https://raw.githubusercontent.com/Trumpingtons/PollisDatasets.jl/ade6f348c24833c8644074acbde72a51aa495adb/data"
read_csv(name) = Tables.columntable(CSV.File(take!(Downloads.download("$data/$name.csv", IOBuffer()))))

# Any series: returns or growth rates as a vector, or a table and the column of its levels
prices = read_csv("sp500")
x = growth(prices, :close)[end-499:end]                      # the last 500 daily returns, in per cent
standardized_lr(x, :intercept; p = 1, grids = [0.5:0.1:0.9]) # Hansen's LR*, one regime against two
lr_pvalues(x, :intercept; p = 1, grids = [0.5:0.1:0.9])      # and its p-values, M = 0 to 4
chpbootstrap(x; p = 1, switching = [:mean, :variance])       # Carrasco, Hu and Ploberger, with p-values

# The articles' tables, on US real GNP
gnp82, gnpc96 = read_csv("gnp82"), read_csv("gnpc96")
standardized_lr(gnp82, :hamilton; grids = 1:3, mud = 0.1:0.1:2.0)  # Hansen's erratum, Table III
table3mean(gnp82, gnpc96)                                          # Carrasco, Hu and Ploberger, Table III

# The regimes and their dates: A the regime of the higher mean, B the other
smoothed_probabilities(gnp82; mud = 0.1:0.1:2.0)                   # P(A), P(B) of each quarter, given all the data
regime_spells(gnp82; mud = 0.1:0.1:2.0, dates = (:year, :quarter)) # A, B, A, ...: each spell's start and stop
turning_points(gnp82; mud = 0.1:0.1:2.0)                           # Hamilton, Table II: the recessions
regime_spells(x, :meanvariance; p = 1, dates = prices.date[end-499:end]) # calm and turbulent spells of the S&P 500
```

## Vignette

[Tests of one regime against two, and the dating of the regimes, in Markov switching models of GNP](vignette/MarkovRegimeTest.tex), its LaTeX source: Pollis compiles it in Help > Documentation > Package Vignettes.

## License

MIT (see [LICENSE](LICENSE)).

## Tests

The tests compute the published tables below from their data and check every printed value, rounded to the digits printed:

- Carrasco, Hu and Ploberger (2014), Table III, p. 782: switch in mean, test statistics (`test/data/targets/chp_table3mean.csv`)
- Carrasco, Hu and Ploberger (2014), Table III, p. 782: switch in mean and variance, test statistics (`test/data/targets/chp_table3meanvariance.csv`)
- Carrasco, Hu and Ploberger (2014), Table III, p. 782: bootstrap p-values (`test/data/targets/chp_table3pvalues.csv`)
- Carrasco, Hu and Ploberger (2014), supplement, Table A-I, p. 10 (`test/data/targets/chp_tableA1.csv`)
- Carrasco, Hu and Ploberger (2014), supplement, Table A-II, p. 10 (`test/data/targets/chp_tableA2.csv`)
- Carrasco, Hu and Ploberger (2014), supplement, Table A-III, p. 11 (`test/data/targets/chp_tableA3.csv`)
- Hamilton (1989), the output of his GAUSS program for the article (EXAMPLE1.OUT of Markov1.zip): filtered probabilities (`test/data/targets/hamilton_filtered.csv`)
- Hamilton (1989), the output of his GAUSS program for the article (EXAMPLE1.OUT of Markov1.zip): full-sample smoothed probabilities (`test/data/targets/hamilton_smoothed.csv`)
- Hamilton (1989), Table I, p. 372 (`test/data/targets/hamilton_table1.csv`)
- Hamilton (1989), Table II, p. 374: dates by the full-sample smoother (`test/data/targets/hamilton_table2.csv`)
- Hansen (1996), erratum, Table X, p. 197: standardized LR statistics (`test/data/targets/hansen_erratum10.csv`)
- Hansen (1996), erratum, Table X, p. 197: p-values (`test/data/targets/hansen_erratum10pvalues.csv`)
- Hansen (1996), erratum, Table III, p. 196: standardized LR statistics (`test/data/targets/hansen_erratum3.csv`)
- Hansen (1996), erratum, Table III, p. 196: p-values (`test/data/targets/hansen_erratum3pvalues.csv`)
- Hansen (1996), erratum, Table VI, p. 197: standardized LR statistics (`test/data/targets/hansen_erratum6.csv`)
- Hansen (1996), erratum, Table VI, p. 197: p-values (`test/data/targets/hansen_erratum6pvalues.csv`)
- Hansen (1992), Table I, p. S71 (`test/data/targets/hansen_table1.csv`)
- Hansen (1992), Table II, p. S72 (`test/data/targets/hansen_table2.csv`)
- Hansen (1992), Table III, p. S73: p-values (M = 0) (`test/data/targets/hansen_table3pvalues.csv`)
- Hansen (1992), Table V, p. S75 (`test/data/targets/hansen_table5.csv`)
- Hansen (1992), Table VI, p. S76: p-values (M = 0) (`test/data/targets/hansen_table6pvalues.csv`)
- Hansen (1992), Table VIII, p. S78 (`test/data/targets/hansen_table8.csv`)
- Hansen (1992), Table IX, p. S78 (`test/data/targets/hansen_table9.csv`)

The data, downloaded from its links when the tests run or kept in `test/data/inputs`:

- `gnp82`: <https://raw.githubusercontent.com/Trumpingtons/PollisDatasets.jl/ade6f348c24833c8644074acbde72a51aa495adb/data/gnp82.csv>, PollisDatasets: Hamilton's (1989) US real GNP, quarterly levels from 1951Q1 to 1984Q4, the file of Carrasco, Hu and Ploberger's replication programs
- `gnpc96`: <https://raw.githubusercontent.com/Trumpingtons/PollisDatasets.jl/ade6f348c24833c8644074acbde72a51aa495adb/data/gnpc96.csv>, PollisDatasets: US real GNP, the vintage of the FRED series GNPC96 that Carrasco, Hu and Ploberger (2014) used, quarterly levels from 1951Q1 to 2010Q4

Run them with `using Pkg; Pkg.test("MarkovRegimeTest")`.

Made with [Pollis](https://pollis.org).
