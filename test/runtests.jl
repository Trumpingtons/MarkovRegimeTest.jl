# Made by Pollis (Compose > New Methods > Package and Publish) from method.toml: change the method, not this file.
using Test, CSV, Downloads, Tables, MarkovRegimeTest

const DATA = joinpath(@__DIR__, "data")

"An input table: a file of the tests, or downloaded from its link into memory."
function input(source; delim = ',', columns = nothing, header = true, skipto = nothing, limit = nothing)
	bytes = startswith(source, r"https?://") ? take!(Downloads.download(source, IOBuffer())) : read(joinpath(DATA, source))
	named = columns === nothing && header
	header = columns !== nothing ? Symbol.(columns) : named ? 1 : false
	rows = (; skipto = something(skipto, named ? 2 : 1), limit)
	file = delim == "whitespace" ? CSV.File(bytes; header, rows..., delim = ' ', ignorerepeated = true, stripwhitespace = true) : CSV.File(bytes; header, rows..., delim = delim == "tab" ? '\t' : only(delim))
	return Tables.columntable(file)
end

"The published table: its row labels and its values as printed, `nothing` where a cell is empty."
function target(file)
	table = CSV.File(joinpath(DATA, "targets", file * ".csv"); types = String, delim = ',')
	columns = Tables.columnnames(table)
	rows = [string(coalesce(row[columns[1]], "")) for row in table]
	values = [let cell = row[column]; cell === missing ? nothing : strip(cell) end for row in table, column in columns[2:end]]
	return rows, string.(columns[2:end]), values
end

"Tests each printed value against the result rounded to the digits printed, but the untested cells, each (row label, column name)."
function check(result, file, tolerance, untested = Tuple{String, String}[])
	rows, columns, values = target(file)
	result = result isa AbstractVecOrMat ? result : Tables.matrix(result)
	result = reshape(result, size(result, 1), :)
	@test size(result) == size(values)
	@test isempty(setdiff(untested, [(row, column) for row in rows for column in columns]))
	for i in axes(values, 1), j in axes(values, 2)
		printed = values[i, j]
		(printed === nothing || isempty(printed) || (rows[i], columns[j]) in untested) && continue
		mantissa, exponent = occursin(r"[eE]", printed) ? split(printed, r"[eE]") : (printed, "0")
		digits = (occursin('.', mantissa) ? length(split(mantissa, '.')[2]) : 0) - parse(Int, exponent)
		published = parse(Float64, printed)
		computed = round(result[i, j]; digits)
		@testset "$(rows[i]), $(columns[j])" begin
			@test isapprox(computed, published; atol = (tolerance + 1e-6) * 10.0^-digits)
		end
	end
end

@testset "MarkovRegimeTest" begin
	@testset "Carrasco, Hu and Ploberger (2014), Table III, p. 782: switch in mean, test statistics" begin
		gnp82 = input("https://raw.githubusercontent.com/Trumpingtons/PollisDatasets.jl/ade6f348c24833c8644074acbde72a51aa495adb/data/gnp82.csv")
		gnpc96 = input("https://raw.githubusercontent.com/Trumpingtons/PollisDatasets.jl/ade6f348c24833c8644074acbde72a51aa495adb/data/gnpc96.csv")
		check(table3mean(gnp82, gnpc96), "chp_table3mean", 0)
	end
	@testset "Carrasco, Hu and Ploberger (2014), Table III, p. 782: switch in mean and variance, test statistics" begin
		gnp82 = input("https://raw.githubusercontent.com/Trumpingtons/PollisDatasets.jl/ade6f348c24833c8644074acbde72a51aa495adb/data/gnp82.csv")
		gnpc96 = input("https://raw.githubusercontent.com/Trumpingtons/PollisDatasets.jl/ade6f348c24833c8644074acbde72a51aa495adb/data/gnpc96.csv")
		check(table3meanvariance(gnp82, gnpc96), "chp_table3meanvariance", 3, [("expTS", "1952Q2 to 1984Q4"), ("expTS", "1952Q2 to 2010Q4")])
	end
	@testset "Carrasco, Hu and Ploberger (2014), Table III, p. 782: bootstrap p-values" begin
		gnp82 = input("https://raw.githubusercontent.com/Trumpingtons/PollisDatasets.jl/ade6f348c24833c8644074acbde72a51aa495adb/data/gnp82.csv")
		gnpc96 = input("https://raw.githubusercontent.com/Trumpingtons/PollisDatasets.jl/ade6f348c24833c8644074acbde72a51aa495adb/data/gnpc96.csv")
		check(table3pvalues(gnp82, gnpc96), "chp_table3pvalues", 3)
	end
	@testset "Carrasco, Hu and Ploberger (2014), supplement, Table A-I, p. 10" begin
		check(tableA1(), "chp_tableA1", 3, [("rho in [-0.7, 0.7], 99%", "supTS, empirical"), ("rho in [-0.7, 0.7], 99%", "expTS, empirical"), ("rho in [-0.98, 0.98], 99%", "supTS, asymptotic"), ("rho in [-0.98, 0.98], 99%", "supTS, empirical"), ("rho in [-0.98, 0.98], 99%", "expTS, asymptotic"), ("rho in [-0.98, 0.98], 99%", "expTS, empirical"), ("rho in [-0.98, 0.98], 95%", "supTS, empirical")])
	end
	@testset "Carrasco, Hu and Ploberger (2014), supplement, Table A-II, p. 10" begin
		gnp82 = input("https://raw.githubusercontent.com/Trumpingtons/PollisDatasets.jl/ade6f348c24833c8644074acbde72a51aa495adb/data/gnp82.csv")
		gnpc96 = input("https://raw.githubusercontent.com/Trumpingtons/PollisDatasets.jl/ade6f348c24833c8644074acbde72a51aa495adb/data/gnpc96.csv")
		check(tableA2(gnp82, gnpc96), "chp_tableA2", 0, [("mu", "1952Q2 to 1984Q4, standard error"), ("sigma", "1952Q2 to 1984Q4, standard error")])
	end
	@testset "Carrasco, Hu and Ploberger (2014), supplement, Table A-III, p. 11" begin
		gnpc96 = input("https://raw.githubusercontent.com/Trumpingtons/PollisDatasets.jl/ade6f348c24833c8644074acbde72a51aa495adb/data/gnpc96.csv")
		check(tableA3(gnpc96), "chp_tableA3", 0, [("mu", "Standard error"), ("mu1", "Standard error"), ("phi1", "Standard error"), ("sigma", "Standard error")])
	end
	@testset "Hansen (1996), erratum, Table X, p. 197: standardized LR statistics" begin
		gnp82 = input("https://raw.githubusercontent.com/Trumpingtons/PollisDatasets.jl/ade6f348c24833c8644074acbde72a51aa495adb/data/gnp82.csv")
		check(vcat(standardized_lr(gnp82, :modelA; grids = 1:4), standardized_lr(gnp82, :modelB; grids = 1:4)), "hansen_erratum10", 0, [("Model B, Grid 2", "LR*")])
	end
	@testset "Hansen (1996), erratum, Table X, p. 197: p-values" begin
		gnp82 = input("https://raw.githubusercontent.com/Trumpingtons/PollisDatasets.jl/ade6f348c24833c8644074acbde72a51aa495adb/data/gnp82.csv")
		check(vcat(lr_pvalues(gnp82, :modelA; grids = 1:4), lr_pvalues(gnp82, :modelB; grids = 1:4)), "hansen_erratum10pvalues", 3)
	end
	@testset "Hansen (1996), erratum, Table III, p. 196: standardized LR statistics" begin
		gnp82 = input("https://raw.githubusercontent.com/Trumpingtons/PollisDatasets.jl/ade6f348c24833c8644074acbde72a51aa495adb/data/gnp82.csv")
		check(standardized_lr(gnp82, :hamilton; grids = 1:3), "hansen_erratum3", 0)
	end
	@testset "Hansen (1996), erratum, Table III, p. 196: p-values" begin
		gnp82 = input("https://raw.githubusercontent.com/Trumpingtons/PollisDatasets.jl/ade6f348c24833c8644074acbde72a51aa495adb/data/gnp82.csv")
		check(lr_pvalues(gnp82, :hamilton; grids = 1:3), "hansen_erratum3pvalues", 3, [("Grid 1", "M = 3")])
	end
	@testset "Hansen (1996), erratum, Table VI, p. 197: standardized LR statistics" begin
		gnp82 = input("https://raw.githubusercontent.com/Trumpingtons/PollisDatasets.jl/ade6f348c24833c8644074acbde72a51aa495adb/data/gnp82.csv")
		check(standardized_lr(gnp82, :intercept; grids = 1:5), "hansen_erratum6", 0)
	end
	@testset "Hansen (1996), erratum, Table VI, p. 197: p-values" begin
		gnp82 = input("https://raw.githubusercontent.com/Trumpingtons/PollisDatasets.jl/ade6f348c24833c8644074acbde72a51aa495adb/data/gnp82.csv")
		check(lr_pvalues(gnp82, :intercept; grids = 1:5), "hansen_erratum6pvalues", 3, [("Grid 1", "M = 0"), ("Grid 1", "M = 2"), ("Grid 2", "M = 1"), ("Grid 2", "M = 2"), ("Grid 2", "M = 3"), ("Grid 2", "M = 4"), ("Grid 3", "M = 2"), ("Grid 3", "M = 3"), ("Grid 3", "M = 4"), ("Grid 4", "M = 3"), ("Grid 4", "M = 4")])
	end
	@testset "Hansen (1992), Table I, p. S71" begin
		gnp82 = input("https://raw.githubusercontent.com/Trumpingtons/PollisDatasets.jl/ade6f348c24833c8644074acbde72a51aa495adb/data/gnp82.csv")
		check(ar_estimates(gnp82), "hansen_table1", 0)
	end
	@testset "Hansen (1992), Table II, p. S72" begin
		gnp82 = input("https://raw.githubusercontent.com/Trumpingtons/PollisDatasets.jl/ade6f348c24833c8644074acbde72a51aa495adb/data/gnp82.csv")
		check(hamilton_estimates(gnp82), "hansen_table2", 0, [("mu", "Standard error")])
	end
	@testset "Hansen (1992), Table III, p. S73: p-values (M = 0)" begin
		gnp82 = input("https://raw.githubusercontent.com/Trumpingtons/PollisDatasets.jl/ade6f348c24833c8644074acbde72a51aa495adb/data/gnp82.csv")
		check(lr_pvalues(gnp82, :hamilton; grids = 1:3, bandwidths = 0:0), "hansen_table3pvalues", 3, [("Grid 2", "p-value")])
	end
	@testset "Hansen (1992), Table V, p. S75" begin
		gnp82 = input("https://raw.githubusercontent.com/Trumpingtons/PollisDatasets.jl/ade6f348c24833c8644074acbde72a51aa495adb/data/gnp82.csv")
		check(intercept_estimates(gnp82), "hansen_table5", 0, [("p", "Estimate"), ("q", "Estimate")])
	end
	@testset "Hansen (1992), Table VI, p. S76: p-values (M = 0)" begin
		gnp82 = input("https://raw.githubusercontent.com/Trumpingtons/PollisDatasets.jl/ade6f348c24833c8644074acbde72a51aa495adb/data/gnp82.csv")
		check(lr_pvalues(gnp82, :intercept; grids = 1:5, bandwidths = 0:0), "hansen_table6pvalues", 3, [("Grid 5", "p-value")])
	end
	@testset "Hansen (1992), Table VIII, p. S78" begin
		gnp82 = input("https://raw.githubusercontent.com/Trumpingtons/PollisDatasets.jl/ade6f348c24833c8644074acbde72a51aa495adb/data/gnp82.csv")
		check(switching_estimates(gnp82), "hansen_table8", 0)
	end
	@testset "Hansen (1992), Table IX, p. S78" begin
		gnp82 = input("https://raw.githubusercontent.com/Trumpingtons/PollisDatasets.jl/ade6f348c24833c8644074acbde72a51aa495adb/data/gnp82.csv")
		check(wald_tests(gnp82), "hansen_table9", 0)
	end
end
