# Before running this, please make sure to activate and instantiate the environment
# corresponding to [this `Project.toml`](https://raw.githubusercontent.com/alan-turing-institute/MLJTutorials/master/Project.toml) and [this `Manifest.toml`](https://raw.githubusercontent.com/alan-turing-institute/MLJTutorials/master/Manifest.toml)
# so that you get an environment which matches the one used to generate the tutorials:
#
# ```julia
# cd("MLJTutorials") # cd to folder with the *.toml
# using Pkg; Pkg.activate("."); Pkg.instantiate()
# ```

using MLJ, RDatasets, PrettyPrinting

import Distributions
const D = Distributions

@load LinearRegressor pkg=MLJLinearModels
@load RidgeRegressor pkg=MLJLinearModels
@load LassoRegressor pkg=MLJLinearModels

hitters = dataset("ISLR", "Hitters")
@show size(hitters)
names(hitters) |> pprint

y, X = unpack(hitters, ==(:Salary), col->true);

no_miss = .!ismissing.(y)
y = collect(skipmissing(y))
X = X[no_miss, :]
train, test = partition(eachindex(y), 0.5, shuffle=true, rng=424);

using PyPlot

figure(figsize=(8,6))
plot(y, ls="none", marker="o")

xticks(fontsize=12); yticks(fontsize=12)
xlabel("Index", fontsize=14), ylabel("Salary", fontsize=14)



figure(figsize=(8,6))
hist(y, bins=50, density=true)

xticks(fontsize=12); yticks(fontsize=12)
xlabel("Salary", fontsize=14); ylabel("Density", fontsize=14)

edfit = D.fit_mle(D.Exponential, y)
xx = range(minimum(y), 2500, length=100)
yy = pdf.(edfit, xx)
plot(xx, yy, lw=3, label="Exponential distribution fit")

legend(fontsize=12)



Xc = coerce(X, autotype(X, rules=(:discrete_to_continuous,)))
scitype(Xc)

@pipeline RegPipe(std = Standardizer(),
                  hot = OneHotEncoder(),
                  reg = LinearRegressor())

model = RegPipe()
pipe  = machine(model, Xc, y)
fit!(pipe, rows=train)
ŷ = predict(pipe, rows=test)
round(rms(ŷ, y[test])^2, sigdigits=4)

figure(figsize=(8,6))

res = ŷ .- y[test]
stem(res)

xticks(fontsize=12); yticks(fontsize=12)
xlabel("Index", fontsize=14); ylabel("Residual (ŷ - y)", fontsize=14)

ylim([-1300, 1000])



figure(figsize=(8,6))
hist(res, bins=30, density=true, color="green")

xx = range(-1100, 1100, length=100)
ndfit = D.fit_mle(D.Normal, res)
lfit  = D.fit_mle(D.Laplace, res)

plot(xx, pdf.(ndfit, xx), lw=3, color="orange", label="Normal fit")
plot(xx, pdf.(lfit, xx), lw=3, color="magenta", label="Laplace fit")

legend(fontsize=12)

xticks(fontsize=12); yticks(fontsize=12)
xlabel("Residual (ŷ - y)", fontsize=14); ylabel("Density", fontsize=14)
xlim([-1100, 1100])



pipe.model.reg = RidgeRegressor()
fit!(pipe, rows=train)
ŷ = predict(pipe, rows=test)
round(rms(ŷ, y[test])^2, sigdigits=4)

r  = range(model, :(reg.lambda), lower=1e-2, upper=100_000, scale=:log10)
tm = TunedModel(model=model, ranges=r, tuning=Grid(resolution=50),
                resampling=CV(nfolds=3, rng=4141), measure=rms)
mtm = machine(tm, Xc, y)
fit!(mtm, rows=train)

best_mdl = fitted_params(mtm).best_model
round(best_mdl.reg.lambda, sigdigits=4)

ŷ = predict(mtm, rows=test)
round(rms(ŷ, y[test])^2, sigdigits=4)

figure(figsize=(8,6))

res = ŷ .- y[test]
stem(res)

xticks(fontsize=12); yticks(fontsize=12)
xlabel("Index", fontsize=14); ylabel("Residual (ŷ - y)", fontsize=14)

ylim([-1300, 1000])



mtm.model.model.reg = LassoRegressor()
mtm.model.range = range(model, :(reg.lambda), lower=500, upper=100_000, scale=:log10)
fit!(mtm, rows=train)

best_mdl = fitted_params(mtm).best_model
round(best_mdl.reg.lambda, sigdigits=4)

ŷ = predict(mtm, rows=test)
round(rms(ŷ, y[test])^2, sigdigits=4)

coefs, intercept = fitted_params(mtm.fitresult.fitresult.machine)
round.(coefs, sigdigits=2)

sum(coefs .≈ 0) / length(coefs)

figure(figsize=(8,6))
stem(coefs)

# name of the features including one-hot-encoded ones
all_names = [:AtBat, :Hits, :HmRun, :Runs, :RBI, :Walks, :Years,
             :CAtBat, :CHits, :CHmRun, :CRuns, :CRBI, :CWalks,
             :League__A, :League__N, :Div_E, :Div_W,
             :PutOuts, :Assists, :Errors, :NewLeague_A, :NewLeague_N]

idxshow = collect(1:length(coefs))[abs.(coefs) .> 10]
xticks(idxshow .- 1, all_names[idxshow], rotation=45, fontsize=12)
yticks(fontsize=12)
ylabel("Amplitude", fontsize=14)



@load ElasticNetRegressor pkg=MLJLinearModels

mtm.model.model.reg = ElasticNetRegressor()
mtm.model.range = [range(model, :(reg.lambda), lower=0.1, upper=100, scale=:log10),
                    range(model, :(reg.gamma),  lower=500, upper=10_000, scale=:log10)]
mtm.model.tuning = Grid(resolution=10)
fit!(mtm, rows=train)

best_mdl = fitted_params(mtm).best_model
@show round(best_mdl.reg.lambda, sigdigits=4)
@show round(best_mdl.reg.gamma, sigdigits=4)

ŷ = predict(mtm, rows=test)
round(rms(ŷ, y[test])^2, sigdigits=4)



# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

