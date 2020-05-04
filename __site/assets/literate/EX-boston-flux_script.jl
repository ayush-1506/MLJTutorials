# This file was generated, do not modify it.

using MLJFlux, MLJ, DataFrames, Statistics, Flux

MLJ.color_off() # hide
Random.seed!(11)

features, targets = @load_boston
features = DataFrame(features)
@show size(features)
@show targets[1:3]
first(features, 3) |> pretty

train, test = partition(eachindex(targets), 0.70, shuffle=true, rng=52)

mutable struct MyNNRegressor <: MLJFlux.Builder
    n1::Int #Number of cells in the first hidden layer
    n2::Int #Number of cells in the second hidden layer
end

function MLJFlux.fit(model::MyNNRegressor, ip, op)
    return Flux.Chain(Flux.Dense(ip, model.n1), Flux.Dense(model.n1, model.n2), Flux.Dense(model.n2, op))
end

myregressor = MyNNRegressor(20, 10)

nnregressor = NeuralNetworkRegressor(builder=myregressor, epochs=10)

mach = machine(nnregressor, features, targets)

fit!(mach, 3, rows=train, vebosity=3)

predict(mach, rows=test)

nnregressor.epochs = 15

fit!(mach, rows=train, verbosity=3)

nnregressor.batch_size = 2
fit!(mach, rows=train, verbosity=3)

# Tuning

bs = range(nnregressor, :batch_size, lower=1, upper=5)

tm = TunedModel(model=nnregressor, ranges=[bs, ], measure=Flux.mse)

m = machine(tm, features, targets)

fit!(m)

fitted_params(m).best_model.batch_size

