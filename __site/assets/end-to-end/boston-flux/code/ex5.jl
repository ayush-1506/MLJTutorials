# This file was generated, do not modify it. # hide
function MLJFlux.fit(model::MyNNRegressor, ip, op)
    return Flux.Chain(Flux.Dense(ip, model.n1), Flux.Dense(model.n1, model.n2), Flux.Dense(model.n2, op))
end