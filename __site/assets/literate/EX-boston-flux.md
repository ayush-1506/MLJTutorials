<!--This file was generated, do not modify it.-->
**Main author**: Ayush Shridhar (ayush-1506).

## Getting started

```julia:ex1
using MLJFlux, MLJ, DataFrames, Statistics, Flux

MLJ.color_off() # hide
Random.seed!(11)
```

Loading the Boston dataset. Our aim will be to implement a
neural network regressor to predict the price of a house,
given a number of features.

```julia:ex2
features, targets = @load_boston
features = DataFrame(features)
@show size(features)
@show targets[1:3]
first(features, 3) |> pretty
```

Next obvious steps: partitioning into train and test set

```julia:ex3
train, test = partition(eachindex(targets), 0.70, shuffle=true, rng=52)
```

Let us try to implement an Neural Network regressor using
Flux.jl. MLJFlux.jl provides an MLJ interface to the Flux.jl
deep learning framework. The package provides four essential
models: `NeuralNetworkRegressor, MultivariateNeuralNetworkRegressor,
NeuralNetworkClassifier` and `ImageClassifier`.

At the heart of these models is a neural network. This is specified using
the `builder` parameter. Creating a builder object consists of two steps:
1. Creating a new struct inherited from `MLJFlux.Builder`. `MLJFlux.Builder`
is an abstract structure used for the purpose of dispatching. Suppose we define
a new struct called `MyNNRegressor`. This can contain any attributed required to
build the model later. (Step 2). Let's use Dense Neural Network with 2 hidden layers.

```julia:ex4
mutable struct MyNNRegressor <: MLJFlux.Builder
    n1::Int #Number of cells in the first hidden layer
    n2::Int #Number of cells in the second hidden layer
end
```

Step 2: Building the neural network from this object.
Extend the `MLJFlux.fit` function. This takes in 3 arguments: The object of
`MyNNRegressor`, input dimension (ip) and output dimension (op).

```julia:ex5
function MLJFlux.fit(model::MyNNRegressor, ip, op)
    return Flux.Chain(Flux.Dense(ip, model.n1), Flux.Dense(model.n1, model.n2), Flux.Dense(model.n2, op))
end
```

Will all definitions ready, let us create an oject of this:

```julia:ex6
myregressor = MyNNRegressor(20, 10)
```

Since the boston dataset is a regression problem, we'll be using
`NeuralNetworkRegressor` here. One thing to remember is that
a `NeuralNetworkRegressor` object works seamlessly like any other
MLJ model: you can wrap it in an  MLJ `machine` and to anything
you'd do otherwise.

Let's start by defining our NeuralNetworkRegressor object, that takes `myregressor`
as it's parameter.

```julia:ex7
nnregressor = NeuralNetworkRegressor(builder=myregressor, epochs=10)
```

Other parameters that `NeuralNetworkRegressor` takes in are:
optimiser = ADAM(0.001, (0.9, 0.999), IdDict{Any,Any}())
loss = Flux.mse,
epochs = 10,
batch_size = 1,
lambda = 0.0,
alpha = 0.0,
optimiser_changes_trigger_retraining = false

`nnregressor` now acts like any other MLJ model. Let's try wrapping it in a
MLJ machine and calling `fit!, predict!`.

```julia:ex8
mach = machine(nnregressor, features, targets)
```

Let's fit this on the train set

```julia:ex9
fit!(mach, 3, rows=train, vebosity=3)
```

As we can see, the loss decreases at each epoch, showing the the neural network
is gradually learning form the training set.

```julia:ex10
predict(mach, rows=test)
```

Now let's retrain our model. One thing to remember is that retrainig may OR may not
re-initialize our neural network model parameters. For example, changing the number of
epochs to 15 will not causes the model to train to 15 epcohs, but just 5 additional
epochs.

```julia:ex11
nnregressor.epochs = 15

fit!(mach, rows=train, verbosity=3)
```

You can always specify that you want to retrain the model from scratch using the force=true
parameter. (Look at documentation for `fit!` for more).

However, chaning parameters such as batch_size will necesarrily cause re-training from scratch.

```julia:ex12
nnregressor.batch_size = 2
fit!(mach, rows=train, verbosity=3)
```

Another bit to remember here is that changing the optimiser doesn't cause retaining by default.
However, the `optimiser_changes_trigger_retraining` in NeuralNetworkRegressor can be toggled to
accomodate this.

```julia:ex13
# Tuning
```

As mentioned above, `nnregressor` can act like any other MLJ model. Let's try to tune the
batch_size parameter.

```julia:ex14
bs = range(nnregressor, :batch_size, lower=1, upper=5)

tm = TunedModel(model=nnregressor, ranges=[bs, ], measure=Flux.mse)
```

For more on tuning, refer to the model-tuning tutorial.

```julia:ex15
m = machine(tm, features, targets)

fit!(m)
```

This evaluated the model at each value of our range.
The best value is:

```julia:ex16
fitted_params(m).best_model.batch_size
```

