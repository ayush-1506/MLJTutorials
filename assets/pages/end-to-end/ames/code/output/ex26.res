MLJ.DeterministicTunedModel(model = JuDoc.KNNRidgeBlend(knn_model = KNNRegressor @ 3…35,
                                                        ridge_model = RidgeRegressor @ 8…37,
                                                        knn_weight = 0.3,),
                            tuning = Grid(resolution = 3,
                                          parallel = true,),
                            resampling = CV(nfolds = 6,
                                            shuffle = false,
                                            rng = Random._GLOBAL_RNG(),),
                            measure = MLJBase.RMSL(),
                            weights = nothing,
                            operation = StatsBase.predict,
                            ranges = MLJ.NumericRange{T,Symbol} where T[NumericRange @ 1…43, NumericRange @ 1…94, NumericRange @ 2…43],
                            full_report = true,
                            train_best = true,) @ 4…71