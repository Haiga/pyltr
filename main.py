import pyltr
import sys

if __name__ == "__main__":
    BD = sys.argv[1]
    outpath = sys.argv[2]
    ntrees = int(sys.argv[3])
    features = sys.argv[4].split(",")

    with open(BD + 'Norm.train.txt') as trainfile, open(BD + 'Norm.vali.txt') as valifile, open(
            BD + 'Norm.test.txt') as evalfile:
        TX, Ty, Tqids, _ = pyltr.data.letor.read_dataset(trainfile)
        VX, Vy, Vqids, _ = pyltr.data.letor.read_dataset(valifile)
        EX, Ey, Eqids, _ = pyltr.data.letor.read_dataset(evalfile)

    metric = pyltr.metrics.NDCG(k=10)

    monitor = pyltr.models.monitors.ValidationMonitor(
        VX, Vy, Vqids, metric=metric, stop_after=250)

    model = pyltr.models.LambdaMART(
        metric=metric,
        # n_estimators=1000,
        n_estimators=ntrees,
        learning_rate=0.05,
        max_features=0.1,
        query_subsample=0.1,
        max_leaf_nodes=5,
        min_samples_leaf=150,
        verbose=1,
        features_risk=features
    )

    model.fit(TX, Ty, Tqids, monitor=monitor)

    Epred = model.predict(EX)
    print('Random ranking:', metric.calc_mean_random(Eqids, Ey))
    print('Our model:', metric.calc_mean(Eqids, Ey, Epred))
    print('Some baseline model:', metric.calc_mean(Eqids, Ey, EX[:, int(features[0])]))

    with open(outpath + 'predictions.txt', 'w') as fo:
        for v in Epred:
            fo.write(str(v) + "\n")
