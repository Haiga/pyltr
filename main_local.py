import pyltr
import sys

if __name__ == "__main__":
    # for fold in ['1', '2', '3', '4', '5']:
    for fold in ['1']:
        BD = "C:\\Users\\pedro\\Downloads\\ml5k\\BD\\real\\"
        outpath = r"C:\Users\pedro\Downloads\ml5k\grisk\\fold" + fold + "\\"
        ntrees = 1000
        features = "106,111"

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
            n_estimators=1000,
            learning_rate=0.01,
            max_features=0.1,
            query_subsample=0.1,
            max_leaf_nodes=10,
            min_samples_leaf=150,
            verbose=1,
            features_risk=features,
            random_state=int(fold) + 42
        )

        model.fit(TX, Ty, Tqids, monitor=monitor)

        Epred = model.predict(EX)
        print('Random ranking:', metric.calc_mean_random(Eqids, Ey))
        print('Our model:', metric.calc_mean(Eqids, Ey, Epred))
        print('Some baseline model:', metric.calc_mean(Eqids, Ey, EX[:, int(features[0])]))

        with open(outpath + 'predictions.txt', 'w') as fo:
            for v in Epred:
                fo.write(str(v) + "\n")
