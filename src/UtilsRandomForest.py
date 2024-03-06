from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.metrics import f1_score
import pickle
from UtilsFunctions import topFeatureSelect, NumberOfTopPCSelect

seed = 42
np.random.seed(seed)

"""
randomizaion = max_features{“sqrt”, “log2”, None}
The number of features to consider when looking for the best split:


max_samplesint or float, default=None
If bootstrap is True, the number of samples to draw from X to train each base estimator.


"""


def randomForestLearner(x, y, n_tree, c, rand, bootstrap_s):
    rlf = RandomForestClassifier(n_estimators=n_tree, criterion=c,
                                 max_features=rand, max_samples=bootstrap_s, random_state=seed)

    rlf.fit(x, np.ravel(y))

    return rlf


def determineRFkFoldConfigurationMutualInfo(ListXTrain, ListYTrain, ListXTest, ListYTest, rank, min_t, max_t, step):

    print("\nComputing best configuration with Mutual Info on Random Forest...\n")

    script_path = Path(__file__)

    # Crea il percorso completo al file utilizzando pathlib
    serialize_dir = script_path.parent.parent / \
        "Serialized" / "BestConfigurationMutualInfoRandomForest.pkl"

    # Verifica se il file esiste
    if os.path.exists(serialize_dir):
        # Se il file esiste, leggi i parametri
        with open(serialize_dir, "rb") as f:
            bestConfiguration = pickle.load(f)

        best_criterion = bestConfiguration["best_criterion"]
        best_TH = bestConfiguration["best_TH"]
        bestN = bestConfiguration["bestN"]
        best_fscore = bestConfiguration["best_fscore"]
        best_n_tree = bestConfiguration["best_n_tree"]
        best_rand = bestConfiguration["best_rand"]
        best_bootstrap_s = bestConfiguration["best_bootstrap_s"]

        print("\nCompleted!\n")
        return best_criterion, best_TH, bestN, best_fscore, best_n_tree, best_rand, best_bootstrap_s

    else:

        best_criterion = None
        best_TH = None
        bestN = None
        best_fscore = 0
        best_n_tree = 0
        best_rand = 0
        best_bootstrap_s = 0

        criterion = ['gini', 'entropy']
        randomization = ['sqrt', 'log2']
        number_of_trees = [10, 20, 30]
        bootstrap_size = [0.7, 0.8, 0.9]

        for criteria in criterion:
            for rand in randomization:
                for n_tree in number_of_trees:
                    for b_size in bootstrap_size:
                        for thre in np.arange(min_t, max_t, step):
                            avg_fscore = 0
                            fscores = []
                            selectedFeatures = topFeatureSelect(rank, thre)
                            if (len(selectedFeatures) > 0):
                                # Utilizzo la lunghezza di ListXTrain poichè è la stessa di ListXTest
                                for i in range(len(ListXTrain)):
                                    x_train_feature_selected = ListXTrain[i].loc[:,
                                                                                 selectedFeatures]
                                    x_test = ListXTest[i].loc[:,
                                                              selectedFeatures]
                                    rlf = randomForestLearner(
                                        x_train_feature_selected, ListYTrain[i], n_tree, criteria, rand, b_size)

                                    y_pred = rlf.predict(x_test)
                                    fscores.append(
                                        f1_score(ListYTest[i], y_pred))

                            if (len(fscores) > 1):
                                avg_fscore = np.mean(fscores)
                                print(f"Average F1 score: '{avg_fscore}'")
                                if avg_fscore > best_fscore:
                                    best_fscore = avg_fscore
                                    best_criterion = criteria
                                    best_TH = thre
                                    bestN = selectedFeatures
                                    best_n_tree = n_tree
                                    best_rand = rand
                                    best_bootstrap_s = b_size

                            if avg_fscore == best_fscore:
                                if (len(selectedFeatures) < len(bestN)):  # ??
                                    best_fscore = avg_fscore
                                    best_criterion = criteria
                                    best_TH = thre
                                    bestN = selectedFeatures
                                    best_n_tree = n_tree
                                    best_rand = rand
                                    best_bootstrap_s = b_size

        # Salva le variabili in un dizionario
        BestConfiguration = {"best_criterion": best_criterion, "best_TH": best_TH,
                             "bestN": bestN, "best_fscore": best_fscore, "best_n_tree": best_n_tree,
                             "best_rand": best_rand, "best_bootstrap_s": best_bootstrap_s}

        # Salva il dizionario in un file usando pickle
        with open(serialize_dir, "wb") as f:
            pickle.dump(BestConfiguration, f)
        print("\nCompleted!\n")

        return best_criterion, best_TH, bestN, best_fscore, best_n_tree, best_rand, best_bootstrap_s


def determineRFkFoldConfigurationPCA(ListXTrain, ListYTrain, ListXTest, ListYTest, explained_variance, min_t, max_t, step):
    print("\nComputing best configuration with PCA on Random Forest...\n")

    script_path = Path(__file__)

    # Crea il percorso completo al file utilizzando pathlib
    serialize_dir = script_path.parent.parent / \
        "Serialized" / "BestConfigurationPCARandomForest.pkl"

    # Verifica se il file esiste
    if os.path.exists(serialize_dir):
        # Se il file esiste, leggi i parametri
        with open(serialize_dir, "rb") as f:
            bestConfiguration = pickle.load(f)

        best_criterionPCA = bestConfiguration["best_criterionPCA"]
        best_THPCA = bestConfiguration["best_THPCA"]
        bestNPCA = bestConfiguration["bestNPCA"]
        best_fscorePCA = bestConfiguration["best_fscorePCA"]
        best_n_treePCA = bestConfiguration["best_n_treePCA"]
        best_randPCA = bestConfiguration["best_randPCA"]
        best_bootstrap_sPCA = bestConfiguration["best_bootstrap_sPCA"]

        print("\nCompleted!\n")
        return best_criterionPCA, best_THPCA, bestNPCA, best_fscorePCA, best_n_treePCA, best_randPCA, best_bootstrap_sPCA

    else:

        best_criterionPCA = None
        best_THPCA = None
        bestNPCA = None
        best_fscorePCA = 0
        best_n_treePCA = 0
        best_randPCA = 0
        best_bootstrap_sPCA = 0

        criterion = ['gini', 'entropy']
        randomization = ['sqrt', 'log2']
        number_of_trees = [10, 20, 30]
        bootstrap_size = [0.7, 0.8, 0.9]

        for criteria in criterion:
            for rand in randomization:
                for n_tree in number_of_trees:
                    for b_size in bootstrap_size:
                        for thre in np.arange(min_t, max_t, step):
                            avg_fscore = 0
                            fscores = []
                            n = NumberOfTopPCSelect(explained_variance, thre)
                            if (n > 0):
                                # Utilizzo la lunghezza di ListXTrain poichè è la stessa di ListXTest
                                for i in range(len(ListXTrain)):
                                    x_train_feature_selected = ListXTrain[i].iloc[:, 1:(
                                        n+1)]

                                    x_test = ListXTest[i].iloc[:, 1:(
                                        n+1)]

                                    rlf = randomForestLearner(
                                        x_train_feature_selected, ListYTrain[i], n_tree, criteria, rand, b_size)

                                    y_pred = rlf.predict(x_test)
                                    fscores.append(
                                        f1_score(ListYTest[i], y_pred))

                            if (len(fscores) > 1):
                                avg_fscore = np.mean(fscores)
                                print(f"Average F1 score: '{avg_fscore}'")
                                if avg_fscore > best_fscorePCA:
                                    best_fscorePCA = avg_fscore
                                    best_criterionPCA = criteria
                                    best_THPCA = thre
                                    bestNPCA = n
                                    best_n_treePCA = n_tree
                                    best_randPCA = rand
                                    best_bootstrap_sPCA = b_size

                            if avg_fscore == best_fscorePCA:
                                if (n < bestNPCA):
                                    best_fscorePCA = avg_fscore
                                    best_criterionPCA = criteria
                                    best_THPCA = thre
                                    bestNPCA = n
                                    best_n_treePCA = n_tree
                                    best_randPCA = rand
                                    best_bootstrap_sPCA = b_size

        # Salva le variabili in un dizionario
        BestConfiguration = {"best_criterionPCA": best_criterionPCA, "best_THPCA": best_THPCA,
                             "bestNPCA": bestNPCA, "best_fscorePCA": best_fscorePCA, "best_n_treePCA": best_n_treePCA,
                             "best_randPCA": best_randPCA, "best_bootstrap_sPCA": best_bootstrap_sPCA}

        # Salva il dizionario in un file usando pickle
        with open(serialize_dir, "wb") as f:
            pickle.dump(BestConfiguration, f)
        print("\nCompleted!\n")

        return best_criterionPCA, best_THPCA, bestNPCA, best_fscorePCA, best_n_treePCA, best_randPCA, best_bootstrap_sPCA
