import os
from pathlib import Path
import pickle
import numpy as np
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from UtilsFunctions import NumberOfTopPCSelect, topFeatureSelect


def knnLearner(x, y, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x, np.ravel(y))
    return knn


def determineKNNkFoldConfigurationMutualInfo(ListXTrain, ListYTrain, ListXTest, ListYTest, rank, min_t, max_t, step):

    print("\nComputing best configuration with Mutual Info on KNN...\n")

    script_path = Path(__file__)

    # Crea il percorso completo al file utilizzando pathlib
    serialize_dir = script_path.parent.parent / \
        "Serialized" / "BestConfigurationMutualInfoKNN.pkl"

    # Verifica se il file esiste
    if os.path.exists(serialize_dir):
        # Se il file esiste, leggi i parametri
        with open(serialize_dir, "rb") as f:
            bestConfiguration = pickle.load(f)

        best_TH = bestConfiguration["best_TH"]
        bestN = bestConfiguration["bestN"]
        best_fscore = bestConfiguration["best_fscore"]
        best_Kneighbors = bestConfiguration["best_Kneighbors"]

        print("\nCompleted!\n")
        return best_TH, bestN, best_fscore, best_Kneighbors

    else:

        best_TH = None
        bestN = None
        best_fscore = 0
        best_Kneighbors = 0
        neighbors = [1, 2, 3]

        for neighbor in neighbors:
            for thre in np.arange(min_t, max_t, step):
                avg_fscore = 0
                fscores = []
                selectedFeatures = topFeatureSelect(rank, thre)
                if (len(selectedFeatures) > 0):
                    # Utilizzo la lunghezza di ListXTrain poichè è la stessa di ListXTest
                    for i in range(len(ListXTrain)):
                        x_train_feature_selected = ListXTrain[i].loc[:,
                                                                     selectedFeatures]
                        x_test = ListXTest[i].loc[:, selectedFeatures]
                        knn = knnLearner(
                            x_train_feature_selected, ListYTrain[i], neighbor)

                        y_pred = knn.predict(x_test)
                        fscores.append(f1_score(ListYTest[i], y_pred))

                if (len(fscores) > 1):
                    avg_fscore = np.mean(fscores)
                    print(f"Average F1 score: '{avg_fscore}'")
                    if avg_fscore == best_fscore:
                        if (len(selectedFeatures) < len(bestN)):  # ??
                            best_fscore = avg_fscore
                            best_TH = thre
                            bestN = selectedFeatures
                            best_Kneighbors = neighbor

                    if avg_fscore > best_fscore:
                        best_fscore = avg_fscore
                        best_TH = thre
                        bestN = selectedFeatures
                        best_Kneighbors = neighbor

        # Salva le variabili in un dizionario
        BestConfiguration = {"best_TH": best_TH,
                             "bestN": bestN, "best_fscore": best_fscore, "best_Kneighbors": best_Kneighbors}

        # Salva il dizionario in un file usando pickle
        with open(serialize_dir, "wb") as f:
            pickle.dump(BestConfiguration, f)
        print("\nCompleted!\n")

        return best_TH, bestN, best_fscore, best_Kneighbors


def determineKNNkFoldConfigurationPCA(ListXTrain, ListYTrain, ListXTest, ListYTest, explained_variance, min_t, max_t, step):

    print("\nComputing best configuration with PCA on KNN..\n")

    script_path = Path(__file__)

    # Crea il percorso completo al file utilizzando pathlib
    serialize_dir = script_path.parent.parent / \
        "Serialized" / "BestConfigurationPCAKNN.pkl"

    # Verifica se il file esiste
    if os.path.exists(serialize_dir):
        # Se il file esiste, leggi i parametri
        with open(serialize_dir, "rb") as f:
            bestConfiguration = pickle.load(f)

        bestTHPCA = bestConfiguration["bestTHPCA"]
        bestNPCA = bestConfiguration["bestNPCA"]
        bestEvalPCA = bestConfiguration["bestEvalPCA"]
        best_KneighborsPCA = bestConfiguration["best_KneighborsPCA"]

        print("\nCompleted!\n")
        return bestTHPCA, bestNPCA, bestEvalPCA, best_KneighborsPCA

    else:
        bestTHPCA = None
        bestNPCA = None
        bestEvalPCA = 0
        best_KneighborsPCA = 0
        neighbors = [1, 2, 3]

        for neighbor in neighbors:
            for thre in np.arange(min_t, max_t, step):
                avg_fscore = 0
                fscores = []
                n = NumberOfTopPCSelect(explained_variance, thre)
                if (n > 0):
                    # Utilizzo la lunghezza di ListXTrain poichè è la stessa di ListXTest
                    for i in range(len(ListXTrain)):
                        # indicizzazione di tipo :n in Python seleziona gli elementi dall’indice 0 all’indice n-1
                        x_train_feature_selected = ListXTrain[i].iloc[:, :n]
                        x_test = ListXTest[i].iloc[:, :n]
                        knn = knnLearner(
                            x_train_feature_selected, ListYTrain[i], neighbor)

                        y_pred = knn.predict(x_test)
                        fscores.append(f1_score(ListYTest[i], y_pred))

                if (len(fscores) > 1):
                    avg_fscore = np.mean(fscores)
                    print(f"Average F1 score: '{avg_fscore}'")

                    if avg_fscore == bestEvalPCA:
                        if (n < bestNPCA):
                            bestEvalPCA = avg_fscore
                            bestTHPCA = thre
                            bestNPCA = n
                            best_KneighborsPCA = neighbor

                    if avg_fscore > bestEvalPCA:
                        bestEvalPCA = avg_fscore
                        bestTHPCA = thre
                        bestNPCA = n
                        best_KneighborsPCA = neighbor

        # Salva le variabili in un dizionario
        BestConfiguration = {"bestTHPCA": bestTHPCA,
                             "bestNPCA": bestNPCA, "bestEvalPCA": bestEvalPCA, "best_KneighborsPCA": best_KneighborsPCA}

        # Salva il dizionario in un file usando pickle
        with open(serialize_dir, "wb") as f:
            pickle.dump(BestConfiguration, f)
        print("\nCompleted!\n")

        return bestTHPCA, bestNPCA, bestEvalPCA, best_KneighborsPCA


def determineKNNkFoldConfigurationMIPCA(ListXTrain, ListYTrain, ListXTest, ListYTest, explained_variance, min_t, max_t, step):

    print("\nComputing best configuration with Mutual Info and PCA on KNN..\n")

    script_path = Path(__file__)

    # Crea il percorso completo al file utilizzando pathlib
    serialize_dir = script_path.parent.parent / \
        "Serialized" / "BestConfigurationMIPCAKNN.pkl"

    # Verifica se il file esiste
    if os.path.exists(serialize_dir):
        # Se il file esiste, leggi i parametri
        with open(serialize_dir, "rb") as f:
            bestConfiguration = pickle.load(f)

        bestTHPCA = bestConfiguration["bestTHPCA"]
        bestNPCA = bestConfiguration["bestNPCA"]
        bestEvalPCA = bestConfiguration["bestEvalPCA"]
        best_KneighborsPCA = bestConfiguration["best_KneighborsPCA"]

        print("\nCompleted!\n")
        return bestTHPCA, bestNPCA, bestEvalPCA, best_KneighborsPCA

    else:
        bestTHPCA = None
        bestNPCA = None
        bestEvalPCA = 0
        best_KneighborsPCA = 0
        neighbors = [1, 2, 3]

        for neighbor in neighbors:
            for thre in np.arange(min_t, max_t, step):
                avg_fscore = 0
                fscores = []
                n = NumberOfTopPCSelect(explained_variance, thre)
                if (n > 0):
                    # Utilizzo la lunghezza di ListXTrain poichè è la stessa di ListXTest
                    for i in range(len(ListXTrain)):
                        # indicizzazione di tipo :n in Python seleziona gli elementi dall’indice 0 all’indice n-1
                        x_train_feature_selected = ListXTrain[i].iloc[:, :n]
                        x_test = ListXTest[i].iloc[:, :n]
                        knn = knnLearner(
                            x_train_feature_selected, ListYTrain[i], neighbor)

                        y_pred = knn.predict(x_test)
                        fscores.append(f1_score(ListYTest[i], y_pred))

                if (len(fscores) > 1):
                    avg_fscore = np.mean(fscores)
                    print(f"Average F1 score: '{avg_fscore}'")

                    if avg_fscore == bestEvalPCA:
                        if (n < bestNPCA):
                            bestEvalPCA = avg_fscore
                            bestTHPCA = thre
                            bestNPCA = n
                            best_KneighborsPCA = neighbor

                    if avg_fscore > bestEvalPCA:
                        bestEvalPCA = avg_fscore
                        bestTHPCA = thre
                        bestNPCA = n
                        best_KneighborsPCA = neighbor

        # Salva le variabili in un dizionario
        BestConfiguration = {"bestTHPCA": bestTHPCA,
                             "bestNPCA": bestNPCA, "bestEvalPCA": bestEvalPCA, "best_KneighborsPCA": best_KneighborsPCA}

        # Salva il dizionario in un file usando pickle
        with open(serialize_dir, "wb") as f:
            pickle.dump(BestConfiguration, f)
        print("\nCompleted!\n")

        return bestTHPCA, bestNPCA, bestEvalPCA, best_KneighborsPCA
