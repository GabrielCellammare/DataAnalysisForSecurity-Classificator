import os
from pathlib import Path
import pickle
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score
from UtilsFunctions import NumberOfTopPCSelect, topFeatureSelect


def EnsambleLearner(x, y, clf1, clf2, clf3):
    eclf = VotingClassifier(
        estimators=[('dt', clf1), ('rf', clf2), ('knn', clf3)], voting='hard')

    eclf.fit(x, np.ravel(y))
    return eclf


def determineEnsamblekFoldConfigurationMutualInfo(ListXTrain, ListYTrain, ListXTest, ListYTest, rank, min_t, max_t, step, clf1, clf2, clf3):

    print("\nComputing best configuration with Mutual Info on Ensamble with DecisionTree, RandomForest and Knn...\n")

    script_path = Path(__file__)

    # Crea il percorso completo al file utilizzando pathlib
    serialize_dir = script_path.parent.parent / \
        "Serialized" / "BestConfigurationMutualInfoEnsemble.pkl"

    # Verifica se il file esiste
    if os.path.exists(serialize_dir):
        # Se il file esiste, leggi i parametri
        with open(serialize_dir, "rb") as f:
            bestConfiguration = pickle.load(f)

        best_TH = bestConfiguration["best_TH"]
        bestN = bestConfiguration["bestN"]
        best_fscore = bestConfiguration["best_fscore"]

        print("\nCompleted!\n")
        return best_TH, bestN, best_fscore

    else:

        best_TH = None
        bestN = None
        best_fscore = 0

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
                    eclf = EnsambleLearner(
                        x_train_feature_selected, ListYTrain[i], clf1, clf2, clf3)

                    y_pred = eclf.predict(x_test)
                    fscores.append(f1_score(ListYTest[i], y_pred))

            if (len(fscores) > 1):
                avg_fscore = np.mean(fscores)
                print(f"Average F1 score: '{avg_fscore}'")
                if avg_fscore == best_fscore:
                    if (len(selectedFeatures) < len(bestN)):
                        best_fscore = avg_fscore
                        best_TH = thre
                        bestN = selectedFeatures

                if avg_fscore > best_fscore:
                    best_fscore = avg_fscore
                    best_TH = thre
                    bestN = selectedFeatures

        # Salva le variabili in un dizionario
        BestConfiguration = {"best_TH": best_TH,
                             "bestN": bestN, "best_fscore": best_fscore}

        # Salva il dizionario in un file usando pickle
        with open(serialize_dir, "wb") as f:
            pickle.dump(BestConfiguration, f)
        print("\nCompleted!\n")

        return best_TH, bestN, best_fscore


def determineEnsamblekFoldConfigurationPCA(ListXTrain, ListYTrain, ListXTest, ListYTest, explained_variance, min_t, max_t, step, clf1, clf2, clf3):

    print("\nComputing best configuration with PCA on Ensemble with Decision Tree, Random Forest and KNN...\n")

    script_path = Path(__file__)

    # Crea il percorso completo al file utilizzando pathlib
    serialize_dir = script_path.parent.parent / \
        "Serialized" / "BestConfigurationPCAEnsemble.pkl"

    # Verifica se il file esiste
    if os.path.exists(serialize_dir):
        # Se il file esiste, leggi i parametri
        with open(serialize_dir, "rb") as f:
            bestConfiguration = pickle.load(f)

        bestTHPCA = bestConfiguration["bestTHPCA"]
        bestNPCA = bestConfiguration["bestNPCA"]
        bestEvalPCA = bestConfiguration["bestEvalPCA"]

        print("\nCompleted!\n")
        return bestTHPCA, bestNPCA, bestEvalPCA

    else:
        bestTHPCA = None
        bestNPCA = None
        bestEvalPCA = 0

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
                    eclf = EnsambleLearner(
                        x_train_feature_selected, ListYTrain[i], clf1, clf2, clf3)

                    y_pred = eclf.predict(x_test)
                    fscores.append(f1_score(ListYTest[i], y_pred))

            if (len(fscores) > 1):
                avg_fscore = np.mean(fscores)
                print(f"Average F1 score: '{avg_fscore}'")

                if avg_fscore == bestEvalPCA:
                    if (n < bestNPCA):
                        bestEvalPCA = avg_fscore
                        bestTHPCA = thre
                        bestNPCA = n

                if avg_fscore > bestEvalPCA:
                    bestEvalPCA = avg_fscore
                    bestTHPCA = thre
                    bestNPCA = n

        # Salva le variabili in un dizionario
        BestConfiguration = {"bestTHPCA": bestTHPCA,
                             "bestNPCA": bestNPCA, "bestEvalPCA": bestEvalPCA}

        # Salva il dizionario in un file usando pickle
        with open(serialize_dir, "wb") as f:
            pickle.dump(BestConfiguration, f)
        print("\nCompleted!\n")

        return bestTHPCA, bestNPCA, bestEvalPCA
