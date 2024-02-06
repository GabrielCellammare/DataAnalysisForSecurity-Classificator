import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import numpy as np
from sklearn.metrics import f1_score
import pickle
from pathlib import Path
from UtilsFunctions import topFeatureSelect, NumberOfTopPCSelect

seed = 42
np.random.seed(seed)


def decisionTreeLearner(X, Y, c='entropy'):
    clf = DecisionTreeClassifier(criterion=c, random_state=seed)
    clf.min_samples_split = 500  # Numero minimo di esempi per mettere uno split
    # Tree_ dentro c'è l'albero addestrata
    clf.fit(X, Y)

    print(f"Number of nodes: {clf.tree_.node_count}")
    print(f"Number of leaves: {clf.get_n_leaves()} ")
    # Return the number of leaves of the decision tree.)
    return clf


"""
CLF Oggetto Decision Tree
Funzione che stampa l'albero e lo salva nell'apposita cartella
"""


def showTree(clf, script_pathTreeFolder, TreeType):

    plt.figure(figsize=(15, 10))
    tree.plot_tree(clf,
                   filled=True,
                   rounded=True)
    file_name = os.path.join(script_pathTreeFolder,
                             f'TreeFigOutput{TreeType}.pdf')
    plt.savefig(file_name,
                format='pdf', dpi=1000)
    plt.show()


def determineDecisionTreekFoldConfigurationMutualInfo(ListXTrain, ListYTrain, ListXTest, ListYTest, rank, min_t, max_t, step):

    print("\nComputing best configuration with Mutual Info on Decision Tree...\n")

    script_path = Path(__file__)

    # Crea il percorso completo al file utilizzando pathlib
    serialize_dir = script_path.parent.parent / \
        "Serialized" / "BestConfigurationMutualInfoDecisionTree.pkl"

    # Verifica se il file esiste
    if os.path.exists(serialize_dir):
        # Se il file esiste, leggi i parametri
        with open(serialize_dir, "rb") as f:
            bestConfiguration = pickle.load(f)

        best_criterion = bestConfiguration["best_criterion"]
        best_TH = bestConfiguration["best_TH"]
        bestN = bestConfiguration["bestN"]
        best_fscore = bestConfiguration["best_fscore"]

        print("\nCompleted!\n")
        return best_criterion, best_TH, bestN, best_fscore

    else:

        best_criterion = None
        best_TH = None
        bestN = None
        best_fscore = 0

        criterion = ['gini', 'entropy']

        for criteria in criterion:
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
                        clf = decisionTreeLearner(
                            x_train_feature_selected, ListYTrain[i], criteria)

                        y_pred = clf.predict(x_test)
                        fscores.append(f1_score(ListYTest[i], y_pred))

                if (len(fscores) > 1):
                    avg_fscore = np.mean(fscores)
                    print(f"Average F1 score: '{avg_fscore}'")
                    if avg_fscore > best_fscore:
                        best_fscore = avg_fscore
                        best_criterion = criteria
                        best_TH = thre
                        bestN = selectedFeatures

                    if avg_fscore == best_fscore:
                        if (len(selectedFeatures) < len(bestN)):  # ??
                            best_fscore = avg_fscore
                            best_criterion = criteria
                            best_TH = thre
                            bestN = selectedFeatures

        # Salva le variabili in un dizionario
        BestConfiguration = {"best_criterion": best_criterion, "best_TH": best_TH,
                             "bestN": bestN, "best_fscore": best_fscore, }

        # Salva il dizionario in un file usando pickle
        with open(serialize_dir, "wb") as f:
            pickle.dump(BestConfiguration, f)
        print("\nCompleted!\n")

        return best_criterion, best_TH, bestN, best_fscore


def determineDecisionTreekFoldConfigurationPCA(ListXTrain, ListYTrain, ListXTest, ListYTest, explained_variance, min_t, max_t, step):

    print("\nComputing best configuration with PCA on Decision Tree...\n")

    script_path = Path(__file__)

    # Crea il percorso completo al file utilizzando pathlib
    serialize_dir = script_path.parent.parent / \
        "Serialized" / "BestConfigurationPCADecisionTree.pkl"

    # Verifica se il file esiste
    if os.path.exists(serialize_dir):
        # Se il file esiste, leggi i parametri
        with open(serialize_dir, "rb") as f:
            bestConfiguration = pickle.load(f)

        best_criterionPCA = bestConfiguration["best_criterionPCA"]
        bestTHPCA = bestConfiguration["bestTHPCA"]
        bestNPCA = bestConfiguration["bestNPCA"]
        bestEvalPCA = bestConfiguration["bestEvalPCA"]

        print("\nCompleted!\n")
        return best_criterionPCA, bestTHPCA, bestNPCA, bestEvalPCA

    else:
        best_criterionPCA = None
        bestTHPCA = None
        bestNPCA = None
        bestEvalPCA = 0

        criterion = ['gini', 'entropy']

        for criteria in criterion:
            for thre in np.arange(min_t, max_t, step):
                avg_fscore = 0
                fscores = []
                n = NumberOfTopPCSelect(explained_variance, thre)
                if (n > 0):
                    # Utilizzo la lunghezza di ListXTrain poichè è la stessa di ListXTest
                    for i in range(len(ListXTrain)):
                        # indicizzazione di tipo :n in Python seleziona gli elementi dall’indice 0 all’indice n-1
                        x_train_feature_selected = ListXTrain[i].iloc[:, 1:(
                            n+1)]
                        x_test = ListXTest[i].iloc[:, 1:(
                            n+1)]
                        clf = decisionTreeLearner(
                            x_train_feature_selected, ListYTrain[i], criteria)

                        y_pred = clf.predict(x_test)
                        fscores.append(f1_score(ListYTest[i], y_pred))

                if (len(fscores) > 1):
                    avg_fscore = np.mean(fscores)
                    print(f"Average F1 score: '{avg_fscore}'")
                    if avg_fscore > bestEvalPCA:
                        bestEvalPCA = avg_fscore
                        best_criterionPCA = criteria
                        bestTHPCA = thre
                        bestNPCA = n

                    if avg_fscore == bestEvalPCA:
                        if (n < bestNPCA):  # ??
                            bestEvalPCA = avg_fscore
                            best_criterionPCA = criteria
                            bestTHPCA = thre
                            bestNPCA = n

        # Salva le variabili in un dizionario
        BestConfiguration = {"best_criterionPCA": best_criterionPCA, "bestTHPCA": bestTHPCA,
                             "bestNPCA": bestNPCA, "bestEvalPCA": bestEvalPCA, }

        # Salva il dizionario in un file usando pickle
        with open(serialize_dir, "wb") as f:
            pickle.dump(BestConfiguration, f)
        print("\nCompleted!\n")

        return best_criterionPCA, bestTHPCA, bestNPCA, bestEvalPCA
