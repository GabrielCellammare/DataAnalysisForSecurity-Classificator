from UtilsFunctions import *
from sklearn.metrics import classification_report
from UtilsDecisionTree import decisionTreeLearner, determineDecisionTreekFoldMixed, showTree
import pickle


def DecisionTreeMutualInfoPCA(x, y, script_path, x_test_cleaned, y_test):
    folds = 5
    # Crea il percorso completo al file utilizzando pathlib
    serialize_dir = script_path.parent.parent / \
        "Serialized" / "BestConfigurationMutualInfoDecisionTree.pkl"

    # Verifica se il file esiste
    if os.path.exists(serialize_dir):
        # Se il file esiste, leggi i parametri
        with open(serialize_dir, "rb") as f:
            bestConfiguration = pickle.load(f)

        best_TH = bestConfiguration["best_TH"]

        print("\nCompleted!\n")

    serialize_dir = script_path.parent.parent / \
        "Serialized" / "MutualInfoTraining.pkl"

    # Verifica se il file esiste
    if os.path.exists(serialize_dir):
        # Se il file esiste, leggi i parametri
        with open(serialize_dir, "rb") as f:
            serialize_dir = pickle.load(f)
            rank = serialize_dir

    else:
        rank = mutualInfoRank(x, y)
        MutualInfoTraining = rank
        print(f"X mutual_info: '{rank}'\n")
        # Salva il dizionario in un file usando pickle
        with open(serialize_dir, "wb") as f:
            pickle.dump(MutualInfoTraining, f)

        boxPlotDirMutualInfo = script_path.parent.parent / "BoxPlotMutualInfo"

        BoxPlotAnalysisDataMutualInfo(
            x, y, boxPlotDirMutualInfo, rank, 10)

    toplist = topFeatureSelect(rank, best_TH)
    x_train_mutualInfo = x.loc[:, toplist]

    # Crea il percorso completo al file utilizzando pathlib
    serialize_dir = script_path.parent.parent / \
        "Serialized" / "BestConfigurationPCADecisionTree.pkl"

    # Verifica se il file esiste
    if os.path.exists(serialize_dir):
        # Se il file esiste, leggi i parametri
        with open(serialize_dir, "rb") as f:
            bestConfiguration = pickle.load(f)

        bestNPCA = bestConfiguration["bestNPCA"]

        print("\nCompleted!\n")

    pcaObj, pcalist, explained_variancePCA = pca(
        x_train_mutualInfo)

    print("\nExplained variance: ", explained_variancePCA, "\n")
    # Applicazione sull'intero dataset di training
    XPCA = applyPCA(x_train_mutualInfo, pcaObj, pcalist)
    XPCA_train_selected = XPCA.iloc[:,
                                    :bestNPCA]

    ListXTrain, ListXTest, ListYTrain, ListYTest = stratifiedKFold(
        XPCA_train_selected, y, folds)

    print("\n\nListXTrain")
    printFolds(ListXTrain)
    print("\n\nListYTrain")
    printFolds(ListYTrain)
    print("\n\nListXTest")
    printFolds(ListXTest)
    print("\n\nListYTest")
    printFolds(ListYTest)

    bestCriterion, bestEval = determineDecisionTreekFoldMixed(
        ListXTrain, ListYTrain, ListXTest, ListYTest)

    print('Feature Ranking by MI and PCA:\n',
          'Best criterion = ', bestCriterion, "\n", 'Best CV F = ', bestEval, "\n")

    DTMixed = decisionTreeLearner(XPCA_train_selected, y, bestCriterion)
    script_pathTreeFolder = script_path.parent.parent / "TreeFigOutput"
    showTree(DTMixed, script_pathTreeFolder, "DecisionTreeMutualInfoPCA")

    x_test_cleaned_featureMutualInfo = x_test_cleaned.loc[:, toplist]
    X_TestPCA = applyPCA(x_test_cleaned_featureMutualInfo, pcaObj, pcalist)

    x_TestPCA_cleaned_feature = X_TestPCA.iloc[:,
                                               :bestNPCA]

    y_pred = DTMixed.predict(x_TestPCA_cleaned_feature)
    target_names = ['class 0 - GoodWare ', 'class 1 - Malware']

    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)

    nome_file = "classification_report(DecisionTree_MutualInfoPCA).txt"
    script_pathClassification = script_path.parent.parent / \
        "ClassificationReport" / "DecisionTree"

    percorso_file = os.path.join(script_pathClassification, nome_file)

    # Apre il file in modalità scrittura
    with open(percorso_file, 'w') as file:
        # Scrive il classification report nel file
        file.write(report)

    script_pathFolder = script_path.parent.parent / "ConfusionMatrix" / "DecisionTree"
    ConfusionMatrixBuilder(
        DTMixed, y_pred, y_test, script_pathFolder, "DecisionTreeMutualInfoPCA")

    return DTMixed
