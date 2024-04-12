import pickle
from UtilsFunctions import *
from sklearn.metrics import classification_report
from UtilsKNN import knnLearner, determineKNNkFoldConfigurationMIPCA

# adopt the stratified CV to determine the best decision tree configuration on the pcs


def KNNMIPCA(x, y, script_path, x_test_cleaned, y_test):

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
        # print(f"X mutual_info: '{rank}'\n")
        # Salva il dizionario in un file usando pickle
        with open(serialize_dir, "wb") as f:
            pickle.dump(MutualInfoTraining, f)

        boxPlotDirMutualInfo = script_path.parent.parent / "BoxPlotMutualInfo"

        if (not os.listdir(boxPlotDirMutualInfo)):
            BoxPlotAnalysisDataMutualInfo(
                x, y, boxPlotDirMutualInfo, rank, 10)

    serialize_dir = script_path.parent.parent / \
        "Serialized" / "BestConfigurationMutualInfoKNN.pkl"

    # Verifica se il file esiste
    if os.path.exists(serialize_dir):
        # Se il file esiste, leggi i parametri
        with open(serialize_dir, "rb") as f:
            bestConfiguration = pickle.load(f)

        best_TH = bestConfiguration["best_TH"]

    # Prendo le feature migliori (Mutual info)
    toplist = topFeatureSelect(rank, best_TH)

    x_new = x.loc[:, toplist]
    x_test_cleaned_new = x_test_cleaned.loc[:, toplist]

    minThresholdPCA = 0.95
    stepThresholdPCA = 0.01
    maxThresholdPCA = 1.01
    folds = 5

    pcaObj, pcalist, explained_variancePCA = pca(
        x_new)
    # print("\nExplained variance: ", explained_variancePCA, "\n")
    XPCA = applyPCA(x_new, pcaObj, pcalist)

    ListXTrainPCA, ListXTestPCA, ListYTrainPCA, ListYTestPCA = stratifiedKFold(
        XPCA, y, folds)

    bestTHPCA, bestNPCA, bestEvalPCA, best_KneighborsPCA = determineKNNkFoldConfigurationMIPCA(
        ListXTrainPCA, ListYTrainPCA, ListXTestPCA, ListYTestPCA, explained_variancePCA, minThresholdPCA, maxThresholdPCA, stepThresholdPCA)

    print('Feature Ranking by Mutual Info and PCA on KNN:', 'Best neighbors', best_KneighborsPCA,

          'best PCA threshold', bestTHPCA, 'best N', bestNPCA, 'Best CV F', bestEvalPCA)

    KNNPCA = knnLearner(
        XPCA.iloc[:, :bestNPCA], y, best_KneighborsPCA)

    X_TestPCA = applyPCA(x_test_cleaned_new, pcaObj, pcalist)

    x_TestPCA_cleaned_feature = X_TestPCA.iloc[:, :bestNPCA]

    print(f"Data Training: Nuova lista di attributi con dimensione: '{
        x_new.iloc[:, :bestNPCA].shape}'", "\n")
    print(f"Data Test: Nuova lista di attributi con dimensione: '{
        x_TestPCA_cleaned_feature.shape}'\n")

    y_pred = KNNPCA.predict(x_TestPCA_cleaned_feature)
    target_names = ['class 0 - GoodWare ', 'class 1 - Malware']

    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)

    nome_file = "classification_report(KNN_MIPCA).txt"
    script_pathClassification = script_path.parent.parent / \
        "ClassificationReport" / "KNN"

    percorso_file = os.path.join(script_pathClassification, nome_file)

    # Apre il file in modalità scrittura
    with open(percorso_file, 'w') as file:
        # Scrive il classification report nel file
        file.write(report)

    script_pathFolder = script_path.parent.parent / "ConfusionMatrix" / "KNN"
    ConfusionMatrixBuilder(KNNPCA, y_pred, y_test,
                           script_pathFolder, "KNNMIPCA")

    return KNNPCA
