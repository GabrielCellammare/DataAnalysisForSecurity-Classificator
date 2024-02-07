from UtilsFunctions import *
from sklearn.metrics import classification_report
from UtilsKNN import knnLearner, determineKNNkFoldConfigurationMutualInfo
import pickle


def KNNMutualInfo(x, y, script_path, removed_columnsMinMax):
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
            x, y, boxPlotDirMutualInfo, rank)

    minThreshold = 0
    maxMutualInfo = 0.0

    for key in rank:
        if (key[1] >= maxMutualInfo):
            maxMutualInfo = key[1]

    print(f"Max mutual info = '{maxMutualInfo}'")

    stepThreshold = 0.05

    maxThreshold = maxMutualInfo+stepThreshold
    folds = 5
    ListXTrain, ListXTest, ListYTrain, ListYTest = stratifiedKFold(
        x, y, folds)

    print("\n\nListXTrain")
    printFolds(ListXTrain)
    print("\n\nListYTrain")
    printFolds(ListYTrain)
    print("\n\nListXTest")
    printFolds(ListXTest)
    print("\n\nListYTest")
    printFolds(ListYTest)

    best_TH, bestN, best_fscore, best_Kneighbors = determineKNNkFoldConfigurationMutualInfo(
        ListXTrain, ListYTrain, ListXTest, ListYTest, rank, minThreshold, maxThreshold, stepThreshold)

    print('Feature Ranking by MI:\n',
          'Best Neighbours= ', best_Kneighbors, "\n"
          'best MI threshold = ', best_TH, "\n", 'best N = ', bestN, "\n", 'Best CV F = ', best_fscore)

    # Prendo le feature migliori (Mutual info)
    toplist = topFeatureSelect(rank, best_TH)
    KNN = knnLearner(x.loc[:, toplist], y, best_Kneighbors)

    # Laboratorio 6

    data_dir = script_path.parent.parent / "Data"
    pathTestX = data_dir / "EmberXTest.csv"
    pathTestY = data_dir / "EmberYTest.csv"

    x_test = loadData(pathTestX)
    y_test = loadData(pathTestY)

    print("\nShape di train_x:", x_test.shape)
    print("\nShape di train_y:", y_test.shape)

    # Utilizzo il tree migliore

    # Rimozione delle colonne inutili. Vengono rimosse le colonne con Min=Max (Dati uguali con x)
    x_test_cleaned = removeColumnsWithMinMaxEqualTest(
        x_test, removed_columnsMinMax)
    # Stampa dei nomi delle colonne rimosse e della dimensione della lista con le colonne rimanenti

    print(f"Data Training: Nuova lista di attributi con dimensione: '{
        x.shape}'\n")
    print(f"Data Test: Nuova lista di attributi con dimensione: '{
        x_test_cleaned.shape}'\n")

    x_test_cleaned_feature = x_test_cleaned.loc[:, toplist]

    print(f"Data Training: Nuova lista di attributi con dimensione: '{
        x.loc[:, toplist].shape}'\n")
    print(f"Data Test: Nuova lista di attributi con dimensione: '{
        x_test_cleaned_feature.shape}'\n")

    y_pred = KNN.predict(x_test_cleaned_feature)
    target_names = ['class 0', 'class 1']
    print(classification_report(y_test, y_pred, target_names=target_names))

    script_pathFolder = script_path.parent.parent / "ConfusionMatrix"
    ConfusionMatrixBuilder(
        KNN, y_pred, y_test, script_pathFolder, "KNNMutualInfo")
