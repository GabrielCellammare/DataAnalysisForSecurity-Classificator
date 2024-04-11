from UtilsEnsamble import EnsambleLearner, determineEnsamblekFoldConfigurationMutualInfo
from UtilsFunctions import *
from sklearn.metrics import classification_report
import pickle


def EnsembleMutualInfo(x, y, script_path, x_test_cleaned, y_test, clf1, clf2, clf3):
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

    minThreshold = 0
    maxMutualInfo = 0.0
    for key in rank:
        if (key[1] >= maxMutualInfo):
            maxMutualInfo = key[1]
    # print(f"Max mutual info = '{maxMutualInfo}'")
    stepThreshold = 0.02
    maxThreshold = maxMutualInfo+stepThreshold

    folds = 5
    ListXTrain, ListXTest, ListYTrain, ListYTest = stratifiedKFold(
        x, y, folds)

    """
    print("\n\nListXTrain")
    printFolds(ListXTrain)
    print("\n\nListYTrain")
    printFolds(ListYTrain)
    print("\n\nListXTest")
    printFolds(ListXTest)
    print("\n\nListYTest")
    printFolds(ListYTest)
    """

    bestTH, bestN, bestEval = determineEnsamblekFoldConfigurationMutualInfo(
        ListXTrain, ListYTrain, ListXTest, ListYTest, rank, minThreshold, maxThreshold, stepThreshold, clf1, clf2, clf3)

    print('Feature Ranking by MI on Ensemble:\n',
          'best MI threshold = ', bestTH, "\n", 'best N = ', bestN, "\n", 'Best CV F = ', bestEval)

    # Prendo le feature migliori (Mutual info)
    toplist = topFeatureSelect(rank, bestTH)
    ECLF = EnsambleLearner(x.loc[:, toplist], y, clf1, clf2, clf3)

    x_test_cleaned_feature = x_test_cleaned.loc[:, toplist]

    print(f"Data Training: Nuova lista di attributi con dimensione: '{
        x.loc[:, toplist].shape}'\n")
    print(f"Data Test: Nuova lista di attributi con dimensione: '{
        x_test_cleaned_feature.shape}'\n")

    y_pred = ECLF.predict(x_test_cleaned_feature)
    target_names = ['class 0 - GoodWare ', 'class 1 - Malware']

    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)

    nome_file = "classification_report(EnsembleMutualInfo).txt"
    script_pathClassification = script_path.parent.parent / \
        "ClassificationReport" / "Ensamble"

    percorso_file = os.path.join(script_pathClassification, nome_file)

    # Apre il file in modalità scrittura
    with open(percorso_file, 'w') as file:
        # Scrive il classification report nel file
        file.write(report)

    script_pathFolder = script_path.parent.parent / "ConfusionMatrix" / "Ensamble"
    ConfusionMatrixBuilder(
        ECLF, y_pred, y_test, script_pathFolder, "EnsambleMutualInfo")

    return ECLF
