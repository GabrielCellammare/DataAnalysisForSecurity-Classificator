from UtilsFunctions import *
from sklearn.metrics import classification_report
from UtilsRandomForest import determineRFkFoldConfigurationPCA, randomForestLearner

# adopt the stratified CV to determine the best decision tree configuration on the pcs


def RandomForestPCA(x, y, script_path, x_test_cleaned, y_test):
    minThresholdPCA = 0.95
    stepThresholdPCA = 0.01
    maxThresholdPCA = 1.01
    folds = 5

    pcaObj, pcalist, explained_variancePCA = pca(
        x)
    print("\nExplained variance: ", explained_variancePCA, "\n")
    XPCA = applyPCA(x, pcaObj, pcalist)  # Applicazione sull'intero dataset

    ListXTrainPCA, ListXTestPCA, ListYTrainPCA, ListYTestPCA = stratifiedKFold(
        XPCA, y, folds)

    best_criterionPCA, best_THPCA, bestNPCA, best_fscorePCA, best_n_treePCA, best_randPCA, best_bootstrap_sPCA = determineRFkFoldConfigurationPCA(
        ListXTrainPCA, ListYTrainPCA, ListXTestPCA, ListYTestPCA, explained_variancePCA, minThresholdPCA, maxThresholdPCA, stepThresholdPCA)

    print("\nbest_criterionPCA: ", best_criterionPCA, "\nbest_THPCA:", best_THPCA,
          "\nbestNPCA: ", bestNPCA, "\nbest_fscorePCA: ", best_fscorePCA, "\nbest_n_treePCA: ", best_n_treePCA,
          "\nbest_randPCA: ", best_randPCA, "\nbest_bootstrap_sPCA: ", best_bootstrap_sPCA)

    RFPCA = randomForestLearner(
        XPCA.iloc[:, :bestNPCA], y, best_n_treePCA, best_criterionPCA, best_randPCA, best_bootstrap_sPCA)

    X_TestPCA = applyPCA(x_test_cleaned, pcaObj, pcalist)

    x_TestPCA_cleaned_feature = X_TestPCA.iloc[:, :bestNPCA]

    print(f"Data Training: Nuova lista di attributi con dimensione: '{
        x.iloc[:, :bestNPCA].shape}'\n")
    print(f"Data Test: Nuova lista di attributi con dimensione: '{
        x_TestPCA_cleaned_feature.shape}'\n")

    y_pred = RFPCA.predict(x_TestPCA_cleaned_feature)
    target_names = ['class 0 - GoodWare ', 'class 1 - Malware']

    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)

    nome_file = "classification_report(RandomForest_PCA).txt"
    script_pathClassification = script_path.parent.parent / \
        "ClassificationReport" / "RandomForest"

    percorso_file = os.path.join(script_pathClassification, nome_file)

    # Apre il file in modalità scrittura
    with open(percorso_file, 'w') as file:
        # Scrive il classification report nel file
        file.write(report)

    script_pathFolder = script_path.parent.parent / "ConfusionMatrix" / "RandomForest"
    ConfusionMatrixBuilder(RFPCA, y_pred, y_test,
                           script_pathFolder, "RandomForestPCA")

    return RFPCA
