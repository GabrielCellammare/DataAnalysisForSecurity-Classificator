from UtilsFunctions import *
from sklearn.metrics import classification_report
from UtilsEnsamble import EnsambleLearner, determineEnsamblekFoldConfigurationPCA

# adopt the stratified CV to determine the best decision tree configuration on the pcs


def EnsemblePCA(x, y, script_path, x_test_cleaned, y_test, clf1, clf2, clf3):
    minThresholdPCA = 0.95
    stepThresholdPCA = 0.01
    maxThresholdPCA = 1.01
    folds = 5

    pcaObj, pcalist, explained_variancePCA = pca(
        x)  # valutare anche x_mutual info
    print("\nExplained variance: ", explained_variancePCA, "\n")
    XPCA = applyPCA(x, pcaObj, pcalist)  # Applicazione sull'intero dataset

    ListXTrainPCA, ListXTestPCA, ListYTrainPCA, ListYTestPCA = stratifiedKFold(
        XPCA, y, folds)

    bestTHPCA, bestNPCA, bestEvalPCA = determineEnsamblekFoldConfigurationPCA(
        ListXTrainPCA, ListYTrainPCA, ListXTestPCA, ListYTestPCA, explained_variancePCA, minThresholdPCA, maxThresholdPCA, stepThresholdPCA, clf1, clf2, clf3)

    print('Feature Ranking by MI:',

          'best MI threshold', bestTHPCA, 'best N', bestNPCA, 'Best CV F', bestEvalPCA)

    ECLFPCA = EnsambleLearner(
        XPCA.iloc[:, :bestNPCA], y, clf1, clf2, clf3)

    X_TestPCA = applyPCA(x_test_cleaned, pcaObj, pcalist)

    x_TestPCA_cleaned_feature = X_TestPCA.iloc[:, :bestNPCA]

    print(f"Data Training: Nuova lista di attributi con dimensione: '{
        x.iloc[:, :bestNPCA].shape}'\n")
    print(f"Data Test: Nuova lista di attributi con dimensione: '{
        x_TestPCA_cleaned_feature.shape}'\n")

    y_pred = ECLFPCA.predict(x_TestPCA_cleaned_feature)

    target_names = ['class 0 - GoodWare ', 'class 1 - Malware']

    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)

    nome_file = "classification_report(Ensamble_PCA).txt"
    script_pathClassification = script_path.parent.parent / \
        "ClassificationReport" / "Ensamble"

    percorso_file = os.path.join(script_pathClassification, nome_file)

    # Apre il file in modalità scrittura
    with open(percorso_file, 'w') as file:
        # Scrive il classification report nel file
        file.write(report)

    script_pathFolder = script_path.parent.parent / "ConfusionMatrix" / "Ensamble"
    ConfusionMatrixBuilder(ECLFPCA, y_pred, y_test,
                           script_pathFolder, "EnsamblePCA")

    return ECLFPCA
