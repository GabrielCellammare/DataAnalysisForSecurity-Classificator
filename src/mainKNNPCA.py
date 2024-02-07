from UtilsFunctions import *
from sklearn.metrics import classification_report
from UtilsKNN import determineDecisionTreekFoldConfigurationPCA, knnLearner

# adopt the stratified CV to determine the best decision tree configuration on the pcs


def KNNPCA(x, y, script_path, x_test_cleaned, y_test):
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

    bestTHPCA, bestNPCA, bestEvalPCA, best_KneighborsPCA = determineDecisionTreekFoldConfigurationPCA(
        ListXTrainPCA, ListYTrainPCA, ListXTestPCA, ListYTestPCA, explained_variancePCA, minThresholdPCA, maxThresholdPCA, stepThresholdPCA)

    print('Feature Ranking by MI:', 'Best neighbors', best_KneighborsPCA,

          'best MI threshold', bestTHPCA, 'best N', bestNPCA, 'Best CV F', bestEvalPCA)

    KNNPCA = knnLearner(
        XPCA.iloc[:, 1:(
            bestNPCA+1)], y, best_KneighborsPCA)

    X_TestPCA = applyPCA(x_test_cleaned, pcaObj, pcalist)

    x_TestPCA_cleaned_feature = X_TestPCA.iloc[:, 1:(
        bestNPCA+1)]

    print(f"Data Training: Nuova lista di attributi con dimensione: '{
        x.iloc[:, :bestNPCA].shape}'\n")
    print(f"Data Test: Nuova lista di attributi con dimensione: '{
        x_TestPCA_cleaned_feature.shape}'\n")

    y_pred = KNNPCA.predict(x_TestPCA_cleaned_feature)
    target_names = ['class 0', 'class 1']
    print(classification_report(y_test, y_pred, target_names=target_names))

    script_pathFolder = script_path.parent.parent / "ConfusionMatrix"
    ConfusionMatrixBuilder(KNNPCA, y_pred, y_test,
                           script_pathFolder, "KNNPCA")

    return KNNPCA
