from UtilsFunctions import *
from sklearn.metrics import classification_report
from UtilsDecisionTree import determineDecisionTreekFoldConfigurationPCA, decisionTreeLearner, showTree

# adopt the stratified CV to determine the best decision tree configuration on the pcs


def DecisionTreePCA(x, y, script_path, x_test_cleaned, y_test):
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

    bestCriterionPCA, bestTHPCA, bestNPCA, bestEvalPCA = determineDecisionTreekFoldConfigurationPCA(
        ListXTrainPCA, ListYTrainPCA, ListXTestPCA, ListYTestPCA, explained_variancePCA, minThresholdPCA, maxThresholdPCA, stepThresholdPCA)

    print('Feature Ranking by MI:', 'Best criterion', bestCriterionPCA,

          'best MI threshold', bestTHPCA, 'best N', bestNPCA, 'Best CV F', bestEvalPCA)

    DTPCA = decisionTreeLearner(
        XPCA.iloc[:,
                  :bestNPCA], y, bestCriterionPCA)
    script_pathTreeFolder = script_path.parent.parent / "TreeFigOutput"
    showTree(DTPCA, script_pathTreeFolder, "DecisionTreePCA")

    X_TestPCA = applyPCA(x_test_cleaned, pcaObj, pcalist)

    x_TestPCA_cleaned_feature = X_TestPCA.iloc[:,
                                               :bestNPCA]

    print(f"Data Training: Nuova lista di attributi con dimensione: '{
        x.iloc[:, :bestNPCA].shape}'\n")
    print(f"Data Test: Nuova lista di attributi con dimensione: '{
        x_TestPCA_cleaned_feature.shape}'\n")

    y_pred = DTPCA.predict(x_TestPCA_cleaned_feature)
    target_names = ['class 0', 'class 1']
    print(classification_report(y_test, y_pred, target_names=target_names))

    script_pathFolder = script_path.parent.parent / "ConfusionMatrix"
    ConfusionMatrixBuilder(DTPCA, y_pred, y_test,
                           script_pathFolder, "DecisionTreePCA")

    return DTPCA
