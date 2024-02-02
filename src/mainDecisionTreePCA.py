from UtilsFunctions import *
from sklearn.metrics import classification_report

# adopt the stratified CV to determine the best decision tree configuration on the pcs


def DecisionTreePCA(x, y, script_path, removed_columns):
    minThresholdPCA = 0.95
    stepThresholdPCA = 0.01
    maxThresholdPCA = 1.01
    folds = 5

    pcaObj, pcalist, explained_variancePCA = pca(
        x)  # valutare anche x_mutual info
    print("\nExplained variance: ", explained_variancePCA, "\n")
    XPCA = applyPCA(x, pcaObj, pcalist)

    ListXTrainPCA, ListXTestPCA, ListYTrainPCA, ListYTestPCA = stratifiedKFold(
        XPCA, y, folds)

    bestCriterionPCA, bestTHPCA, bestNPCA, bestEvalPCA = determineDecisionTreekFoldConfigurationPCA(
        ListXTrainPCA, ListYTrainPCA, ListXTestPCA, ListYTestPCA, explained_variancePCA, minThresholdPCA, maxThresholdPCA, stepThresholdPCA)

    print('Feature Ranking by MI:', 'Best criterion', bestCriterionPCA,

          'best MI threshold', bestTHPCA, 'best N', bestNPCA, 'Best CV F', bestEvalPCA)

    DTPCA = decisionTreeLearner(
        XPCA.iloc[:, :bestNPCA], y, bestCriterionPCA)
    script_pathTreeFolder = script_path.parent.parent / "TreeFigOutput"
    showTree(DTPCA, script_pathTreeFolder, "PCA")

    data_dir = script_path.parent.parent / "Data"
    pathTestX = data_dir / "EmberXTest.csv"
    pathTestY = data_dir / "EmberYTest.csv"

    x_test = loadData(pathTestX)
    y_test = loadData(pathTestY)

    print("\nShape di train_x:", x_test.shape)
    print("\nShape di train_y:", y_test.shape)

    # Utilizzo il tree migliore

    # Rimozione delle colonne inutili. Vengono rimosse le colonne con Min=Max (Dati uguali con x)
    x_test_cleaned = removeColumnsWithMinMaxEqualTest(x_test, removed_columns)

    # Stampa dei nomi delle colonne rimosse e della dimensione della lista con le colonne rimanenti

    print(f"Data Training: Nuova lista di attributi con dimensione: '{
        x.shape}'\n")
    print(f"Data Test: Nuova lista di attributi con dimensione: '{
        x_test_cleaned.shape}'\n")

    X_TestPCA = applyPCA(x_test_cleaned, pcaObj, pcalist)
    x_TestPCA_cleaned_feature = X_TestPCA.iloc[:, :bestNPCA]

    print(f"Data Training: Nuova lista di attributi con dimensione: '{
        x.iloc[:, :bestNPCA].shape}'\n")
    print(f"Data Test: Nuova lista di attributi con dimensione: '{
        x_TestPCA_cleaned_feature.shape}'\n")

    y_pred = DTPCA.predict(x_TestPCA_cleaned_feature)
    target_names = ['class 0', 'class 1']
    print(classification_report(y_test, y_pred, target_names=target_names))

    ConfusionMatrixBuilder(DTPCA, y_pred, y_test)
