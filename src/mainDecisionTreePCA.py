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
        x)
    # print("\nExplained variance: ", explained_variancePCA, "\n")
    # Applicazione sull'intero dataset di training
    XPCA = applyPCA(x, pcaObj, pcalist)

    ListXTrainPCA, ListXTestPCA, ListYTrainPCA, ListYTestPCA = stratifiedKFold(
        XPCA, y, folds)

    bestCriterionPCA, bestTHPCA, bestNPCA, bestEvalPCA = determineDecisionTreekFoldConfigurationPCA(
        ListXTrainPCA, ListYTrainPCA, ListXTestPCA, ListYTestPCA, explained_variancePCA, minThresholdPCA, maxThresholdPCA, stepThresholdPCA)

    print('Feature Ranking by PCA:', "\n", 'Best criterion = ', bestCriterionPCA, "\n",

          'best PCA threshold = ', bestTHPCA, "\n", 'best N = ', bestNPCA, "\n", 'Best CV F = ', bestEvalPCA)

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
    target_names = ['class 0 - GoodWare ', 'class 1 - Malware']

    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)

    nome_file = "classification_report(DecisionTree_PCA).txt"
    script_pathClassification = script_path.parent.parent / \
        "ClassificationReport" / "DecisionTree"

    percorso_file = os.path.join(script_pathClassification, nome_file)

    # Apre il file in modalità scrittura
    with open(percorso_file, 'w') as file:
        # Scrive il classification report nel file
        file.write(report)

    script_pathFolder = script_path.parent.parent / "ConfusionMatrix" / "DecisionTree"
    ConfusionMatrixBuilder(DTPCA, y_pred, y_test,
                           script_pathFolder, "DecisionTreePCA")

    return DTPCA
