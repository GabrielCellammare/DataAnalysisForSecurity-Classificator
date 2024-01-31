from functions import *
from pathlib import Path
from sklearn.metrics import classification_report


# Variabile che prende il valore del path in cui si trova il file
script_path = Path(__file__)

# Crea il percorso completo al file utilizzando pathlib
data_dir = script_path.parent / "Data"
pathTrainX = data_dir / "EmberXTrain.csv"
pathTrainY = data_dir / "EmberYTrain.csv"


# Carico nelle variabili X ed Y il dataset e le label
x = loadData(pathTrainX)
y = loadData(pathTrainY)

print("\nShape di train_x:", x.shape)
print("\nShape di train_y:", y.shape)


# Visualizzazione di alcune statistiche riguardo le colonne
preElaborationData(x)


# Rimozione delle colonne inutili. Vengono rimosse le colonne con Min=Max (Dati uguali)
x_cleaned, removed_columns = removeColumnsWithMinMaxEqual(x)

# Stampa dei nomi delle colonne rimosse e della dimensione della lista con le colonne rimanenti
print(f"Nomi delle colonne rimosse: '{removed_columns}'\n")
print(f"Nuova lista di attributi con dimensione: '{x_cleaned.shape}'\n")


# Calcolo quante occorrenze per ogni classe ci sono
labelCount = differentClass(y)
print(labelCount, "\n")
# Stampo un istogramma
plotHistogram(labelCount, y)


boxPlotDir = script_path.parent / "BoxPlot"

overlapDir = script_path.parent / "OverlapDir"
# BoxPlotPreAnalysisData1(x_cleaned, y, boxPlotDir, overlapDir)


""""
# boxPlotDirMutualInfo = script_path.parent / "BoxPlotMutualInfo"
x_mutualinfo = mutualInfoRank(x_cleaned, y)
print(f"X mutual_info: '{x_mutualinfo}'\n")

# BoxPlotAnalysisDataMutualInfo(x_cleaned, y, boxPlotDirMutualInfo, x_mutualinfo)


selectedFeatures = topFeatureSelect(x_mutualinfo, 0.1)
print(len(selectedFeatures))

x_mutualInfo = x_cleaned.loc[:, selectedFeatures]


pca, pcalist, explained_variance = pca(
    x_mutualInfo)  # valutare anche x_cleanead

print(pcalist)
print(len(pcalist))
print(explained_variance)


XPCA = applyPCA(x_mutualInfo, pca, pcalist)
n = NumberOfTopPCSelect(explained_variance, 0.99)
print(n)

# create a dataset with the selected PCs
XPCASelected = XPCA.iloc[:, 1:(n+1)]
print(XPCASelected.shape)
"""

ListXTrain, ListXTest, ListYTrain, ListYTest = stratifiedKFold(
    x_cleaned, y, 5)

print("\n\nListXTrain")
printFolds(ListXTrain)
print("\n\nListYTrain")
printFolds(ListYTrain)
print("\n\nListXTest")
printFolds(ListXTest)
print("\n\nListYTest")
printFolds(ListYTest)


# Laboratorio 5 con aggiunta di mutual info

# Oggetto decistion Tree
"""
clf = decisionTreeLearner(x_cleaned, y, 'entropy')


script_pathTreeFolder = script_path.parent / "TreeFigOutput"
showTree(clf, script_pathTreeFolder)
"""

serialize_dir = script_path.parent / "Serialized" / "MutualInfoTraining.pkl"
# Verifica se il file esiste
if os.path.exists(serialize_dir):
    # Se il file esiste, leggi i parametri
    with open(serialize_dir, "rb") as f:
        serialize_dir = pickle.load(f)
        rank = serialize_dir
else:
    rank = mutualInfoRank(x_cleaned, y)
    MutualInfoTraining = rank
    print(f"X mutual_info: '{rank}'\n")
    # Salva il dizionario in un file usando pickle
    with open(serialize_dir, "wb") as f:
        pickle.dump(MutualInfoTraining, f)

minThreshold = 0
max = 0.0

for key in rank:
    if (key[1] >= max):
        max = key[1]

print(f"Max mutual info = '{max}'")

stepThreshold = 0.05

maxThreshold = max+stepThreshold

bestCriterion, bestTH, bestN, bestEval = determineDecisionTreekFoldConfiguration(
    ListXTrain, ListYTrain, ListXTest, ListYTest, rank, minThreshold, maxThreshold, stepThreshold)


print('Feature Ranking by MI:\n',
      'Best criterion = ', bestCriterion, "\n"
      'best MI threshold = ', bestTH, "\n", 'best N = ', bestN, "\n", 'Best CV F = ', bestEval)

# Prendo le feature migliori (Mutual info)
toplist = topFeatureSelect(rank, bestTH)
DT = decisionTreeLearner(x_cleaned.loc[:, toplist], y, bestCriterion)

# Laboratorio 6

data_dir = script_path.parent / "Data"
pathTestX = data_dir / "EmberXTest.csv"
pathTestY = data_dir / "EmberYTest.csv"

x_test = loadData(pathTestX)
y_test = loadData(pathTestY)

print("\nShape di train_x:", x_test.shape)
print("\nShape di train_y:", y_test.shape)


# Utilizzo il tree migliore

# Rimozione delle colonne inutili. Vengono rimosse le colonne con Min=Max (Dati uguali)
x_test_cleaned = removeColumnsWithMinMaxEqualTest(x_test, removed_columns)

# Stampa dei nomi delle colonne rimosse e della dimensione della lista con le colonne rimanenti

print(f"Data Training: Nuova lista di attributi con dimensione: '{
      x_cleaned.shape}'\n")
print(f"Data Test: Nuova lista di attributi con dimensione: '{
      x_test_cleaned.shape}'\n")

x_test_cleaned_feature = x_test.loc[:, toplist]
print(f"Data Training: Nuova lista di attributi con dimensione: '{
      x_cleaned.loc[:, toplist].shape}'\n")
print(f"Data Test: Nuova lista di attributi con dimensione: '{
      x_test_cleaned_feature.shape}'\n")

y_pred = DT.predict(x_test_cleaned_feature)
target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))


ConfusionMatrixBuilder(DT, y_pred, y_test)


# adopt the stratified CV to determine the best decision tree configuration on the pcs
minThreshold = 0.95
stepThreshold = 0.01
maxThreshold = 1.01
folds = 5
ListXTrainPCA, ListXTestPCA, ListYTrainPCA, ListYTestPCA = stratifiedKFold(
    x_cleaned, y, folds)

bestCriterionPCA, bestTHPCA, bestNPCA,
bestEvalPCA = determineDecisionTreekFoldConfigurationPCA(
    ListXTrainPCA, ListYTrainPCA, ListXTestPCA, ListYTestPCA, explained_variance, minThreshold, maxThreshold, stepThreshold)
print('Feature Ranking by MI:', 'Best criterion', bestCriterion,
      'best MI threshold', bestTH, 'best N', bestN, 'Best CV F', bestEval)
DTPCA = decisionTreeLearner(
    x_cleaned.iloc[:, 1:(bestNPCA + 1)], Y, bestCriterionPCA)


"""
# Controllo i valori mancanti verificando se il count è <1200 e li salvo in una lista
columnsWrongCount = missedValue(x)
std_values, mean_values, min_values, max_values = attributeListForPlot(x)
# Crea quattro grafici separati per gli attributi
plotVariablesValues(std_values, mean_values, min_values, max_values)
"""
