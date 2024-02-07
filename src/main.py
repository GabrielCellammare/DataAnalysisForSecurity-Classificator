from UtilsFunctions import loadData, removeColumnsWithMinMaxEqual, removeColumnsWithMinMaxEqualTest
from mainDescribeData import DescribeData
from mainDecisionTreeMutualInfo import DecisionTreeMutualInfo
from mainDecisionTreePCA import DecisionTreePCA
from mainEnsembleMutualInfo import EnsembleMutualInfo
from mainEnsemblePCA import EnsemblePCA
from mainRandomForestMutualInfo import RandomForestMutualInfo
from mainRandomForestPCA import RandomForestPCA
from mainKNNMutualInfo import KNNMutualInfo
from mainKNNPCA import KNNPCA
from pathlib import Path


# Variabile che prende il valore del path in cui si trova il file
script_path = Path(__file__)

# Crea il percorso completo al file utilizzando pathlib
data_dir = script_path.parent.parent / "Data"

pathTrainX = data_dir / "EmberXTrain.csv"
pathTrainY = data_dir / "EmberYTrain.csv"


# Carico nelle variabili X ed Y il dataset e le label
x = loadData(pathTrainX)
y = loadData(pathTrainY)

# Rimozione delle colonne inutili. Vengono rimosse le colonne con Min=Max (Dati uguali)
# Sarà necessario rimuovere le stesse nel test

x_cleaned, removed_columnsMaxMin = removeColumnsWithMinMaxEqual(x)

# Stampa dei nomi delle colonne rimosse e della dimensione della lista con le colonne rimanenti
print(f"Nomi delle colonne rimosse: '{removed_columnsMaxMin}'\n")
print(f"Nuova lista di attributi con dimensione: '{x_cleaned.shape}'\n")

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
    x_test, removed_columnsMaxMin)

"""
# DescribeData(x_cleaned, y, script_path)
clf1DecisionTreeMutualInfo = DecisionTreeMutualInfo(
    x_cleaned, y, script_path, x_test_cleaned, y_test)

clf3KNNMutualInfo = KNNMutualInfo(
    x_cleaned, y, script_path, x_test_cleaned, y_test)

clf2RandomForestMutualInfo = RandomForestMutualInfo(
    x_cleaned, y, script_path, x_test_cleaned, y_test)
eclfMutualInfo = EnsembleMutualInfo(x_cleaned, y, script_path, x_test_cleaned, y_test,
                                    clf1DecisionTreeMutualInfo, clf2RandomForestMutualInfo, clf3KNNMutualInfo)
"""

clf1DecisionTreePCA = DecisionTreePCA(
    x_cleaned, y, script_path, x_test_cleaned, y_test)


clf2RandomForestPCA = RandomForestPCA(
    x_cleaned, y, script_path, x_test_cleaned, y_test)


clf3KNNPCA = KNNPCA(x_cleaned, y, script_path, x_test_cleaned, y_test)


eclfPCA = EnsemblePCA(x_cleaned, y, script_path, x_test_cleaned, y_test,
                      clf1DecisionTreePCA, clf2RandomForestPCA, clf3KNNPCA)
