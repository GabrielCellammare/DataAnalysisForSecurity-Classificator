from UtilsFunctions import loadData, removeColumnsWithMinMaxEqual, removeColumnsWithMinMaxEqualTest
from mainDescribeData import DescribeData
from mainDecisionTreeMutualInfo import DecisionTreeMutualInfo
from mainDecisionTreePCA import DecisionTreePCA
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


# DescribeData(x_cleaned, y, script_path)
# DecisionTreeMutualInfo(x_cleaned, y, script_path,
# removed_columnsMaxMin)
# DecisionTreePCA(x_cleaned, y, script_path, removed_columnsMaxMin)
# RandomForestMutualInfo(x_cleaned, y, script_path, removed_columnsMaxMin)
# RandomForestPCA(x_cleaned, y, script_path, removed_columnsMaxMin)
KNNMutualInfo(x_cleaned, y, script_path, removed_columnsMaxMin)
KNNPCA(x_cleaned, y, script_path, removed_columnsMaxMin)
