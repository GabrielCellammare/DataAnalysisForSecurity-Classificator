import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


"""Funzione che legge il file csv e restituisce un dataframe"""

# Seed per evitare la randomicità
seed = 42
np.random.seed(seed)
goodMalware = "'Goodware - Malware'"


def loadData(pathTrain):
    return pd.read_csv(pathTrain)


"""
Funzione che itera attraverso le colonne del DataFrame "x" e per ogni colonna estrae il contenuto
assegnandolo alla variabile col_data. Con il metodo col_data.describe() vengono poi calcolate diverse
statistiche come conteggio, media, deviazione standard, minimo, massimo e quartili.
Viene restituito un array
"""


def preElaborationData(x):
    for col in x.columns:
        col_data = x[col]
        col_description = col_data.describe()
        print(f"Statistiche della colonna '{
            col}': \n{col_description}")
        print("\n")


"""
Funzione che controlla i valori max e min. Se questi sono uguali aggiunge la colonna in una nuova lista e quando
finisce di cliclare tutte le colonne, rimuove quelle presenti nella lista perchè inutili.
"""


def removeColumnsWithMinMaxEqual(x):
    print("\nRemoving columns with min=max...\n")
    columns_to_remove = []  # Lista per definire i nomi delle colonne rimosse

    for col in x.columns:
        col_data = x[col]
        col_description = col_data.describe()

        if col_description["min"] == col_description["max"]:
            columns_to_remove.append(col)

    x_cleaned = x.drop(columns=columns_to_remove)  # Rimuovi le colonne inutili
    # x drop accetta come parametro columns = ['Colonna A, B, C']
    print("\nCompleted!\n")
    return x_cleaned, columns_to_remove


"""
Funzione che prende in input un set di colonne da rimuovere e rimuove le suddette dal dataset
"""


def removeColumnsWithMinMaxEqualTest(x, columnsremoving):
    print("\nRemoving columns with min=max in Test...\n")
    x_cleaned = x.drop(columns=columnsremoving)  # Rimuovi le colonne inutili
    # x drop accetta come parametro columns = ['Colonna A, B, C']
    print("\nCompleted!\n")
    return x_cleaned


# Ritorna una series
"""
s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])
Series is a one-dimensional labeled array capable of holding any data type (integers, strings, floating point numbers, Python objects, etc.)
    s
    Out[4]: 
    a    0.469112
    b   -0.282863
    c   -1.509059
    d   -1.135632
    e    1.212112
    dtype: float64_
"""


"""
Funziona che conta le occorrenze per ogni classe, restituisce un oggetto series
"""


def differentClass(y):
    conteggio_valori = y['Label'].value_counts()
    return conteggio_valori


"""
Funzione che genera l'istogramma riguardo le occorrenze delle classi
"""


def plotHistogram(labelCount, y):
    percentuale = (labelCount / (len(y) - 1)) * 100
    # Percentuale = series

    plt.bar(percentuale.index, percentuale)
    plt.ylabel('Percentuale (%)')
    plt.title('Distribuzione dei valori nella classi')

    # Aggiungi le etichette delle percentuali sopra le barre
    for index, value in enumerate(percentuale):
        plt.text(index, value, f'{value: .2f}%', ha='center', va='bottom')

    # Rimuovi le etichette sull'asse X
    plt.xticks([])
    plt.show()


"""
Funzione che salva il box plot di ogni variabile indipendente rispetto alle classi
"""


def BoxPlotPreAnalysisData(x, y, boxPlotDir):
    print("\nSaving Box Plot in 'BoxPlot' Folder...\n")
    # Ottieni la lista dei file nella cartella

    x['Label'] = y['Label']

    for col in x.columns:
        if (col != 'Label'):
            x.boxplot(column=col, by='Label')
            # Aggiungi titoli e label per gli assi, se necessario
            plt.title(f'Boxplot della colonna {col}')
            plt.xlabel(goodMalware)
            plt.ylabel('Dati')
            file_name = os.path.join(
                boxPlotDir, f'boxplot_{col}.png')
            plt.savefig(file_name)
            plt.close()

    print("\nCompleted\n")


"""
Funzione che calcola il mutal info e restituisce un dict contenente in sorted_x[0] il nome della colonna, e in sorted_x[1] il valore
del mutual info
"""


def mutualInfoRank(X, Y):
    print("Computing mutual info ranking...")
    # Restituisce i nomi delle colonne
    independentList = list(X.columns.values)
    res = dict(zip
               (independentList,
                   mutual_info_classif
                   (X,  np.ravel(Y), discrete_features=False, random_state=seed)))
    # La funzione zip accoppia ciascun nome di colonna con il corrispondente valore di informazione mutua, creando una serie di tuple.

    sorted_x = sorted(res.items(), key=lambda kv: kv[1], reverse=True)

    print("Computing mutual info ranking...completed")
    # ritorna un dizionario
    return sorted_x


"""
Stampa i box blot delle variabili indipendenti con mutual info elevato e non
nprint=numero di stampe
n=10 stamperà le prime 10 variabili significative e le ultime 10, escludendo quelle con mutual info=0
"""


def BoxPlotAnalysisDataMutualInfo(x, y, boxPlotDir, mutualInfo, n_print=10):
    # Ottieni la lista dei file nella cartella
    print("\nSaving Mutual info variables Box Plot in 'BoxPlotMutualInfo' Folder...\n")

    boxPlotDirMutualInfoFirst = boxPlotDir / "MoreSignificant"
    boxPlotDirMutualInfoLast = boxPlotDir / "LessSignificant"

    if (not os.listdir(boxPlotDirMutualInfoFirst)):
        i = 0
        if (mutualInfo):
            x['Label'] = y['Label']
            for tupla in mutualInfo:
                for col in x.columns:
                    if (tupla[0] == col and tupla[1] != 0):
                        if (i < n_print):
                            x.boxplot(column=col, by='Label')
                            # Aggiungi titoli e label per gli assi, se necessario
                            plt.title(f'Boxplot della colonna {col}')
                            plt.xlabel(goodMalware)
                            plt.ylabel('Dati')
                            file_name = os.path.join(
                                boxPlotDirMutualInfoFirst, f'boxplot_{col}.png')
                            plt.savefig(file_name)
                            plt.close()
                            i += 1
        i = 0
        if (mutualInfo):
            x['Label'] = y['Label']
            for tupla in reversed(mutualInfo):
                for col in x.columns:
                    if (tupla[0] == col and tupla[1] != 0 and i < n_print):
                        x.boxplot(column=col, by='Label')
                        # Aggiungi titoli e label per gli assi, se necessario
                        plt.title(f'Boxplot della colonna {col}')
                        plt.xlabel(goodMalware)
                        plt.ylabel('Dati')
                        file_name = os.path.join(
                            boxPlotDirMutualInfoLast, f'boxplot_{col}.png')
                        plt.savefig(file_name)
                        plt.close()
                        i += 1
        print("\nCompleted!\n")


"""
Funzione che seleziona da un dict (prodotto da mutual info) le feature con un mutual info >= al threesold
Restituisce la liste delle feature
"""


def topFeatureSelect(rank, threesold):
    selectedFeatures = []
    for tupla in rank:
        if (tupla[1] >= threesold):
            selectedFeatures.append(tupla[0])

    return selectedFeatures


"""
Funzione che si occupa di istanziare un oggetto PCA per la selezione delle componenti principali 
fit=calcola autovalori e autovettori, parametri utilizzati per trasformare i dati per il transform
"""


def pca(X):
    print("\nTraining PCA...\n")
    pca = PCA(n_components=len(X.columns))
    pca.fit(X)
    # Nome delle nuove feature del tipo pca0...pcan
    feature_names = pca.get_feature_names_out()
    print("\nCompleted!\n")
    return pca, feature_names, pca.explained_variance_ratio_


"""
Applicazione del PCA tramite un oggetto pca 
"""


def applyPCA(X, pca, pcalist):
    print("\nApplying PCA...\n")
    # Trasforma il DataFrame utilizzando PCA
    transformed = pca.transform(X)
    # Crea un nuovo DataFrame con le componenti principali
    df_pca = pd.DataFrame(transformed, columns=pcalist)
    print("\nCompleted!\n")
    return df_pca


"Selezione di Alcune delle PC la cui somma della varianza superi un determinato threesold"


def NumberOfTopPCSelect(explained_variance, thresold):
    cumulative_variance = 0.0
    num_components = 0

    for variance in explained_variance:
        cumulative_variance += variance
        num_components += 1

        if cumulative_variance > thresold:
            break

    return num_components


"""
Funzione che crea in base al numero di fold, un numero di training set e testing set definito per la Cross Validation

"""


def stratifiedKFold(X, Y, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    ListXTrain = []
    ListXTest = []
    ListYTrain = []
    ListYTest = []
    for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
        print(train_index)
        print("Test: ", test_index, "\n")
        ListXTrain.append(pd.DataFrame(X, index=train_index))
        ListYTrain.append(pd.DataFrame(Y, index=train_index))
        ListXTest.append(pd.DataFrame(X, index=test_index))
        ListYTest.append(pd.DataFrame(Y, index=test_index))

    for i in range(len(ListXTest)-1):
        indici_lista_i = list(ListXTest[i].index)
        indici_lista_i2 = list(ListXTest[i+1].index)
        set1 = set(indici_lista_i)
        set2 = set(indici_lista_i2)
        intersezione = set1 & set2
        """
        if intersezione:
            print("Le liste hanno un'intersezione:", intersezione)
        else:
            print("Le liste non hanno un'intersezione.")
        """

    return ListXTrain, ListXTest, ListYTrain, ListYTest


"""
Funzione che stamperà in maniera predefinita tutti i fold
"""


def printFolds(list):
    for i in range(len(list)):
        print(f"Fold {i} " + str(list[i].shape) + "\n")


"""
Funzione che stampa la matrice di confusione e la salva con uno specifico nome in una specifica cartella
"""


def ConfusionMatrixBuilder(clf, y_pred, y_test, script_pathFolder, typeC):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=clf.classes_)
    disp.plot()
    file_name = os.path.join(script_pathFolder,
                             f'ConfusionMatrixOutput{typeC}.pdf')
    plt.savefig(file_name,
                format='pdf', dpi=1000)
    plt.show()
