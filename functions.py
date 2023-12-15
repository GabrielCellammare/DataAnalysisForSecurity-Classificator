import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import numpy as np
from sklearn.metrics import f1_score


"""Funzione che legge il file csv e restituisce un dataframe"""

seed = 42
np.random.seed(seed)
goodMalware = "'Goodware - Malware'"


def loadData(pathTrain):
    return pd.read_csv(pathTrain)


"""
Funzione che itera attraverso le colonne del DataFrame "x" e per ogni colonna estrae il conenuto
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
    columns_to_remove = []  # Lista per definire i nomi delle colonne rimosse

    for col in x.columns:
        col_data = x[col]
        col_description = col_data.describe()

        if col_description["min"] == col_description["max"]:
            columns_to_remove.append(col)

    x_cleaned = x.drop(columns=columns_to_remove)  # Rimuovi le colonne inutili
    # x drop accetta come parametro columns = ['Colonna A, B, C']

    return x_cleaned, columns_to_remove


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
    print(type(conteggio_valori))
    return conteggio_valori


"""
Funzione che genera l'istogramma riguardo le occorrenze delle classi
"""


def plotHistogram(labelCount, y):
    percentuale = (labelCount / (len(y) - 1)) * 100
    # Percentuale = series

    plt.bar(percentuale.index, percentuale)
    plt.ylabel('Percentuale (%)')
    plt.title('Distribuzione dei valori nella classi"')

    # Aggiungi le etichette delle percentuali sopra le barre
    for index, value in enumerate(percentuale):
        plt.text(index, value, f'{value: .2f}%', ha='center', va='bottom')

    # Rimuovi le etichette sull'asse X
    plt.xticks([])
    plt.show()


"""
Funzione che salva il box plot di ogni variabile indipendente rispetto alle classi
"""


def preBoxPlotAnalysisData(x, y, boxPlotDir):
    # Ottieni la lista dei file nella cartella
    elenco_file = os.listdir(boxPlotDir)

    # Itera attraverso la lista dei file e rimuovili
    for file in elenco_file:
        percorso_completo = os.path.join(boxPlotDir, file)
        os.remove(percorso_completo)

    x['Label'] = y['Label']
    for col in x.columns:
        x.boxplot(column=col, by='Label')
        # Aggiungi titoli e label per gli assi, se necessario
        plt.title(f'Boxplot della colonna {col}')
        plt.xlabel(goodMalware)
        plt.ylabel('Dati')
        file_name = os.path.join(
            boxPlotDir, f'boxplot_{col}.png')
        plt.savefig(file_name)
        plt.close()


"""
Funzione che calcola il mutal info e restituisce un dict contenente in sorted_x[0] il nome della colonna, e in sorted_x[1] il valore
del mutual info
"""


def mutualInfoRank(X, Y):
    print("Computing mutual info ranking...")
    independentList = list(X.columns.values)
    res = dict(zip
               (independentList,
                   mutual_info_classif
                   (X,  np.ravel(Y), discrete_features=False, random_state=seed)))

    sorted_x = sorted(res.items(), key=lambda kv: kv[1], reverse=True)

    print("Computing mutual info ranking...completed")
    # ritorna un dizionario
    return sorted_x


"""Stampa i box blot delle variabili indipendenti con mutual info elevato e non"""


def BoxPlotAnalysisDataMutualInfo(x, y, boxPlotDir, mutualInfo={}, n_print=10):
    # Ottieni la lista dei file nella cartella

    boxPlotDirMutualInfoFirst = boxPlotDir / "MoreSignificant"
    boxPlotDirMutualInfoLast = boxPlotDir / "LessSignificant"

    elenco_file = os.listdir(boxPlotDirMutualInfoFirst)
    # Itera attraverso la lista dei file e rimuovili
    for file in elenco_file:
        percorso_completo = os.path.join(boxPlotDirMutualInfoFirst, file)
        os.remove(percorso_completo)

    elenco_file = os.listdir(boxPlotDirMutualInfoLast)
    # Itera attraverso la lista dei file e rimuovili
    for file in elenco_file:
        percorso_completo = os.path.join(boxPlotDirMutualInfoLast, file)
        os.remove(percorso_completo)

    i = 0
    if (mutualInfo):
        x['Label'] = y['Label']
        for tupla in mutualInfo:
            for col in x.columns:
                if (tupla[0] == col and tupla[1] != 0):  # capire se va 0 considerato o meno
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
"""


def pca(X):
    pca = PCA(n_components=len(X.columns))
    pca.fit(X)
    feature_names = pca.get_feature_names_out()

    return pca, feature_names, pca.explained_variance_ratio_


"""
Applicazione del PCA tramite un oggetto pca 
"""


def applyPCA(X, pca, pcalist):
    # Trasforma il DataFrame utilizzando PCA
    transformed = pca.transform(X)
    # Crea un nuovo DataFrame con le componenti principali
    # Deve essere la stessa?
    df_pca = pd.DataFrame(transformed, columns=pcalist)
    print(f"\npcalist: '{pcalist}'\n")
    print(f"\ntransformed: '{transformed}'\n", )
    return df_pca


"Selezione di Alcune delle PC con un determinato threesold"


def NumberOfTopPCSelect(explained_variance, threesold):
    n = 0
    i = 0
    if (explained_variance[i] < threesold):
        sumVariance = explained_variance[i]
        n = n+1
        i = i+1
        while (sumVariance < threesold) and i < len(explained_variance):
            sumVariance = sumVariance+explained_variance[i]
            n = n+1
            i = i+1
        return n
    else:
        return 0


def stratifiedKFold(X, Y, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    ListXTrain = []
    ListXTest = []
    ListYTrain = []
    ListYTest = []
    for i, (train_index, test_index) in enumerate(skf.split(X, Y)):

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
        if intersezione:
            print("Le liste hanno un'intersezione:", intersezione)
        else:
            print("Le liste non hanno un'intersezione.")

    return ListXTrain, ListXTest, ListYTrain, ListYTest


def printFolds(list):
    for i in range(len(list)):
        print(f"Fold {i} " + str(list[i].shape) + "\n")


def decisionTreeLearner(X, Y, c='gini'):
    clf = DecisionTreeClassifier(criterion=c, random_state=seed)
    clf.min_samples_split = 500  # Numero minimo di esempi per mettere uno split
    # Tree_ dentro c'è l'albero addestrata
    clf.fit(X, Y)

    print(f"Number of nodes: {clf.tree_.node_count}")
    print(f"Number of leaves: {clf.get_n_leaves()} ")
    # Return the number of leaves of the decision tree.)
    return clf


def showTree(clf, script_pathTreeFolder):

    plt.figure(figsize=(15, 10))
    tree.plot_tree(clf,
                   filled=True,
                   rounded=True)
    file_name = os.path.join(script_pathTreeFolder, 'TreeFigOutput.pdf')
    plt.savefig(file_name,
                format='pdf', dpi=1000)
    plt.show()


def determineDecisionTreekFoldConfiguration(ListXTrain, ListYTrain, ListXTest, ListYTest, rank, min_t, max_t, step):

    best_fscore = 0
    best_criterion = None
    best_TH = None
    bestN = None
    criterion = ['gini', 'entropy']

    for criteria in criterion:
        for thre in np.arange(min_t, max_t, step):
            fscores = []
            selectedFeatures = topFeatureSelect(rank, thre)
            if (len(selectedFeatures) > 0):
                for i in range(len(ListXTrain)):
                    x_train_feature_selected = ListXTrain[i].loc[:,
                                                                 selectedFeatures]
                    clf = decisionTreeLearner(
                        x_train_feature_selected, ListYTrain[i], criteria)

                    x_test = ListXTest[i].loc[:, selectedFeatures]
                    y_pred = clf.predict(x_test)

                    fscores.append(f1_score(ListYTest[i], y_pred))

            avg_fscore = np.mean(fscores)
            print(f"Average F1 score: '{avg_fscore}'")
            if avg_fscore > best_fscore:
                best_fscore = avg_fscore
                best_criterion = criteria
                best_TH = thre
                bestN = selectedFeatures

            if avg_fscore == best_fscore:
                if (len(selectedFeatures) < len(bestN)):
                    best_fscore = avg_fscore
                    best_criterion = criteria
                    best_TH = thre
                    bestN = selectedFeatures

    return best_criterion, best_TH, bestN, best_fscore


"""




# Funzione che controlla se ci sono attributi mancanti, andando a controllare il "count"
def missedValue(x):
    columnsWrongCount = []
    for col in x.columns:
        col_data = x[col]
        col_description = col_data.describe()
        if col_description["count"] != 12000:
            count = col_description["count"]
            columnsWrongCount.append(col)
            print(f"La colonna '{
                  col}' ha un count ha degli attributi mancanti, count: '{count}'")
    if len(columnsWrongCount) == 0:
        print("Non ci sono colonne con attributi mancanti")

    return columnsWrongCount

"""
