from UtilsFunctions import *


def DescribeData(x, y, script_path):
    print("\nShape di train_x:", x.shape)
    print("\nShape di train_y:", y.shape)

    # Visualizzazione di alcune statistiche riguardo le colonne
    preElaborationData(x)

    # Calcolo quante occorrenze per ogni classe ci sono
    labelCount = differentClass(y)
    print(labelCount, "\n")
    # Stampo un istogramma
    plotHistogram(labelCount, y)

    # Percorso della cartella contenente Box plot di variabili "significative"
    boxPlotDir = script_path.parent.parent / "BoxPlot"

    # Percorso della cartella contenente Box plot di variabili "non significative"
    overlapDir = script_path.parent.parent / "BoxPlotOverlapDir"

    # if not os.listdir(boxPlotDir):
    # Stampa Box Plot classica
    if (not os.listdir(boxPlotDir)):
        BoxPlotPreAnalysisData(x, y, boxPlotDir)
    """
    # BoxPlotPreAnalysisData(x, y, boxPlotDir)
    columnsToRemove = []
    if (not os.listdir(overlapDir)) and (not os.listdir(boxPlotDir)):
        # Stampa Box plot variabili signficative e non
        columnsToRemove = BoxPlotPreAnalysisDataSelection(
            x, y, boxPlotDir, overlapDir)

    # Ottieni una lista di tutti i nomi dei file nella cartella
    nomi_dei_file = os.listdir(overlapDir)

    # Filtra la lista per includere solo i file che terminano con '.png'
    foto = [file for file in nomi_dei_file if file.endswith('.png')]

    for nome in foto:
        # Dividi il nome del file sulla base del carattere '_'
        parti = nome.split('_')
        # Prendi la seconda e la terza parte (ignora la prima parte 'boxplot')
        nome_estratto = '_'.join(parti[1:])
        # Rimuovi l'estensione del file '.png'
        nome_estratto = nome_estratto.replace('.png', '')
        columnsToRemove.append(nome_estratto)

    return columnsToRemove
    """
