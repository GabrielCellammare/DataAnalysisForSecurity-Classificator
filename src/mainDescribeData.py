from UtilsFunctions import preElaborationData, differentClass, plotHistogram, BoxPlotPreAnalysisData
import os


def DescribeData(x, y, script_path):
    print("\nShape di x:", x.shape)
    print("\nShape di y:", y.shape)

    # Visualizzazione di alcune statistiche riguardo le colonne
    preElaborationData(x)

    # Calcolo quante occorrenze per ogni classe ci sono
    labelCount = differentClass(y)
    print(labelCount, "\n")
    # Stampo un istogramma
    plotHistogram(labelCount, y)

    # Percorso della cartella contenente Box plot di variabili "significative"
    boxPlotDir = script_path.parent.parent / "BoxPlot"

    # if not os.listdir(boxPlotDir):
    # Stampa Box Plot classica
    if (not os.listdir(boxPlotDir)):
        BoxPlotPreAnalysisData(x, y, boxPlotDir)
