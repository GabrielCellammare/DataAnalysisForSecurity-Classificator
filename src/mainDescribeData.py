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

    boxPlotDir = script_path.parent.parent / "BoxPlot"

    overlapDir = script_path.parent.parent / "OverlapDir"

    # BoxPlotPreAnalysisData(x, y, boxPlotDir)
    # BoxPlotPreAnalysisDataFeatureSelection(x, y, boxPlotDir, overlapDir)
