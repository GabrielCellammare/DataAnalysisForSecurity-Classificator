from UtilsFunctions import *


def test_mutualInfo(x, y, script_path):
    x_mutualinfo = mutualInfoRank(x, y)
    print(f"X mutual_info: '{x_mutualinfo}'\n")
    boxPlotDirMutualInfo = script_path.parent.parent / "BoxPlotMutualInfo"
    BoxPlotAnalysisDataMutualInfo(
        x, y, boxPlotDirMutualInfo, x_mutualinfo)

    selectedFeatures = topFeatureSelect(x_mutualinfo, 0.1)
    print(len(selectedFeatures))

    x_mutualinfo = x.loc[:, selectedFeatures]


def test_PCA(x):
    pca, pcalist, explained_variance = pca(
        x)

    print(pcalist)
    print(len(pcalist))
    print(explained_variance)

    XPCA = applyPCA(x, pca, pcalist)
    n = NumberOfTopPCSelect(explained_variance, 0.99)
    print(n)

    # create a dataset with the selected PCs
    XPCASelected = XPCA.iloc[:, 1:(n+1)]
    print(XPCASelected.shape)


def test_PCA(x, y, script_path):
    # Laboratorio 5 con aggiunta di mutual info

    # Oggetto decistion Tree
    clf = decisionTreeLearner(x, y, 'entropy')

    script_pathTreeFolder = script_path.parent.parent / "TreeFigOutput"
    showTree(clf, script_pathTreeFolder)
