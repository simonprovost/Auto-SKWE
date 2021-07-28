import sys

from src.preProcess import *
from src.processing import Processing
from src.utils import getDataFromCsv, readParamsFile


def main(argv):
    ### DATA PRE PROCESSING
    dataset = getDataFromCsv('./datasets/breast-cancer/Breast-cancer-Coimbra.csv')
    input = dataset[1]
    classData = dataset[2]

    data = PreProcess(input, classData, datasetName="Breast-cancer-Coimbra")
    data.redefineColumnsType()

    X = data.input
    y = data.output

    ### AUTO ML PROCESSING
    params = readParamsFile(argv[1])

    autoML = Processing(inputData=X, classLabelData=y, datasetName=data.datasetName, **params)
    autoML.setup()
    autoML.fit_predict()
    models = autoML.getBestModels(numberOfModelToGet=0, display=False)
    autoML.showLatexTable(models[0])


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise ValueError("Need path to the params to use!")
    main(sys.argv)
