import sys

from src.preProcess import *
from src.processing import Processing
from src.utils import getDataFromArff, readParamsFile


def main(argv):
    ### DATA PRE PROCESSING
    dataset = getDataFromArff('./datasets/diabetic-retinopathy-debrecen/diabetic-retinopathy-Debrecen.arff')
    input = dataset[1]
    classData = dataset[2]

    data = PreProcess(input, classData)
    data.redefineColumnsType()

    X = data.input
    y = np.array([1 if x == "1" else 0 for x in data.output])

    ### AUTO ML PROCESSING
    params = readParamsFile(argv[1])

    autoML = Processing(inputData=X, classLabelData=y, datasetName="diabetic-retinopathy-Debrecen", **params)
    autoML.setup()
    autoML.fit_predict()
    models = autoML.getBestModels(numberOfModelToGet=0, display=False)
    autoML.showLatexTable(models[0])


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise ValueError("Need path to the params to use!")
    main(sys.argv)
