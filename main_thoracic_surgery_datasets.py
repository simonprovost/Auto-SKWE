import sys

from src.preProcess import *
from src.processing import Processing
from src.utils import getDataFromArff, readParamsFile


def main(argv):
    ### DATA PRE PROCESSING
    dataset = getDataFromArff('./datasets/thoracic-data/Thoracic-Surgery-binary-survival.arff')
    input = dataset[1]
    classData = dataset[2]

    data = PreProcess(input, classData, datasetName="Thoracic-Surgery-binary-survival")

    data.oneHotEncoding(0, False)
    data.oneHotEncoding(3, False)
    data.oneHotEncoding(9, False)
    data.redefineColumnsType()

    data.input[4] = np.array([0 if x == "F" else 1 for x in data.input[4]])
    data.input[5] = np.array([0 if x == "F" else 1 for x in data.input[5]])
    data.input[6] = np.array([0 if x == "F" else 1 for x in data.input[6]])
    data.input[7] = np.array([0 if x == "F" else 1 for x in data.input[7]])
    data.input[8] = np.array([0 if x == "F" else 1 for x in data.input[8]])
    data.input[10] = np.array([0 if x == "F" else 1 for x in data.input[10]])
    data.input[11] = np.array([0 if x == "F" else 1 for x in data.input[11]])
    data.input[12] = np.array([0 if x == "F" else 1 for x in data.input[12]])
    data.input[13] = np.array([0 if x == "F" else 1 for x in data.input[13]])
    data.input[14] = np.array([0 if x == "F" else 1 for x in data.input[14]])
    data.output = np.array([0 if x == "F" else 1 for x in data.output])

    #//TODO data.reSamplingSMOTE()

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
