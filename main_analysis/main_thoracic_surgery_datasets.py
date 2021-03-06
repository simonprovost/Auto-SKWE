import os
import sys
import inspect

import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.preProcess import *
from src.processing import Processing
from src.utils import getDataFromArff, readParamsFile


def main(argv):
    ### DATA PRE PROCESSING
    dataset = getDataFromArff('datasets/thoracic-data/Thoracic-Surgery-binary-survival.arff')
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

    X = data.input
    y = data.output

    ### AUTO ML PROCESSING
    params = readParamsFile(argv[1])

    datasetCrossVal = None
    if len(argv) == 3 and argv[2] == "cv":
        dataset[0].replace("F", 0, inplace=True)
        dataset[0].replace("T", 1, inplace=True)
        datasetCrossVal = data.redefineColumnsType(selfInput=False, extraData=dataset[0])
    if len(argv) == 4 and argv[3] == "weka":
        datasetCrossVal = data.redefineColumnsType(selfInput=False, extraData=dataset[0])

    autoML = Processing(inputData=X, classLabelData=y, entireDataset=datasetCrossVal, datasetName=data.datasetName,
                        **params)
    autoML.setup()
    if len(argv) == 3 and argv[2] == "cv":
        autoML.k_folds_split(k_folds=10)
        autoML.cross_validation_process("./outputCrossValidationAutoML/Thoracic-Surgery-binary-survival/",
                                        reSampling=True)
        autoML.show_latex_cross_validation_process()
    elif len(argv) == 4 and argv[3] == "weka":
        autoML.k_folds_split(k_folds=10)
        autoML.cross_validation_process("./outputCrossValidationAutoML/Thoracic-Surgery-binary-survival/",
                                        AutoSklearn=False)
    else:
        autoML.fit_predict()
        models = autoML.get_best_models(numberOfModelToGet=0, display=False)
        autoML.show_latex_table(models[0])


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise ValueError("Need path to the params to use!")
    main(sys.argv)
