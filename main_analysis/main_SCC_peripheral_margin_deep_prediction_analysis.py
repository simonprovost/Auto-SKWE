import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.preProcess import *
from src.processing import Processing
from src.utils import getDataFromCsv, readParamsFile


def main(argv):
    ### DATA PRE PROCESSING
    dataset = getDataFromCsv('./datasets/nhs-data/[DATASET_2_SCC]dataset_nhs_peripheral_margin_deep_prediction.csv')
    input = dataset[1]
    classData = dataset[2]

    data = PreProcess(input, classData, datasetName="scc_margin_deep")
    data.redefineColumnsType()

    X = data.input
    y = data.output

    ### AUTO ML PROCESSING

    params = readParamsFile(argv[1])

    datasetCrossVal = None
    if len(argv) == 3 and argv[2] == "cv":
        datasetCrossVal = data.redefineColumnsType(selfInput=False, extraData=dataset[0])
    if len(argv) == 4 and argv[3] == "weka":
        datasetCrossVal = data.redefineColumnsType(selfInput=False, extraData=dataset[0])

    autoML = Processing(inputData=X, classLabelData=y, entireDataset=datasetCrossVal, datasetName=data.datasetName,
                        **params)
    autoML.setup()
    if len(argv) == 3 and argv[2] == "cv":
        autoML.k_folds_split(k_folds=10)
        autoML.cross_validation_process("./outputCrossValidationAutoML/SCC_margin_deep/")
        autoML.show_latex_cross_validation_process()
    elif len(argv) == 4 and argv[3] == "weka":
        autoML.k_folds_split(k_folds=10)
        autoML.cross_validation_process("./outputCrossValidationAutoML/SCC_margin_deep/", AutoSklearn=False)
    else:
        autoML.fit_predict()
        models = autoML.get_best_models(numberOfModelToGet=0, display=False)
        autoML.show_latex_table(models[0])


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise ValueError("Need path to the params to use!")
    main(sys.argv)
