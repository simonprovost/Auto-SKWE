import sys

from src.preProcess import *
from src.processing import Processing
from src.utils import getDataFromCsv, readParamsFile


def main(argv):
    ### DATA PRE PROCESSING
    dataset = getDataFromCsv('datasets/heart-failure/Heart-failure-clinical-records_dataset.csv')
    input = dataset[1]
    classData = dataset[2]

    data = PreProcess(input, classData, datasetName="Heart-failure-clinical-records_dataset")
    data.redefineColumnsType()

    X = data.input
    y = data.output

    ### AUTO ML PROCESSING
    params = readParamsFile(argv[1])

    datasetCrossVal = None
    if len(argv) > 2 and argv[2] == "cv":
        datasetCrossVal = data.redefineColumnsType(selfInput=False, extraData=dataset[0])

    autoML = Processing(inputData=X, classLabelData=y, entireDataset=datasetCrossVal, datasetName=data.datasetName, **params)
    autoML.setup()
    if len(argv) > 2 and argv[2] == "cv":
        autoML.k_folds_split(k_folds=10)
        autoML.cross_validation_process("./outputCrossValidationAutoML/Heart-failure-clinical-records/")
        autoML.show_latex_cross_validation_process()
    else:
        autoML.fit_predict()
        models = autoML.get_best_models(numberOfModelToGet=0, display=False)
        autoML.show_latex_table(models[0])


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise ValueError("Need path to the params to use!")
    main(sys.argv)
