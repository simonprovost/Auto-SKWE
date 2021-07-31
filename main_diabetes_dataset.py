import sys

from src.preProcess import *
from src.processing import Processing
from src.utils import getDataFromArff, readParamsFile


def main(argv):
    ### DATA PRE PROCESSING
    dataset = getDataFromArff('datasets/diabetic-retinopathy-debrecen/diabetic-retinopathy-Debrecen.arff')
    input = dataset[1]
    classData = dataset[2]

    data = PreProcess(input, classData, datasetName="diabetic-retinopathy-Debrecen")
    data.redefineColumnsType()

    X = data.input
    y = np.array([1 if x == "1" else 0 for x in data.output])

    ### AUTO ML PROCESSING
    params = readParamsFile(argv[1])

    datasetCrossVal = None
    if len(argv) > 2 and argv[2] == "cv":
        dataset[0].replace("1", 1, inplace=True)
        dataset[0].replace("0", 0, inplace=True)
        datasetCrossVal = data.redefineColumnsType(selfInput=False, extraData=dataset[0])

    autoML = Processing(inputData=X, classLabelData=y, entireDataset=datasetCrossVal, datasetName=data.datasetName, **params)
    autoML.setup()
    if len(argv) > 2 and argv[2] == "cv":
        autoML.k_folds_split(k_folds=10)
        autoML.cross_validation_process("./outputCrossValidationAutoML/diabetic-retinopathy-Debrecen/")
        autoML.show_latex_cross_validation_process()
    else:
        autoML.fit_predict()
        models = autoML.get_best_models(numberOfModelToGet=0, display=False)
        autoML.show_latex_table(models[0])


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise ValueError("Need path to the params to use!")
    main(sys.argv)
