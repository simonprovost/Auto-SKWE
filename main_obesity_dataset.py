import sys

from src.preProcess import *
from src.processing import Processing
from src.utils import getDataFromCsv, readParamsFile


def main(argv):
    ### DATA PRE PROCESSING
    dataset = getDataFromCsv('datasets/obesity-data/ObesityDataSet_raw_and_data_sinthetic.csv')
    input = dataset[1]
    classData = dataset[2]

    data = PreProcess(input, classData, datasetName="ObesityDataSet_raw_and_data_synthetic")
    data.genericFeatureScaling(inputData=False, columnName="NObeyesdad",
                               initialArray=['Insufficient_Weight',
                                             'Normal_Weight',
                                             'Overweight_Level_I',
                                             'Overweight_Level_II',
                                             'Obesity_Type_I',
                                             'Obesity_Type_II',
                                             'Obesity_Type_III'],
                               outputArray=[0, 1, 2, 3, 4, 5, 6])
    data.redefineColumnsType()

    X = data.input
    y = np.array(data.output['NObeyesdad']).tolist()

    ### AUTO ML PROCESSING

    params = readParamsFile(argv[1])

    datasetCrossVal = None
    if len(argv) > 2 and argv[2] == "cv":
        dataset[0].replace("Insufficient_Weight", 0, inplace=True)
        dataset[0].replace("Normal_Weight", 1, inplace=True)
        dataset[0].replace("Overweight_Level_I", 2, inplace=True)
        dataset[0].replace("Overweight_Level_II", 3, inplace=True)
        dataset[0].replace("Obesity_Type_I", 4, inplace=True)
        dataset[0].replace("Obesity_Type_II", 5, inplace=True)
        dataset[0].replace("Obesity_Type_III", 6, inplace=True)
        datasetCrossVal = data.redefineColumnsType(selfInput=False, extraData=dataset[0])

    autoML = Processing(inputData=X, classLabelData=y, entireDataset=datasetCrossVal, datasetName=data.datasetName, **params)
    autoML.setup()
    if len(argv) > 2 and argv[2] == "cv":
        autoML.k_folds_split(k_folds=10)
        autoML.cross_validation_process("./outputCrossValidationAutoML/ObesityDataset_raw_and_data_synthetic/")
        autoML.show_latex_cross_validation_process()
    else:
        autoML.fit_predict()
        models = autoML.get_best_models(numberOfModelToGet=0, display=False)
        autoML.show_latex_table(models[0])


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise ValueError("Need path to the params to use!")
    main(sys.argv)
