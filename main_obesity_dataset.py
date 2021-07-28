import sys

from src.preProcess import *
from src.processing import Processing
from src.utils import getDataFromCsv, readParamsFile


def main(argv):
    ### DATA PRE PROCESSING
    dataset = getDataFromCsv('./datasets/obesity-data/ObesityDataSet_raw_and_data_sinthetic.csv')
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

    autoML = Processing(inputData=X, classLabelData=y, datasetName=data.datasetName, **params)
    autoML.setup()
    autoML.fit_predict()
    models = autoML.getBestModels(numberOfModelToGet=0, display=False)
    autoML.showLatexTable(models[0])


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise ValueError("Need path to the params to use!")
    main(sys.argv)
