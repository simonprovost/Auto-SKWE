import math
import sys

import numpy as np
from src.preProcess import *
from src.processing import Processing
from src.utils import getDataFromCsv, readParamsFile
import pandas as pd
import numbers


def main(argv):
    dataset = pd.read_csv('../../datasets/nhs-data/initial_dataset/nhs-data-master-sheet-updated-phase-2.csv')

    dataset_bcc = True
    dataset_scc = False

    input = dataset.drop(
        ['Unnamed: 0', 'Unnamed: 0.1', 'Peripheral Margin (Deep)', 'Peripheral Margin (Raw)', 'Specimen Number Full',
         'Maximum dimension'], axis=1)
    # input = dataset.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Peripheral Margin (Deep)', 'Peripheral Margin (Raw)', 'Specimen Number Full', 'Maximum dimension'], axis=1)
    output = dataset[['Peripheral Margin (Raw)', 'Peripheral Margin (Deep)']]

    data = PreProcess(input, output, datasetName="Dataset-NHS")
    data.genericFeatureScaling(inputData=True, columnName="High or low",
                               initialArray=['High', 'Low', 'high'],
                               outputArray=['High', 'Low', 'High'])
    data.genericFeatureScaling(inputData=True, columnName="Tumour Thickness",
                               initialArray=['>6mm', '<2mm', '>2-4mm', '>4mm', '>4-6mm'],
                               outputArray=['6mm', '2-4mm', '2-4mm', '4-6mm', '4-6mm'])
    data.genericFeatureScaling(inputData=True, columnName="Level of Invasion",
                               initialArray=['Clark Level 5', 'Clark Level 3', 'Clark Level 4', 'Clark Level 2',
                                             'Clark 5', '?Clark 3', 'wedge'],
                               outputArray=['Clark Level 5', 'Clark Level 3-4', 'Clark Level 3-4', 'Clark Level 2',
                                            'Clark Level 5', 'Clark Level 3-4', 'wedge'])

    if not dataset_bcc and not dataset_scc:
        data.oneHotEncoding(column="Histo")

    data.oneHotEncoding(column="Grade")
    data.oneHotEncoding(column="Level of Invasion")
    data.oneHotEncoding(column="Anatomical Site")
    data.oneHotEncoding(column="Perineural(Replaced)")
    data.oneHotEncoding(column="Subtype")
    data.redefineColumnsType()

    X = data.input
    y = data.output['Peripheral Margin (Raw)']

    y = [x for x in y if x != 'nan']
    idxToRemove = [idx for idx, x in enumerate(y) if x == 'nan']
    X.drop(idxToRemove, inplace=True)

    for idx, x in enumerate(X):
        if 'nan' in x or 'unknown' in x:
            X.drop(x, inplace=True, axis=1)

    for idx, x in enumerate(X):
        frequency_of_one = np.count_nonzero(X[x])

        if 'x0' in x and frequency_of_one <= 10:
            X.drop(x, inplace=True, axis=1)

    datasetCrossVal = X
    datasetCrossVal['Peripheral Margin (Raw)'] = y

    indexNames = datasetCrossVal[datasetCrossVal['Histo'] == 'melanoma'].index.values
    indexNamesSCC = datasetCrossVal[datasetCrossVal['Histo'] == 'SCC'].index.values

    idxToRemoves = np.concatenate((indexNames, indexNamesSCC), axis=0)
    datasetCrossVal.drop(idxToRemoves, inplace=True)
    datasetCrossVal['Peripheral Margin (Raw)'] = [str(a) for a in datasetCrossVal['Peripheral Margin (Raw)']]

    indexNAClassLabel = datasetCrossVal[ datasetCrossVal['Peripheral Margin (Raw)'] == 'nan'].index.values
    datasetCrossVal.drop(indexNAClassLabel, inplace=True)

    datasetCrossVal['Peripheral Margin (Raw)'] = [float(a) for a in datasetCrossVal['Peripheral Margin (Raw)']]
    datasetCrossVal.drop('Histo', axis=1, inplace=True)

    datasetCrossVal.to_csv("./[DATASET_1_BCC]dataset_nhs_peripheral_margin_raw_prediction.csv", index=1)


if __name__ == "__main__":
    main(sys.argv)
