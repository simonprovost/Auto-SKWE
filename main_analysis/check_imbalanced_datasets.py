import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.preProcess import *
from src.utils import getDataFromCsv, getDataFromArff


def main():
    ### DATA PRE PROCESSING
    dataset = getDataFromCsv('datasets/breast-cancer/Breast-cancer-Coimbra.csv')

    data = PreProcess(dataset[1], dataset[2], datasetName="Breast-cancer-Coimbra")
    data.redefineColumnsType()
    data.checkImbalancedDataset(latex=True)
    print("\n")

    ### DATA PRE PROCESSING
    dataset = getDataFromArff('datasets/diabetic-retinopathy-debrecen/diabetic-retinopathy-Debrecen.arff')

    data = PreProcess(dataset[1], dataset[2], datasetName="diabetic-retinopathy-Debrecen")
    data.redefineColumnsType()
    data.checkImbalancedDataset(latex=True)
    print("\n")

    ### DATA PRE PROCESSING
    dataset = getDataFromCsv('datasets/heart-failure/Heart-failure-clinical-records_dataset.csv')

    data = PreProcess(dataset[1], dataset[2], datasetName="Heart-failure-clinical-records_dataset")
    data.redefineColumnsType()
    data.checkImbalancedDataset(latex=True)
    print("\n")

    ### DATA PRE PROCESSING
    dataset = getDataFromCsv('datasets/obesity-data/ObesityDataSet_raw_and_data_sinthetic.csv')

    data = PreProcess(dataset[1], dataset[2], datasetName="ObesityDataSet_raw_and_data_synthetic")
    data.redefineColumnsType()
    data.checkImbalancedDataset(latex=True)
    print("\n")

    ### DATA PRE PROCESSING
    dataset = getDataFromArff('datasets/thoracic-data/Thoracic-Surgery-binary-survival.arff')

    data = PreProcess(dataset[1], dataset[2], datasetName="Thoracic-Surgery-binary-survival")
    data.redefineColumnsType()
    data.checkImbalancedDataset(latex=True)


if __name__ == "__main__":
    main()
