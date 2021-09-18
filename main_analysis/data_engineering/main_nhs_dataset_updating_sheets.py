import math
import sys

import numpy as np
from src.preProcess import *
from src.processing import Processing
from src.utils import getDataFromCsv, readParamsFile
import pandas as pd
import numbers


def main(argv):
    datasetMaster = pd.read_csv('../../datasets/nhs-data/initial_dataset/nhs-data-master-sheet.csv')
    datasetUpdate = pd.read_csv('../../datasets/nhs-data/initial_dataset/nhs-data-update-margins.csv')

    known_subjects = []
    all_data_subjects = []
    internal_index = 0

    for idx, x in enumerate(datasetUpdate['Specimen Number Full']):
        if x not in known_subjects:
            known_subjects.append(x)
            all_data_subjects.append([x, [None, None]])
            for idx_j, j in enumerate(datasetUpdate['Specimen Number Full']):
                if x == j:
                    if (
                            isinstance(datasetUpdate['Peripheral final'][idx_j], str)
                            and datasetUpdate['Peripheral final'][idx_j] != 'uncertain'
                            and not isinstance(all_data_subjects[internal_index][1][0], float)
                    ):
                        all_data_subjects[internal_index][1][0] = float(datasetUpdate['Peripheral final'][idx_j])
                    if (
                            isinstance(datasetUpdate['Deep final'][idx_j], str)
                            and datasetUpdate['Deep final'][idx_j] != 'uncertain'
                            and not isinstance(all_data_subjects[internal_index][1][1], float)
                    ):
                        all_data_subjects[internal_index][1][1] = float(datasetUpdate['Deep final'][idx_j])
            internal_index += 1

    for idx, x in enumerate(datasetMaster['Specimen Number Full']):
        for j in all_data_subjects:
            if x in j:
                if math.isnan(datasetMaster['Peripheral Margin (Raw)'][idx]) and isinstance(j[1][0], float):
                    datasetMaster['Peripheral Margin (Raw)'][idx] = j[1][0]
                if math.isnan(datasetMaster['Peripheral Margin (Deep)'][idx]) and isinstance(j[1][1], float):
                    datasetMaster['Peripheral Margin (Deep)'][idx] = j[1][1]

    datasetMaster['Peripheral Margin (Raw)'] = np.array(
        [0 if x <= 0.5 else x if math.isnan(x) else 1 for x in datasetMaster['Peripheral Margin (Raw)']])
    datasetMaster['Peripheral Margin (Deep)'] = np.array(
        [0 if x <= 0.5 else x if math.isnan(x) else 1 for x in datasetMaster['Peripheral Margin (Deep)']])

    datasetMaster.to_csv("./datasets/nhs-data/nhs-data-master-sheet-updated-phase-2.csv", sep=',')

    check_sample_proportion = False
    if check_sample_proportion:
        for x in datasetMaster:
            lenTotal = len(datasetMaster[x])
            lenNA = len([0 for j in datasetMaster[x] if not isinstance(j, str) and math.isnan(j)])


if __name__ == "__main__":
    main(sys.argv)
