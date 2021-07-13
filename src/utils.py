import arff
import numpy as np
from os import path
import pandas as pd
import os, os.path
import random
import ast


def __checkPathExists(filepath, extension=None, errorMessage='Filepath given should be a path that exists!'):
    if not path.exists(filepath):
        raise ValueError(errorMessage)
    if extension:
        file_name, file_extension = path.splitext(filepath)
        if file_extension != extension:
            raise ValueError(
                "the extension of the file: `" + file_name + file_extension + "` is not what it was required: " + extension)


def setSeedEnvironement(seedValue):
    if not seedValue:
        return

    os.environ['PYTHONHASHSEED'] = str(seedValue)
    random.seed(seedValue)
    np.random.seed(seedValue)


def getDataFromCsv(filepath, extension='.csv'):
    __checkPathExists(filepath, extension)

    dataset = pd.read_csv(filepath)
    input = dataset.iloc[:, :-1]
    output = dataset.iloc[:, -1:]

    return dataset, input, output


def getDataFromArff(filepath, extension='.arff', openOption='rt'):
    __checkPathExists(filepath, extension)

    dataset = arff.load(open(filepath, openOption))
    fullData = pd.DataFrame(dataset['data'])
    x = fullData.drop(fullData.columns[-1],axis=1)
    y = fullData.iloc[:, -1]

    return fullData, x, y


def readParamsFile(filePath):
    params = {}

    if not os.path.isfile(filePath):
        raise ValueError("Filepath's not a propoer file to analyse!'")

    fileParams = open(filePath)
    for line in fileParams:
        key, value = line.split()
        params[key] = int(value) if value.isnumeric() else value

    if params.get('resampling_strategy_arguments') is not None:
        params['resampling_strategy_arguments'] = ast.literal_eval(params['resampling_strategy_arguments'])
    return params
