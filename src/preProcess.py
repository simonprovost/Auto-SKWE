from collections import Counter
from collections import defaultdict

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder


class LabelEncoder:
    """
    The LabelEncoder class instance below is a derived of the sklearn.preprocessing.LabelEncoder() class, forked for
    a better application of the current context. Given the data its running a [label encoding method](https://www.geeksforgeeks.org/ml-label-encoding-of-datasets-in-python/)
    for having all features in a numerical way for feeding any machine learning algorithms.

    Init
    ----------
    Nothing is necessary to give during the instantiation of the class.

    Returns
    -------
    An instance of the LabelEncoder() with all the public methods available below.

    """

    def __init__(self):
        self.columns = None
        self.led = defaultdict(preprocessing.LabelEncoder)

    def fit(self, X):
        """
        The fit method is looping through the data to be sure that NaN value are changed in "None" value (string).

        Parameters
        ----------
        X: Pandas.series()
            X is the dataframe that you are looking for being fit with the method.

        """
        self.columns = X.columns
        for col in self.columns:
            val = X[col].unique()
            val = [x if x is not None else "None" for x in val]
            self.led[col].fit(val)
        return self

    def fit_transform(self, X):
        """
        The fit_transform method is a faster way of doing LabelEncoder().fit ; LabelEncoder().transform.

        Parameters
        ----------
        X: Pandas.series()
            X is the dataframe that you are looking for being fit with the method.

        """
        if self.columns is None:
            self.fit(X)
        return self.transform(X)

    def transform(self, X):
        """
        The transform method is moving all X's data into a label encoded.

        Parameters
        ----------
        X: Pandas.series()
            X is the dataframe that you are looking for being fit with the method.

        """
        return X.apply(lambda x: self.led[x.name].transform(x.apply(lambda e: e if e is not None else "None")))


class PreProcess:
    def __init__(self, inputData, classLabelData, datasetName):
        self.input = inputData
        self.output = classLabelData
        self.datasetName = datasetName
        self.encodedData = None

    def genericFeatureScaling(self, inputData, columnName, initialArray, outputArray):
        if not initialArray or not outputArray or len(initialArray) != len(outputArray):
            raise ValueError("initialArray && OutputArray (with the same size) is needed to continue the process!")

        for idx, value in enumerate(initialArray):
            if inputData:
                self.input[columnName] = np.where(
                    (self.input[columnName] == value), outputArray[idx], self.input[columnName]
                )
            else:
                self.output[columnName] = np.where(
                    (self.output[columnName] == value), outputArray[idx], self.output[columnName]
                )

    def labelEncoding(self, columns):
        self.encodedData = LabelEncoder()
        self.encodedData.fit(self.input)
        transformed = self.encodedData.transform(self.input[columns])

        self.input[columns] = transformed.values

    def oneHotEncoding(self, column, columnString=True):
        self.encodedData = OneHotEncoder()

        columnName = str(column) if not columnString else column
        x_Data = self.encodedData.fit_transform(self.input[column].values.reshape(-1, 1)).toarray()

        onehencColumnNames = self.encodedData.get_feature_names()
        dfOneHot = pd.DataFrame(x_Data, columns=[columnName + "_" + onehencColumnNames[i] for i in range(x_Data.shape[1])])
        self.input = pd.concat([self.input, dfOneHot], axis=1)
        self.input.drop(column, axis=1, inplace=True)

    def convertsToIntegerType(self, columns):
        for x in columns:
            self.input[x] = pd.to_numeric(self.input[x])

    def reSamplingSMOTE(self):
        counter = Counter(self.output)
        oversample = SMOTE()
        self.input, self.output = oversample.fit_resample(self.input, self.output)
        counter = Counter(self.output)

    def __checkFullFloat(self, array):
        for x in array:
            try:
                isinstance(float(x), float)
            except ValueError:
                return False
            if isinstance(float(x), float):
                continue
            return False
        return True

    def redefineColumnsType(self, selfInput=True, extraData=None):
        if selfInput is not True and extraData is not None:
            results = extraData.copy()
            for x in results:
                if np.dtype(results[x]) != np.object:
                    results[x] = pd.to_numeric(results[x])
                elif np.dtype(results[x]) == np.object:
                    results[x] = results[x].astype("category", copy=False)
            return results
        else:
            for idx, x in enumerate(self.input):
                if np.dtype(self.input[x]) != np.object:
                    self.input[x] = pd.to_numeric(self.input[x])
                elif np.dtype(self.input[x]) == np.object:
                    self.input[x] = self.input[x].astype("category")

    def checkImbalancedDataset(self, latex=False):
        classData = np.unique(self.output.values, return_counts=True)
        total_count = sum(classData[1])
        results = {
            "predictive_feature_" + str(x): classData[1][idx] * 100 / total_count
            for idx, x in enumerate(classData[0])
        }
        results['dataset_name'] = self.datasetName
        results['total_number_of_instances'] = total_count
        results['number_predictive_features'] = len(classData[0])

        if len(classData[0]) > 2:
            results['classification_format'] = "multi-class"
        elif len(classData[0]) == 2:
            results['classification_format'] = "binary-classification"
        else:
            raise ValueError("Class data should have at least two unique values.")

        print(results)
        if latex:
            print(pd.DataFrame(results, index=[0]).to_latex(index=False))
