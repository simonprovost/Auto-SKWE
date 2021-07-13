import os
import os.path

import autosklearn.classification
import numpy
import pandas as pd
import sklearn.datasets
import sklearn.datasets
import sklearn.model_selection
import sklearn.model_selection

from src.utils import setSeedEnvironement


### //TODO Disclaimer: Uncoment only if you are using Jupyter.
# import PipelineProfiler


class Processing:
    def __init__(self, inputData, classLabelData, datasetName, tmp_folder="./outputRuns/config_log_files/autosklearn_run", output_folder="./outputRuns/prediction_optional_tests/autosklearn_run", outputDir="./outputRuns", **kwargs):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.autoML = None
        self.predictions = None

        self.input = inputData
        self.classLabel = classLabelData

        numberOfRunsTmp = len([name for name in os.listdir(outputDir + '/config_log_files') if
                                    os.path.isdir(os.path.join(outputDir + '/config_log_files', name))]) + 1
        numberOfRunsOutput = len([name for name in os.listdir(outputDir + '/prediction_optional_tests') if
                                       os.path.isdir(os.path.join(outputDir + '/prediction_optional_tests', name))]) + 1

        self.params = kwargs
        self.datasetName = datasetName or "N/A"
        self.tmp_folder = tmp_folder + '_' + str(numberOfRunsTmp)
        self.output_folder = output_folder + '_' + str(numberOfRunsOutput)

    def setup(self):
        setSeedEnvironement(self.params.get('seed', None))

        self.X_train, self.X_test, self.y_train, self.y_test = \
            sklearn.model_selection.train_test_split(self.input, self.classLabel, random_state=None)

        self.autoML = autosklearn.classification.AutoSklearnClassifier(
            **self.params,
            tmp_folder=self.tmp_folder,
            output_folder=self.output_folder,
        )

    def fit_predict(self, showModels=False):
        self.__isAutoMLSetup()

        self.autoML.fit(X=self.X_train, y=self.y_train, dataset_name=self.datasetName)

        if showModels:
            print(self.autoML.show_models())
        self.predictions = self.autoML.predict(X=self.X_test)
        self.predictions = self.predictions.astype(str).astype(int)

    # METRICS

    ### //TODO Disclaimer: Uncoment only if you are using Jupyter.
    #def showVisualisationPipeline(self):
    #    profiler_data = PipelineProfiler.import_autosklearn(self.autoML)
    #    PipelineProfiler.plot_pipeline_matrix(profiler_data)

    def getBestModels(self, numberOfModelToGet=0, display=True):
        models = []

        losses_and_configurations = [
            (run_value.cost, run_key.config_id, run_value.time, )
            for run_key, run_value in self.autoML.automl_.runhistory_.data.items()
        ]
        losses_and_configurations.sort()

        if display:
            print("Lowest loss:", losses_and_configurations[0][0])

        for idx, x in enumerate(losses_and_configurations):
            if idx > numberOfModelToGet != -1:
                break
            for key, value in self.autoML.automl_.runhistory_.ids_config.items():
                if key == x[1]:
                    values = value.get_dictionary()
                    models.append({
                        'model_number': key,
                        'loss': x[0],
                        'time(s)': x[2],
                        'experiment_time': self.params.get("time_left_for_this_task", "N/A"),
                        'params': values
                    })
                    if display:
                        print(value)
        if numberOfModelToGet == 0:
            models[0]['accuracy'] = sklearn.metrics.accuracy_score(self.y_test, self.predictions)
            models[0]['error_rate'] = 100 * (1 - sklearn.metrics.accuracy_score(self.y_test, self.predictions))
        return models

    def showLatexTable(self, model):
        hyperParams = [(key, value) for key, value in model['params'].items() if key != 'classifier:__choice__'],

        df = pd.DataFrame({
            "Dataset name": self.datasetName,
            "Classifier": model['params'].get('classifier:__choice__', "None"),
            "Search Time limit": model.get('experiment_time', None),
            "Algorithm time run (s)": model.get('time(s)', "None"),
            "Seed": self.params.get("seed", "N/A"),
            "Score accuracy": model.get('accuracy', "None"),
            "Error rate": model.get('error_rate', "None"),
        }, index=[0])

        with pd.option_context("max_colwidth", 1000):
            print(df.to_latex(index=False))

        for key, value in model['params'].items():
            print("Param name:" + str(key) + " Value: " + str(value) + "\n")

    def showMetrics(self, level=None, targetNames=None):
        self.__isAutoMLSetup()

        print(self.autoML.sprint_statistics())
        print("Accuracy score", sklearn.metrics.accuracy_score(self.y_test, self.predictions))

        if level >= 1 or level == -1:
            print("\nClassification Report: \n",
                  sklearn.metrics.classification_report(self.y_test, self.predictions, target_names=targetNames))

        if level >= 2 or level == -1:
            print("\nConfusion Matrix: \n", sklearn.metrics.confusion_matrix(self.y_test, self.predictions))

        if level >= 3 or level == -1:
            print("Error Rate: {:0.5f}%".format(
                100 * (1 - sklearn.metrics.accuracy_score(self.y_test, self.predictions))))

        if level >= 4 or level == -1:
            print("Balanced Error Rate: {:0.5f}%".format(
                self.__balancedErrorRate(sklearn.metrics.confusion_matrix(self.y_test, self.predictions))))

    def __balancedErrorRate(self, confusionMatrix):
        classesAcc = []

        for i in range(len(confusionMatrix)):
            for j in range(len(confusionMatrix[0])):
                if i == j:
                    classesAcc.append(confusionMatrix[i][j] / confusionMatrix[i].sum())
        return 100 * (1 - numpy.mean(classesAcc))

    # UTILS

    def __isAutoMLSetup(self):
        if not self.autoML or [x for x in (self.X_train, self.y_train, self.X_test, self.y_test) if x is None]:
            raise ValueError("If the autoML pipeline has not been configured yet or an error has occurred, "
                             "please process the setup() method.")
