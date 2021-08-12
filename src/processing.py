import pickle
from collections import Counter

import autosklearn.classification
import numpy
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.datasets
import sklearn.model_selection
import sklearn.model_selection
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import roc_auc_score, recall_score, f1_score

from src.utils import setSeedEnvironement


### //TODO Disclaimer: Uncoment only if you are using Jupyter.
# import PipelineProfiler


class Processing:
    def __init__(self, inputData, classLabelData, entireDataset, datasetName,
                 tmp_folder="./outputRuns/config_log_files/autosklearn_run",
                 output_folder="./outputRuns/prediction_optional_tests/autosklearn_run", outputDir="./outputRuns",
                 **kwargs):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.autoML = None
        self.predictions = None
        self.all_10_folds = None
        self.all_best_models = []
        self.ensemble_results = []

        self.entireDataset = entireDataset
        self.input = inputData
        self.classLabel = classLabelData

        self.params = kwargs
        self.datasetName = datasetName or "N/A"

    def setup(self, personalisedInput=None, personalisedOutput=None):
        setSeedEnvironement(self.params.get('seed', None))

        input = self.input if personalisedInput is None else personalisedInput
        output = self.classLabel if personalisedOutput is None else personalisedOutput

        self.X_train, self.X_test, self.y_train, self.y_test = \
            sklearn.model_selection.train_test_split(input, output, random_state=None)

        self.autoML = autosklearn.classification.AutoSklearnClassifier(
            **self.params,
        )

    def fit_predict(self, showModels=False):
        self.__is_auto_ml_setup()

        self.autoML.fit(X=self.X_train, y=self.y_train, dataset_name=self.datasetName)

        if showModels:
            print(self.autoML.show_models())
        self.predictions = self.autoML.predict(X=self.X_test)
        self.predictions = self.predictions.astype(str).astype(int)

    # CROSS VALIDATION

    def k_folds_split(self, k_folds):
        if self.entireDataset is None:
            raise ValueError("Entire dataset must be passed when the creation of an instance of this class.")
        data = self.entireDataset.copy()
        data = data.sample(frac=1)

        self.all_10_folds = np.array_split(data, k_folds)

    def reSamplingSMOTE(self, training_set, inputData, outputData, display=False):
        counter = Counter(outputData)
        categoricalFeatures = []
        for x in training_set[0]:
            if training_set[0][x].dtype.str == '|O08':
                categoricalFeatures.append(True)
            else:
                categoricalFeatures.append(False)

        oversample = SMOTENC(categorical_features=categoricalFeatures, random_state=0)
        counter = Counter(outputData)
        if display:
            print(counter)
        inputResampled, outputResampled = oversample.fit_resample(inputData, outputData)
        counter = Counter(outputResampled)
        if display:
            print("#######")
            print(counter)
        return inputResampled, outputResampled

    def cross_validation_process(self, model_output_path=None, reSampling=False):
        if not self.all_10_folds:
            raise ValueError(
                "Please split your dataset with the k_folds_split method before going through the cross-validation method.")
        for idx, fold in enumerate(self.all_10_folds):
            training_set = self.all_10_folds.copy()
            training_set.pop(idx)
            test_set = fold

            input, output = self.__get_inputs_outputs_from_folds(training_set)
            if reSampling:
                input, output = self.reSamplingSMOTE(training_set, input, output, display=True)

            self.setup(personalisedInput=input, personalisedOutput=output)
            self.fit_predict()

            filepath = model_output_path + 'classifier_' + self.datasetName + '_' + str(idx) + '.pkl'
            self.save_model(path=filepath)
            loaded_classifier = self.load_model(path=filepath)
            self.all_best_models.append(loaded_classifier)

            crossValUnseenDataOutput = self.__predict_unseen_data_cross_validation_process(test_set, loaded_classifier)
            self.ensemble_results.append(crossValUnseenDataOutput[0])

    def show_latex_cross_validation_process(self, display=False):
        for model in self.ensemble_results:
            self.show_latex_table(model)
            if display:
                print(model)

    def __predict_unseen_data_cross_validation_process(self, test_set, classifier):
        InputUnseenData = test_set.iloc[:, :-1]
        OutputUnseenData = test_set.iloc[:, -1]
        yPredUnseenData = classifier.predict(InputUnseenData)

        return self.get_best_models(numberOfModelToGet=0, display=False, personalisedModel=classifier,
                                    personalisedXTest=InputUnseenData, personalisedYTest=OutputUnseenData,
                                    personalisedPredictions=yPredUnseenData)

    def __get_inputs_outputs_from_folds(self, data):
        results = pd.concat(data)
        return results.iloc[:, :-1], results.iloc[:, -1]

    # METRICS

    ### //TODO Disclaimer: Uncoment only if you are using Jupyter.
    # def showVisualisationPipeline(self):
    #    profiler_data = PipelineProfiler.import_autosklearn(self.autoML)
    #    PipelineProfiler.plot_pipeline_matrix(profiler_data)

    def get_best_models(self, numberOfModelToGet=0, display=True, metrics=True, personalisedModel=None,
                      personalisedXTest=None, personalisedYTest=None, personalisedPredictions=None):
        models = []
        autoML = self.autoML if personalisedModel is None else personalisedModel
        X_test = self.X_test if personalisedXTest is None else personalisedXTest
        y_test = self.y_test if personalisedYTest is None else personalisedYTest
        predictions = self.predictions if personalisedPredictions is None else personalisedPredictions

        losses_and_configurations = [
            (run_value.cost, run_key.config_id, run_value.time,)
            for run_key, run_value in autoML.automl_.runhistory_.data.items()
        ]
        losses_and_configurations.sort()

        if display:
            print("Lowest loss:", losses_and_configurations[0][0])

        for idx, x in enumerate(losses_and_configurations):
            if idx > numberOfModelToGet != -1:
                break
            for key, value in autoML.automl_.runhistory_.ids_config.items():
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
        if numberOfModelToGet == 0 and metrics:
            multi_class_string = 'ovr' if self.__is_multi_class_problem() else 'raise'
            y_prob = autoML.predict_proba(X_test) if self.__is_multi_class_problem() else predictions.tolist()

            models[0]['recall_score'] = recall_score(y_test, predictions, average="macro")
            models[0]['f1_score_macro'] = f1_score(y_test, predictions, average="macro")
            models[0]['f1_score_micro'] = f1_score(y_test, predictions, average="micro")
            # TODO add ovo for imbalanced dataset.
            if multi_class_string == 'ovr':
                models[0]['auroc'] = roc_auc_score(y_test, y_prob, multi_class=multi_class_string)
            else:
                models[0]['auroc'] = roc_auc_score(y_test, y_prob)

            models[0]['accuracy'] = sklearn.metrics.accuracy_score(y_test, predictions)
            models[0]['error_rate'] = 100 * (1 - sklearn.metrics.accuracy_score(y_test, predictions))

        return models

    def show_latex_table(self, model):
        hyperParams = [(key, value) for key, value in model['params'].items() if key != 'classifier:__choice__'],

        df = pd.DataFrame({
            "Dataset name": self.datasetName,
            "Classifier": model['params'].get('classifier:__choice__', "None"),
            "Search Time limit": model.get('experiment_time', None),
            "Algorithm time run (s)": model.get('time(s)', "None"),
            "Seed": self.params.get("seed", "N/A"),
            "Score accuracy": model.get('accuracy', "None"),
            "Error rate": model.get('error_rate', "None"),
            "recall_score": model.get('recall_score', "None"),
            "f1 score macro": model.get('f1_score_macro', "None"),
            "f1 score micro": model.get('f1_score_micro', "None"),
            "auroc": model.get('auroc', "None")
        }, index=[0])

        with pd.option_context("max_colwidth", 1000):
            print(df.to_latex(index=False))

        for key, value in model['params'].items():
            print(str(key) + " & " + str(value) + " \\\\ \\hline \n")

    def show_metrics(self, level=None, targetNames=None):
        self.__is_auto_ml_setup()

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
                self.__balanced_error_rate(sklearn.metrics.confusion_matrix(self.y_test, self.predictions))))

    def __balanced_error_rate(self, confusionMatrix):
        classesAcc = []

        for i in range(len(confusionMatrix)):
            for j in range(len(confusionMatrix[0])):
                if i == j:
                    classesAcc.append(confusionMatrix[i][j] / confusionMatrix[i].sum())
        return 100 * (1 - numpy.mean(classesAcc))

    # UTILS

    def __is_auto_ml_setup(self):
        if not self.autoML or [x for x in (self.X_train, self.y_train, self.X_test, self.y_test) if x is None]:
            raise ValueError("If the autoML pipeline has not been configured yet or an error has occurred, "
                             "please process the setup() method.")

    def save_model(self, path, rights="wb"):
        with open(path, rights) as f:
            pickle.dump(self.autoML, f)

    def load_model(self, path, rights="rb"):
        with open(path, rights) as f:
            loaded_classifier = pickle.load(f)
        return loaded_classifier

    def __is_multi_class_problem(self):
        return len(self.__get_unique_values(self.classLabel)) > 2

    def __get_unique_values(self, values):

        uniqueAttr = set(values)

        return [x for x in uniqueAttr]
