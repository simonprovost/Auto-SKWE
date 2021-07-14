# AutoML (Auto-Sklearn & Auto-WEKA)

<p align="justify">
    The following repository contains an analysis of medical datasets using AUTO-WEKA and AUTO-SKLEARN. 
    At the moment, only AUTO-SKLEARN is used in the current version. The purpose of this study is to determine
    whether Auto-WEKA or Auto-Sklearn produces more accurate results while maintaining a reproducible model on medical datasets.
    <br /><br />
    The following sections will discuss the architecture of the project, how to run it, and provide some examples.
</p>

**Auto-WEKA**: https://github.com/automl/autoweka \
**Auto-Sklearn**: https://github.com/automl/auto-sklearn

## Architecture of the repository
```
./
├── datasets
│   ├── diabetic-retinopathy-debrecen
│   ├── obesity-data
│   └── thoracic-data
├── outputAutoML
│   ├── ObesityDataset_raw_and_data_synthetic
│   ├── Thoracic-Surgery-binary-survival
│   └── diabetic-retinopathy-Debrecen
├── outputRuns
│   ├── config_log_files
│   └── prediction_optional_tests
├── params
└── src

17 directories
```

### Root:

The repository's root directory contains two critical components: the *run.sh* script, which enables you to run the analysis
on all main.py files contained within the script and to log the results of the auto-ML pipeline to the appropriate location.
The three *main_X_dataset.py* files contain the pre-processing, and the AutoML pipeline, from opening and reading the dataset
to feed the data into the `PreProcess class` to ensure that the data meets the Auto-Sklearn requirements and running
the Auto-ML tools to extract relevant information thanks to the `Process class` to determine if the data has been properly classified.

### Datasets:

The datasets folder contains all of the datasets currently being used for analysis, along with their associated features
definitions (.docx /.pdf files).

### Params:

The params folder contains all of the parameter files used to feed the AutoML pipeline. You are welcome to use those
that are already available or to add new ones to the folder. Instead of modifying the `main.py`, we add some
configuration files that contain the parameters for the AutoML pipeline that will be fed.
The files function as a `KEY VALUE` like a Python dictionary. If the `VALUE` is also a dictionary, write it as a
Python dictionary string with the following syntax: `{'fold'=5}` without whitespace, and everything should be properly
parsed to feed the AutoML correctly.

### SRC (sources):

The SRC directory contains all the Python classes and files required to run the project. There should be a `PreProcess` class
in addition to a `Process` class, as well as an `utils.py` file containing all the project's utility tools.

### OutputAutoML:

The outputAutoML folder will contain the relevant information obtained from the AutoML pipeline, such as a latex table
about the results that are copied and pasted into some `.log` files by the `run.sh` script.

The popularized format of an output file is as follows:

```
[A Latex table showing the following results: Dataset name &        Classifier &  Search Time limit &  Algorithm time run (s) &  Seed &  Score accuracy &  Error rate]

[All the Hyper Parameters chosen by the AutoML Pipeline]
```


### OutputRuns:

The outputAutoML folder will contain all the AutoML outcomes (debug, info, warning, config, etc.) through all the runs
you would have made, annotated by the number of the run such as *auto-sklearn-run_x*  `x=1 or 2 and so on`


## How to Install / Run the project

### Install

Before all, you have to download and install auto-sklearn for python: https://automl.github.io/auto-sklearn/master/installation.html
\
\
For MacOSX users: we have concocted a list of steps to follow in order to have auto-sklearn working on your machine: https://gist.github.com/simonprovost/051952533680026b67fa58c3552b8a7b 

All the remaining external packages to install can be downloaded as follows:
```
 pip3 install -r requirements.txt # or pip install -r requirements.txt
```

### Run

As soon as everything's downloaded, you could run the project as follows without modification:

```
./run.sh 
```

If you want to provide new datasets/params files and their appropriate `main_X_datasets.py`, feel free to add
a new line within the `run.sh` script and or comment some. Every output will be moved to where you asked in the `run.sh` script,
and for those already there in the `OutputAutoML/` folder.

#### How to add a new line to the run.sh Script

```
# [python_interpreter] [main_X_dataset.py you would like to run] [params file to use to feed into the AutoML Pipeline] [OPTIONAL: a redirection to somewhere]
python3 main_diabetes_dataset.py "./params/params_seed_42_one_hour.params" > ./outputAutoML/diabetic-retinopathy-Debrecen/one_hour_experiment_seed_42.log
```

### Documentation

// In progress refers to the examples bellow for a few help to start. https://github.com/simonprovost/AutoML/projects/1#card-64929428 

### A few example of how to use the available methods through all classes


#### [0] How to use Arff file ?

```python
dataset = getDataFromArff(your_path_to_the_dataset)
input = dataset[1]
classData = dataset[2]
# dataset[0] is the entire dataset.
```

#### [1] How to use CSV file ?

```python
dataset = getDataFromCsv(your_path_to_the_dataset)
input = dataset[1]
classData = dataset[2]
# dataset[0] is the entire dataset.
```

#### [2] How to use the PreProcess class ?

```python
data = PreProcess(input, classData) #input and classData are the one previously instantiated (exemple 0/1)

### Adapt the dataset to fit the requirement of Scikit-Learn in terms of column data-type. Float64 into Pandas numeric and object into "category"
data.redefineColumnsType()

### Perform a SMOTE method on the output data.https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/ 
data.reSamplingSMOTE()

### Perform a one Hot encoding on the column you asked to on the input data directly. Either a column string or a number can be used see below.
data.oneHotEncoding(column="test") or data.oneHotEncoding(column=0, columnString=False)

### Perform a label encoder on all the input data columns name passed through the parameter. At least one is necessary.
data.labelEncoding(columns=["test", "test2"])

### Perform a change on the input data so that every value that meet one of the one provided through the `InitialArray` param. will be changed into its equivalent available through the `outputArray` param (list in order F goes to 0 and T to 1)
data.genericFeatureScaling(inputData=True, columnName="Test", initialArray=["F", "T"], outputArray=[0, 1])
### Can also be performed on output data via passing false ot the `InputData` param.
data.genericFeatureScaling(inputData=False, columnName="Class", initialArray=["F", "T"], outputArray=[0, 1])
```

#### [3] How to get parameters from a params file ?

````python
params = readParamsFile(argv[1])
print(params) #should be a dictionary `like`.
````

#### [4] How to use the Process class ?

````python
### The following instantiate the class with as parameters: the Input/ClassLabel data you would have obtained through the PreProcess or even before (see examples [0] [1] [2]).
### The params parameter is one of the most important is it is the params that is going to be feed to the AutoML pipeline. See example [3].
### The datasetName is simply to get a better output at the very end.
autoML = Processing(inputData=X, classLabelData=y, datasetName="name_of_the_dataset_for_the_output", **params)

### The setup is a necessary step before proceeding further. This method establishes the AutoML pipeline, splits the data, and also establishes the seed value across the whole run.
autoML.fit_predict()

### To obtain a list of "n" models or the best one use the following call. The display parameter is there to see if you want the program to output the hyper_parameters in the console.
models = autoML.get_best_models(numberOfModelToGet=0, display=False)

### The showLatexTable method allows us to get the Latex Table results from the best model obtained previously.
autoML.show_latex_table(models[0])

### The showMetrics methods provides evluation metrics according to the level of debugging you set in parameter. The target name is to let the classification_report() printing the one you provide instead of numeric one.
### Level 1 = Classification report (f1...).
### Level 2 = Confusion Matrix.
### Level 3 = Error Rate.
### Level -1 or 4 = Balanced Error Rate.
### Higher is the level more metrics is printing.
autoML.show_metrics(level=4, targetNames=["Class1", "Class2"])
````



