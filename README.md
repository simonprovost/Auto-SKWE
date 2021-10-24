# Auto-SKWE (Auto-Sklearn vs. Auto-WEKA) 🔬

<p align="justify">
    The following repository contains an analysis of medical datasets using AUTO-WEKA and AUTO-SKLEARN. The purpose of
    this study is to determine whether Auto-WEKA or Auto-Sklearn produces more accurate results while maintaining a
    reproducible model on medical datasets.
    <br /><br />
    The following sections will discuss the architecture of the project, how to run it, and provide some examples on how to use the whole program made for the experiments.
</p>

**Auto-WEKA**: https://github.com/automl/autoweka \
**Auto-Sklearn**: https://github.com/automl/auto-sklearn

## 💡 Final Dissertation

[Click here to read the final dissertation!](https://github.com/simonprovost/AutoML/blob/main/report/Master_Thesis_Provost_Simon_sgp28.pdf)

**Final grade: 75 out of 100. Which enabled me to earn a distinction degree for my master's degree.**

## Architecture of the repository
```
./
├── README.md
├── datasets
│   ├── breast-cancer
│   ├── diabetic-retinopathy-debrecen
│   ├── heart-failure
│   ├── nhs-data [PRIVATE]
│   ├── obesity-data
│   └── thoracic-data
├── main_analysis
├── outputAutoML
├── outputCrossValidationAutoML
├── params
├── report
├── scripts
└── src
```

### Root:

The root directory of the repository contains one crucial component: the scripts folder, which contains bash scripts that enable you to conduct the analysis you wish to run.

### Datasets:

The datasets folder contains all datasets that are currently available for public usage and being analysed, as well as their related feature definitions (.docx /.pdf files). Occasionally, the dataset is in csv/arff or both formats; csv is advantageous for Auto-Sklearn, while arff is advantageous for Auto-Weka, despite the fact that arff works with Auto-Sklearn throughout this program's implementation.

### Params:

The params directory contains all parameter files that are used to feed the AutoML pipeline. You are welcome to utilise any of the existing ones or to contribute new ones to the folder. Rather of altering '*.py', we add some configuration files containing the settings for the AutoML pipeline that will be fed.
The files function as a `KEY VALUE` like a Python dictionary. If the `VALUE` is also a dictionary, write it as a
Python dictionary string with the following syntax: `{'fold'=5}` without whitespace, and everything should be properly
parsed to feed the AutoML correctly.

### SRC (sources):

All Python classes and files required to run the project are contained in the SRC directory. Additionally to the 'Process' class, there should be a 'PreProcess' class and a 'utils' file holding all of the project's utility tools.

### OutputAutoML:

The outputAutoML folder will contain the pertinent information gathered from the AutoML pipeline, such as a latex table with the results that will be copied and pasted into some '.log' files by the script you selected.

Nota bene: At the moment, the output formats for Auto-Sklearn and Auto-WEKA are distinct due to the frameworks' separation. However, the result will eventually be generalised to be identical later on the year.

### OutputCrossValidationAutoML:

Once the Auto-ML programme has fitted the models, they are kept in this directory. 
Nota bene: No model is currently stored with Auto-analysis Weka's because the framework does not support it.

## How to Install / Run the project

### 🎼 Install

#### Step (1): Before all, you have to download and install the following Framework/Library:

* Auto-sklearn for python: https://automl.github.io/auto-sklearn/master/installation.html _(v. auto-sklearn 0.13.0)_.
* Scikit-Learn _(see requirement.txt/step(2) for the version+installation)_.
* Auto-weka : http://www.cs.ubc.ca/labs/beta/Projects/autoweka/manual.pdf _(v. 2.6.3)_.
* Weka : https://waikato.github.io/weka-wiki/downloading_weka/ _(v. weka-3-8-5)_.

NB:

> For MacOSX users: we have concocted a list of steps to follow in order to have auto-sklearn working on your machine: > https://gist.github.com/simonprovost/051952533680026b67fa58c3552b8a7b 

> Unfortunately for MacOSX users we did not found a way to make auto-weka being able to be run on OSX (i.e: see this issue: > https://github.com/automl/autoweka/issues/95). Fortunately, it does work on Ubuntu 16/18 LTS. _Note: We use a Google Cloud Compute Engine to make everything work well._

#### Step (2): All the remaining external packages to install can be downloaded as follows:

```shell
 pip install -r requirements.txt # or pip install -r requirements.txt
```
_Note: For those who use python3/pip3, add the following subtitute to pip --> pip3_

### ⚙️ Run

As soon as everything's downloaded, you could run the project as follows without modification:

Searching of the best classifier given all the datasets available on the entire dataset using Auto-Sklearn:
```shell
./run_simple_auto_ml.sh
```

Searching of the best classifier given all the datasets available with a 10-fold cross validation using Auto-Sklearn/Auto-Weka:

Auto-Sklearn:
```shell
./run_cross_validation_auto_ml.sh
```

Auto-Weka:
```shell
./run_cross_validation_auto_ml.sh weka
```

NHS Analysis: **PRIVATE. Not accessible for public.**

If you want to provide new datasets/params files and their appropriate `main_X_datasets.py`, feel free to add
a new line within the appropriate script and or comment some. Every output will be moved to where you asked to be outputted in the appropriate script.

#### 🪛 How to add a new line to the a script

```
# [python_interpreter] [main_X_dataset.py you would like to run] [params file to use to feed into the AutoML Pipeline] [OPTIONAL: a redirection to somewhere] [OPTIONAL: use weka or not in adding "weka" as a parameter of the running command]
python main_diabetes_dataset.py "./params/params_seed_42_one_hour.params" > ./outputAutoML/diabetic-retinopathy-Debrecen/one_hour_experiment_seed_42.log $1 #the argument 1 could be "weka" and while you run the script you add ./myScript weka and it will run an analysis of Auto-WEKA instead of Auto-Sklearn
```
_Note: For those who use python3/pip3, add the following subtitute to python --> python3_


### 🔖 A few example of how to use the available methods through all classes

_Note: Additional material will be released in the future (proper documentation) at which point this section will be removed. If you have any questions or encounter an issue, I will assist you via Github Issue._

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

# do not forget to autoML.setup().
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


#### [5] How to use the Process class cross validation method ?

````python
### The following instantiate the class with as parameters: the Input/ClassLabel data you would have obtained through the PreProcess or even before (see examples [0] [1] [2]).
### The params parameter is one of the most important is it is the params that is going to be feed to the AutoML pipeline. See example [3].
### The datasetName is simply to get a better output at the very end.
autoML = Processing(inputData=X, classLabelData=y, datasetName="name_of_the_dataset_for_the_output", **params)

# Do not forget to autoML.setup().
### Firt and foremost the k_folds split will split the dataset into k folds.
autoML.k_folds_split(k_folds=10)

### the cross validation process will take care of fitting 10 models using the cross-validation process chosen (i.e: 9 folds for the training one for the testingà).
### Will also do a resample if request (see parameters). Which will at the end either test with Scikit Learn for Auto Sklearn the fitted model or
### run auto-weka with the folds splitted and results the outcome of Auto-Weka.
autoML.cross_validation_process("./outputCrossValidationAutoML/name_of_the_dataset_for_the_output/")
````

## 🤝 Contributions

Anyone can contribute anything that can be used to detect irregularities or patterns in movement data *(sleeping humans)*.



