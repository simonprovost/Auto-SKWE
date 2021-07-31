#!/bin/bash

#################

# Check for imbalanced datasets.

#################

echo "Generate latex table from Datasets characteristics {imbalanced dataset check}"

python3 check_imbalanced_datasets.py > ./outputAutoML/datasets_characteristics.log

#################

# DATASET NAME: ObesityDataset_raw_and_data_synthetic

#################

echo "Cross validation ObesityDataset_raw_and_data_synthetic analysis..."

echo "First experiment seed 85 (one hour):"
python3 main_obesity_dataset.py "./params/params_seed_85_one_hour.params" > ./outputAutoML/ObesityDataset_raw_and_data_synthetic/one_hour_experiment_seed_85.log cv

#################

# DATASET NAME: diabetic-retinopathy-Debrecen

#################
echo "Cross validation diabetic-retinopathy-Debrecen analysis..."

echo "First experiment seed 85 (one hour):"
python3 main_diabetes_dataset.py "./params/params_seed_85_one_hour.params" > ./outputAutoML/diabetic-retinopathy-Debrecen/one_hour_experiment_seed_85.log cv

#################

# DATASET NAME: Thoracic-Surgery-binary-survival

#################

echo "Cross validation Thoracic-Surgery-binary-survival analysis..."

echo "First experiment seed 85 (one hour):"
python3 main_thoracic_surgery_datasets.py "./params/params_seed_85_one_hour.params" > ./outputAutoML/Thoracic-Surgery-binary-survival/one_hour_experiment_seed_85.log cv

#################

# DATASET NAME: Breast-cancer-Coimbra

#################

echo "Cross validation Breast-cancer-Coimbra analysis..."

echo "First experiment seed 85 (one hour):"
python3 main_breast_cancer_dataset.py "./params/params_seed_85_one_hour.params" > ./outputAutoML/Breast-cancer-Coimbra/one_hour_experiment_seed_85.log cv

#################

# DATASET NAME: Heart-failure

#################

echo "Cross validation Heart-failure analysis..."

echo "First experiment seed 85 (one hour):"
python3 main_hearth_failure_dataset.py "./params/params_seed_85_one_hour.params" > ./outputAutoML/Heart-failure-clinical-records/one_hour_experiment_seed_85.log cv

