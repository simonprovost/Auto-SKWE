#!/bin/bash

#################

# Check for imbalanced datasets.

#################

echo "Generate latex table from Datasets characteristics {imbalanced dataset check}"

python3 check_imbalanced_datasets.py > ./outputAutoML/datasets_characteristics.log

#################

# DATASET NAME: ObesityDataset_raw_and_data_synthetic

#################

echo "ObesityDataset_raw_and_data_synthetic analysis..."

echo "First experiment seed 42 (one hour):"
python3 main_obesity_dataset.py "./params/params_seed_42_one_hour.params" > ./outputAutoML/ObesityDataset_raw_and_data_synthetic/one_hour_experiment_seed_42.log

echo "Second experiment seed 21 (one hour):"
python3 main_obesity_dataset.py "./params/params_seed_21_one_hour.params" > ./outputAutoML/ObesityDataset_raw_and_data_synthetic/one_hour_experiment_seed_21.log

echo "Third experiment seed 11 (one hour):"
python3 main_obesity_dataset.py "./params/params_seed_11_one_hour.params" > ./outputAutoML/ObesityDataset_raw_and_data_synthetic/one_hour_experiment_seed_11.log

echo "Fourth experiment seed 2 (one hour):"
python3 main_obesity_dataset.py "./params/params_seed_2_one_hour.params" > ./outputAutoML/ObesityDataset_raw_and_data_synthetic/one_hour_experiment_seed_2.log

echo "Fifth experiment seed 85 (one hour):"
python3 main_obesity_dataset.py "./params/params_seed_85_one_hour.params" > ./outputAutoML/ObesityDataset_raw_and_data_synthetic/one_hour_experiment_seed_85.log

#################

# DATASET NAME: diabetic-retinopathy-Debrecen

#################
echo "diabetic-retinopathy-Debrecen analysis..."

echo "First experiment seed 42 (one hour):"
python3 main_diabetes_dataset.py "./params/params_seed_42_one_hour.params" > ./outputAutoML/diabetic-retinopathy-Debrecen/one_hour_experiment_seed_42.log

echo "Second experiment seed 21 (one hour):"
python3 main_diabetes_dataset.py "./params/params_seed_21_one_hour.params" > ./outputAutoML/diabetic-retinopathy-Debrecen/one_hour_experiment_seed_21.log

echo "Third experiment seed 11 (one hour):"
python3 main_diabetes_dataset.py "./params/params_seed_11_one_hour.params" > ./outputAutoML/diabetic-retinopathy-Debrecen/one_hour_experiment_seed_11.log

echo "Second experiment seed 2 (one hour):"
python3 main_diabetes_dataset.py "./params/params_seed_2_one_hour.params" > ./outputAutoML/diabetic-retinopathy-Debrecen/one_hour_experiment_seed_2.log

echo "Third experiment seed 85 (one hour):"
python3 main_diabetes_dataset.py "./params/params_seed_85_one_hour.params" > ./outputAutoML/diabetic-retinopathy-Debrecen/one_hour_experiment_seed_85.log

#################

# DATASET NAME: Thoracic-Surgery-binary-survival

#################

echo "Thoracic-Surgery-binary-survival analysis..."

echo "First experiment seed 42 (one hour):"
python3 main_thoracic_surgery_datasets.py "./params/params_seed_42_one_hour.params" > ./outputAutoML/Thoracic-Surgery-binary-survival/one_hour_experiment_seed_42.log

echo "Second experiment seed 21 (one hour):"
python3 main_thoracic_surgery_datasets.py "./params/params_seed_21_one_hour.params" > ./outputAutoML/Thoracic-Surgery-binary-survival/one_hour_experiment_seed_21.log

echo "Third experiment seed 11 (one hour):"
python3 main_thoracic_surgery_datasets.py "./params/params_seed_11_one_hour.params" > ./outputAutoML/Thoracic-Surgery-binary-survival/one_hour_experiment_seed_11.log

echo "Second experiment seed 2 (one hour):"
python3 main_thoracic_surgery_datasets.py "./params/params_seed_2_one_hour.params" > ./outputAutoML/Thoracic-Surgery-binary-survival/one_hour_experiment_seed_2.log

echo "Third experiment seed 85 (one hour):"
python3 main_thoracic_surgery_datasets.py "./params/params_seed_85_one_hour.params" > ./outputAutoML/Thoracic-Surgery-binary-survival/one_hour_experiment_seed_85.log

#################

# DATASET NAME: Breast-cancer-Coimbra

#################

echo "Breast-cancer-Coimbra analysis..."

echo "First experiment seed 42 (one hour):"
python3 main_breast_cancer_dataset.py "./params/params_seed_42_one_hour.params" > ./outputAutoML/Breast-cancer-Coimbra/one_hour_experiment_seed_42.log

echo "Second experiment seed 21 (one hour):"
python3 main_breast_cancer_dataset.py "./params/params_seed_21_one_hour.params" > ./outputAutoML/Breast-cancer-Coimbra/one_hour_experiment_seed_21.log

echo "Third experiment seed 11 (one hour):"
python3 main_breast_cancer_dataset.py "./params/params_seed_11_one_hour.params" > ./outputAutoML/Breast-cancer-Coimbra/one_hour_experiment_seed_11.log

echo "Second experiment seed 2 (one hour):"
python3 main_breast_cancer_dataset.py "./params/params_seed_2_one_hour.params" > ./outputAutoML/Breast-cancer-Coimbra/one_hour_experiment_seed_2.log

echo "Third experiment seed 85 (one hour):"
python3 main_breast_cancer_dataset.py "./params/params_seed_85_one_hour.params" > ./outputAutoML/Breast-cancer-Coimbra/one_hour_experiment_seed_85.log

#################

# DATASET NAME: Heart-failure

#################

echo "Heart-failure analysis..."

echo "First experiment seed 42 (one hour):"
python3 main_hearth_failure_dataset.py "./params/params_seed_42_one_hour.params" > ./outputAutoML/Heart-failure-clinical-records/one_hour_experiment_seed_42.log

echo "Second experiment seed 21 (one hour):"
python3 main_hearth_failure_dataset.py "./params/params_seed_21_one_hour.params" > ./outputAutoML/Heart-failure-clinical-records/one_hour_experiment_seed_21.log

echo "Third experiment seed 11 (one hour):"
python3 main_hearth_failure_dataset.py "./params/params_seed_11_one_hour.params" > ./outputAutoML/Heart-failure-clinical-records/one_hour_experiment_seed_11.log

echo "Second experiment seed 2 (one hour):"
python3 main_hearth_failure_dataset.py "./params/params_seed_2_one_hour.params" > ./outputAutoML/Heart-failure-clinical-records/one_hour_experiment_seed_2.log

echo "Third experiment seed 85 (one hour):"
python3 main_hearth_failure_dataset.py "./params/params_seed_85_one_hour.params" > ./outputAutoML/Heart-failure-clinical-records/one_hour_experiment_seed_85.log

