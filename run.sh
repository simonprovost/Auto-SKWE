#!/bin/bash

echo "ObesityDataset_raw_and_data_synthetic analysis..."

echo "First experiment seed 42 (one hour):"
python3 main_obesity_dataset.py "./params/params_seed_42_one_hour.params" > ./outputAutoML/ObesityDataset_raw_and_data_synthetic/one_hour_experiment_seed_42.log

echo "Second experiment seed 21 (one hour):"
python3 main_obesity_dataset.py "./params/params_seed_21_one_hour.params" > ./outputAutoML/ObesityDataset_raw_and_data_synthetic/one_hour_experiment_seed_21.log

echo "Third experiment seed 11 (one hour):"
python3 main_obesity_dataset.py "./params/params_seed_11_one_hour.params" > ./outputAutoML/ObesityDataset_raw_and_data_synthetic/one_hour_experiment_seed_11.log


echo "diabetic-retinopathy-Debrecen analysis..."

echo "First experiment seed 42 (one hour):"
python3 main_diabetes_dataset.py "./params/params_seed_42_one_hour.params" > ./outputAutoML/diabetic-retinopathy-Debrecen/one_hour_experiment_seed_42.log

echo "Second experiment seed 21 (one hour):"
python3 main_diabetes_dataset.py "./params/params_seed_21_one_hour.params" > ./outputAutoML/diabetic-retinopathy-Debrecen/one_hour_experiment_seed_21.log

echo "Third experiment seed 11 (one hour):"
python3 main_diabetes_dataset.py "./params/params_seed_11_one_hour.params" > ./outputAutoML/diabetic-retinopathy-Debrecen/one_hour_experiment_seed_11.log



echo "Thoracic-Surgery-binary-survival analysis..."

echo "First experiment seed 42 (one hour):"
python3 main_thoracic_surgery_datasets.py "./params/params_seed_42_one_hour.params" > ./outputAutoML/Thoracic-Surgery-binary-survival/one_hour_experiment_seed_42.log

echo "Second experiment seed 21 (one hour):"
python3 main_thoracic_surgery_datasets.py "./params/params_seed_21_one_hour.params" > ./outputAutoML/Thoracic-Surgery-binary-survival/one_hour_experiment_seed_21.log

echo "Third experiment seed 11 (one hour):"
python3 main_thoracic_surgery_datasets.py "./params/params_seed_11_one_hour.params" > ./outputAutoML/Thoracic-Surgery-binary-survival/one_hour_experiment_seed_11.log

