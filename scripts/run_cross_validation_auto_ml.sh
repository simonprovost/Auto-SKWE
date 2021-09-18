#!/bin/bash

wekaCV () {
  if [[ "$1" == "weka" ]]; then
    for i in {0..9}
    do
      FILE_TRAIN="./training_set_fold$i.arff"
      FILE_TEST="./test_set_fold$i.arff"
      if test -f "$FILE_TRAIN"; then
          echo "$FILE_TRAIN exists."
      else
        continue
      fi
      if test -f "$FILE_TEST"; then
          echo "$FILE_TEST exists."
      else
        continue
      fi
      java -cp ../autoweka/temps/autoweka.jar:/home/simonprovost34430/weka-3-8-5/weka.jar weka.classifiers.meta.AutoWEKAClassifier -t $FILE_TRAIN -T $FILE_TEST -timeLimit 1 -seed 85 -memLimit 5000 -nBestConfigs 50 > "$2$i.log"
    done
  fi
}



#################

# Check for imbalanced datasets.

#################

wekaCV weka "./outputAutoML/NHS-dataset/one_hour_experiment_seed_85_"


echo "Generate latex table from Datasets characteristics {imbalanced dataset check}"

#python3 check_imbalanced_datasets.py > ./outputAutoML/datasets_characteristics.log

#################

# DATASET NAME: ObesityDataset_raw_and_data_synthetic

#################

echo "Cross validation ObesityDataset_raw_and_data_synthetic analysis..."

echo "First experiment seed 85 (one hour):"
python3 ./main_analysis/main_obesity_dataset.py "./params/params_seed_85_one_hour.params" > ./outputAutoML/ObesityDataset_raw_and_data_synthetic/one_hour_experiment_seed_85.log cv $1
wekaCV $1 "./outputAutoML/ObesityDataset_raw_and_data_synthetic/one_hour_experiment_seed_85_"


#################

# DATASET NAME: diabetic-retinopathy-Debrecen

#################
echo "Cross validation diabetic-retinopathy-Debrecen analysis..."

echo "First experiment seed 85 (one hour):"
python3 ./main_analysis/main_diabetes_dataset.py "./params/params_seed_85_one_hour.params" > ./outputAutoML/diabetic-retinopathy-Debrecen/one_hour_experiment_seed_85.log cv $1
wekaCV $1 "./outputAutoML/diabetic-retinopathy-Debrecen/one_hour_experiment_seed_85_"

#################

# DATASET NAME: Thoracic-Surgery-binary-survival

#################

echo "Cross validation Thoracic-Surgery-binary-survival analysis..."

echo "First experiment seed 85 (one hour):"
python3 ./main_analysis/main_thoracic_surgery_datasets.py "./params/params_seed_85_one_hour.params" > ./outputAutoML/Thoracic-Surgery-binary-survival/one_hour_experiment_seed_85.log cv $1
wekaCV $1 "./outputAutoML/Thoracic-Surgery-binary-survival/one_hour_experiment_seed_85_"

#################

# DATASET NAME: Breast-cancer-Coimbra

#################

echo "Cross validation Breast-cancer-Coimbra analysis..."

echo "First experiment seed 85 (one hour):"
python3 ./main_analysis/main_breast_cancer_dataset.py "./params/params_seed_85_one_hour.params" > ./outputAutoML/Breast-cancer-Coimbra/one_hour_experiment_seed_85.log cv $1
wekaCV $1 "./outputAutoML/Breast-cancer-Coimbra/one_hour_experiment_seed_85_"

#################

# DATASET NAME: Heart-failure

#################

echo "Cross validation Heart-failure analysis..."

echo "First experiment seed 85 (one hour):"
python3 ./main_analysis/main_hearth_failure_dataset.py "./params/params_seed_85_one_hour.params" > ./outputAutoML/Heart-failure-clinical-records/one_hour_experiment_seed_85.log cv $1
wekaCV $1 "./outputAutoML/Heart-failure-clinical-records/one_hour_experiment_seed_85_"

