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
      java -cp ../autoweka/temps/autoweka.jar:/home/simonprovost34430/weka-3-8-5/weka.jar weka.classifiers.meta.AutoWEKAClassifier -t $FILE_TRAIN -T $FILE_TEST -timeLimit 60 -seed 85 -memLimit 5000 -nBestConfigs 50 > "$2$i.log"
    done
  fi
}



#################

# DATASET NAME:

#################

echo "Cross validation analysis..."

echo "First experiment seed 85 (one hour):"
python3 ./main_analysis/main_BCC_and_SCC_peripheral_margin_deep_prediction_analysis.py "./params/params_seed_85_one_hour.params" > ./outputAutoML/BCC_SCC_deep/one_hour_experiment_seed_85.log $1
wekaCV $1 "./outputAutoML/BCC_SCC_deep/one_hour_experiment_seed_85_"


#################

# DATASET NAME:

#################

echo "Cross validation analysis..."

echo "First experiment seed 85 (one hour):"
python3 ./main_analysis/main_BCC_and_SCC_peripheral_margin_raw_prediction_analysis.py "./params/params_seed_85_one_hour.params" > ./outputAutoML/BCC_SCC_raw/one_hour_experiment_seed_85.log $1
wekaCV $1 "./outputAutoML/BCC_SCC_raw/one_hour_experiment_seed_85_"


#################

# DATASET NAME:

#################

echo "Cross validation analysis..."

echo "First experiment seed 85 (one hour):"
python3 ./main_analysis/main_BCC_peripheral_margin_deep_prediction_analysis.py "./params/params_seed_85_one_hour.params" > ./outputAutoML/BCC_deep/one_hour_experiment_seed_85.log $1
wekaCV $1 "./outputAutoML/BCC_deep/one_hour_experiment_seed_85_"

#################

# DATASET NAME:

#################

echo "Cross validation analysis..."

echo "First experiment seed 85 (one hour):"
python3 ./main_analysis/main_BCC_peripheral_margin_raw_prediction_analysis.py "./params/params_seed_85_one_hour.params" > ./outputAutoML/BCC_raw/one_hour_experiment_seed_85.log $1
wekaCV $1 "./outputAutoML/BCC_raw/one_hour_experiment_seed_85_"

#################

# DATASET NAME:

#################

echo "Cross validation analysis..."

echo "First experiment seed 85 (one hour):"
python3 ./main_analysis/main_SCC_peripheral_margin_deep_prediction_analysis.py "./params/params_seed_85_one_hour.params" > ./outputAutoML/SCC_deep/one_hour_experiment_seed_85.log $1
wekaCV $1 "./outputAutoML/SCC_deep/one_hour_experiment_seed_85_"

#################

# DATASET NAME:

#################

echo "Cross validation analysis..."

echo "First experiment seed 85 (one hour):"
python3 ./main_analysis/main_SCC_peripheral_margin_raw_prediction_analysis.py "./params/params_seed_85_one_hour.params" > ./outputAutoML/SCC_raw/one_hour_experiment_seed_85.log $1
wekaCV $1 "./outputAutoML/SCC_raw/one_hour_experiment_seed_85_"
