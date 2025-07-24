#gnn @input: selected_pairs file + features (from idb ?)

cp ../../DBs/Dataset-Muaz/testing_Dataset-Muaz.csv ./testing_Dataset-Muaz.csv
cp ../../DBs/Dataset-Muaz/pairs/pairs_testing_Dataset-Muaz.csv ./pairs_testing_Dataset-Muaz.csv
# Preprocess data
# Generate the preprocessed (extract) pickle files
# -> ./datautils/extract
cd ./datautils
./setup_preprocess.sh
cd -

# Default arguments should be correctly updated
#
python3 eval_save.py

# @output: csv file with each line: sim score

python3 ./convert_results_jtrans.py

cp pairs_results_Dataset-Muaz_jtr.csv ../../Results/csv/
