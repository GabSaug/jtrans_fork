# @input: selected_pairs file + features (from idb ?)

# Preprocess data
# Generate the preprocessed (extract) pickle files
# -> ./datautils/extract
cd ./datautils
./setup_preprocess.sh

# Default arguments should be correctly updated
#
python3 eval_save.py

# @output: csv file with each line: sim score
