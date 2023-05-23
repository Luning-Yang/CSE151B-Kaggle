# folder="data"
mkdir -p data

dataset="ucsd-cse-151b-class-competition"

# download the dataset
kaggle competitions download -c "$dataset" -p data/

# unzip the dataset
zip_path="data/${dataset}.zip"
unzip "$zip_path" -d data/

# remove the zipped file
rm "$zip_path"
mv data/archive/* data/
rm -r data/archive
rm data/*.ipynb