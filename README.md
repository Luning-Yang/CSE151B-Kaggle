# CSE151B-Kaggle

Run the following commond to download the train and test set. It will create a folder called *data* and move all downloaded datasets to that folder. 
```
bash load_data.sh
```

Run the following command to perform the feature engineering. The processed dataset called *processed_train.csv* will also be move to *data*. 
```
python feature-engineering.py
```

All the models are saved in src/models.py

Models' corresponding weights are saved at ./saved_model_x.pt, where x can be mlp1, mlp2 or CNN.

Currently CNN has the best performance, the other codes are saved in workplace.ipynb with detailed instructions. 
