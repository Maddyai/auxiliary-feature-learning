# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# df = pd.read_csv('../data/wine.data')
# df = pd.read_csv('../data/dataR2.csv')
# print(df.head())


# sns.distplot(df.values[1], kde=False)
# sns.distplot(df["Classification"], kde=False, bins=40)

# sns.swarmplot(x="Classification", y="BMI", data=df)
# sns.violinplot(x = "BMI", y="Age",hue = 'Classification', data = df)
# sns.regplot(x="BMI", y="Age",data=df)
# plt.show()
"""Old code
"""
from data import DataLoader
from model import AutoEncoder
from model import NeuralNetwork
from helper import lineplot


def run(data_obj, training_size):
    data_obj.split_dataset(training_size)
    data_obj.preprocess()

    ae_model = AutoEncoder(data_obj.x_train_scaled.shape[1],
                           training_size, data_obj.name)
    ae_model.train(data_obj.x_train_scaled, data_obj.x_val_scaled)

    x_train_encoded, x_val_encoded, x_test_encoded = ae_model.encoded_data(
        data_obj.x_train_scaled, data_obj.x_val_scaled, data_obj.x_test_scaled)

    # print((data_obj.y_train.shape[1]))
    # print(len(data_obj.y_train['target'].unique()))
    # print("-"*80)
    # print(data_obj.y_train)
    nn_model = NeuralNetwork(
        data_obj.x_train_scaled.shape[1], data_obj.y_train.shape[1],
        training_size, data_obj.name)
    nn_model.train(
        x_train_encoded, data_obj.y_train, x_val_encoded, data_obj.y_val)
    nn_model.evaluate(x_test_encoded, data_obj.y_test)

    data_obj.reset_scalar()

    return nn_model.result()


if __name__ == '__main__':

    from data import preprocess_dataR2, preprocess_wine, preprocess_audit 

    dataset_config = [
        ('breast_cancer', 'data/dataR2.csv', preprocess_dataR2),
        ('wine', 'data/wine.data', preprocess_wine),
        ('audit_risk', 'data/audit_data/audit_risk.csv', preprocess_audit),
        ('audit_trial', 'data/audit_data/trial.csv', preprocess_audit),
     ]

    for dataset_name, dataset_path, preprocess in dataset_config:
        data_obj = DataLoader(dataset_name, dataset_path, preprocess)

        val_result, test_result = [], []
        training_sizes = [.2, .3, .4, .5, .6, .7, 0.8, 0.9]

        for training_size in training_sizes:
            temp = run(data_obj, training_size)
            val_result.append(temp[0])
            test_result.append(temp[1])

        lineplot(training_sizes, val_result, test_result)
    # print(dataset)

"""
Notes:
Dataset analysis:
 - dataset distribution for different class
 - feature distribution
 - explority analysis
 ? correlation between feature (PCA)

Model building:
 -AutoEncoder:
  * standarde size for autoencoder layers and number of neurons
  * 
 - Neural netork:
  * single hidden layer
  * equal to the number of bottle neck layers neurons
  *   

TODO:
 - new dataset apapter


Done
- compare results
- diffferent size of dataset split to give to auto encoder 
- save the best model
- plot the accuracy for val and test
- get the features from the different endoded versions
- train the models
"""
