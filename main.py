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


def run(data_obj, training_size):
    data_obj.split_dataset(training_size)
    data_obj.preprocess()

    ae_model = AutoEncoder(data_obj.x_train_scaled.shape[1], training_size)
    ae_model.train(data_obj.x_train_scaled, data_obj.x_val_scaled)

    x_train_encoded, x_val_encoded, x_test_encoded = ae_model.encoded_data(
        data_obj.x_train_scaled, data_obj.x_val_scaled, data_obj.x_test_scaled)

    # print(len(data_obj.y_train['target'].unique()))
    nn_model = NeuralNetwork(data_obj.x_train_scaled.shape[1], 1, training_size)
    #len(data_obj.y_train.unique()))
    nn_model.train(x_train_encoded, data_obj.y_train, x_val_encoded, data_obj.y_val)
    nn_model.evaluate(x_test_encoded, data_obj.y_test)

    data_obj.reset_scalar()

    return nn_model.result()

if __name__ == '__main__':

    # results = []

    training_sizes = [.15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65,
                      .7, .75, 0.8, 0.85, 0.9]

    val_result = []
    test_result = []
    dataset_path = 'data/dataR2.csv'
    data_obj = DataLoader(dataset_path)

    for training_size in training_sizes:
        print(training_size)
        temp = run(data_obj, training_size)
        val_result.append(temp[0])
        test_result.append(temp[1])

    import pandas as pd
    dataset = pd.DataFrame({'Training Size': training_sizes + training_sizes,
                            'Accuracy': val_result + test_result,
                            'Dataset': [
                                'Validation' for _ in range(len(training_sizes))] +
                            ['Test' for _ in range(len(training_sizes))],
                            })

    # from helper import lineplot
    # lineplot(dataset)
    print(dataset)

"""
auxiliary features:
    -use standared size for training and
      -get the features from the different endoded versions
      -and train

done
    -diffferent size of dataset give to auto encoder and save them
    plot the accuracy for val and test
"""