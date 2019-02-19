# Python code to Standardize data (0 mean, 1 stdev)
import pandas
import numpy
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import StratifiedKFold
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import tensorflow as tf

seed = 7
numpy.random.seed(seed)

# PREPROCESSING

url = "./data/titanic/train.csv"
a = pandas.read_csv(url)
del a['passengerid']
del a['cabin']
del a['name']

dataframe = a.dropna(subset=['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked'])

target = dataframe['survived'].values
features = dataframe[['pclass', 'sex', 'age', 'fare', 'embarked', 'sibsp', 'parch']]

preprocess = make_column_transformer(
    (StandardScaler(), ['age', 'fare', 'sibsp', 'parch']),
    (OneHotEncoder(), ['pclass', 'sex', 'embarked'])
)

dataframe = preprocess.fit_transform(features)
print('the final frame is', dataframe)

# MODEL TRAINING
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

def experiment(dim, activation, init, epochs, file):
    x = 0
    print(init, activation)
    cvscores = []
    c = 0
    for train, test in kfold.split(dataframe, target):
        print('this is split', c)
        c += 1
        for z in range(0, 5):
            # create model
            model = Sequential()
            model.add(Dense(dim[0], input_dim=12, kernel_initializer=init, activation=activation))
            for c in range(1, len(dim) - 1):
                model.add(Dense(dim[0], kernel_initializer=init, activation=activation))
            model.add(Dense(dim[-1], kernel_initializer=init, activation=activation))

            # Compile model
            model.compile(loss='mean_squared_error',
                          optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False),
                          metrics=['accuracy'])

            # Fit the model
            model.fit(dataframe[train], target[train], epochs=epochs, batch_size=128, verbose=0,
                      callbacks=[EarlyStopping(monitor='loss',
                                               min_delta=-.000001,
                                               patience=30,
                                               verbose=0, mode='auto')])

            # evaluate the model
            scores = model.evaluate(dataframe[test], target[test], verbose=0)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
            cvscores.append(scores[1] * 100)

            model.save(file+'_split'+str(c)+'_'+str(z)+'.h5')
            x += 1
    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))


print('small')
experiment([5, 1], 'sigmoid', 'random_normal', 3000, './titanic_results/small/rand_sig')
experiment([5, 1], 'relu', 'random_normal', 3000, './titanic_results/small/rand_relu')
experiment([5, 1], 'sigmoid', 'glorot_normal', 3000, './titanic_results/small/xavier')
experiment([5, 1], 'relu', 'he_normal', 3000, './titanic_results/small/he')

print('medium')
experiment([10, 1], 'sigmoid', 'random_normal', 3000, './titanic_results/medium/rand_sig')
experiment([10, 1], 'relu', 'random_normal', 3000, './titanic_results/medium/rand_relu')
experiment([10, 1], 'sigmoid', 'glorot_normal', 3000, './titanic_results/medium/xavier')
experiment([10, 1], 'relu', 'he_normal', 3000, './titanic_results/medium/he')

print('large')
experiment([10, 5, 1], 'sigmoid', 'random_normal', 6000, './titanic_results/large/rand_sig')
experiment([10, 5, 1], 'relu', 'random_normal', 6000, './titanic_results/large/rand_relu')
experiment([10, 5, 1], 'sigmoid', 'glorot_normal', 6000, './titanic_results/large/xavier')
experiment([10, 5, 1], 'relu', 'he_normal', 6000, './titanic_results/large/he')
