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

print('BANK ////////////////////////////////////////////')

seed = 7
numpy.random.seed(seed)

# PREPROCESSING

a = pandas.read_csv('D:/PycharmProjects/ap-research/bank/bank-additional-full.csv')
del a['duration']
del a['contact']
del a['month']
del a['day_of_week']

dataframe = a.dropna(
    subset=["age", "job", "marital", "education", "default", "housing", "loan", "campaign", "pdays", "previous",
            "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y"])
print('the starter dataframe is ', dataframe.shape)
target = dataframe['y'].values
features = dataframe[
    ["age", "job", "marital", "education", "default", "housing", "loan", "campaign", "pdays", "previous", "poutcome",
     "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]]

preprocess = make_column_transformer(
    (StandardScaler(),
     ['age', "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m",
      "nr.employed"]),
    (OneHotEncoder(), ["job", "marital", "education"])
)

dataframe = preprocess.fit_transform(features)
print('the final frame is', dataframe)

# MODEL TRAINING
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cross_valid = list(kfold.split(dataframe, target))


def experiment(dim, activation, init, epochs, file):
    x = 0
    print(init, activation)
    cvscores = []
    c = 0
    for train, test in cross_valid:
        print('this is split', c)
        c += 1
        for z in range(0, 5):
            # create model
            model = Sequential()
            model.add(Dense(dim[0], input_dim=30, kernel_initializer=init, activation=activation))
            for c in range(1, len(dim) - 1):
                model.add(Dense(dim[c], kernel_initializer=init, activation=activation))
            model.add(Dense(dim[-1], kernel_initializer=init, activation=activation))

            # Compile model
            model.compile(loss='mean_squared_error',
                          optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False),
                          metrics=['accuracy'])

            model.save('D:/PycharmProjects/ap-research/bank/weights/'+file + '_split' + str(c) + '_' + str(z) + '.h5')

            # Fit the model
            model.fit(dataframe[train], target[train], validation_data=(dataframe[test], target[test]), epochs=epochs,
                      batch_size=128, verbose=0,
                      callbacks=[EarlyStopping(monitor='val_loss',
                                               min_delta=-.000001,
                                               patience=3,
                                               verbose=1, mode='min', restore_best_weights=True)])

            # evaluate the model
            scores = model.evaluate(dataframe[test], target[test], verbose=0)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
            cvscores.append(scores[1] * 100)

            model.save('D:/PycharmProjects/ap-research/bank/bank_results/'+file + '_split' + str(c) + '_' + str(z) + '.h5')
            x += 1
    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))


print('small')
experiment([20, 1], 'sigmoid', 'random_normal', 100000, 'small/rand_sig')
experiment([20, 1], 'relu', 'random_normal', 100000, 'small/rand_relu')
experiment([20, 1], 'sigmoid', 'glorot_normal', 100000, 'small/xavier')
experiment([20, 1], 'relu', 'he_normal', 100000, 'small/he')

print('medium')
experiment([30, 1], 'sigmoid', 'random_normal', 100000, 'medium/rand_sig')
experiment([30, 1], 'relu', 'random_normal', 100000, 'medium/rand_relu')
experiment([30, 1], 'sigmoid', 'glorot_normal', 100000, 'medium/xavier')
experiment([30, 1], 'relu', 'he_normal', 100000, 'medium/he')

print('large')
experiment([30, 10, 1], 'sigmoid', 'random_normal', 100000, 'large/rand_sig')
experiment([30, 10, 1], 'relu', 'random_normal', 100000, 'large/rand_relu')
experiment([30, 10, 1], 'sigmoid', 'glorot_normal', 100000, 'large/xavier')
experiment([30, 10, 1], 'relu', 'he_normal', 100000, 'large/he')
