# Python code to Standardize data (0 mean, 1 stdev)
import pandas
import numpy
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import StratifiedKFold
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

print('TITANIC ////////////////////////////////////////////')
seed = 7
numpy.random.seed(seed)

# PREPROCESSING

a = pandas.read_csv('D:/PycharmProjects/ap-research/titanic/titanic.csv')
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
cross_valid = list(kfold.split(dataframe, target))

print(len(cross_valid))
def experiment(dim, activation, init, epochs, file):
    x = 0
    print(init, activation)
    cvscores = []
    c = 0
    for train, test in cross_valid:
        c += 1
        for z in range(0, 5):
            # create model
            model = Sequential()
            model.add(Dense(dim[0], input_dim=12, kernel_initializer=init, activation=activation))
            for d in range(1, len(dim) - 1):
                model.add(Dense(dim[d], kernel_initializer=init, activation=activation))

            model.add(Dense(dim[-1], kernel_initializer=init, activation=activation))

            # Compile model
            model.compile(loss='mean_squared_error',
                          optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False),
                          metrics=['accuracy']) #lowered the learning rate from .01 for large

            model.save('D:/PycharmProjects/ap-research/titanic/weights/'+file + '_split' + str(c) + '_' + str(z) + '.h5')

            # Fit the model
            model.fit(dataframe[train], target[train], validation_data=(dataframe[test], target[test]), epochs=epochs, batch_size=128, verbose=0,
                      callbacks=[EarlyStopping(monitor='val_loss',
                                               min_delta=0,
                                               patience=10,
                                               verbose=1, mode='min', restore_best_weights=True)]) #-.000001 3 for others

            # evaluate the model
            scores = model.evaluate(dataframe[test], target[test], verbose=0)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
            cvscores.append(scores[1] * 100)

            model.save('D:/PycharmProjects/ap-research/titanic/titanic_results/'+file + '_split' + str(c) + '_' + str(z) + '.h5')
            x += 1
    print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))


'''print('small')
experiment([5, 1], 'sigmoid', 'random_normal', 100000, 'small/rand_sig')
experiment([5, 1], 'relu', 'random_normal', 100000, 'small/rand_relu')
experiment([5, 1], 'sigmoid', 'glorot_normal', 100000, 'small/xavier')
experiment([5, 1], 'relu', 'he_normal', 100000, 'small/he')

print('medium')
experiment([10, 1], 'sigmoid', 'random_normal', 100000, 'medium/rand_sig')
experiment([10, 1], 'relu', 'random_normal', 100000, 'medium/rand_relu')
experiment([10, 1], 'sigmoid', 'glorot_normal', 100000, 'medium/xavier')
experiment([10, 1], 'relu', 'he_normal', 100000, 'medium/he')'''

experiment([10, 1], 'relu', 'random_normal', 100000, 'medium/rand_relu')

print('large')
#experiment([10, 5, 1], 'sigmoid', 'random_normal', 100000, 'large/rand_sig')
#experiment([10, 5, 1], 'relu', 'random_normal', 100000, 'large/rand_relu')
#experiment([10, 5, 1], 'sigmoid', 'glorot_normal', 100000, 'large/xavier')
#experiment([10, 5, 1], 'relu', 'he_normal', 100000, 'large/he')
