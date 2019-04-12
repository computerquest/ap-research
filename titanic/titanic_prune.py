import keras
from keras.models import load_model
# Gmatch4py use networkx graph
import networkx as nx
# import the GED using the munkres algorithm
import gmatch4py as gm
import matplotlib.pyplot as plt
import csv
import pandas
from keras.models import load_model
import numpy as np

import pandas
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

np.random.seed(19680801)

num_param = -1

seed = 7

# PREPROCESSING

a = pandas.read_csv('/home/jstigter/PycharmProjects/ap-research/titanic/titanic.csv')
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

before = 0
missing_index = []


def create_pruned_graphs(totNode, num_input, model_size, model_type, model_split, model_rep):
    global  dataframe
    global target

    min_weight = .01
    modelb = load_model(
        '/home/jstigter/PycharmProjects/ap-research/titanic/titanic_results/' + model_size + '/' + model_type + '_split' + str(
            model_split) + '_' + str(
            model_rep) + '.h5')

    # Compile model
    modelb.compile(loss='mean_squared_error',
                   optimizer=keras.optimizers.SGD(lr=0.005, momentum=0.0, decay=0.0, nesterov=False),
                   metrics=['accuracy'])  # lowered the learning rate from .01 for large

    # modelb.summary()
    fit = round(modelb.evaluate(dataframe, target, steps=1)[1] * 100)
    print('fit is ', fit)
    if fit <= 66:
        return None

    totNode += 1  # this is because keras treats all biases like additional nodes
    totNode += num_input
    # print(modelb.get_weights())

    prev_len = 0
    graph = nx.DiGraph()

    for x in range(0, totNode):
        graph.add_node(x)

    bias = []
    # print('starting to make the graph')
    for x in modelb.layers:
        weights = x.get_weights()[0]
        bias.extend(x.get_weights()[1])
        # this gets the weights for a given node
        for i in range(0, len(weights)):
            node_weight = weights[i]
            for z in range(0, len(node_weight)):
                current_weight = node_weight[z]
                if abs(current_weight) > min_weight:
                    # print('adding edge between', (prev_len + i, prev_len + len(weights) + z), abs(current_weight),
                    #      current_weight)
                    graph.add_edge(prev_len + i, prev_len + len(
                        weights) + z)  # (all previous nodes and this ones position; all previous nodes, this round of nodes, and the posotion on the next layer
                # else:
                # print('didnt add', (prev_len + i, prev_len + len(weights) + z), abs(current_weight),
                # current_weight)
        prev_len += len(weights)

    for i in range(0, len(bias)):
        if abs(bias[i]) > min_weight:
            # print('adding edge between', (totNode - 1, num_input + i), bias[i])

            graph.add_edge(totNode - 1, num_input + i)
        # else:
        # print('didnt add', (totNode - 1, num_input + i), bias[i])
    # nx.draw(graph)
    # plt.show()

    return graph


def test_split(totNode, num_input, model_size, model_type, model_split):
    g = []

    for i in range(0, 5):
        graph = create_pruned_graphs(totNode, num_input, model_size, model_type, model_split, i)

        if graph != None:
            print('adding type', type(graph))
            g.append(graph)
        else:
            missing_index.append(i + before * 5)

    ged = gm.GraphEditDistance(1, 1, 1, 1)  # all edit costs are equal to 1
    result = ged.compare(g, None)
    print(result)

    return g


def test_type(totNode, num_input, model_size, model_type):
    g = []

    for x in range(1, 6):
        for i in range(0, 5):
            graph = create_pruned_graphs(totNode, num_input, model_size, model_type, x, i)

            if graph != None:
                print('adding type', type(graph))
                g.append(g)
    ged = gm.GraphEditDistance(1, 1, 1, 1)  # all edit costs are equal to 1
    result = ged.compare(g, None)
    print(result)

    return g


'''print('he')
all_net.extend(test_type(6, 12, 'small', 'he'))

print('rand_relu')
all_net.extend(test_type(6, 12, 'small', 'rand_relu'))

print('xavier')
all_net.extend(test_type(6, 12, 'small', 'xavier'))

print('rand_sig')
all_net.extend(test_type(6, 12, 'small', 'rand_sig'))

print('all together')
ged = gm.GraphEditDistance(1, 1, 1, 1)  # all edit costs are equal to 1
result = ged.compare(all_net, None)
print(*result)'''


def generate_files(size, num_node, num_input):
    global before
    for x in range(1, 6):
        all_net = []

        print('split ///////////////////////////////////////', x)
        print('he')
        all_net.extend(test_split(num_node, num_input, size, 'he', x))

        before += 1

        print('rand_relu')
        all_net.extend(test_split(num_node, num_input, size, 'rand_relu', x))

        before += 1

        print('xavier')
        all_net.extend(test_split(num_node, num_input, size, 'xavier', x))

        before += 1

        print('rand_sig')
        all_net.extend(test_split(num_node, num_input, size, 'rand_sig', x))

        before += 1

        print('all together')
        ged = gm.GraphEditDistance(1, 1, 1, 1)  # all edit costs are equal to 1
        result = ged.compare(all_net, None)

        pd = pandas.DataFrame(result)

        for i in missing_index:
            pd.insert(i, column=str(-i), value=[0 for z in result])

        pd.to_csv('prune_results/' + size + '.' + str(x) + '.csv')

        print(*result)

        before = 0

generate_files('large', 16, 12)

generate_files('small', 6, 12)
generate_files('medium', 11, 12)

print('done')
