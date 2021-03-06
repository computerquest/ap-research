import keras
from keras.models import load_model
# Gmatch4py use networkx graph
import networkx as nx
# import the GED using the munkres algorithm
import gmatch4py as gm
import matplotlib.pyplot as plt
import csv
import pandas

import keras
from keras.models import load_model
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pandas
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

np.random.seed(19680801)

seed = 7

# PREPROCESSING

a = pandas.read_csv('/home/jstigter/PycharmProjects/ap-research/bank/bank-additional-full.csv')
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

before = 0
missing_index = []


def create_pruned_graphs(totNode, num_input, model_size, model_type, model_split, model_rep):
    global  dataframe
    global target

    min_weight = .01
    modelb = load_model(
        '/home/jstigter/PycharmProjects/ap-research/bank/bank_results/' + model_size + '/' + model_type + '_split' + str(
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
    global before

    global missing_index
    for i in range(0, 5):
        graph = create_pruned_graphs(totNode, num_input, model_size, model_type, model_split, i)

        if graph != None:
            print('adding type', type(graph))
            g.append(graph)
        else:
            missing_index.append(i + round * 5)

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
    for x in range(1, 6):
        all_net = []

        print('split ///////////////////////////////////////', x)
        print('he')
        all_net.extend(test_split(num_node, num_input, size, 'he', x))

        print('rand_relu')
        all_net.extend(test_split(num_node, num_input, size, 'rand_relu', x))

        print('xavier')
        all_net.extend(test_split(num_node, num_input, size, 'xavier', x))

        print('rand_sig')
        all_net.extend(test_split(num_node, num_input, size, 'rand_sig', x))

        print('all together')
        ged = gm.GraphEditDistance(1, 1, 1, 1)  # all edit costs are equal to 1
        result = ged.compare(all_net, None)

        pd = pandas.DataFrame(result)

        for i in missing_index:
            pd.insert(i, column=str(-i), value=[0 for z in result])

        pd.to_csv('prune_results/' + size + '.' + str(x) + '.csv')

        print(*result)


#generate_files('small', 21, 30)
generate_files('medium', 31, 30)
generate_files('large', 41, 30)

print('done')
