import keras
from keras.models import load_model
# Gmatch4py use networkx graph
import networkx as nx
# import the GED using the munkres algorithm
import gmatch4py as gm
import matplotlib.pyplot as plt
import csv
import pandas


def create_pruned_graphs(totNode, num_input, model_size, model_type, model_split, model_rep):
    min_weight = .01
    modelb = load_model(
        '/home/jstigter/PycharmProjects/ap-research/bank/bank_results/' + model_size + '/' + model_type + '_split' + str(
            model_split) + '_' + str(
            model_rep) + '.h5')

    # modelb.summary()

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
        g.append(create_pruned_graphs(totNode, num_input, model_size, model_type, model_split, i))

    ged = gm.GraphEditDistance(1, 1, 1, 1)  # all edit costs are equal to 1
    result = ged.compare(g, None)
    print(result)

    return g


def test_type(totNode, num_input, model_size, model_type):
    g = []

    for x in range(1, 6):
        for i in range(0, 5):
            g.append(create_pruned_graphs(totNode, num_input, model_size, model_type, x, i))

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
        pd.to_csv('prune_results/' + size + '.' + str(x) + '.csv')

        print(*result)


generate_files('small', 21, 30)
generate_files('medium', 31, 30)
generate_files('large', 41, 30)

print('done')
