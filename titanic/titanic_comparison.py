from keras.models import load_model
import numpy as np


def weight_delta(size, main):
    ans = []
    for split in range(1,6):
        total_weights = np.zeros(shape=(4,))
        for x in range(0, 5):
            modela = load_model(
                '/home/jstigter/PycharmProjects/ap-research/titanic/titanic_results/' + size + '/' + main + '_split' + str(split) + '_' + str(
                    x) + '.h5')
            #print(modela.summary())

            modelb = load_model(
                '/home/jstigter/PycharmProjects/ap-research/titanic/weights/' + size + '/' + main + '_split' + str(split) + '_' + str(
                    x) + '.h5')
            #print(modelb.summary())

            difs = np.subtract(modelb.get_weights(), modela.get_weights())
            total_weights = np.add(total_weights, difs)

        temp = np.divide(total_weights, 5)

        for c in temp:
            flat_list = []
            for sublist in c.tolist():
                if isinstance(sublist, (list,)):
                    for item in sublist:
                        flat_list.append(item)
                else:
                    flat_list.append(sublist)
            print('adding', flat_list)
            ans.extend(flat_list)


    return ans

print(weight_delta('small', 'he'))
