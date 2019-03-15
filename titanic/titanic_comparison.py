from keras.models import load_model
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def weight_delta(size, main):
    ans = []
    for split in range(1,6):
        temp = []
        for x in range(0, 5):
            modelb = load_model(
                '/home/jstigter/PycharmProjects/ap-research/titanic/titanic_results/' + size + '/' + main + '_split' + str(split) + '_' + str(
                    x) + '.h5')

            modela = load_model(
                '/home/jstigter/PycharmProjects/ap-research/titanic/weights/' + size + '/' + main + '_split' + str(split) + '_' + str(
                    x) + '.h5')

            difs = np.subtract(modela.get_weights(), modelb.get_weights())

            for c in difs:
                flat_list = []
                for sublist in c.tolist():
                    if isinstance(sublist, (list,)):
                        for item in sublist:
                            flat_list.append(item)
                    else:
                        flat_list.append(sublist)
                temp.extend(flat_list)

        ans.append(temp)

    return ans

a4_dims = (12, 12)
fig, axs = plt.subplots(figsize=a4_dims, ncols=2, sharey=True, sharex=True)


fig.text(0.5, 0.01, 'Δ Weight', fontsize=12,  va='center', ha='center')
fig.text(0.01, 0.5, 'Frequency', fontsize=12, va='center', rotation='vertical')

axs[0].set_title('Folds Separate')
axs[1].set_title('Folds Together')

plt.ylim(0, 15)

axs[0].set_xlim([-2, 2])


axs[1].set_xlim([-2, 2])

results = weight_delta('small', 'he')

sns.distplot(results[0], bins=30, ax=axs[0])
sns.distplot(results[1], bins=30, ax=axs[0])
sns.distplot(results[2], bins=30, ax=axs[0])
sns.distplot(results[3], bins=30, ax=axs[0])
sns.distplot(results[4], bins=30, ax=axs[0])

sns.distplot([item for sublist in results for item in sublist], bins=30, ax=axs[1])

plt.margins(1)
plt.subplots_adjust(bottom=.4, right=.2) # or whatever
plt.show()
