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

def weight_delta(size, main):
    global num_param

    ans = []
    column_names = []
    for split in range(1,6):
        temp = []
        for x in range(0, 5):
            modelb = load_model(
                '/home/jstigter/PycharmProjects/ap-research/bank/bank_results/' + size + '/' + main + '_split' + str(split) + '_' + str(
                    x) + '.h5')

            # Compile model
            modelb.compile(loss='mean_squared_error',
                          optimizer=keras.optimizers.SGD(lr=0.005, momentum=0.0, decay=0.0, nesterov=False),
                          metrics=['accuracy']) #lowered the learning rate from .01 for large

            difs = modelb.get_weights()

            column_names.append('Fold: '+str(split)+' '+' '+ str(x)+' '+str(round(modelb.evaluate(dataframe,target)[1]*100)))

            this_dif = []
            for c in difs:
                flat_list = []
                for sublist in c.tolist():
                    if isinstance(sublist, (list,)):
                        for item in sublist:
                            flat_list.append(item)
                    else:
                        flat_list.append(sublist)
                this_dif.extend(flat_list)
            temp.append(this_dif)
        print('temp size is ', len(temp))
        ans.extend(temp)


    dt = pandas.DataFrame(ans, index=column_names)
    print(dt, column_names)
    return dt.T

def create_figures(size, init):
    results = weight_delta(size, init)

    a4_dims = (12, 12)

    f = plt.figure(1, figsize=a4_dims)

    plt.xlabel('Weights', fontsize=18)
    plt.ylabel('Folds', fontsize=16)

    n_bins = 30
    '''for data in results:
        print(data, [x / num_param for x in data], num_param)
        sns.distplot(data, hist_kws={'weights':np.zeros_like(np.array(data)) + 1. / len(data)}, bins=n_bins, ax=axs[0])'''

    print(results)
    plt.ylim([-1,1])
    plt.xlim([-5,5])
    #plt.xticks(results, list(results.columns.values), rotation='vertical')
    sns.boxplot(data=results, whis=1.5, orient='h')
    plt.xticks()
    f.show()
    f.savefig('/home/jstigter/PycharmProjects/ap-research/bank/graphs/after_box/'+size+'.'+init+'_box.png')
    f.clear()

    f2 = plt.figure(2, figsize=a4_dims)

    combined_data = results.values.flatten()

    plt.xlim([-3,3])
    plt.ylim([0,len(combined_data)])


    plt.xlabel('Weight', fontsize=18)
    plt.ylabel('Frequency', fontsize=16)

    print(sorted(combined_data, key=abs, reverse=True))
    plt.hist(combined_data, bins=30, edgecolor='black')

    f2.show()
    f2.savefig('/home/jstigter/PycharmProjects/ap-research/bank/graphs/after_hist/'+size+'.'+init+'_hist.png')
    f2.clear()

create_figures('large', 'he')
create_figures('large', 'rand_sig')
create_figures('large', 'rand_relu')
create_figures('large', 'xavier')

create_figures('medium', 'he')
create_figures('medium', 'rand_sig')
create_figures('medium', 'rand_relu')
create_figures('medium', 'xavier')

create_figures('small', 'he')
create_figures('small', 'rand_sig')
create_figures('small', 'rand_relu')
create_figures('small', 'xavier')
