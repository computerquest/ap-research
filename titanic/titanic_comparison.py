from keras.models import load_model

model = load_model('D:/PycharmProjects/ap-research/titanic/titanic_results/small/rand_sig_split1_0.h5')
print(model.get_weights(), len(model.layers))
print(model.summary())

print('other model //////////////////////////////////')
model = load_model('D:/PycharmProjects/ap-research/titanic/weights/small/rand_sig_split1_0.h5')

print(model.get_weights(), len(model.layers))
print(model.summary())