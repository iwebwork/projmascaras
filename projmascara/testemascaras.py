from IPython.display import Image
import tensorflow as tf
import keras as K
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#iniciando rede neural convolucional CNN
classifier=Sequential()

#primeira camada de convoluçao
#convertendo imagens para 64 pixel em um array 3D pois as imagens sao coloridas
#com 3 camadas de cores
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3) , activation='relu'))


#aplicando agrupamento pooling para reduzir o tamanho do mapa
classifier.add(MaxPooling2D(pool_size=(2,2)))

#adicionando 2 camada de convoluçao
classifier.add(Conv2D(32,(3,3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))


#aplicando achatamento = flatten para converter estruturas de dados 2D resultado
#da camada anterior em uma estrutura 1D

classifier.add(Flatten())

#conectando as camadas usando a funçao de ativaçao retificadore "relu"
#e depois uma funçao de sigmóide para obter a probabilidade de cada imagem conter
# mascara ou sem mascara

#full connection
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(1,activation ='sigmoid'))

#compilando rede usando o algoritmo adam , e a funçao log loss com a binary_crossentropy

classifier.compile(optimizer='adam' , loss='binary_crossentropy',metrics=['accuracy'])
#rede neural construida




# Criando os objetos train_datagen e validation_datagen 
#com as regras de pré-processamento das imagens

from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale= 1./255,
                                 shear_range = 0.2,
                                 zoom_range = 0.2,
                                 horizontal_flip = True
                                        )
validation_datagen = ImageDataGenerator(rescale= 1./255)

#pre processamento de dados e validaçao
training_set = train_datagen.flow_from_directory('treino', target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode='binary')

validation_set = validation_datagen.flow_from_directory('validacao', target_size=(64,64),
                                                        batch_size = 32,
                                                        class_mode = 'binary')

#executando o treinamento (processo pode levar tempo , dependendo do seu pc)
classifier.fit_generator(training_set,
                         steps_per_epoch = 32,
                         epochs = 4,
                         validation_data=validation_set,
                         validation_steps= 8 )



#fazendo as previsoes
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('teste/simon-ma-X-B8Gdg4-4E-unsplash.jpg', target_size=
                          (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'sem mascara'
else:
    prediction = 'com mascara'
    
Image(filename='teste/simon-ma-X-B8Gdg4-4E-unsplash.jpg')
        

prediction

