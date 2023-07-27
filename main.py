from keras.layers import Input
from keras.layers import  Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import UpSampling2D
from keras.layers import Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.layers import TimeDistributed
from keras_radam import RAdam
from keras_radam.training import RAdamOptimizer
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os
import numpy as np
import cv2
import glob



def build_generator():
    noise_shape = (100,)
    noise = Input(shape=noise_shape)
    
    model = Sequential()
    model.add(Dense(64 * 4 * 4,activation="relu", input_shape=noise_shape))
    model.add(Reshape((4,4,64)))
    model.add(BatchNormalization(momentum=0.8))
    
    for layer in [128,256,128,64]:
        model.add(UpSampling2D())
        model.add(Conv2D(layer, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(16, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))
    model.summary()
    output_img = model(noise)
    return Model(noise, output_img)


def build_discriminator():
    img = Input(shape=img_size) 
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_size, padding="same")) #32,32
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    
    for layer in [64,128,128,128]:
        model.add(Conv2D(layer, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    validity = model(img)
    return Model(img, validity)

def build_discriminator():
    img = Input(shape=img_size)
    model = Sequential()
    model.add(ResNet50(include_top=False, pooling='avg',input_shape=img_size, weights=None))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))    
    model.layers[0].trainable = False
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='adam')
    validity = model(img)
    return Model(img, validity)
    

def build_discriminator():
    img = Input(shape=img_size)
    model=Sequential()
    model.add(LSTM(32,return_sequences=True,input_shape=img_size))
    model.add(LSTM(32,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(16, activation='relu')))
    model.add(TimeDistributed(Dense(8, activation='relu')))
    model.add(Flatten())
    model.add(Dense(1, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer=RAdamOptimizer(learning_rate=1e-3))
    return model


def build_model():
    generator = build_generator()
    generator.compile(loss='binary_crossentropy', optimizer=RAdamOptimizer(learning_rate=1e-3))
    noise = Input(shape=(100,))
    generated_image = generator(noise)
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=RAdamOptimizer(learning_rate=1e-3), metrics=['accuracy'])
    discriminator.trainable = False
    validity = discriminator(generated_image)
    gan = Model(noise, validity)
    gan.compile(loss='binary_crossentropy', optimizer=RAdamOptimizer(learning_rate=1e-3))
    return discriminator,generator,gan


def load_dataset(batch_size, image_shape, data_dir=None):
    sample_dim = (batch_size,) + image_shape
    sample = np.empty(sample_dim, dtype=np.float32)
    all_data_dirlist = list(glob.glob(data_dir))
    sample_imgs_paths = np.random.choice(all_data_dirlist,batch_size)
    for index,img_filename in enumerate(sample_imgs_paths):
        image = Image.open(img_filename)
        image = image.resize(image_shape[:-1])
        image = np.asarray(image)
        image = (image/127.5) -1 
        sample[index,...] = image
    return sample


def read_data():
    image_shape=(64,64,3)
    X_train = load_dataset(300, (64,64,3), "ex/*.jpeg")
    print('data loaded')
    return X_train


def train(epochs, batch_size=128, save_interval=50):
    global discriminator, generator, gan
    half_batch = int(batch_size / 2)
    for epoch in range(epochs):
        real_ids = np.random.randint(0, X_train.shape[0], half_batch)
        real_imgs = X_train[real_ids]
        noise = np.random.normal(0, 1, (half_batch, 100))
        gen_imgs = generator.predict(noise)
        real_loss = discriminator.train_on_batch(real_imgs, np.ones((half_batch, 1)))
        fake_loss = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(real_loss, fake_loss)
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
        if epoch % save_interval == 0:
            save_imgs(epoch)


def save_imgs(epoch):
    global discriminator, generator, gan
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(r, c)
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[i*c+j])
            axs[i,j].axis('off') 
    fig.savefig("results/animes_%d.jpeg" % epoch, dpi=600, bbox_inches='tight')
    plt.close()    
    generator.save("models/generator_%d.h5" % epoch)
    discriminator.save("models/discriminator_%d.h5" % epoch)


img_rows = 64
img_cols = 64
channels = 3
img_size = (img_rows, img_cols,channels)
X_train = read_data()

dataDir = 'data set path'
files = glob.glob(os.path.join(dataDir, '*.jpg'))
print('Total number of images in the dataset', len(files))
fig, axes = plt.subplots(1, 3, figsize=(15, 10))
i = 0
for ax in axes:
    idx = np.random.randint(0, len(files))
    filename = files[idx]
    img = mpimg.imread(filename)
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title('Brain MRI with size: {}'.format(img.shape))
    i += 1
    a=img.shape


i=0
for a in range(len(files)):
    idx = np.random.randint(0, len(files))
    filename = files[idx]
    img = mpimg.imread(filename)
    a=img.shape
    if len(img.shape)==3:
        i=i+1
        print(filename)
        print(i)


def main(): 
    inPath ="folder containing the raw images"
    i=0
    outPath ="folder that will contain the modified image"
    
    for imagePath in os.listdir(inPath): 
        inputPath = os.path.join(inPath, imagePath) 
        img = mpimg.imread(inputPath)
        if len(img.shape)!=3:
            img = Image.open(inputPath) 
            fullOutPath = os.path.join(outPath, 'covid_'+imagePath) 
            img.save(fullOutPath) 
            i=i+1
            print(fullOutPath)
            print(i)
     
if __name__ == '__main__': 
    main()

discriminator, generator, gan = build_model()
train(epochs=100000, batch_size=256, save_interval=200)


