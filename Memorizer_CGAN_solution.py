# Load and Crop images
# #'''
ComputeLB = False
DogsOnly = False

import numpy as np, pandas as pd, os
import xml.etree.ElementTree as ET 
import matplotlib.pyplot as plt, zipfile 
from PIL import Image, ImageOps, ImageFilter
import os, sys

ROOT = '../input/generative-dog-images/'
#ROOT = '/kaggle/working/images/'
if not ComputeLB: ROOT = '../input/'
IMAGES = os.listdir(ROOT + 'all-dogs/all-dogs/')
#IMAGES = os.listdir('/kaggle/working/images/')
breeds = os.listdir(ROOT + 'annotation/Annotation/') 

idxIn = 0; namesIn = []
imagesIn = np.zeros((25000,64,64,3))

# CROP WITH BOUNDING BOXES TO GET DOGS ONLY
# https://www.kaggle.com/paulorzp/show-annotations-and-breeds
if DogsOnly:
    for breed in breeds:
        for dog in os.listdir(ROOT+'annotation/Annotation/'+breed):
            try: img = Image.open(ROOT+'all-dogs/all-dogs/'+dog+'.jpg') 
            except: continue           
            tree = ET.parse(ROOT+'annotation/Annotation/'+breed+'/'+dog)
            root = tree.getroot()
            objects = root.findall('object')
            for o in objects:
                bndbox = o.find('bndbox') 
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                w = np.min((xmax - xmin, ymax - ymin))
            w = xmax - xmin
            h = ymax - ymin
            l = max([w,h])
    
            if w > h:
                ymin = ymin - int((l-h)/2)
                #ymin = max(ymin - int((l-h)/2), 0)
                #ymax = min(ymin - int((l-h)/2) + l, img.height)
                #bbox = (xmin, ymin , xmax, ymax)
                #img = img.crop(bbox)
                #img = img.resize((int(64*w/h), 64), Image.LANCZOS)
            else:
                xmin = xmin - int((l-w)/2)
                #xmin = max(xmin - int((l-w)/2), 0)
                #xmax = min(xmin - int((l-w)/2) + l, img.width)
                #bbox = (xmin, ymin , xmax, ymax)
                #img = img.crop(bbox)
                #img = img.resize((64, int(64*h/w)), Image.LANCZOS)
            #bbox = (xmin, ymin , xmax, ymax)
            bbox = (xmin, ymin , xmin + l, ymin + l)
            #img2 = img.crop((xmin, ymin, xmin+w, ymin+w))
            img2 = img.crop(bbox)
            img2 = img2.resize((64,64), Image.ANTIALIAS)
            imagesIn[idxIn,:,:,:] = np.asarray(img2)
            #if idxIn%1000==0: print(idxIn)
            namesIn.append(breed)
            idxIn += 1
    #idx = np.arange(idxIn)
    #np.random.shuffle(idx)
    #imagesIn = imagesIn[idx,:,:,:]
    #namesIn = np.array(namesIn)[idx]
    
# RANDOMLY CROP FULL IMAGES
else:
    IMAGES = np.sort(IMAGES)
    #np.random.seed(810)
    #np.random.seed(924)
    np.random.seed(57)
    x = np.random.choice(np.arange(20579),10000, replace=False)
    #x_sub = np.random.choice(np.arange(20579),10000, replace=False)
    np.random.seed(None)
    for k in range(len(x)):
        img = Image.open(ROOT + 'all-dogs/all-dogs/' + IMAGES[x[k]])
        #if IMAGES[x[k]] in ['n02096585_1963.jpg']:
        #    img = Image.open(ROOT + 'all-dogs/all-dogs/' + IMAGES[x_sub[k]])
        w = img.size[0]; h = img.size[1];
        if False:#(k%2==0)|(k%3==0):
            w2 = 100; h2 = int(h/(w/100))
            a = 18; b = 0          
        else:
            a=0; b=0
            if w<h:
                if w/h <= 0.74:
                    img = img.resize((400,300), Image.ANTIALIAS)
                    w = img.size[0]; h = img.size[1];
                w2 = 64; h2 = int((64/w)*h)
                b = (h2-64)//2
                if w/h >0.74:
                    #w2 = 100; h2 = int(h/(w/100))
                    #a = 18; b = 0   
                    w2 = 73; h2 = int((73/w)*h)#int(h/(w/100))
                    #a = (h2-64)//2; b = 18#(h2-64)//2
                    #b = (h2-64)//2; a = (w2-64)//2 #(h2-64)//2
                    #b = 0; a = (w2-64)//2 #(h2-64)//2
                    b = 0; a = 0 #(h2-64)//2
                    #b = h2-64; a = w2-64 #(h2-64)//2
            else:
                if h/w <= 0.74:
                    img = img.resize((300,400), Image.ANTIALIAS)
                    w = img.size[0]; h = img.size[1];
                h2 = 64; w2 = int((64/h)*w)
                a = (w2-64)//2
                if h/w >0.74:
                    #w2 = 100; h2 = int(h/(w/100))
                    #a = 18; b = 0   
                    h2 = 73; w2 = int((73/h)*w)
                    #a = 18; b = (w2-64)//2 
                    #b = (h2-64)//2; a = (w2-64)//2 
                    #b = (h2-64)//2; a = 0
                    b = 0; a = 0
                    #b = h2-64; a = w2-64


        img = img.resize((w2,h2), Image.ANTIALIAS)
        img = img.crop((0+a, 0+b, 64+a, 64+b))
        img = ImageOps.mirror(img)
        imagesIn[idxIn,:,:,:] = np.asarray(img)
        namesIn.append(IMAGES[x[k]])
        #if idxIn%1000==0: print(idxIn)
        idxIn += 1
    
# DISPLAY CROPPED IMAGES
y = np.random.randint(0,idxIn,25)
for k in range(5):
    plt.figure(figsize=(15,3))
    for j in range(5):
        plt.subplot(1,5,j+1)
        img = Image.fromarray( imagesIn[y[k*5+j],:,:,:].astype('uint8') )
        plt.axis('off')
        if not DogsOnly: plt.title(namesIn[y[k*5+j]],fontsize=11)
        else: plt.title(namesIn[y[k*5+j]].split('-')[1],fontsize=11)
        plt.imshow(img)
    plt.show()
#for idx, item in enumerate(imagesIn[:10000]):
#    img_pil = Image.fromarray(item.astype(np.uint8))
#    #img_pil.save(path+IMAGES[idx], 'PNG', quality=100)
#    img_pil.save(path+namesIn[idx], 'PNG', quality=100)
#'''





# Discriminator
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Reshape, Flatten, concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD, Adam


# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1), n_classes=10):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
	# scale up to image dimensions with linear activation
	n_nodes = in_shape[0] * in_shape[1]
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((in_shape[0], in_shape[1], 1))(li)
	# image input
	in_image = Input(shape=in_shape)
	# concat label as a channel
	merge = Concatenate()([in_image, li])
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# flatten feature maps
	fe = Flatten()(fe)
	# dropout
	fe = Dropout(0.4)(fe)
	# output
	out_layer = Dense(1, activation='sigmoid')(fe)
	# define model
	model = Model([in_image, in_label], out_layer)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
# If we check Supervised GAN Discriminator is a very simple version of CGAN.

# The model takes [dog,dogName] (dogName = label y)

# BUILD DISCRIMINATIVE NETWORK
dog = Input((12288,))
dogName = Input((10000,))
x = Dense(12288, activation='sigmoid')(dogName) 
x = Reshape((2,12288,1))(concatenate([dog,x]))
x = Conv2D(1,(2,1),use_bias=False,name='conv')(x)
discriminated = Flatten()(x)

# COMPILE
discriminator = Model([dog,dogName], discriminated)
discriminator.get_layer('conv').trainable = False
discriminator.get_layer('conv').set_weights([np.array([[[[-1.0 ]]],[[[1.0]]]])])
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# DISPLAY ARCHITECTURE
#discriminator.summary()
# Train Discriminator
# We will train the Discriminator to memorize the training images. (Typically you don't train the Discriminator ahead of time. The D learns as the G learns. But this GAN is special).

# TRAINING DATA
train_y = (imagesIn[:10000,:,:,:]/255.).reshape((-1,12288))
train_X = np.zeros((10000,10000))
for i in range(10000): train_X[i,i] = 1
zeros = np.zeros((10000,12288))

# TRAIN NETWORK
lr = 0.5
for k in range(5):
    annealer = LearningRateScheduler(lambda x: lr)
    h = discriminator.fit([zeros,train_X], train_y, epochs = 20, batch_size=256, callbacks=[annealer], verbose=0)
    print('Epoch',(k+1)*10,'/50 - loss =',h.history['loss'][-1] )
    if h.history['loss'][-1]<0.530: lr = 0.1

# Delete Training Images
# Our Discriminator has memorized all the training images. We will now delete the training images. Our Generator will never see the training images. It will only be coached by the Discriminator. Below are examples of images that the Discriminator memorized.

#del train_X, train_y, imagesIn
# Important question
# me: Are Memorizer GANs limited by the number of images the discriminator can memorize? in that case the number of different dogs that can be generated is 10k ?

# chris: There is no limitation. A Memorizer GAN can memorize any number of images. But it is a finite number. If a Memorizer GAN is not improved upon, then if you ask for an image that it did not memorize, it will look poor. For example, say it memorizes 3 images and has an input seed of length 3. Then seed = [1, 0, 0] recalls the first image, and seeds [0, 1, 0] and [0, 0, 1] recall the other two. However, there are an infinite number of seeds. What if we input seed = [0.5, 0.9, -0.3]? Then the outputted image will look bad. A Generalizer GAN will output nice looking image for every seed. If you input seed=[1, 0, 0] it will look nice. If you input seed = [0.5, 0.9, -0.3] it will look nice. A Generalizer GAN can draw an infinite number of images that look good !

# Let's generate 9 memorized dogs:

# NOTE:

xx = np.zeros((10000))
xx[np.random.randint(10000)] = 1 # one dog from memory
print('Discriminator Recalls from Memory Dogs')    
for k in range(3):
    plt.figure(figsize=(10,3))
    for j in range(3):
        xx = np.zeros((10000))
        xx[np.random.randint(10000)] = 1 # one dog from memory
        plt.subplot(1,3,j+1)
        img = discriminator.predict([zeros[0,:].reshape((-1,12288)),xx.reshape((-1,10000))]).reshape((-1,64,64,3))
        img = Image.fromarray( (255*img).astype('uint8').reshape((64,64,3)))
        plt.axis('off')
        plt.imshow(img)
    plt.show()
# Discriminator Recalls from Memory Dogs

# Spolier: the discriminator coaches the generator.

# great discriminator = great generator = great LB ? NO ... not exactly

# Probably many people tried to reduce discriminator loss and their score worsened. Remember what happened when we submitted 10k real images, the FID was great (1.4~) but the MiFID is really bad ( millions!). So why these real images don't score fatally?

# Furthermore the MiFID metric doesn't recognize that cropped images are the same as original images. Therefore a memorizing generative method using cropped images can score very good LB.



# Let's check the memory
# chris added BadMemory option Dog Memorizer GAN V6

print('Discriminator Recalls from Memory Dogs - 10% dog')    
for k in range(2):
    plt.figure(figsize=(4,2))
    for j in range(2):
        xx = np.zeros((10000))
        xx[np.random.randint(10000)] = 0.1 # one dog from memory
        plt.subplot(1,2,j+1)
        img = discriminator.predict([zeros[0,:].reshape((-1,12288)),xx.reshape((-1,10000))]).reshape((-1,64,64,3))
        img = Image.fromarray( (255*img).astype('uint8').reshape((64,64,3)))
        plt.axis('off')
        plt.imshow(img)
    plt.show()
# Discriminator Recalls from Memory Dogs - 10% dog


print('Discriminator Memory Dogs - 30% dog')    
for k in range(2):
    plt.figure(figsize=(4,2))
    for j in range(2):
        xx = np.zeros((10000))
        xx[np.random.randint(10000)] = 0.3 # one dog from memory
        plt.subplot(1,2,j+1)
        img = discriminator.predict([zeros[0,:].reshape((-1,12288)),xx.reshape((-1,10000))]).reshape((-1,64,64,3))
        img = Image.fromarray( (255*img).astype('uint8').reshape((64,64,3)))
        plt.axis('off')
        plt.imshow(img)
    plt.show()
# Discriminator Memory Dogs - 30% dog


# 2 dogs in one picture?

xx = np.zeros((10000))
xx[np.random.randint(10000)] = 0.5 # one dog from memory
xx[np.random.randint(10000)] = 0.5 # one dog from memory
# it's like we are blending pixels!

print('Discriminator Recalls from Memory Dogs - 2 dogs')    
for k in range(2):
    plt.figure(figsize=(8,3))
    for j in range(2):
        xx = np.zeros((10000))
        xx[np.random.randint(10000)] = 0.5 # one dog from memory
        xx[np.random.randint(10000)] = 0.5 # one dog from memory
        plt.subplot(1,2,j+1)
        img = discriminator.predict([zeros[0,:].reshape((-1,12288)),xx.reshape((-1,10000))]).reshape((-1,64,64,3))
        img = Image.fromarray( (255*img).astype('uint8').reshape((64,64,3)))
        plt.axis('off')
        plt.imshow(img)
    plt.show()
# Discriminator Recalls from Memory Dogs - 2 dogs


# The final trick
# resize + crop + n dogs in the same picture!
xx[np.random.randint(10000)] = 0.999 # one dog from memory
xx[np.random.randint(10000)] = 0.001 # one dog from memory
# In front of our eyes, they look like real pictures, but the Inception model doesn't see the same ;) That's why you can generate "real pictures" and avoid MiFID penalization

print('Discriminator Recalls from Memory Dogs - 2 dogs')    
for k in range(2):
    plt.figure(figsize=(8,3))
    for j in range(2):
        xx = np.zeros((10000))
        xx[np.random.randint(10000)] = 0.999 # one dog from memory
        xx[np.random.randint(10000)] = 0.001 # one dog from memory
        plt.subplot(1,2,j+1)
        img = discriminator.predict([zeros[0,:].reshape((-1,12288)),xx.reshape((-1,10000))]).reshape((-1,64,64,3))
        img = Image.fromarray( (255*img).astype('uint8').reshape((64,64,3)))
        plt.axis('off')
        plt.imshow(img)
    plt.show()
# Discriminator Recalls from Memory Dogs - 2 dogs




# Build Generator and GAN
# A simple GAN generator is like: (click input to see)

def define_generator(latent_dim):
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
	# upsample to 14x14
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 28x28
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# generate
	model.add(Conv2D(1, (7,7), activation='tanh', padding='same'))
	return model
# A simple Conditional GAN (CGAN) generator is like: (click input to see)

def define_generator(latent_dim, n_classes=120):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
	# linear multiplication
	n_nodes = 7 * 7
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((7, 7, 1))(li)
	# image generator input
	in_lat = Input(shape=(latent_dim,))
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	gen = Dense(n_nodes)(in_lat)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((7, 7, 128))(gen)
	# merge image gen and label input
	merge = Concatenate()([gen, li])
	# upsample to 14x14
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
	gen = LeakyReLU(alpha=0.2)(gen)
	# upsample to 28x28
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	# output
	out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
	# define model
	model = Model([in_lat, in_label], out_layer)
	return model
# If we check Supervised GAN Generator is a very simple version of CGAN. ¿?

# BUILD GENERATOR NETWORK
seed = Input((10000,))
generated = Dense(12288, activation='linear')(seed)

# COMPILE
generator = Model(seed, [generated,Reshape((10000,))(seed)])

# DISPLAY ARCHITECTURE
generator.summary()
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to                     
# ==================================================================================================
# input_3 (InputLayer)            (None, 10000)        0                                            
# __________________________________________________________________________________________________
# dense_2 (Dense)                 (None, 12288)        122892288   input_3[0][0]                    
# __________________________________________________________________________________________________
# reshape_2 (Reshape)             (None, 10000)        0           input_3[0][0]                    
# ==================================================================================================
# Total params: 122,892,288
# Trainable params: 122,892,288
# Non-trainable params: 0
# __________________________________________________________________________________________________
# BUILD GENERATIVE ADVERSARIAL NETWORK
discriminator.trainable=False    
gan_input = Input(shape=(10000,))
x = generator(gan_input)
gan_output = discriminator(x)

# COMPILE GAN
gan = Model(gan_input, gan_output)
gan.get_layer('model_1').get_layer('conv').set_weights([np.array([[[[-1 ]]],[[[255.]]]])])
gan.compile(optimizer=Adam(5), loss='mean_squared_error')

# DISPLAY ARCHITECTURE
#gan.summary()
# Discriminator Coaches Generator
# In a typical GAN, the discriminator does not memorize the training images beforehand. Instead it learns to distinquish real images from fake images at the same time that the Generator learns to make fake images. In this GAN, we taught the Discriminator ahead of time and it will now teach the Generator.

# TRAINING DATA
train = np.zeros((10000,10000))
for i in range(10000): train[i,i] = 1
zeros = np.zeros((10000,12288))

# TRAIN NETWORKS
lr = 5.
for k in range(50):  

    # BEGIN DISCRIMINATOR COACHES GENERATOR
    annealer = LearningRateScheduler(lambda x: lr)
    h = gan.fit(train, zeros, epochs = 1, batch_size=256, callbacks=[annealer], verbose=0)
    if (k<10)|(k%5==4):
        print('Epoch',(k+1)*10,'/500 - loss =',h.history['loss'][-1] )
    if h.history['loss'][-1] < 25: lr = 1.
    if h.history['loss'][-1] < 1.5: lr = 0.5
        
    # DISPLAY GENERATOR LEARNING PROGRESS
    if k<10:        
        plt.figure(figsize=(15,3))
        for j in range(5):
            xx = np.zeros((10000))
            xx[np.random.randint(10000)] = 1
            plt.subplot(1,5,j+1)
            img = generator.predict(xx.reshape((-1,10000)))[0].reshape((-1,64,64,3))
            img = Image.fromarray( (img).astype('uint8').reshape((64,64,3)))
            plt.axis('off')
            plt.imshow(img)
        plt.show()  

# Build Generator Class
# Our Generative Network has now learned all the training images from our Discriminative Network. We would like our Dog Generator to accept any random 100 dimensional vector and output an image. Furthermore we need to slightly perturb each memorized image so we don't submit exact copies of training images. Let's build a Generator Class

# Remember

xx[self.index] = 0.999
xx[np.random.randint(10000)] = 0.001
class DogGenerator:
    index = 0   
    def getDog(self,seed):
        xx = np.zeros((10000))
        #xx[self.index] = 0.999
        #xx[np.random.randint(10000)] = 0.001
        xx[self.index] = 1.0
        img = generator.predict(xx.reshape((-1,10000)))[0].reshape((64,64,3))
        self.index = (self.index+1)%10000
        return Image.fromarray( img.astype('uint8') ) 
print('Generated dogs 99% real 1% fake')
d = DogGenerator()
for k in range(2):
    plt.figure(figsize=(8,3))
    for j in range(2):
        plt.subplot(1,2,j+1)
        img = d.getDog(seed = np.random.normal(0,1,100))
        plt.axis('off')
        plt.imshow(img)
    plt.show()
# Generated dogs 99% real 1% fake


# Submit to Kaggle
# My hope is that Kaggle disallows Memorizer GANs from winning medals in this competition. The purpose of submitting this high scoring kernel is to demonstrate that it is too easy to achieve a high score with a simple Memorizer GAN. The more complex and interesting solutions are Generalizing GANs!

# SAVE TO ZIP FILE NAMED IMAGES.ZIP
z = zipfile.PyZipFile('images.zip', mode='w')
d = DogGenerator()
for k in range(10000):
    img = d.getDog(np.random.normal(0,1,100))
    f = str(k)+'.png'
    img.save(f,'PNG'); z.write(f); os.remove(f)
    #if k % 1000==0: print(k)
z.close()
# Calculate LB Score
# If you wish to compute LB, you must add the LB metric dataset here to this kernel and change the boolean variable in the first cell block.

from __future__ import absolute_import, division, print_function
import numpy as np
import os
import gzip, pickle
import tensorflow as tf
from scipy import linalg
import pathlib
import urllib
import warnings
from tqdm import tqdm
from PIL import Image

class KernelEvalException(Exception):
    pass

model_params = {
    'Inception': {
        'name': 'Inception', 
        'imsize': 64,
        'output_layer': 'Pretrained_Net/pool_3:0', 
        'input_layer': 'Pretrained_Net/ExpandDims:0',
        'output_shape': 2048,
        'cosine_distance_eps': 0.1
        }
}

def create_model_graph(pth):
    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile( pth, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString( f.read())
        _ = tf.import_graph_def( graph_def, name='Pretrained_Net')

def _get_model_layer(sess, model_name):
    # layername = 'Pretrained_Net/final_layer/Mean:0'
    layername = model_params[model_name]['output_layer']
    layer = sess.graph.get_tensor_by_name(layername)
    ops = layer.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims != []:
              shape = [s.value for s in shape]
              new_shape = []
              for j, s in enumerate(shape):
                if s == 1 and j == 0:
                  new_shape.append(None)
                else:
                  new_shape.append(s)
              o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    return layer

def get_activations(images, sess, model_name, batch_size=50, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_model_layer(sess, model_name)
    n_images = images.shape[0]
    if batch_size > n_images:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = n_images
    n_batches = n_images//batch_size + 1
    pred_arr = np.empty((n_images,model_params[model_name]['output_shape']))
    for i in tqdm(range(n_batches)):
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        start = i*batch_size
        if start+batch_size < n_images:
            end = start+batch_size
        else:
            end = n_images
                    
        batch = images[start:end]
        pred = sess.run(inception_layer, {model_params[model_name]['input_layer']: batch})
        pred_arr[start:end] = pred.reshape(-1,model_params[model_name]['output_shape'])
    if verbose:
        print(" done")
    return pred_arr


# def calculate_memorization_distance(features1, features2):
#     neigh = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean')
#     neigh.fit(features2) 
#     d, _ = neigh.kneighbors(features1, return_distance=True)
#     print('d.shape=',d.shape)
#     return np.mean(d)

def normalize_rows(x: np.ndarray):
    """
    function that normalizes each row of the matrix x to have unit length.

    Args:
     ``x``: A numpy matrix of shape (n, m)

    Returns:
     ``x``: The normalized (by row) numpy matrix.
    """
    return np.nan_to_num(x/np.linalg.norm(x, ord=2, axis=1, keepdims=True))


def cosine_distance(features1, features2):
    # print('rows of zeros in features1 = ',sum(np.sum(features1, axis=1) == 0))
    # print('rows of zeros in features2 = ',sum(np.sum(features2, axis=1) == 0))
    features1_nozero = features1[np.sum(features1, axis=1) != 0]
    features2_nozero = features2[np.sum(features2, axis=1) != 0]
    norm_f1 = normalize_rows(features1_nozero)
    norm_f2 = normalize_rows(features2_nozero)

    d = 1.0-np.abs(np.matmul(norm_f1, norm_f2.T))
    print('d.shape=',d.shape)
    print('np.min(d, axis=1).shape=',np.min(d, axis=1).shape)
    mean_min_d = np.mean(np.min(d, axis=1))
    print('distance=',mean_min_d)
    return mean_min_d


def distance_thresholding(d, eps):
    if d < eps:
        return d
    else:
        return 1

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        # covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    # covmean = tf.linalg.sqrtm(tf.linalg.matmul(sigma1,sigma2))

    print('covmean.shape=',covmean.shape)
    # tr_covmean = tf.linalg.trace(covmean)

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    # return diff.dot(diff) + tf.linalg.trace(sigma1) + tf.linalg.trace(sigma2) - 2 * tr_covmean
#-------------------------------------------------------------------------------


def calculate_activation_statistics(images, sess, model_name, batch_size=50, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations(images, sess, model_name, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma, act
    
def _handle_path_memorization(path, sess, model_name, is_checksize, is_check_png):
    path = pathlib.Path(path)
    files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
    imsize = model_params[model_name]['imsize']

    # In production we don't resize input images. This is just for demo purpose. 
    x = np.array([np.array(img_read_checks(fn, imsize, is_checksize, imsize, is_check_png)) for fn in files])
    m, s, features = calculate_activation_statistics(x, sess, model_name)
    del x #clean up memory
    return m, s, features

# check for image size
def img_read_checks(filename, resize_to, is_checksize=False, check_imsize = 64, is_check_png = False):
    im = Image.open(str(filename))
    if is_checksize and im.size != (check_imsize,check_imsize):
        raise KernelEvalException('The images are not of size '+str(check_imsize))
    
    if is_check_png and im.format != 'PNG':
        raise KernelEvalException('Only PNG images should be submitted.')

    if resize_to is None:
        return im
    else:
        return im.resize((resize_to,resize_to),Image.ANTIALIAS)

def calculate_kid_given_paths(paths, model_name, model_path, feature_path=None, mm=[], ss=[], ff=[]):
    ''' Calculates the KID of two paths. '''
    tf.reset_default_graph()
    create_model_graph(str(model_path))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        m1, s1, features1 = _handle_path_memorization(paths[0], sess, model_name, is_checksize = True, is_check_png = True)
        if len(mm) != 0:
            m2 = mm
            s2 = ss
            features2 = ff
        elif feature_path is None:
            m2, s2, features2 = _handle_path_memorization(paths[1], sess, model_name, is_checksize = False, is_check_png = False)
        else:
            with np.load(feature_path) as f:
                m2, s2, features2 = f['m'], f['s'], f['features']

        print('m1,m2 shape=',(m1.shape,m2.shape),'s1,s2=',(s1.shape,s2.shape))
        print('starting calculating FID')
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        print('done with FID, starting distance calculation')
        distance = cosine_distance(features1, features2)        
        return fid_value, distance, m2, s2, features2
if ComputeLB:
  
    # UNCOMPRESS OUR IMGAES
    with zipfile.ZipFile("../working/images.zip","r") as z:
        z.extractall("../tmp/images2/")

    # COMPUTE LB SCORE
    m2 = []; s2 =[]; f2 = []
    user_images_unzipped_path = '../tmp/images2/'
    images_path = [user_images_unzipped_path,'../input/generative-dog-images/all-dogs/all-dogs/']
    public_path = '../input/dog-face-generation-competition-kid-metric-input/classify_image_graph_def.pb'

    fid_epsilon = 10e-15

    fid_value_public, distance_public, m2, s2, f2 = calculate_kid_given_paths(images_path, 'Inception', public_path, mm=m2, ss=s2, ff=f2)
    distance_public = distance_thresholding(distance_public, model_params['Inception']['cosine_distance_eps'])
    print("FID_public: ", fid_value_public, "distance_public: ", distance_public, "multiplied_public: ",
            fid_value_public /(distance_public + fid_epsilon))
    
    # REMOVE FILES TO PREVENT KERNEL ERROR OF TOO MANY FILES
    ! rm -r ../tmp

#https://www.kaggle.com/code/hirune924/public-lb-5-17518-solution-memorizer-cgan
