develop = True
if develop:
    get_ipython().system(u'wget http://www.stanford.edu/~kterao/public_data/v0.1.0/2d/classification/five_particles/practice_train_5k.root -O InputFiles/classification_train.root')
    get_ipython().system(u'wget http://www.stanford.edu/~kterao/public_data/v0.1.0/2d/classification/five_particles/practice_test_5k.root -O InputFiles/classification_test.root')
    get_ipython().system(u'wget http://www.stanford.edu/~kterao/public_data/v0.1.0/2d/segmentation/multipvtx/practice_train_2k.root -O InputFiles/segmentation_train.root')
    get_ipython().system(u'wget http://www.stanford.edu/~kterao/public_data/v0.1.0/2d/segmentation/multipvtx/practice_test_2k.root -O InputFiles/segmentation_test.root')
else:
    get_ipython().system(u'wget http://www.stanford.edu/~kterao/public_data/v0.1.0/2d/classification/five_particles/train_50k.root -O InputFiles/classification_train.root')
    get_ipython().system(u'wget http://www.stanford.edu/~kterao/public_data/v0.1.0/2d/classification/five_particles/test_40k.root -O InputFiles/classification_test.root')
    get_ipython().system(u'wget http://www.stanford.edu/~kterao/public_data/v0.1.0/2d/segmentation/multipvtx/train_15k.root -O InputFiles/segmentation_train.root')
    get_ipython().system(u'wget http://www.stanford.edu/~kterao/public_data/v0.1.0/2d/segmentation/multipvtx/test_10k.root -O InputFiles/segmentation_test.root')


# ## Classification

import ROOT
from larcv import larcv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PlottingTools as pt
get_ipython().magic(u'matplotlib inline')

from tqdm import tqdm
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True          

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3 
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.optimizers import SGD

from keras import backend as K
print '\nData format for Keras is:', K.image_data_format()

get_ipython().system(u'rm -rf Plots/*')


# Common variables for all models
epochs = 20
batchSize = 25

imageLength = 256
imageHeight = 256

# Stop training when a monitored quantity has stopped improving after 20 epochs
earlyStop = EarlyStopping(patience=20, verbose=1)

# Reduce learning rate when a metric has stopped improving
reduceLR = ReduceLROnPlateau(factor=0.3, patience=3, cooldown=3, verbose=1)

pdgToCategory =  {11   : 0, #electron
                  22   : 1, #gamma
                  13   : 2, #muon
                  211  : 3, #pion
                  2212 : 4} #proton

pdgToParticle =  {11   : 'Electron',
                  22   : 'Gamma',
                  13   : 'Muon',
                  211  : 'Pion',
                  2212 : 'Proton'}

location = [0, 1, 2, 3, 4]
labels = ['electron', 'gamma', 'muon', 'pion', 'proton']


# ### Load the training and test datasets

train_image_chain = ROOT.TChain("image2d_data_tree")
train_image_chain.AddFile('InputFiles/classification_train.root')
print 'Found', train_image_chain.GetEntries(), 'images in training dataset!'

test_image_chain = ROOT.TChain("image2d_data_tree")
test_image_chain.AddFile('InputFiles/classification_test.root')
print 'Found', test_image_chain.GetEntries(), 'images in test dataset!'

train_label_chain = ROOT.TChain("particle_mctruth_tree")
train_label_chain.AddFile('InputFiles/classification_train.root')
print 'Found', train_label_chain.GetEntries(), 'labels in training dataset!'

test_label_chain = ROOT.TChain("particle_mctruth_tree")
test_label_chain.AddFile('InputFiles/classification_test.root')
print 'Found', test_label_chain.GetEntries(), 'labels in test dataset!'


# ### Visualize training images in detail

def get_view_range(image2d):
    nz_pixels=np.where(image2d>=0.0)
    ylim = (np.min(nz_pixels[0])-5,np.max(nz_pixels[0])+5)
    xlim = (np.min(nz_pixels[1])-5,np.max(nz_pixels[1])+5)
    # Adjust for allowed image range
    ylim = (np.max((ylim[0],0)), np.min((ylim[1],image2d.shape[1]-1)))
    xlim = (np.max((xlim[0],0)), np.min((xlim[1],image2d.shape[0]-1)))
    return (xlim,ylim)
xAxis=['x', 'y', 'z']
yAxis=['y', 'z', 'x']

entryList = [x*1002 for x in range(5)]
for entry in entryList:
    train_image_chain.GetEntry(entry)
    image_object = train_image_chain.image2d_data_branch
    image_vector = image_object.as_vector()

    fig = plt.figure(figsize=(20,5))
    for index, image in enumerate(image_vector):
        ax = fig.add_subplot(1, 3, index+1)
        numpyImage = larcv.as_ndarray(image)
        ax.imshow(numpyImage, interpolation='none',cmap='ocean_r', origin='lower')
        
        ax.set_xlabel(xAxis[index], fontsize=20)
        ax.set_ylabel(yAxis[index], fontsize=20)
        
        xlim, ylim = get_view_range(numpyImage)
        
        # Set range
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)

    train_label_chain.GetEntry(entry)
    label_object = train_label_chain.particle_mctruth_branch
    for particle in label_object.as_vector():
        fig.suptitle('Training images: {}'.format(pdgToParticle[particle.pdg_code()]), fontsize=20)
        plt.show()
        fig.savefig('Plots/ClassificationTrainingImage{}.pdf'.format(pdgToParticle[particle.pdg_code()]), bbox_inches='tight')


# ### Tools to preprocess the images and labels for Keras

def imageToTensor(imageToConvert):
    # imageToConvert is a C++ class larcv::EventImage2D exposed to python interpreter
    # Note here that std::vectors in pyroot are iterable
    array2D = imageToConvert.as_vector()
    
    array3D = []
    for _, image in enumerate(array2D):
        # larcv has a helper function to convert std::vector to numpy array, so we can use that:
        numpyImage = larcv.as_ndarray(image)
        array3D.append(numpyImage)        
    tensor3D = np.dstack((array3D[0], array3D[1], array3D[2]))

    # Convert 3D tensor to 4D tensor with shape (1, imageLength, imageHeight, 3) and return 4D tensor
    return tensor3D#np.expand_dims(tensor3D, axis=0)

def imageChainToTensor(imageChain):
    listOfTensors = []
    for entry in tqdm(xrange(imageChain.GetEntries())):
        imageChain.GetEntry(entry)
        entryData = imageChain.image2d_data_branch
        listOfTensors.append(imageToTensor(entryData))
    return np.array(listOfTensors)

def labelChainToArray(labelChain):
    listOfLabels = []
    for entry in tqdm(xrange(labelChain.GetEntries())):
        labelChain.GetEntry(entry)
        entryData = labelChain.particle_mctruth_branch
        arrayParticle = entryData.as_vector()
        for _, particle in enumerate(arrayParticle):
            listOfLabels.append(pdgToCategory[particle.pdg_code()])
    return np.array(listOfLabels)


# Convert every images to 4D tensors
X_train = imageChainToTensor(train_image_chain)
X_test = imageChainToTensor(test_image_chain)

print X_train[0].shape
print X_test[0].shape


# There are 5 different particle types in the dataset
num_classes = 5
trainLabelArray = labelChainToArray(train_label_chain)
testLabelArray = labelChainToArray(test_label_chain)

y_train = np_utils.to_categorical(trainLabelArray, num_classes)
y_test = np_utils.to_categorical(testLabelArray, num_classes)
print y_train[:10]


# ## Benchmark model
benchmarkModel = Sequential()

benchmarkModel.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu',input_shape=X_train[0].shape))
benchmarkModel.add(BatchNormalization())
benchmarkModel.add(MaxPooling2D(pool_size=2))
benchmarkModel.add(Dropout(0.65))

benchmarkModel.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
benchmarkModel.add(BatchNormalization())
benchmarkModel.add(MaxPooling2D(pool_size=2))
benchmarkModel.add(Dropout(0.65))

benchmarkModel.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
benchmarkModel.add(BatchNormalization())
benchmarkModel.add(MaxPooling2D(pool_size=2))
benchmarkModel.add(Dropout(0.65))

benchmarkModel.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
benchmarkModel.add(BatchNormalization())
benchmarkModel.add(MaxPooling2D(pool_size=2))
benchmarkModel.add(Dropout(0.65))

benchmarkModel.add(Flatten())         
benchmarkModel.add(Dense(500, activation='relu'))
benchmarkModel.add(BatchNormalization())
benchmarkModel.add(Dropout(0.5))

benchmarkModel.add(Dense(num_classes, activation='softmax'))

benchmarkModel.summary()


# ### Compile the code
benchmarkModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# ### Calculate the classification accuracy of the benchmark model (before training)
# Evaluate the test accuracy
score = benchmarkModel.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]

# Print the test accuracy
print('Test accuracy of the benchmark model (before training): %.4f%%' % accuracy)


# ### Train the model
# Save the best model after every epoch
checkPoint = ModelCheckpoint(filepath='SavedModels/ClassificationBenchmarkBest.hdf5', 
                               verbose=1, save_best_only=True)
history = benchmarkModel.fit(X_train, y_train, batch_size=batchSize, epochs=epochs,
          validation_split=0.20, callbacks=[checkPoint, earlyStop, reduceLR],
          verbose=0, shuffle=True)

# Summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Benchmark model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.savefig('Plots/ClassificationBenchmarkAccuracy.pdf', bbox_inches='tight')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Benchmark model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.savefig('Plots/ClassificationBenchmarkLoss.pdf', bbox_inches='tight')
plt.show()


# ### Calculate the classification accuracy of the model (after training)
# Load the model with the best classification accuracy on the validation set
benchmarkModel.load_weights('SavedModels/ClassificationBenchmarkBest.hdf5')

# Calculate the classification accuracy on the test set
score = benchmarkModel.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]

# Print the test accuracy
print('Test accuracy of the benchmark model (after training): %.4f%%' % accuracy)


# ### Visualize the result of the model
y_pred = benchmarkModel.predict(X_test, verbose = False)

y_p = []
for y in y_pred:
    y_p.append(np.argmax(y))
y_t = []
for t in y_test:
    y_t.append(np.argmax(t))

fig, ax1 = plt.subplots(1,1, figsize = (6,6))
ax1.hist2d(y_t, y_p, bins=(25, 25), cmap=plt.cm.Greys)
ax1.plot(y_t, y_t, 'r-', label = 'True')
plt.xticks(location, labels)
plt.yticks(location, labels)

ax1.legend()
ax1.set_title('Benchmark model', fontsize=25)
ax1.set_xlabel('True particle', fontsize=20)
ax1.set_ylabel('Predicted particle', fontsize=20)
pt.setTicks(ax1)

fig.savefig('Plots/ClassificationBenchmarkResult.pdf', bbox_inches='tight')


# ## Model based on InceptionV3
# Load InceptionV3 model + remove final classification layers
baseModel = InceptionV3(weights='imagenet', include_top=False, input_shape=(imageLength, imageHeight, 3))

# Add a global average pooling layer
myLayers = baseModel.output
myLayers = GlobalAveragePooling2D()(myLayers)
myLayers = Dropout(0.5)(myLayers)

# A fully connected layer
myLayers = Dense(1000, activation='relu')(myLayers)
myLayers = Dropout(0.5)(myLayers)

# Output layer
predictions = Dense(num_classes, activation='softmax')(myLayers)

# The model
InceptionV3Model = Model(inputs=baseModel.input, outputs=predictions)

# Freeze all convolutional InceptionV3 layers
for layer in baseModel.layers:
    layer.trainable = False
    
InceptionV3Model.summary()


# ### Compile the code
InceptionV3Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# ### Train the model
# Save the best model after every epoch
checkPoint = ModelCheckpoint(filepath='SavedModels/ClassificationInceptionV3BottleneckBest.hdf5', 
                               verbose=1, save_best_only=True)
history = InceptionV3Model.fit(X_train, y_train, batch_size=batchSize, epochs=epochs,
                               validation_split=0.20, callbacks=[checkPoint, earlyStop, reduceLR],
                               verbose=0, shuffle=True)

# Summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('InceptionV3 bottleneck model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.savefig('Plots/ClassificationInceptionV3BottleneckAccuracy.pdf', bbox_inches='tight')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('InceptionV3 bottleneck model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.savefig('Plots/ClassificationInceptionV3BottleneckLoss.pdf', bbox_inches='tight')
plt.show()


# ### Calculate the classification accuracy of the model (after training)
# Load the model with the best classification accuracy on the validation set
InceptionV3Model.load_weights('SavedModels/ClassificationInceptionV3BottleneckBest.hdf5')

# Calculate the classification accuracy on the test set
score = InceptionV3Model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]

# Print the test accuracy
print('Test accuracy of the model (after training): %.4f%%' % accuracy)


# ### Visualize the result of the model
y_pred = InceptionV3Model.predict(X_test, verbose = False)

y_p = []
for y in y_pred:
    y_p.append(np.argmax(y))
y_t = []
for t in y_test:
    y_t.append(np.argmax(t))

fig, ax1 = plt.subplots(1,1, figsize = (6,6))
ax1.hist2d(y_t, y_p, bins=(25, 25), cmap=plt.cm.Greys)
ax1.plot(y_t, y_t, 'r-', label = 'True')
plt.xticks(location, labels)
plt.yticks(location, labels)

ax1.legend()
ax1.set_title('InceptionV3 bottleneck model', fontsize=25)
ax1.set_xlabel('True particle', fontsize=20)
ax1.set_ylabel('Predicted particle', fontsize=20)
pt.setTicks(ax1)

fig.savefig('Plots/ClassificationInceptionV3BottleneckResult.pdf', bbox_inches='tight')


# ### Visualize layer names and layer indices to see how many layers to freeze (to fine-tune the model)
for i, layer in enumerate(baseModel.layers):
   print i, layer.name


# ### Train the top few inception blocks
# Let's train the top 3 inception blocks
# Freeze the first 229 layers
for layer in InceptionV3Model.layers[:229]:
   layer.trainable = False
for layer in InceptionV3Model.layers[229:]:
   layer.trainable = True


# ### Compile the code
InceptionV3Model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.005, momentum=0.9), metrics=['accuracy'])


# ### Train the model
# Save the best model after every epoch
checkPoint = ModelCheckpoint(filepath='SavedModels/ClassificationInceptionV3FineTunedBest.hdf5', 
                               verbose=1, save_best_only=True)

history = InceptionV3Model.fit(X_train, y_train, batch_size=batchSize, epochs=epochs,
                               validation_split=0.20, callbacks=[checkPoint, earlyStop, reduceLR],
                               verbose=0, shuffle=True)

# Summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('InceptionV3 fine-tuned model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.savefig('Plots/ClassificationInceptionV3FineTunedAccuracy.pdf', bbox_inches='tight')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('InceptionV3 fine-tune model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.savefig('Plots/ClassificationInceptionV3FineTunedAccuracy.pdf', bbox_inches='tight')
plt.show()


# ### Calculate the classification accuracy of the model (after training)
# Load the model with the best classification accuracy on the validation set
InceptionV3Model.load_weights('SavedModels/ClassificationInceptionV3FineTunedBest.hdf5')

# Calculate the classification accuracy on the test set
score = InceptionV3Model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]

# Print the test accuracy
print('Test accuracy of the model (after training): %.4f%%' % accuracy)


# ### Visualize the result of the model
y_pred = InceptionV3Model.predict(X_test, verbose = False)

y_p = []
for y in y_pred:
    y_p.append(np.argmax(y))
y_t = []
for t in y_test:
    y_t.append(np.argmax(t))

fig, ax1 = plt.subplots(1,1, figsize = (6,6))
ax1.hist2d(y_t, y_p, bins=(25, 25), cmap=plt.cm.Greys)
ax1.plot(y_t, y_t, 'r-', label = 'True')
plt.xticks(location, labels)
plt.yticks(location, labels)

ax1.legend()
ax1.set_title('InceptionV3 fine-tune model', fontsize=25)
ax1.set_xlabel('True particle', fontsize=20)
ax1.set_ylabel('Predicted particle', fontsize=20)
pt.setTicks(ax1)

fig.savefig('Plots/ClassificationInceptionV3FineTunedResult.pdf', bbox_inches='tight')


# # Semantic Segmentation
# ### Load the training and test datasets
train_image_chain = ROOT.TChain("image2d_data_tree")
train_image_chain.AddFile('InputFiles/segmentation_train.root')
print 'Found', train_image_chain.GetEntries(), 'images in training dataset!'

test_image_chain = ROOT.TChain("image2d_data_tree")
test_image_chain.AddFile('InputFiles/segmentation_test.root')
print 'Found', test_image_chain.GetEntries(), 'images in test dataset!'

train_label_chain = ROOT.TChain("image2d_segment_tree")
train_label_chain.AddFile('InputFiles/segmentation_train.root')
print 'Found', train_label_chain.GetEntries(), 'labels in training dataset!'

test_label_chain = ROOT.TChain("image2d_segment_tree")
test_label_chain.AddFile('InputFiles/segmentation_test.root')
print 'Found', test_label_chain.GetEntries(), 'labels in test dataset!'


# ### Visualize a single training image: Background + Shower + Track
entry = 552
train_image_chain.GetEntry(entry)
train_label_chain.GetEntry(entry)

# Let's grab a specific projection (1st one)
image2d = larcv.as_ndarray(train_image_chain.image2d_data_branch.as_vector().front())
label2d = larcv.as_ndarray(train_label_chain.image2d_segment_branch.as_vector().front())

categories = ['Background','Shower','Track']
unique_values, unique_counts = np.unique(label2d, return_counts=True)

fig, axes = plt.subplots(1, len(unique_values), figsize=(18,12), facecolor='w')
xlim,ylim = get_view_range(image2d)
for index, value in enumerate(unique_values):
    ax = axes[index]
    mask = (label2d == value)
    ax.imshow(image2d * mask, interpolation='none', cmap='gist_heat_r', origin='lower')
    ax.set_title(categories[index],fontsize=20,fontname='Georgia',fontweight='bold')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
plt.show()
fig.savefig('Plots/SegmentationTrainingImageSeparated.pdf', bbox_inches='tight')


# ### Visualize a single training image and label in detail
# Dump images
fig, (ax0,ax1) = plt.subplots(1, 2, figsize=(15,10), facecolor='w')
ax0.imshow(image2d, interpolation='none', cmap='gist_heat_r', origin='lower')
ax1.imshow(label2d, interpolation='none', cmap='gist_heat_r', origin='lower',vmin=0., vmax=2.0)
ax0.set_title('Data',fontsize=20,fontname='Georgia',fontweight='bold')
ax0.set_xlim(xlim)
ax0.set_ylim(ylim)
ax1.set_title('Label',fontsize=20,fontname='Georgia',fontweight='bold')
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
plt.show()
fig.savefig('Plots/SegmentationTrainingImageAndLabel.pdf', bbox_inches='tight')

