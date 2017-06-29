"""
    Name:           inception_train.py
    Created:        2/4/2017
    Description:    Fine-tune inception v3 on specific Modastylz data.
"""
#==============================================
#                   Modules
#==============================================
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import pandas as pd
import time
import gzip
import pickle
from collections import Counter
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Model, model_from_json
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.imagenet_utils import decode_predictions
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
#==============================================
#                   Files
#==============================================


#==============================================
#                   Functions
#==============================================
def instantiate(n_classes, n_dense=2048, inception_json="inceptionv3_mod.json", verbose=1):
    """
    Instantiate the inception v3.
    """

    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(n_dense, activation='relu')(x)
    # and a final logistic layer
    predictions = Dense(n_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # serialize model to json
    model_json = model.to_json()
    with open(inception_json, "w") as iOF:
        iOF.write(model_json)

    return base_model, model




def finetune(base_model, model, X_train, y_train, X_val, y_val,
             epochs_1=1000, epochs_2=2000, patience_1=1, patience_2=1, batch_size=32,
             nb_train_samples=41000, nb_validation_samples=7611,
             img_width=299, img_height=299, class_imbalance=False,
             inception_h5_1="inceptionv3_fine_tuned_1.h5", inception_h5_2="inceptionv3_fine_tuned_2.h5",
             inception_h5_check_point_1="inceptionv3_fine_tuned_check_point_1.h5", inception_h5_check_point_2="inceptionv3_fine_tuned_check_point_2.h5",
             layer_names_file="inceptionv3_mod_layer_names.txt", verbose=1):
    """
    Finetune the inception v3.
    """

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # define train & val data generators
    train_generator = train_datagen.flow(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True)

    validation_generator = test_datagen.flow(
        X_val,
        y_val,
        batch_size=batch_size,
        shuffle=True)

    # get class weights
    if class_imbalance:
        class_weight = get_class_weights(train_generator.classes, smooth_factor=0.1)
    else:
        class_weight = None

    if verbose >= 2:
        class_name_dict = {val:key for key,val in train_generator.class_indices.items()}
        print({class_name_dict[key]:val for key,val in class_weight.items()})

    # train the model on the new data for a few epochs on the batches generated by datagen.flow().
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs_1,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[EarlyStopping(monitor='val_loss', patience=patience_1),
                   ModelCheckpoint(filepath=inception_h5_check_point_1, verbose=1, save_best_only=True)],
        class_weight=class_weight)

    # save weights just in case
    model.save_weights(inception_h5_1)

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    with open(layer_names_file, "w") as iOF:
        for ix, layer in enumerate(base_model.layers):
            iOF.write("%d, %s\n"%(ix, layer.name))
            if verbose >= 2: print(ix, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 172 layers and unfreeze the rest:
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs_2,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[EarlyStopping(monitor='val_loss', patience=patience_2),
                   ModelCheckpoint(filepath=inception_h5_check_point_2, verbose=1, save_best_only=True)],
        class_weight=class_weight)

    # save final weights
    model.save_weights(inception_h5_2)




def finetune_from_saved(inception_h5_load_from, inception_h5_save_to,
             inception_json, X_train, y_train, X_val, y_val, nb_freeze=0,
             epochs=5000, patience=2, batch_size=32,
             nb_train_samples=85639, nb_validation_samples=10694,
             img_width=299, img_height=299, class_imbalance=False,
             inception_h5_check_point="inceptionv3_fine_tuned_check_point_3.h5", verbose=1):
    """
    Finetune the inception v3 from already fine-tuned one.
    """

    # load json and create model
    with open(inception_json, 'r') as iOF:
        loaded_model_json = iOF.read()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(inception_h5_load_from)
    if verbose >= 1: print("Loaded model from disk")

    # we freeze the first nb_freeze layers and unfreeze the rest:
    for layer in loaded_model.layers[:nb_freeze]:
        layer.trainable = False
    for layer in loaded_model.layers[nb_freeze:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    loaded_model.compile(optimizer=SGD(lr=0.00001, momentum=0.9), loss='categorical_crossentropy')

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # define train & val data generators
    train_generator = train_datagen.flow(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True)

    validation_generator = test_datagen.flow(
        X_val,
        y_val,
        batch_size=batch_size,
        shuffle=True)

    # get class weights
    if class_imbalance:
        class_weight = get_class_weights(train_generator.classes, smooth_factor=0.1)
    else:
        class_weight = None

    # train the model on the new data for a few epochs on the batches generated by datagen.flow().
    loaded_model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[EarlyStopping(monitor='val_loss', patience=patience),
                   ModelCheckpoint(filepath=inception_h5_check_point, verbose=1, save_best_only=True)],
        class_weight=class_weight)

    # save weights
    loaded_model.save_weights(inception_h5_save_to)





def preprocess_input(x):
    """
    Preprocessing step for inception v3.
    """
    x /= 255.
    x -= 0.5
    x *= 2.
    return x




def get_class_weights(y, smooth_factor=0):
    """
    Returns the weights for each class based on the frequencies of the samples
    :param smooth_factor: factor that smooths extremely uneven weights
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """
    counter = Counter(y)

    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p

    majority = max(counter.values())

    return {clss: float(majority / cnt) for clss, cnt in counter.items()}




def train_for_each_task(df_labels_train, df_labels_val, target_size=(299,299),
                        model_dir="../data/imaterialist/models/",
                        train_dir="../data/imaterialist/train_images/",
                        validation_dir="../data/imaterialist/validation_images/",
                        verbose=1):
    """
    Train an Inception V3 for each task.
    """

    ### get all task ids
    different_tasks = set(df_labels_train.taskId)

    ### loop over tasks
    for tid in different_tasks:
        # for task 40 dress gender is always female

        if tid > 41:

            if verbose >= 1: print("Training for task %d..."%tid)

            ### Get number of classes
            df_task_train = df_labels_train[df_labels_train.taskId == tid]
            le = LabelEncoder()
            le.fit(df_task_train.labelId)
            different_classes = le.classes_
            n_classes = len(different_classes)

            ### Store LabelEncoder
            with gzip.open(model_dir+'label_encoder_%d.pkl'%tid, 'wb') as iOF:
                pickle.dump(le, iOF)

            ### Create model
            if verbose >= 1: print("\tInstantiating Inception V3 (task %d)..."%tid)
            base_model, model = instantiate(n_classes, inception_json=model_dir+"inceptionv3_mod_%d.json"%tid, verbose=verbose)

            ### Load images
            if verbose >= 1: print("\tLoading images into RAM (task %d)..."%tid)
            grouped_df_train = df_task_train.groupby(['imageId'])['labelId'].apply(list)
            grouped_df_val = df_labels_val[df_labels_val.taskId == tid].groupby(['imageId'])['labelId'].apply(list)
            X_train, y_train = [], []
            X_val, y_val = [], []
            # for train and validation
            for gdf, X, y, img_dir in [(grouped_df_train, X_train, y_train, train_dir), (grouped_df_val, X_val, y_val, validation_dir)]:
                for image_id in gdf.index:
                    image_path = img_dir+str(image_id)+".jpg"
                    if os.path.exists(image_path):
                        try:
                            # get X
                            img = load_img(image_path, target_size=target_size)
                            arr = img_to_array(img)
                            X.append(arr)
                            # get y
                            y_pos = le.transform(gdf[image_id])
                            y_lab = np.zeros((n_classes,), dtype=int)
                            y_lab[y_pos] = 1
                            y.append(y_lab)
                        except OSError:
                            if verbose >= 2: print("OSError on image %s."%image_path)
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_val = np.array(X_val)
            y_val = np.array(y_val)

            print(X_train.shape)
            print(y_train.shape)
            print(X_val.shape)
            print(y_val.shape)
            print(y_train)
            print(y_val)

            ### Train model
            if verbose >= 1: print("\tFine-tuning Inception V3 first two passes (task %d)..."%tid)
            finetune(base_model, model, X_train, y_train, X_val, y_val, batch_size=32,
                     nb_train_samples=len(y_train), nb_validation_samples=len(y_val),
                     patience_1=1, patience_2=2,
                     inception_h5_1=model_dir+"inceptionv3_fine_tuned_1_%d.h5"%tid,
                     inception_h5_2=model_dir+"inceptionv3_fine_tuned_2_%d.h5"%tid,
                     inception_h5_check_point_1=model_dir+"inceptionv3_fine_tuned_check_point_1_%d.h5"%tid,
                     inception_h5_check_point_2=model_dir+"inceptionv3_fine_tuned_check_point_2_%d.h5"%tid,
                     layer_names_file=model_dir+"inceptionv3_mod_layer_names.txt",
                     verbose=verbose)
            del(base_model)
            del(model)
            if verbose >= 1: print("\tFine-tuning Inception V3 third pass (task %d)..."%tid)
            finetune_from_saved(model_dir+"inceptionv3_fine_tuned_check_point_2_%d.h5"%tid,
                                model_dir+"inceptionv3_fine_tuned_3_%d.h5"%tid,
                                model_dir+"inceptionv3_mod_%d.json"%tid,
                                X_train, y_train, X_val, y_val, batch_size=32,
                                patience=5,
                                nb_train_samples=len(y_train), nb_validation_samples=len(y_val),
                                inception_h5_check_point=model_dir+"inceptionv3_fine_tuned_check_point_3_%d.h5"%tid,
                                verbose=verbose)
            K.clear_session()




#==============================================
#                   Main
#==============================================
if __name__ == '__main__':
    df_labels_train = pd.read_csv("../data/imaterialist/fgvc4_iMat.train.data.csv")
    df_labels_val = pd.read_csv("../data/imaterialist/fgvc4_iMat.validation.data.csv")
    train_for_each_task(df_labels_train, df_labels_val)
