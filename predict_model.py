"""
    Name:           predict_model.py
    Created:        29/6/2017
    Description:    Predict for inception v3.
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
from tqdm import tqdm
#==============================================
#                   Files
#==============================================
from train_model import preprocess_input

#==============================================
#                   Functions
#==============================================
def load_model(chosen_metrics=['top_k_categorical_accuracy', 'categorical_accuracy', 'mse', 'mape', 'cosine'],
                inception_json="inceptionv3_mod.json", inception_h5="inceptionv3_fine_tuned_2.h5", verbose=1):
    """
    Load the inception v3 and trained weights from disk.
    """

    # load json and create model
    with open(inception_json, 'r') as iOF:
        loaded_model_json = iOF.read()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(inception_h5)
    if verbose >= 1: print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                         metrics=chosen_metrics)

    return loaded_model





def infer(model, X_test, y_test, batch_size=32, img_width=299, img_height=299, verbose=1):
    """
    Infer with the inception v3.
    """

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # number of samples
    nb_test_samples = len(y_test)

    # add fake entries to get batch size multiple
    nb_add = batch_size - nb_test_samples%batch_size
    y_test_add = y_test[-nb_add:, ...]
    X_test_add = X_test[-nb_add:, ...]
    y_test_stacked = np.vstack([y_test, y_test_add])
    X_test_stacked = np.vstack([X_test, X_test_add])

    # define test data generators
    test_generator = test_datagen.flow(
        X_test_stacked,
        y_test_stacked,
        batch_size=batch_size,
        shuffle=False)

    y_pred = model.predict_generator(
        test_generator,
        steps=(nb_test_samples + nb_add) // batch_size)

    # get predictions
    y_pred = y_pred[:-nb_add, ...]
    y_pred = np.array([np.argmax(yp) for yp in y_pred])

    return y_pred





def predict_for_each_task(df_labels_test, df_labels_train, target_size=(299,299),
                          model_dir="../data/imaterialist/models/",
                          test_dir="../data/imaterialist/train_images/",
                          pred_csv_name="../data/imaterialist/imaterialist_submission_001.csv",
                          verbose=1):
    """
    Train an Inception V3 for each task.
    """

    ### define dictionary
    pred_dict = {"id": [], "expected": []}

    ### get all task ids
    different_tasks = set(df_labels_train.taskId)

    ### Load images
    if verbose >= 1: print("Loading images into RAM...")
    X_test, id_test, id_error = [], [], []
    for image_id in tqdm(df_labels_test['imageId'], miniters=100):
        image_path = test_dir+str(image_id)+".jpg"
        if os.path.exists(image_path):
            try:
                # get X
                img = load_img(image_path, target_size=target_size)
                arr = img_to_array(img)
                X_test.append(arr)
                # get id
                id_test.append(image_id)
            except OSError:
                id_error.append(image_id)
                if verbose >= 2: print("OSError on image %s."%image_path)
        else:
            id_error.append(image_id)
    X_test = np.array(X_test)
    print(X_test.shape)

    ### loop over tasks
    for tid in different_tasks:
        # for task 40 dress gender is always 4 (women)
        # for task 42 shooe pump type is always 232 (mary janes)
        # for task 43 ... is always 257
        # for task 45 ... is always 344

        if verbose >= 1: print("Prediction for task %d..."%tid)

        df_task_train = df_labels_train[df_labels_train.taskId == tid]
        n_classes = len(set(df_task_train.labelId))
        most_common_label = df_task_train.labelId.value_counts().idxmax()
        print("Defaulting: ", tid, most_common_label)

        if tid in [40, 42, 43, 45]:
            ##### Default to most common value
            if verbose >= 1: print("\tDefaulting to label %d (task %d)..."%(most_common_label,tid))
            for iid in df_labels_test['imageId']:
                pred_dict["id"].append("%d_%d"%(iid,tid))
                pred_dict["expected"].append(most_common_label)
        else:
            ##### Get predction from trained inception v3 net
            ### Load LabelEncoder
            with gzip.open(model_dir+'label_encoder_%d.pkl'%tid, 'rb') as iOF:
                le = pickle.load(iOF)

            ### Create fake y_test
            y_test = np.zeros((X_test.shape[0], n_classes), dtype=int)

            for iid in id_error:
                pred_dict["id"].append("%d_%d"%(iid,tid))
                pred_dict["expected"].append(most_common_label)

            ### Create model
            if verbose >= 1: print("\tLoading Inception V3 (task %d)..."%tid)
            model = load_model(chosen_metrics=None,
                               inception_json=model_dir+"inceptionv3_mod_%d.json"%tid,
                               inception_h5=model_dir+"inceptionv3_fine_tuned_check_point_3_%d.h5"%tid,
                               verbose=verbose)

            ### Inference with model
            if verbose >= 1: print("\tInference with Inception V3 (task %d)..."%tid)
            y_pred = infer(model, X_test, y_test, verbose=verbose)

            ### Transform with label encodera nd store in dict
            y_pred = le.inverse_transform(y_pred)
            for iid, yp in zip(id_test, y_pred):
                pred_dict["id"].append("%d_%d"%(iid,tid))
                pred_dict["expected"].append(yp)

            K.clear_session()

    pred_df = pd.DataFrame(pred_dict)
    print(pred_df.head())
    pred_df.to_csv(pred_csv_name, index=False, header=True)




#==============================================
#                   Main
#==============================================
if __name__ == '__main__':
    df_labels_test = pd.read_csv("../data/imaterialist/fgvc4_iMat.test.image.csv")
    df_labels_train = pd.read_csv("../data/imaterialist/fgvc4_iMat.train.data.csv")
    predict_for_each_task(df_labels_test, df_labels_train, target_size=(299,299),
                          model_dir="../data/imaterialist/models/",
                          test_dir="../data/imaterialist/train_images/",
                          pred_csv_name="../data/imaterialist/imaterialist_submission_001.csv",
                          verbose=1)
