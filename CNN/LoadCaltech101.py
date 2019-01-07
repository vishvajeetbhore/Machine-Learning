"""
Adapted from https://github.com/yoavz/transfer-learning-keras
"""
from __future__ import print_function
import os, sys, json, h5py
import numpy as np

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

def load_data(datadir="data", h5file="output/Caltech101.h5", target_size=(224, 224, 3), train_test_ratio=0.7, recache=False):

    if recache or not os.path.isfile(h5file):

        all_directories = list(sorted([ x[0] for x in os.walk(datadir) ][1:]))
        labels = {idx: os.path.basename(d) for idx, d in enumerate(all_directories)}
        with open("output/labels.json", "w") as f:
            json.dump(labels, f)
        num_labels = len(labels)

        # Create the h5 file
        f = h5py.File(h5file, mode="w")
        X_train = f.create_dataset('X_train', (0,) + target_size, maxshape=(None,) + target_size, dtype='f', chunks=True)
        Y_train = f.create_dataset('Y_train', (0, num_labels), maxshape=(None, num_labels), dtype='i', chunks=True)
        X_test = f.create_dataset('X_test', (0,) + target_size, maxshape=(None,) + target_size, dtype='f', chunks=True)
        Y_test = f.create_dataset('Y_test', (0, num_labels), maxshape=(None, num_labels), dtype='i', chunks=True)

        nb_train = 0
        nb_test = 0

        # Iterate over all folders
        for class_idx, name  in labels.items():
            print("Loading {} images...".format(name))

            # List images
            class_path = datadir + "/" + name
            files = [os.path.join(class_path, n) for n in os.listdir(class_path)
                        if os.path.isfile(os.path.join(class_path, n))]
            
            # Train/test split
            N = len(files)
            N_train = int(np.floor(N * train_test_ratio))

            # One-hot encoding of the label
            labels = np.zeros(num_labels, np.int)
            labels[int(class_idx)] = 1
            
            # Arrays to store the data
            X_train_arr = np.zeros((N_train,) + target_size)
            Y_train_arr = np.zeros((N_train, num_labels), dtype=np.int)
            X_test_arr = np.zeros((N-N_train,) + target_size)
            Y_test_arr = np.zeros((N-N_train, num_labels), dtype=np.int)

            for img_idx, img_file in enumerate(files):
                # Load the image
                img = image.load_img(img_file, target_size=target_size[:2])
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                if img_idx < N_train:
                    X_train_arr[img_idx, ...] = x
                    Y_train_arr[img_idx, ...] = labels
                else:
                    X_test_arr[img_idx - N_train, ...] = x
                    Y_test_arr[img_idx - N_train, ...] = labels

            # Write to HDF5 file
            X_train.resize((nb_train+len(X_train_arr),) + target_size)
            Y_train.resize((nb_train+len(X_train_arr), num_labels))
            X_train[nb_train:nb_train+len(X_train_arr), ...] = X_train_arr
            Y_train[nb_train:nb_train+len(X_train_arr), ...] = Y_train_arr
            nb_train += len(X_train_arr)

            X_test.resize((nb_test+len(X_test_arr),) + target_size)
            Y_test.resize((nb_test+len(X_test_arr), num_labels))
            X_test[nb_test:nb_test+len(X_test_arr), ...] = X_test_arr
            Y_test[nb_test:nb_test+len(X_test_arr), ...] = Y_test_arr
            nb_test += len(X_test_arr)

        print("Saving data in ", h5file)
        f.close() 
        
    print('Loading data from', h5file)
    f = h5py.File(h5file, mode="r")
    X_train = f["X_train"][:]
    X_test = f["X_test"][:]
    y_train = f["Y_train"][:]
    y_test = f["Y_test"][:]
    f.close()

    return ((X_train, y_train), (X_test, y_test))

if __name__ == "__main__":
    load_data(recache=True)
