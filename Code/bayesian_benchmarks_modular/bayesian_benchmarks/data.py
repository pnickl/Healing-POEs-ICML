# Copyright 2017 Hugh Salimbeni
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import os
import pandas
import logging
from datetime import datetime
from scipy.io import loadmat
import pickle
import shutil
import pandas as pd
import tensorflow as tf

from sklearn.decomposition import PCA
from urllib.request import urlopen
logging.getLogger().setLevel(logging.INFO)
import zipfile
import gzip
import struct
import array

from bayesian_benchmarks.paths import DATA_PATH, BASE_SEED

import scipy.io as sio
from sklearn.preprocessing import StandardScaler
# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')


_ALL_REGRESSION_DATATSETS = {}
_ALL_CLASSIFICATION_DATATSETS = {}



def add_regression(C):
    _ALL_REGRESSION_DATATSETS.update({C.name:C})
    return C

def add_classficiation(C):
    _ALL_CLASSIFICATION_DATATSETS.update({C.name:C})
    return C





def normalize(X):
    X_mean = np.average(X, 0)[None, :]
    X_std = 1e-6 + np.std(X, 0)[None, :]
    return (X - X_mean) / X_std, X_mean, X_std


class Dataset(object):
    def __init__(self, split=0, prop=0.9):
        if self.needs_download:
            self.download()

        X_raw, Y_raw = self.read_data()
        X, Y = self.preprocess_data(X_raw, Y_raw)

        ind = np.arange(self.N)

        np.random.seed(BASE_SEED + split)
        np.random.shuffle(ind)

        n = int(self.N * prop)

        self.X_train = X[ind[:n]]
        self.Y_train = Y[ind[:n]]

        self.X_test = X[ind[n:]]
        self.Y_test = Y[ind[n:]]

    @property
    def datadir(self):
        dir = os.path.join(DATA_PATH, self.name)
        if not os.path.isdir(dir):
            os.mkdir(dir)
        return dir

    @property
    def datapath(self):
        filename = self.url.split('/')[-1]  # this is for the simple case with no zipped files
        #print(filename)
        #print(os.path.join(self.datadir, filename))
        return os.path.join(self.datadir, filename)

    @property
    def needs_download(self):
        return not os.path.isfile(self.datapath)

    def download(self):
        logging.info('donwloading {} data'.format(self.name))

        is_zipped = np.any([z in self.url for z in ['.gz', '.zip', '.tar']])

        if is_zipped:
            filename = os.path.join(self.datadir, self.url.split('/')[-1])
        else:
            filename = self.datapath

        with urlopen(self.url) as response, open(filename, 'wb') as out_file:
            data = response.read()
            out_file.write(data)

        if is_zipped:
            zip_ref = zipfile.ZipFile(filename, 'r')
            zip_ref.extractall(self.datadir)
            zip_ref.close()

            # os.remove(filename)

        logging.info('finished donwloading {} data'.format(self.name))

    def read_data(self):
        raise NotImplementedError

    def preprocess_data(self, X, Y):
        X, self.X_mean, self.X_std = normalize(X)
        Y, self.Y_mean, self.Y_std = normalize(Y)
        return X, Y

class DatasetRobotics(object):
    def __init__(self, split=0, prop=0.9):
        # if self.needs_download:
        #     self.download()

        X_raw, Y_raw, N_train, N_test = self.read_data()
        X, Y = self.preprocess_data(X_raw, Y_raw)

        ind = np.arange(N_train + N_test)

        # # Don't randomize, use provided train/test splits
        # np.random.seed(BASE_SEED + split)
        # np.random.shuffle(ind)

        self.X_train = X[ind[:N_train]]
        self.Y_train = Y[ind[:N_train]]

        self.X_test = X[ind[N_train:]]
        self.Y_test = Y[ind[N_train:]]

    @property
    def datadir(self):
        dir = os.path.join(DATA_PATH, self.name)
        if not os.path.isdir(dir):
            os.mkdir(dir)
        return dir

    @property
    def datapath(self):
        filename = self.url.split('/')[-1]  # this is for the simple case with no zipped files
        #print(filename)
        #print(os.path.join(self.datadir, filename))
        return os.path.join(self.datadir, filename)

    @property
    def needs_download(self):
        return not os.path.isfile(self.datapath)

    def download(self):
        logging.info('donwloading {} data'.format(self.name))

        is_zipped = np.any([z in self.url for z in ['.gz', '.zip', '.tar']])

        if is_zipped:
            filename = os.path.join(self.datadir, self.url.split('/')[-1])
        else:
            filename = self.datapath

        with urlopen(self.url) as response, open(filename, 'wb') as out_file:
            data = response.read()
            out_file.write(data)

        if is_zipped:
            zip_ref = zipfile.ZipFile(filename, 'r')
            zip_ref.extractall(self.datadir)
            zip_ref.close()

            # os.remove(filename)

        logging.info('finished donwloading {} data'.format(self.name))

    def read_data(self):
        raise NotImplementedError

    def preprocess_data(self, X, Y):
        X, self.X_mean, self.X_std = normalize(X)
        Y, self.Y_mean, self.Y_std = normalize(Y)
        return X, Y

uci_base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'


#@add_regression
class Sarcos0(DatasetRobotics):
    N, D, name = 48933, 21, 'sarcos0'

    def needs_download(self):
        return False

    def read_data(self):
        select_output = 0

        train_data = sio.loadmat(DATA_PATH + '/sarcos/sarcos_inv_train.mat')['sarcos_inv']
        train_input = train_data[:, 0:21]
        train_target = train_data[:, 21:28]

        test_data = sio.loadmat(DATA_PATH + '/sarcos/sarcos_inv_test.mat')['sarcos_inv_test']
        test_input = test_data[:, 0:21]
        test_target = test_data[:, 21:28]

        N_train = len(train_input)
        N_test = len(test_input)

        input_data = np.vstack((train_input, test_input))
        target_data = np.vstack((np.expand_dims(train_target[:, select_output], axis=1),
                                 np.expand_dims(test_target[:, select_output], axis=1)))

        return input_data, target_data, N_train, N_test

@add_regression
class Wam0(DatasetRobotics):
    N, D, name = 15000, 12, 'wam0'
    def needs_download(self):
        return False

    def read_data(self):
        select_output = 0

        data = sio.loadmat(DATA_PATH + '/wam_ias/ias_real_barrett_data.mat')
        train_input = data['X_train']
        test_input = data['X_test']
        train_target = data['Y_train']
        test_target = data['Y_test']

        N_train = len(train_input)
        N_test = len(test_input)

        input_data = np.vstack((train_input, test_input))
        target_data = np.vstack((np.expand_dims(train_target[:, select_output], axis=1),
                                 np.expand_dims(test_target[:, select_output], axis=1)))

        return input_data, target_data, N_train, N_test

@add_regression
class Wam1(DatasetRobotics):
    N, D, name = 15000, 12, 'wam1'
    def needs_download(self):
        return False

    def read_data(self):
        select_output = 1

        data = sio.loadmat(DATA_PATH + '/wam_ias/ias_real_barrett_data.mat')
        train_input = data['X_train']
        test_input = data['X_test']
        train_target = data['Y_train']
        test_target = data['Y_test']

        N_train = len(train_input)
        N_test = len(test_input)

        input_data = np.vstack((train_input, test_input))
        target_data = np.vstack((np.expand_dims(train_target[:, select_output], axis=1),
                                 np.expand_dims(test_target[:, select_output], axis=1)))

        return input_data, target_data, N_train, N_test

@add_regression
class Wam2(DatasetRobotics):
    N, D, name = 15000, 12, 'wam2'
    def needs_download(self):
        return False

    def read_data(self):
        select_output = 2

        data = sio.loadmat(DATA_PATH + '/wam_ias/ias_real_barrett_data.mat')
        train_input = data['X_train']
        test_input = data['X_test']
        train_target = data['Y_train']
        test_target = data['Y_test']

        N_train = len(train_input)
        N_test = len(test_input)

        input_data = np.vstack((train_input, test_input))
        target_data = np.vstack((np.expand_dims(train_target[:, select_output], axis=1),
                                 np.expand_dims(test_target[:, select_output], axis=1)))

        return input_data, target_data, N_train, N_test

@add_regression
class Wam3(DatasetRobotics):
    N, D, name = 15000, 12, 'wam3'
    def needs_download(self):
        return False

    def read_data(self):
        select_output = 3

        data = sio.loadmat(DATA_PATH + '/wam_ias/ias_real_barrett_data.mat')
        train_input = data['X_train']
        test_input = data['X_test']
        train_target = data['Y_train']
        test_target = data['Y_test']

        N_train = len(train_input)
        N_test = len(test_input)

        input_data = np.vstack((train_input, test_input))
        target_data = np.vstack((np.expand_dims(train_target[:, select_output], axis=1),
                                 np.expand_dims(test_target[:, select_output], axis=1)))

        return input_data, target_data, N_train, N_test

#@add_regression
class Sarcos1(DatasetRobotics):
    N, D, name = 48933, 21, 'sarcos1'

    def needs_download(self):
        return False

    def read_data(self):
        select_output = 1

        train_data = sio.loadmat(DATA_PATH + '/sarcos/sarcos_inv_train.mat')['sarcos_inv']
        train_input = train_data[:, 0:21]
        train_target = train_data[:, 21:28]

        test_data = sio.loadmat(DATA_PATH + '/sarcos/sarcos_inv_test.mat')['sarcos_inv_test']
        test_input = test_data[:, 0:21]
        test_target = test_data[:, 21:28]

        N_train = len(train_input)
        N_test = len(test_input)

        input_data = np.vstack((train_input, test_input))
        target_data = np.vstack((np.expand_dims(train_target[:, select_output], axis=1),
                                 np.expand_dims(test_target[:, select_output], axis=1)))

        return input_data, target_data, N_train, N_test

#@add_regression
class Sarcos2(DatasetRobotics):
    N, D, name = 48933, 21, 'sarcos2'

    def needs_download(self):
        return False

    def read_data(self):
        select_output = 2

        train_data = sio.loadmat(DATA_PATH + '/sarcos/sarcos_inv_train.mat')['sarcos_inv']
        train_input = train_data[:, 0:21]
        train_target = train_data[:, 21:28]

        test_data = sio.loadmat(DATA_PATH + '/sarcos/sarcos_inv_test.mat')['sarcos_inv_test']
        test_input = test_data[:, 0:21]
        test_target = test_data[:, 21:28]

        N_train = len(train_input)
        N_test = len(test_input)

        input_data = np.vstack((train_input, test_input))
        target_data = np.vstack((np.expand_dims(train_target[:, select_output], axis=1),
                                 np.expand_dims(test_target[:, select_output], axis=1)))

        return input_data, target_data, N_train, N_test

#@add_regression
class Sarcos3(DatasetRobotics):
    N, D, name = 48933, 21, 'sarcos3'

    def needs_download(self):
        return False

    def read_data(self):
        select_output = 3

        train_data = sio.loadmat(DATA_PATH + '/sarcos/sarcos_inv_train.mat')['sarcos_inv']
        train_input = train_data[:, 0:21]
        train_target = train_data[:, 21:28]

        test_data = sio.loadmat(DATA_PATH + '/sarcos/sarcos_inv_test.mat')['sarcos_inv_test']
        test_input = test_data[:, 0:21]
        test_target = test_data[:, 21:28]

        N_train = len(train_input)
        N_test = len(test_input)

        input_data = np.vstack((train_input, test_input))
        target_data = np.vstack((np.expand_dims(train_target[:, select_output], axis=1),
                                 np.expand_dims(test_target[:, select_output], axis=1)))

        return input_data, target_data, N_train, N_test

#@add_regression
class Sarcos4(DatasetRobotics):
    N, D, name = 48933, 21, 'sarcos4'

    def needs_download(self):
        return False

    def read_data(self):
        select_output = 4

        train_data = sio.loadmat(DATA_PATH + '/sarcos/sarcos_inv_train.mat')['sarcos_inv']
        train_input = train_data[:, 0:21]
        train_target = train_data[:, 21:28]

        test_data = sio.loadmat(DATA_PATH + '/sarcos/sarcos_inv_test.mat')['sarcos_inv_test']
        test_input = test_data[:, 0:21]
        test_target = test_data[:, 21:28]

        N_train = len(train_input)
        N_test = len(test_input)

        input_data = np.vstack((train_input, test_input))
        target_data = np.vstack((np.expand_dims(train_target[:, select_output], axis=1),
                                 np.expand_dims(test_target[:, select_output], axis=1)))

        return input_data, target_data, N_train, N_test

#@add_regression
class Sarcos5(DatasetRobotics):
    N, D, name = 48933, 21, 'sarcos5'

    def needs_download(self):
        return False

    def read_data(self):
        select_output = 5

        train_data = sio.loadmat(DATA_PATH + '/sarcos/sarcos_inv_train.mat')['sarcos_inv']
        train_input = train_data[:, 0:21]
        train_target = train_data[:, 21:28]

        test_data = sio.loadmat(DATA_PATH + '/sarcos/sarcos_inv_test.mat')['sarcos_inv_test']
        test_input = test_data[:, 0:21]
        test_target = test_data[:, 21:28]

        N_train = len(train_input)
        N_test = len(test_input)

        input_data = np.vstack((train_input, test_input))
        target_data = np.vstack((np.expand_dims(train_target[:, select_output], axis=1),
                                 np.expand_dims(test_target[:, select_output], axis=1)))

        return input_data, target_data, N_train, N_test

#@add_regression
class Sarcos6(DatasetRobotics):
    N, D, name = 48933, 21, 'sarcos6'

    def needs_download(self):
        return False

    def read_data(self):
        select_output = 6

        train_data = sio.loadmat(DATA_PATH + '/sarcos/sarcos_inv_train.mat')['sarcos_inv']
        train_input = train_data[:, 0:21]
        train_target = train_data[:, 21:28]

        test_data = sio.loadmat(DATA_PATH + '/sarcos/sarcos_inv_test.mat')['sarcos_inv_test']
        test_input = test_data[:, 0:21]
        test_target = test_data[:, 21:28]

        N_train = len(train_input)
        N_test = len(test_input)

        input_data = np.vstack((train_input, test_input))
        target_data = np.vstack((np.expand_dims(train_target[:, select_output], axis=1),
                                 np.expand_dims(test_target[:, select_output], axis=1)))

        return input_data, target_data, N_train, N_test

# @add_regression
# class Wam0(DatasetRobotics):
#     N, D, name = 30000, 12, 'wam0'
#     def needs_download(self):
#         return False
#
#     def read_data(self):
#         select_output = 0
#
#         train_input = np.load(DATA_PATH + '/wam/wam_invdyn_train.npz')['input'][0:25000]
#         train_target = np.load(DATA_PATH + '/wam/wam_invdyn_train.npz')['target'][0:25000]
#         test_input = np.load(DATA_PATH + '/wam/wam_invdyn_test.npz')['input'][0:5000]
#         test_target = np.load(DATA_PATH + '/wam/wam_invdyn_test.npz')['target'][0:5000]
#
#         N_train = len(train_input)
#         N_test = len(test_input)
#
#         input_data = np.vstack((train_input, test_input))
#         target_data = np.vstack((np.expand_dims(train_target[:, select_output], axis=1),
#                                  np.expand_dims(test_target[:, select_output], axis=1)))
#
#         return input_data, target_data, N_train, N_test
#
# @add_regression
# class Wam1(DatasetRobotics):
#     N, D, name = 30000, 12, 'wam1'
#     def needs_download(self):
#         return False
#
#     def read_data(self):
#         select_output = 1
#
#         train_input = np.load(DATA_PATH + '/wam/wam_invdyn_train.npz')['input'][0:25000]
#         train_target = np.load(DATA_PATH + '/wam/wam_invdyn_train.npz')['target'][0:25000]
#         test_input = np.load(DATA_PATH + '/wam/wam_invdyn_test.npz')['input'][0:5000]
#         test_target = np.load(DATA_PATH + '/wam/wam_invdyn_test.npz')['target'][0:5000]
#
#         N_train = len(train_input)
#         N_test = len(test_input)
#
#         input_data = np.vstack((train_input, test_input))
#         target_data = np.vstack((np.expand_dims(train_target[:, select_output], axis=1),
#                                  np.expand_dims(test_target[:, select_output], axis=1)))
#
#         return input_data, target_data, N_train, N_test
#
# @add_regression
# class Wam2(DatasetRobotics):
#     N, D, name = 30000, 12, 'wam2'
#     def needs_download(self):
#         return False
#
#     def read_data(self):
#         select_output = 2
#
#         train_input = np.load(DATA_PATH + '/wam/wam_invdyn_train.npz')['input'][0:25000]
#         train_target = np.load(DATA_PATH + '/wam/wam_invdyn_train.npz')['target'][0:25000]
#         test_input = np.load(DATA_PATH + '/wam/wam_invdyn_test.npz')['input'][0:5000]
#         test_target = np.load(DATA_PATH + '/wam/wam_invdyn_test.npz')['target'][0:5000]
#
#         N_train = len(train_input)
#         N_test = len(test_input)
#
#         input_data = np.vstack((train_input, test_input))
#         target_data = np.vstack((np.expand_dims(train_target[:, select_output], axis=1),
#                                  np.expand_dims(test_target[:, select_output], axis=1)))
#
#         return input_data, target_data, N_train, N_test
#
# @add_regression
# class Wam3(DatasetRobotics):
#     N, D, name = 30000, 12, 'wam3'
#     def needs_download(self):
#         return False
#
#     def read_data(self):
#         select_output = 3
#
#         train_input = np.load(DATA_PATH + '/wam/wam_invdyn_train.npz')['input'][0:25000]
#         train_target = np.load(DATA_PATH + '/wam/wam_invdyn_train.npz')['target'][0:25000]
#         test_input = np.load(DATA_PATH + '/wam/wam_invdyn_test.npz')['input'][0:5000]
#         test_target = np.load(DATA_PATH + '/wam/wam_invdyn_test.npz')['target'][0:5000]
#
#         N_train = len(train_input)
#         N_test = len(test_input)
#
#         input_data = np.vstack((train_input, test_input))
#         target_data = np.vstack((np.expand_dims(train_target[:, select_output], axis=1),
#                                  np.expand_dims(test_target[:, select_output], axis=1)))
#
#         return input_data, target_data, N_train, N_test

#@add_regression
class Concrete(Dataset):
    N, D, name = 1030, 8, 'concrete'
    url = uci_base_url + 'concrete/compressive/Concrete_Data.xls'

    def read_data(self):
        data = pandas.read_excel(self.datapath).values
        return data[:, :-1], data[:, -1].reshape(-1, 1)


#@add_regression
class Airline(Dataset):
    N, D, name = 800000, 7, 'airline'
    url = '/DelayedFlights_all.csv'

    def read_data(self):
        print(self.datapath)
        
        data=pd.read_csv(self.datapath)
        data=data[['DayofMonth','DayOfWeek','Month','AirTime','DepTime','ArrTime','Distance','ArrDelay']].dropna().iloc[:800000,:].values

        return data[:, :-1], data[:, -1].reshape(-1, 1)    
    
#@add_regression
class Power(Dataset):
    N, D, name = 9568, 4, 'power'
    url = uci_base_url + '00294/CCPP.zip'

    @property
    def datapath(self):
        return os.path.join(self.datadir, 'CCPP/Folds5x2_pp.xlsx')

    def read_data(self):
        data = pandas.read_excel(self.datapath).values
        return data[:, :-1], data[:, -1].reshape(-1, 1)


# Andrew Wilson's datasets
#https://drive.google.com/open?id=0BxWe_IuTnMFcYXhxdUNwRHBKTlU
class WilsonDataset(Dataset):
    @property
    def datapath(self):
        n = self.name[len('wilson_'):]
        return '{}/uci/{}/{}.mat'.format(DATA_PATH, n, n)

    def read_data(self):
        data = loadmat(self.datapath)['data']
        return data[:, :-1], data[:, -1, None]


#@add_regression
class Wilson_airfoil(WilsonDataset):
    name, N, D = 'wilson_airfoil', 1503, 5



#@add_regression
class Wilson_parkinsons(WilsonDataset):
    name, N, D = 'wilson_parkinsons', 5875, 20


#@add_regression
class Wilson_kin40k(WilsonDataset):
    name, N, D = 'wilson_kin40k', 40000, 8

#@add_regression
class Wilson_protein(WilsonDataset):
    name, N, D = 'wilson_protein', 45730, 9

    
    
class Classification(Dataset):
    def preprocess_data(self, X, Y):
        X, self.X_mean, self.X_std = normalize(X)
        return X, Y

    @property
    def needs_download(self):
        if os.path.isfile(os.path.join(DATA_PATH, 'classification_data', 'iris', 'iris_R.dat')):
            return False
        else:
            return True
    print(1)
    def download(self):
        pass


    def read_data(self, components = None):
        datapath = os.path.join(DATA_PATH, 'classification_data', self.name, self.name + '_R.dat')
        if os.path.isfile(datapath):
            data = np.array(pandas.read_csv(datapath, header=0, delimiter='\t').values).astype(float)
        else:
            data_path1 = os.path.join(DATA_PATH, 'classification_data', self.name, self.name + '_train_R.dat')
            data1 = np.array(pandas.read_csv(data_path1, header=0, delimiter='\t').values).astype(float)

            data_path2 = os.path.join(DATA_PATH, 'classification_data', self.name, self.name + '_test_R.dat')
            data2 = np.array(pandas.read_csv(data_path2, header=0, delimiter='\t').values).astype(float)

            data = np.concatenate([data1, data2], 0)

        return data[:, :-1], data[:, -1].reshape(-1, 1)


rescale = lambda x, a, b: b[0] + (b[1] - b[0]) * x / (a[1] - a[0])


@add_classficiation
class MNIST(Classification):

     name = 'MNIST'
     N = 70000
     D = 20
     K = 10
     #needs_download = False
    
   
     


     def read_data(self, components = 20):
        """Reads data from tf.keras and concatenates training and testing sets for randomisation
        - Creates features as prinicipal components (PCA) of pixel values  
        - Output: (feature matrix: 700000 X n_components , labels vector(ordinal) : 70000 x 1

        """
        def parse_images(filename):

            with gzip.open(filename, 'rb') as fh:
                magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
                return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

        def parse_labels(filename):
            with gzip.open(filename, 'rb') as fh:
                magic, num_data = struct.unpack(">II", fh.read(8))
                return np.array(array.array("B", fh.read()), dtype=np.uint8)
           
        train_images = parse_images('./bayesian_benchmarks/data/mnist/train-images-idx3-ubyte.gz')
        train_labels = parse_labels('./bayesian_benchmarks/data/mnist/train-labels-idx1-ubyte.gz')
        test_images  = parse_images('./bayesian_benchmarks/data/mnist/t10k-images-idx3-ubyte.gz')
        test_labels  = parse_labels('./bayesian_benchmarks/data/mnist/t10k-labels-idx1-ubyte.gz')

        train_images = train_images / 255.0
        test_images = test_images / 255.0
        train_images = np.reshape(train_images, (60000, 784))
        test_images = np.reshape(test_images, (10000, 784))
        train_images = tf.concat([train_images, test_images], 0)
        train_labels = tf.concat([train_labels, test_labels], 0)
 
        pca = PCA(n_components = components)
        pca.fit(train_images)
        train_image_pca = pca.transform(train_images)
        train_labels = tf.dtypes.cast(train_labels, tf.int64)
        train_labels = tf.reshape(train_labels, [-1, 1])

        return train_image_pca, train_labels.numpy()


##########################

regression_datasets = list(_ALL_REGRESSION_DATATSETS.keys())
regression_datasets.sort()

classification_datasets = list(_ALL_CLASSIFICATION_DATATSETS.keys())
classification_datasets.sort()

def get_regression_data(name, *args, **kwargs):
    return _ALL_REGRESSION_DATATSETS[name](*args, **kwargs)


def get_classification_data(name, *args, **kwargs):
    return _ALL_CLASSIFICATION_DATATSETS[name](*args, **kwargs)

