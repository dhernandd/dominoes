# Copyright 2017, Daniel Hernandez Diaz, Columbia University. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ======================================================================
from __future__ import print_function

import numpy as np
import h5py

import tensorflow as tf

# from runner import simulate_after_state
# from utils import make_afterstate_dict

# print(f.keys())
# d = f['game4']
# data = [data1]
# for game in f:
#     num_dominoes = f[game]['num_dominoes'][:]
#     if not all(num_dominoes):
#         print(game, num_dominoes)
#         del f[game]
#         ctr +=1
# 
# print(d['board_state'][:])
# b = d['board_state'][:]

def to_new_board_state(datafile_name):
    """
    """
    datafile = h5py.File(datafile_name,'a')
    for k in datafile.keys():
        data = datafile[k]
        board_state = data['board_state'][:]
        
        bs = np.zeros_like(board_state[:,2:])
        bs[0] = board_state[0,2:]
        h1 = bs[0,0]
        h2 = bs[0,1]
        for n, m in enumerate(board_state[1:], 1):
            d = m[2:]
            p = m[1]
            if p == 2:
                d = d if d[0] == h2 else d[::-1]
                h2 = d[1]
                bs[n] = np.array([p, h2])
            elif p == 1:
                d = d if d[1] == h1 else d[::-1]
                h1 = d[0]
                bs[n] = np.array([p, h1])
            elif p == -1:
                bs[n] = np.array([-1, 0])
            else:
                bs[n] = np.zeros(2)
        
        data['new_bs'] = bs


def flatten_input(datafile_name):
    """
    """
    datafile = h5py.File(datafile_name,'a')
    for name in datafile.keys():
        data = datafile[name]
    
        board_state = data['new_bs'][:]
        num_dominoes = np.array(data['num_dominoes'][:])
        passes = data['passes'][:]
        player_hand = data['player_hand'][:]
    
        flat_input = np.expand_dims(np.concatenate((np.array(num_dominoes), player_hand.flatten(),
                                 passes.flatten(), board_state.flatten())), 0)

        data['input'] = flat_input


def make_dict_database(file_name):
    f = h5py.File(file_name,'r')

    num_games = len(f.keys())
    db = {'input' : np.zeros((num_games, 181)), 'output' : np.zeros((num_games, 2))}
    for i, g in enumerate(f.keys()):
        db['input'][i] = f[g]['input'][:]
        db['output'][i] = f[g]['win_probs'][:]
    
    return db


def make_total_dict_db(list_files):
    
    for i, file_name in enumerate(list_files):
        print('Processing file', file_name)
        db = make_dict_database(file_name)
        if not i:
            total_input = db['input']
            total_output = db['output']
        else:
            total_input = np.concatenate((total_input, db['input']))
            total_output = np.concatenate((total_output, db['output']))
    
    d = {'input' : total_input, 'output' : total_output}        
    
    import cPickle as pickle
    pickle.dump(d, open('data/domino_main.db', "w+"))


def add_draw_prob(y_data):
    new_y = np.zeros((len(y_data), len(y_data[0])+1))
    for i, y in enumerate(y_data):
        new_y[i,:2] = y_data[i]
        new_y[i,2] = 1 - np.sum(y_data[i])
    
    return new_y

# import pickle                 
# db = pickle.load(open('data/domino_main.db'))
# print(db['input'][2000], db['output'][2000])
# print(len(db['input']))
 
# db = make_dict_database('data/set1.hdf5')
# print(db['input'][0:2], db['output'][0:2])

# to_new_board_state('data/set17.hdf5')
# to_new_board_state('data/set19.hdf5')
# flatten_input('data/set17.hdf5')
# flatten_input('data/set19.hdf5')
# list_files = ['data/set1.hdf5', 'data/set10.hdf5', 'data/set11.hdf5',
#               'data/set12.hdf5','data/set13.hdf5',
#               'data/set14.hdf5',  
#               'data/set17.hdf5', 'data/set19.hdf5']
# make_total_dict_db(list_files)


# import pickle                 
# db = pickle.load(open('data/domino_main.db'))
# x, y = db['input'][41:51], db['output'][41:51]
# print(len(db['input']))

# y = add_draw_prob(y)
# 
# with tf.Session() as sess:
#     new_saver = tf.train.import_meta_graph('saves/nn_0.0/model_0.0.meta')
#     new_saver.restore(sess, tf.train.latest_checkpoint('saves/nn_0.0/'))
# 
#     print(sess.run('softmax/softmax:0', feed_dict={'Inputs:0' : x}))
#     print(y)


    
# 
f = h5py.File('data/set19.hdf5','a')
g = f['game3213']
# for g in f.keys():
key = unicode('win_probs', "utf-8")
print(g[key][:])
# print(g_name)
# print(g.keys())
# print(g['new_bs'][:])
# print(g['player_hand'][:])
# to_new_board_state(f)
# flatten_input(f)

# g = f['game1']
# print(g.keys())
# print(g['board_state'][:])
# print(g['win_probs'][:])
# print(len(g['input'][:][0]))
    
# for g in f.keys():
#     data = f[g]
#     board_state = data['board_state'][:]
#     data['new_bs'] = to_new_board_state(board_state)

# del g['new_key']
# print(g.keys())
# print(g['new_bs'][:])
# print('Done')
    

# print('Deteled', ctr, 'elems')
# print(d.keys())
# print(d['player'][()])
# print(d['player_hand'][:])
# print(d['num_dominoes'][:])
# print(d['passes'][:])
# print(d['board_state'][:])

# after_state_dict = make_afterstate_dict(d,)
# simulate_after_state(after_state_dict, hand_size=7, num_sims=10000)
# print(d['win_probs'][:])