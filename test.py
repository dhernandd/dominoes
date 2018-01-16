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

# from runner import simulate_after_state
# from utils import make_afterstate_dict

f = h5py.File('data/set49.hdf5','a')

print(f.keys())
# d = f['game1033']
# data = [data1]
ctr = 0
for game in f:
    num_dominoes = f[game]['num_dominoes'][:]
    if not all(num_dominoes):
        print(game, num_dominoes)
        del f[game]
        ctr +=1

print('Deteled', ctr, 'elems')
# print(d.keys())
# print(d['player'][()])
# print(d['player_hand'][:])
# print(d['num_dominoes'][:])
# print(d['passes'][:])
# print(d['board_state'][:])

# after_state_dict = make_afterstate_dict(d,)
# simulate_after_state(after_state_dict, hand_size=7, num_sims=10000)
# print(d['win_probs'][:])

