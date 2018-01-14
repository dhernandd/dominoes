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
from __future__ import division

# import sys
# sys.

import numpy as np
import h5py

from utils import draw_init, draw_mc, roll_game, get_usable_dominoes, draw_arbitrary
from dominoes import Dominoes
from players import RandomDominoPlayer

TOPNUM = 7
DOMINO = [[i, j] for i in range(1,TOPNUM+1) for j in range(i, TOPNUM+1)]

NUM_PLAYERS = 4
HAND_SIZE = 7

DATA_FILE = 'set1.hd5f'

STARTING_DOMINOES = [[[7, 7], [4, 6], [6, 7], [3, 3], [1, 2], [4, 5], [1, 5]],
 [[2, 6], [5, 5], [1, 4], [4, 4], [6, 6], [5, 6], [3, 7]],
 [[3, 6], [2, 2], [2, 5], [3, 5], [2, 4], [1, 6], [5, 7]],
 [[3, 4], [1, 3], [1, 1], [4, 7], [2, 3], [1, 7], [2, 7]]]


def make_database(num_files=50):
    """
    """
    for i in range(num_files):
        print('\n\n\nCreating set', str(i), '\n\n\n')
        s_file = 'set' + str(i) + '.hdf5' 
        play_dominoes(save_file=s_file)

 
def play_dominoes(draw=draw_init, save_file=None):
    list_victories = [0]*(NUM_PLAYERS+1)
    game = Dominoes(NUM_PLAYERS, HAND_SIZE, TOPNUM, save_file=save_file)
    player1 = RandomDominoPlayer(1, game)
    player2 = RandomDominoPlayer(2, game)
    player3 = RandomDominoPlayer(3, game)
    player4 = RandomDominoPlayer(4, game)
#     list_players = (player1, player2)
#     list_players = (player1, player2, player3)
    list_players = (player1, player2, player3, player4)
    game.setup(list_players)

    for _ in range(1):
        
        game.reset()
        print('\n\nNew game!')
        starting_dominoes = draw(NUM_PLAYERS, DOMINO, HAND_SIZE)
        print('\nStarting_dominoes')
        print(starting_dominoes[0])
        print(starting_dominoes[1])
        print(starting_dominoes[2])
        print(starting_dominoes[3])
        for player, dominoes in zip(list_players, starting_dominoes):
            player.reset(start_dominoes=list(dominoes))
            
        winner = roll_game(game)
        list_victories[winner] += 1
        
        print('Salida:', game.salida)
        print('Game End!\n\n')
            
    print('\n', list_victories, '\t', list_victories[1]+list_victories[3], 
              list_victories[2]+list_victories[4])


def simulate_after_state(after_state_dict, hand_size=HAND_SIZE, num_sims=1):
    """
    """
    list_victories = [0]*(NUM_PLAYERS+1)
    
    from copy import deepcopy

    num_players = len(after_state_dict['num_dominoes'])
    
    board_state = after_state_dict['board_state']
    last_player_id = np.trim_zeros(board_state[:, 0])[-1]
    
    player_hand_one_hot = after_state_dict['player_hand']
    topnum = len(player_hand_one_hot)
    
    game = Dominoes(num_players, hand_size, topnum)
    list_players_ids = list(range(1, num_players+1))
    list_players = []
    for _id in list_players_ids:
        pl = RandomDominoPlayer(_id, game)
        list_players.append(pl)
    game.setup(list_players)
    

    for _ in range(num_sims):
        print('New simulation!')
        
        asd_copy = deepcopy(after_state_dict)
        game.reset(asd_copy)
        
        # Sanity check. The game-stored last player is the same as the one
        # deduced from the board state
        last_player = game.last_player
        assert last_player.num_player == last_player_id, "The last player is not what it should be"
        last_player.reset(np.array(player_hand_one_hot))
        print('Player dominoes:', last_player.dominoes)
        
        usable_dominoes_pp = get_usable_dominoes(list(DOMINO), asd_copy)
        other_player_ids = [i for i in range(1, num_players+1) if i != last_player.num_player]
        
        try:
            draw_dict = draw_arbitrary(other_player_ids, list(usable_dominoes_pp), 
                                   asd_copy['num_dominoes'])
        except:
            continue
#         print('\nStarting_dominoes')
#         print(draw_dict)
        for pl_id in other_player_ids:
            player = game.players_dict[pl_id]
            player.reset(start_dominoes=draw_dict[player.num_player], one_hot=False)
            
        winner = roll_game(game)
        list_victories[winner] += 1
        
        print('Salida:', game.salida)
        print('Game End!\n\n')
            
    print('\n', list_victories, '\t', list_victories[1]+list_victories[3], 
              list_victories[2]+list_victories[4])
    
    return (list_victories[1]+list_victories[3])/num_sims, \
        (list_victories[2]+list_victories[4])/num_sims


def add_probs_to_dataset(data_file):
    """
    """
    pass

if __name__ ==  '__main__':
#     play_dominoes()
    f = h5py.File('data/' + DATA_FILE,'a')
    
    ctr = 0
    import time
    t1 = time.time()
    for name in f.keys():
        print('Game', name)
        data = f[name]
        after_state_dict = {}
        
        after_state_dict['board_state'] = data['board_state'][:]
        after_state_dict['num_dominoes'] = data['num_dominoes'][:]
        after_state_dict['passes'] = data['passes'][:]
        after_state_dict['player'] = data['player'][()]
        after_state_dict['player_hand'] = data['player_hand'][:]
        
        if sum(after_state_dict['num_dominoes']) == HAND_SIZE*NUM_PLAYERS - 1:
            n = 10000
        elif sum(after_state_dict['num_dominoes']) >= HAND_SIZE*NUM_PLAYERS/2:
            n = 5000
        else:
            n = 1000
        probs = simulate_after_state(after_state_dict, num_sims=n)
        
        if 'win_probs' in data.keys(): del data['win_probs']
        data['win_probs']  = probs
        
        ctr += 1
        if ctr == 10:
            break
    t2 = time.time()
    print('Total time', t2-t1)

    
        
    