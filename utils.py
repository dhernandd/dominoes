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
from dask.dataframe.methods import assign


TOPNUM = 7
DOMINO = [[i, j] for i in range(1,TOPNUM+1) for j in range(i, TOPNUM+1)]

NUM_PLAYERS = 2
HAND_SIZE = 3

def draw_init(num_players, domino, hand_size):
    """
    Draws initial hands of size hand_size for num_players players from the set
    domino, in order to start a domino game.
    
    Output:
        start_dominoes: List of domino hands.
    """
    # Take a random permutation of the domino set.
    permuted = np.random.permutation(domino).tolist()
    start_dominoes = []
    
    # Starting from the beginning, scan the permutation extracting a hand for
    # each subsequent player
    for i in range(num_players):
        start_dominoes.append(permuted[i*hand_size:(i+1)*hand_size])
    return start_dominoes


def draw_mc(dominoes, zipped_plyrs_doms, pass_list=None):
    """
    Given a game state, and a player, draw hands for the other players.
    """
    permuted = np.random.permutation(dominoes).tolist()
    num_players = len(zipped_plyrs_doms)
    start_dominoes = []
    ctr = 0
    for i in range(num_players):
        nxt = ctr + zipped_plyrs_doms[i][1]
        this_plyr_dominoes = permuted[ctr:nxt]
        start_dominoes.append((zipped_plyrs_doms[i][0], this_plyr_dominoes))
        ctr = ctr + zipped_plyrs_doms[i][1]
    
    return start_dominoes
    
def make_afterstate_dict(data):
    """
    """
    after_state_dict = {}
        
    after_state_dict['board_state'] = data['board_state'][:]
    after_state_dict['num_dominoes'] = data['num_dominoes'][:]
    after_state_dict['passes'] = data['passes'][:]
    after_state_dict['player'] = data['player'][()]
    after_state_dict['player_hand'] = data['player_hand'][:]
    
    return after_state_dict
    
    
def roll_game(game, init_move=None):
    """
    """
    while True:
        pl = game.get_cur_player()
        if init_move is not None:
            m = pl.get_move(identity_dom=init_move)
            init_move = None
        else:
            m = pl.get_move()
#         print('Game board, move', game.game_board[:game.num_moves], m[1:])
        game.play(m[0], m[1], save=False)
        if game.finished:
#             game.display_game()
            winner = game.winner
            break
    return winner


def get_usable_dominoes(domino, afterstate):
    """
    """
    board_state = afterstate['board_state'][:]
#     print(board_state[:])
    
    # Remove all dominoes in the board
    board_used_dominoes = [d.tolist() for d in board_state[:,2:] if d.any()]
#     print(board_used_dominoes)
    for d in board_used_dominoes:
        domino.remove(d)
#     print(domino)
    
    # Remove all dominoes in the player hand
    player_hand_one_hot = afterstate['player_hand']
#     print(player_hand_one_hot[:])
    topnum = len(player_hand_one_hot)
    player_cur_dominoes = [[i,j] for i in range(1,topnum+1) for j in range(1, topnum+1) 
                           if player_hand_one_hot[i-1][j-1] == 1 and i <= j] 
#     print(player_cur_dominoes)
    for d in player_cur_dominoes:
        domino.remove(d)
#     print(domino)
    
    player_passes_one_hot = afterstate['passes']
    num_players = len(player_passes_one_hot)
    other_player_ids = [i for i in range(1, num_players+1) if i != afterstate['player'][()]]
#     print(num_players, other_player_ids)
    l = [0]*len(domino)
    # Player may have d in her hand if she is not passed to any of the heads of d
    player_may_have_d = lambda pl, d, pss : not any([pss[pl-1][x-1] for x in d])
    for i, d in enumerate(domino):
        possible_players = [x for x in list(other_player_ids) if
                            player_may_have_d(x, d, player_passes_one_hot)]
        l[i] = (possible_players, d) 
        
    return l
        
     
    
def draw_arbitrary(player_ids, usable_dominoes, hand_sizes):
    """
    """
    drawn_hands = { i : [] for i in player_ids}
    list_full = []
    usable_dominoes = np.random.permutation(usable_dominoes)
    num_hands = len(player_ids)
    for j in range(num_hands):
        for pls, d in usable_dominoes:
    #         print(pls, d)
            if len(pls) == j+1:
    #         if len(pls) == 0:
    #             raw_input()
    #         print('Looking at', d)
                for x in player_ids:
                    if x in list_full and x in pls: 
                        pls.remove(x)
                assign_pl = np.random.choice(pls)
                drawn_hands[assign_pl].append(d)
#         print('Assigning', d, 'to', assign_pl)
        
                if len(drawn_hands[assign_pl]) == hand_sizes[assign_pl-1]:
                    list_full.append(assign_pl)
#             print('Adding', assign_pl, 'to full list')
    
    return drawn_hands

    
    
if __name__ == '__main__':
    f = h5py.File('set1.hdf5','a')
    
    data = f['game2696']
    print(data.keys())
    
    num_dominoes = data['num_dominoes'][:]
    passes = data['passes'][:]
    us_ds = get_usable_dominoes(DOMINO, data)
    print(us_ds)
#     x = np.random.permutation(us_ds)
    print(num_dominoes, passes)
    d = draw_arbitrary([2,3,4], us_ds, num_dominoes)
    print(d)
    

    

