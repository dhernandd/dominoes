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

RUN_TYPE = 'sim'

TOPNUM = 7
DOMINO = [[i, j] for i in range(1,TOPNUM+1) for j in range(i, TOPNUM+1)]

NUM_PLAYERS = 4
HAND_SIZE = 7

DATA_FILE = 'set15.hdf5'

STARTING_DOMINOES = [[[7, 7], [4, 6], [6, 7], [3, 3], [1, 2], [4, 5], [1, 5]],
 [[2, 6], [5, 5], [1, 4], [4, 4], [6, 6], [5, 6], [3, 7]],
 [[3, 6], [2, 2], [2, 5], [3, 5], [2, 4], [1, 6], [5, 7]],
 [[3, 4], [1, 3], [1, 1], [4, 7], [2, 3], [1, 7], [2, 7]]]


def make_database(num_files=1):
    """
    Creates a database of domino boards without any probability of winning
    associated to them. This must be added later.
    """
    for i in range(num_files):
        print('\n\n\nCreating set', str(i), '\n\n\n')
        s_file = 'set' + str(i) + '.hdf5' 
        play_dominoes(save_file=s_file)

 
def play_dominoes(draw=draw_init, save_file=None, num_games=10, save=True):
    """
    Plays games of dominoes.
    
    Args:
        draw: The function that draws the starting dominoes.         
        save_file: If this is not None, the Dominoes instance will save the positions thus
        creating a database.
        num_games: The number of games to enter play
        
    """
    list_victories = [0]*(NUM_PLAYERS+1)
    game = Dominoes(NUM_PLAYERS, HAND_SIZE, TOPNUM, save_file=save_file)
    
    # Setups the game. This includes initializing the player instances.
    player1 = RandomDominoPlayer(1, game)
    player2 = RandomDominoPlayer(2, game)
    player3 = RandomDominoPlayer(3, game)
    player4 = RandomDominoPlayer(4, game)
    list_players = (player1, player2, player3, player4)
    game.setup(list_players)

    # Play!
    for i in range(num_games):
        # Cleans the game state
        game.reset()
        
        # Draws the starting dominoes.
        starting_dominoes = draw(NUM_PLAYERS, DOMINO, HAND_SIZE)
#         print('\nStarting_dominoes')
#         print(starting_dominoes[0])
#         print(starting_dominoes[1])
#         print(starting_dominoes[2])
#         print(starting_dominoes[3])
        
        # Assigns the starting dominoes to each player
        for player, dominoes in zip(list_players, starting_dominoes):
            player.reset(start_dominoes=list(dominoes))
        
        # Rolls the game!
        winner = roll_game(game, save=save)
        list_victories[winner] += 1
        if (i*20) % num_games == 0:
            print(i*100/num_games, '% processed.')
            
#         print('Salida:', game.salida)
            
    print('\n', list_victories, '\t', list_victories[1]+list_victories[3], 
              list_victories[2]+list_victories[4])


def simulate_after_state(after_state_dict, hand_size=HAND_SIZE, num_sims=1):
    """
    Takes an afterstate and simulates onwards.
    
    Args:
        after_state_dict: The dictionary with all the info for afterstate of
        player P. It includes entries for the game board, the remaining hand of
        player P, the number of dominoes in the hands of the other players and a
        one hot encoding of the known 'fallos'.
        
        hand_size: The initial hand sizes for this game. This is necessary to
        start the Dominoes instance
        
        num_sims: The number of simulations to carry for this afterstate. This
        should generally be smaller, the lower the total number of remaining
        dominoes in the hands of all players.
        
    Returns:
        (p1, p2): A tuple with the probabilities of team 1 and team 2 winning.
        The probability of a draw is 1-p1-p2.
    """
    from copy import deepcopy

    list_victories = [0]*(NUM_PLAYERS+1)

    # A number of important parameters can be obtained from the afterstate...
    num_players = len(after_state_dict['num_dominoes'])
    
    board_state = after_state_dict['board_state']
    last_player_id = np.trim_zeros(board_state[:, 0])[-1]
    
    player_hand_one_hot = after_state_dict['player_hand']
    topnum = len(player_hand_one_hot)
    
    # Setups the game to be played from the afterstate onwards.
    game = Dominoes(num_players, hand_size, topnum)
    list_players_ids = list(range(1, num_players+1))
    list_players = []
    for _id in list_players_ids:
        pl = RandomDominoPlayer(_id, game)
        list_players.append(pl)
    game.setup(list_players)
    

    for _ in range(num_sims):
        # Pass a game state to reset. This sets the initial position on the
        # board.
        asd_copy = deepcopy(after_state_dict)
        game.reset(asd_copy)
        
        # Sanity check. The game-stored last player is the same as the one
        # deduced from the board state.
        last_player = game.last_player
        assert last_player.num_player == last_player_id, "The last player is not what it should be"
        
        # Give the last player her stored hand.
        last_player.reset(np.array(player_hand_one_hot), one_hot=True)
        
        # Make lists of the dominoes each player could have.
        usable_dominoes_pp = get_usable_dominoes(list(DOMINO), asd_copy)
        other_player_ids = [i for i in range(1, num_players+1) if i != last_player.num_player]
        
        # Try drawing using the draw_arbitrary function. There is still a bug
        # here that makes this fail in some rare cases so whatever, just catch
        # the exception and move on.
        try:
            draw_dict = draw_arbitrary(other_player_ids, list(usable_dominoes_pp), 
                                   asd_copy['num_dominoes'])
        except:
            continue
        
        # At last, give the hands to the respective players...
        for pl_id in other_player_ids:
            player = game.players_dict[pl_id]
            player.reset(start_dominoes=draw_dict[player.num_player], one_hot=False)
        
        # ...and roll
        winner = roll_game(game)
        list_victories[winner] += 1
                
    print(list_victories, '\t', list_victories[1]+list_victories[3], 
              list_victories[2]+list_victories[4])
    
    return (list_victories[1]+list_victories[3])/num_sims, \
        (list_victories[2]+list_victories[4])/num_sims


def add_probs_to_dataset(data_file='set1.hdf5'):
    """
    Adds the probabilities of winning for the last player to each board state in
    a database.
    
    Args:
        data_file: The database of board_states
    """
    file_path = data_file
    print('Processing data from file', file_path)
    f = h5py.File(file_path,'a')

    import time
    t1 = time.time()
    ctr = 0
    print(f.keys())
    for name in f.keys():
        print('\n\nGame:', name)
        data = f[name]
        after_state_dict = {}
        
        after_state_dict['board_state'] = data['board_state'][:]
        after_state_dict['num_dominoes'] = data['num_dominoes'][:]
        after_state_dict['passes'] = data['passes'][:]
        after_state_dict['player'] = data['player'][()]
        after_state_dict['player_hand'] = data['player_hand'][:]
        
        # Depending on how many dominoes remain in the hands of the players,
        # decide how many simulations
        if sum(after_state_dict['num_dominoes']) == HAND_SIZE*NUM_PLAYERS - 1:
            n = 10000
        elif sum(after_state_dict['num_dominoes']) >= HAND_SIZE*NUM_PLAYERS/2:
            n = 5000
        else:
            n = 1000
        probs = simulate_after_state(after_state_dict, num_sims=n)
        
        try:
            if 'win_probs' in data.keys(): del data['win_probs']
            data['win_probs']  = probs
        except:
            del f[name]
        print('Win probs:', probs)
        ctr += 1
        if ctr % 50 == 0: print('Processed', ctr, 'games')
        
    t2 = time.time()
    print('Total time', t2-t1)


def to_new_board_state(datafile_name):
    """
    Defines a more efficient board state.
    """
    datafile = h5py.File(datafile_name,'a')
    for k in datafile.keys():
        data = datafile[k]
        board_state = data['board_state'][:]
        
        # The new board state is temporarily stored in bs. bs is a Mx2 tensor.
        # bs[0] is simply the 'salida' (first move).
        bs = np.zeros_like(board_state[:,2:])
        bs[0] = board_state[0,2:]
        h1 = bs[0,0]
        h2 = bs[0,1]
        
        # The point of the new coding is for each move to ONLY store the
        # position of the move (1, 2, -1) and the change in the corresponding
        # head. This is what is done below for each.
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
        
        

if __name__ ==  '__main__':
    for i in range(1, 10):
        sf = 'set' + str(i) +'.hdf5'
        play_dominoes(save_file=sf, num_games=1000)
#     add_probs_to_dataset()
    

    
        
    