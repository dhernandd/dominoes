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

import tensorflow as tf
import h5py

from dominoes import Dominoes
from utils import draw_mc, roll_game

TOPNUM = 7
DOMINO = [[i, j] for i in range(1,TOPNUM+1) for j in range(i, TOPNUM+1)]

NUM_PLAYERS = 4
HAND_SIZE = 7

DATA_FILE = 'set1.hdf5'


def variable_in_cpu(name, shape, initializer):
    """
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, dtype=tf.float32, 
                              initializer=initializer)
    return var


def main_iterator(input_x, input_y, batch_size):
    assert len(input_x) == len(input_y)
    l = len(input_x)
    
    from random import shuffle
    data = zip(input_x, input_y)
    shuffle(data)
    
#     rem_indices = range(l)
    ctr = 0
    while l > (ctr+1)*batch_size:
#         inds = rem_indices[:batch_size]
#         rem_indices = rem_indices[batch_size:]
#         l = len(rem_indices)
        ctr += 1
#         print(np.array([ np.append(d[1], 1.0-sum(d[1])) for d in data[(ctr-1)*batch_size:ctr*batch_size]]))
        yield ( np.array([d[0] for d in data[(ctr-1)*batch_size:ctr*batch_size]]), 
                np.array([ np.append(d[1], 1.0-sum(d[1])) for d in data[(ctr-1)*batch_size:ctr*batch_size]]) )



class DominoPlayer():
    """
    """
    def __init__(self, num_player, game, topnum=7):
        """
        """
        self.game = game
        self.num_player = num_player
        self.topnum = topnum
        
        self.all_dominoes = [[i, j] for i in range(1,self.topnum+1) 
                             for j in range(i, self.topnum+1)]
        
        self.is_watching = False


    def define_moves_dict(self, topnum):
        """
        Creates a dictionary where the keys are integers corresponding to the
        domino heads and the values are all the dominoes in the current player
        hand with those numbers.
        """
        d = {i : [] for i in range(1, topnum+1)}
        for domino in self.dominoes:
            d[domino[0]].append(domino)
            if domino[0] != domino[1]:
                d[domino[1]].append(domino)
        return d


    def reset(self, start_dominoes, one_hot=False):
        """
        TODO: Implement a reset that resets to an arbitrary game state.
        """
        if one_hot:
            topnum = len(start_dominoes)
            self.hand_one_hot = start_dominoes
            self.dominoes = [[i+1, j+1] for i in range(topnum) for j in range(topnum) 
                             if start_dominoes[i][j] == 1 and i <= j]
        else:
            self.dominoes = start_dominoes
            self.hand_one_hot = np.zeros((self.topnum, self.topnum))
            for d in self.dominoes:
                self.hand_one_hot[d[0]-1, d[1]-1] = 1.0
                self.hand_one_hot[d[1]-1, d[0]-1] = 1.0
        
        self.hand_size = len(self.dominoes)
        assert self.game.num_dominoes_per_player[self.num_player-1] == self.hand_size, \
                    'The number of dominoes the game reports does not coincide with the \
                    hand size obtained by counting the player dominoes.'
        
        self.dominoes_dict = self.define_moves_dict(self.topnum)
        other_players = [pl.num_player for pl in self.game.players]
                
#         self.others_passes_one_hot = {pl.num_player : [0]*self.topnum 
#                                       for pl in self.game.players}
#         self.others_rem_numdoms = {pl.num_player : self.hand_size for pl in self.game.players}
        list_domino = list(self.all_dominoes)
        for dom in self.dominoes:
            list_domino.remove(dom)
        
        self.others_possible_doms = {tuple(d) : other_players for d in list_domino}
        
        self.is_ready = True
    
    
    def get_possible_moves(self):
        """
        Extracts the possible moves from the game board
        """
        if not self.game.game_started:
            possible_moves_dict = {1 : [(self, (1, d)) for d in self.dominoes], 
                                         2 : []}
        else:
            possible_moves_dict = {1 : [(self, (1, d)) for d in self.dominoes_dict[self.game.heads[0]]], 
                                   2 : [(self, (2, d)) for d in self.dominoes_dict[self.game.heads[1]]]}

        return possible_moves_dict
    
    
    def get_specified_move(self, specified_move):
        """
        Return the specified move from the player hand.
        """
#         print('Player', self.num_player)

        # Sanity check: Is the specified domino in the player hand?
        assert specified_move[1] in self.dominoes, "That domino is not in-hand!"
        
#         print('Playing', specified_move[1], 'from', self.dominoes)
        move = (self, (specified_move[0], specified_move[1]))
        dom = move[1][1]
        
        # Clean up. Remove the domino from the player's hand and from the
        # dictionary contaning the player's possible moves for every
        # possible 'head'. .
        self.dominoes.remove(dom)
        self.dominoes_dict[dom[0]].remove(dom)
        if dom[0] != dom[1]:
            self.dominoes_dict[dom[1]].remove(dom)
            
        self.hand_one_hot[dom[0]-1, dom[1]-1] = 0.0
        self.hand_one_hot[dom[1]-1, dom[0]-1] = 0.0

    
        return move
            


class RandomDominoPlayer(DominoPlayer):
    """
    """
    def __init__(self, num_player, game):
        """
        """
        DominoPlayer.__init__(self, num_player, game)
        
    
    def get_move(self, identity_dom=None, save=False):
        """
        """
        if identity_dom is not None:
            move = self.get_specified_move(identity_dom)
            self.save = True
        else:
            possible_moves = self.get_possible_moves()
            
            # If more than one move possible, maybe save.
            if len(possible_moves[1]) + len(possible_moves[2]) > 1:
                self.save = True
            else:
                self.save = False
            
            if not possible_moves[1] and not possible_moves[2]:
                # Must pass
#                 print('Must pass!')
                move = (self, (-1, [0,0]))
            else:
    #             print('possible moves', possible_moves)
                if not self.game.game_started:
                    choice = np.random.choice(len(possible_moves[1]))
                    move = possible_moves[1][choice]
                else:
                    if not possible_moves[1]:
                        choice = np.random.choice(len(possible_moves[2]))
                        move = possible_moves[2][choice]
                    elif not possible_moves[2]:
                        choice = np.random.choice(len(possible_moves[1]))
                        move = possible_moves[1][choice]
                    else:
                        f_or_b = 2*np.random.randint(2) - 1 
                        if f_or_b == -1:
                            choice = np.random.choice(len(possible_moves[2]))
                            move = possible_moves[2][choice]
                        else:
                            choice = np.random.choice(len(possible_moves[1]))
                            move = possible_moves[1][choice]
    #             print('Current dominoes', self.dominoes, 
    #                   'playing:', move[2])
    #             print('Game State', self.game.game_state[:self.game.num_moves])
                
                #  Remove from the player hand
                dom = move[1][1]
                self.dominoes.remove(dom)
                self.dominoes_dict[dom[0]].remove(dom)
                if dom[0] != dom[1]:
                    self.dominoes_dict[dom[1]].remove(dom)
#                 self.dominoes.remove(move[2])
#                 self.dominoes_dict[move[2][0]].remove(move[2])
#                 if move[2][0] != move[2][1]:
#                     self.dominoes_dict[move[2][1]].remove(move[2])

        #         print('\nPossible_moves', self.num_player, possible_moves)
                
                # Remove from the one-hot representation as well
                self.hand_one_hot[dom[0]-1, dom[1]-1] = 0.0
                self.hand_one_hot[dom[1]-1, dom[0]-1] = 0.0
    
        return move


class GreedyDominoPlayer(DominoPlayer):
    """
    El BotaGorda
    """
    def __init__(self, num_player, game):
        """
        """
        DominoPlayer.__init__(self, num_player, game)

    
    def get_move(self):
        """
        This player selects, among her possible moves, the one with the highest
        score. If there is more than one move with highest score, she chooses at
        random among them.
        """
        possible_moves = self.get_possible_moves()
#         print('\nPossible_moves', self.num_player, possible_moves)
        if not possible_moves[1] and not possible_moves[2]:
            # Must pass
#             print('Must pass!')
            move = (self, -1, [0,0])
        else:
            move_score1 = [(d, sum(d)) for d in possible_moves[1]]
            move_score2 = [(d, sum(d)) for d in possible_moves[2]]
            
            maxscore1 = max(move_score1, key=lambda a: a[1])[1] if any(move_score1) else 0
            bestmoves1 = [d for d in move_score1 if d[1] == maxscore1]
            maxscore2 = max(move_score2, key=lambda a: a[1])[1] if any(move_score2) else 0
            bestmoves2 = [d for d in move_score2 if d[1] == maxscore2]
            
            if maxscore1 > maxscore2:
                choice = np.random.choice(len(bestmoves1))
                move = (self, 1, bestmoves1[choice][0])
            elif maxscore1 < maxscore2:
                choice = np.random.choice(len(bestmoves2))
                move = (self, 2, bestmoves2[choice][0])
            else:
                choice = np.random.choice(len(bestmoves1) + len(bestmoves2))
                if choice < len(bestmoves1):
                    move = (self, 1, bestmoves1[choice][0])
                else:
                    move = (self, 2, bestmoves2[choice-len(bestmoves1)][0])

            self.dominoes.remove(move[2])
            self.dominoes_dict[move[2][0]].remove(move[2])
            if move[2][0] != move[2][1]:
                self.dominoes_dict[move[2][1]].remove(move[2])
            
        return move
    

class MonteCarloDominoPlayer(DominoPlayer):
    """
    """
    def __init__(self, player_id, game, num_mc_games=1, num_players=4, hand_size=7):
        """
        """
        DominoPlayer.__init__(self, player_id, game)

        self.is_watching = True
        self.mc_games = num_mc_games
        self.num_players = num_players
        self.hand_size = hand_size


    def reset(self, start_dominoes):
        """
        """
        DominoPlayer.reset(self, start_dominoes)
        
        self.used_dominoes = list(self.dominoes)
#         self.unused_dominoes = list(DOMINO)
#         for d in self.dominoes: self.unused_dominoes.remove(d)
        
    def get_move(self, identity_dom=None):
        """
        """
        try_both_heads = True
        if self.game.game_started == False: try_both_heads = False
        
        print('Player', self.num_player)
        if identity_dom is not None:
            print(identity_dom, self.dominoes)
            assert identity_dom[1] in self.dominoes, "That domino is not in-hand!"
            move = (self, identity_dom[0], identity_dom[1])
            print('Choosing move:', move[1:])
            
            self.dominoes.remove(move[2])
            self.dominoes_dict[move[2][0]].remove(move[2])
            if move[2][0] != move[2][1]:
                self.dominoes_dict[move[2][1]].remove(move[2])
            return move
                
        possible_moves = self.get_possible_moves()
        if not possible_moves[1] and not possible_moves[2]:
            # Must pass
            print('Must pass!')
            move = (self, -1, [0,0])
        else:
            # TODO: If move is unique, skip this shit, if first move do just one.
            move_wins = []
#             print('Current game board:', self.game.game_board[:self.game.num_moves])
#             print('TBHs', try_both_heads)
            for m in possible_moves[1]:
                print('Current game board:', self.game.game_board[:self.game.num_moves])
                print('Trying move', m, 'in pos', 1)
                move_wins += [(self.simulate_games_after_move(1, m), 1, m)]
            if try_both_heads:
                for m in possible_moves[2]:
                    print('Trying move', m, 'in pos', 2)
                    move_wins += [(self.simulate_games_after_move(2, m), 2, m)]
            move_tuple = max(move_wins, key=lambda m : m[0])
            move = (self, move_tuple[1], move_tuple[2])
            print('Choosing move:', move[1:], 'with vics', move_tuple[0],'\n')
            
            self.dominoes.remove(move[2])
            self.dominoes_dict[move[2][0]].remove(move[2])
            if move[2][0] != move[2][1]:
                self.dominoes_dict[move[2][1]].remove(move[2])

        return move

    def passive_update_after_move(self, numplayer, d):
        """
        """
        # If Player #numplayer can play domino d: ...
        if d != [0, 0]:
            # ... add d to the list of used dominoes, ...
            self.used_dominoes.append(d)
            
            # ... remove it from the dict of possible dominoes in the hands of players and ... 
            self.others_possible_doms.pop(tuple(d))
            # ... update the number of remaining dominoes of the current player.
            self.others_rem_numdoms[numplayer] -= 1
        # If Player #numplayer passes: ...
        else:
            # ... Find the head H of the board. Remove #numplayer from the list
            # of possible players of the dominoes that have H, ...
            head = self.game.game_board[0][0]
            for d in self.others_possible_doms:
                if head in d and numplayer in self.others_possible_doms[head]:
                    self.others_possible_doms[head].remove(numplayer)
            # ... Update the one-hot list of player passes, ...
            self.others_passes_one_hot[numplayer][head-1] = 1
                    
            # ... if the tail T is different from the head, remove #numplayer
            # from the list of possible players of the dominoes that have T
            tail = self.game.game_board[self.game.back_pos][1]
            if tail != head:
                for d in self.others_possible_doms:
                    if tail in d and numplayer in self.others_possible_doms[tail]:
                        self.others_possible_doms[tail].remove(numplayer)
            
                self.others_passes_one_hot[numplayer][tail-1] = 1
            
                
    
    def simulate_games_after_move(self, pos, dom):
        """
        """
        game = Dominoes(self.num_players, self.hand_size)
        mc_player1 = RandomDominoPlayer(1, game)
        mc_player2 = RandomDominoPlayer(2, game)
        mc_player3 = RandomDominoPlayer(3, game)
        mc_player4 = RandomDominoPlayer(4, game)
        list_mc_players = [mc_player1, mc_player2, mc_player3, mc_player4]
        game.setup(*list_mc_players)
        
        # Prepare the MC simulation
        cur_num_moves = self.game.num_moves
        cur_game_state = self.game.game_state[cur_num_moves-1]
        cur_player = game.get_player_from_id(self.num_player)                    
        
        the_other_players = list(range(1, self.num_players+1))
        the_other_players.remove(self.num_player)
#         print('tops', the_other_players)
        if not self.game.game_started:
            plyrs_rem_doms = [self.hand_size for i in the_other_players]
        else:
            plyrs_rem_doms = [self.game.game_state[self.game.num_moves-1][i+5] 
                          for i in the_other_players]
        zip_both = zip(the_other_players, plyrs_rem_doms) 
        num_vics = 0
        for _ in range(self.mc_games):
            if not self.game.game_started:
                game.reset()
            else:
                game.reset(init_state=cur_game_state)

            # Play the proposed move
            cur_player.reset(list(self.dominoes))
            print('MC player dominoes', cur_player.dominoes, 'to try', dom)
            cur_player.dominoes.remove(dom)
            cur_player.dominoes_dict[dom[0]].remove(dom)
            if dom[0] != dom[1]:
                cur_player.dominoes_dict[dom[1]].remove(dom)
            game.play(cur_player, pos, dom)
#             print('Game State:', game.game_state[:game.num_moves])
            
            unused_dominoes = list(self.all_dominoes)
#             print('used/unused dominoes', self.used_dominoes, ',\t', unused_dominoes)
            # TODO: Optimize this?
            for d in self.used_dominoes: unused_dominoes.remove(d)
#             print('used/unused dominoes', self.used_dominoes, ',\t', unused_dominoes)
            starting_dominoes = draw_mc(unused_dominoes, zip_both)
            for plyr_id, start_dominoes in starting_dominoes:
                game.get_player_from_id(plyr_id).reset(start_dominoes)
                
            winner = roll_game(game)
            if winner%2 == self.num_player%2: num_vics += 1 
        print('Number of victories for this move:', num_vics)
        return num_vics
            

class DeepQDominoPlayer(DominoPlayer):
    """
    """
    def __init__(self, num_player, game):
        """
        """
        DominoPlayer.__init__(self, num_player, game)
        
        # TODO: Define here the network that gets a game state and outputs a
        # probability
        self.input = tf.placeholder(tf.float32, [None, 181], 'Inputs')
        self.batch_size = self.input.get_shape().as_list()[0]
        self.targets = tf.placeholder(tf.float32, [self.batch_size, 3], 'Targets')
        
        with tf.variable_scope('full1') as scope:
            weights_full1 = variable_in_cpu('weights', [181, 256], 
                                      initializer=tf.random_normal_initializer())
            biases_full1 = variable_in_cpu('biases', [256], 
                                     initializer=tf.constant_initializer())
            full1 = tf.nn.relu(tf.matmul(self.input, weights_full1) + biases_full1,
                               name=scope.name)

#         with tf.variable_scope('full1b') as scope:
#             weights_full1b = variable_in_cpu('weights', [256, 256], 
#                                       initializer=tf.random_normal_initializer())
#             biases_full1b = variable_in_cpu('biases', [256], 
#                                      initializer=tf.constant_initializer())
#             full1b = tf.nn.relu(tf.matmul(full1, weights_full1b) + biases_full1b,
#                                name=scope.name)
        
        reshape_full1 = tf.reshape(full1, [-1, 16, 16, 1])
        
        with tf.variable_scope('conv1') as scope:
            kernel_conv1 = variable_in_cpu('weights', shape=[3, 3, 1, 64],
                                     initializer=tf.random_normal_initializer())
            conv1 = tf.nn.conv2d(reshape_full1, kernel_conv1, [1, 1, 1, 1], padding='SAME')
            biases_conv1 = variable_in_cpu('biases', [64], tf.constant_initializer())
            pre_activation = tf.nn.bias_add(conv1, biases_conv1)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
#    
#         self.conv1 = conv1
        with tf.variable_scope('conv2') as scope:
            kernel_conv2 = variable_in_cpu('weights', shape=[3, 3, 64, 32],
                                     initializer=tf.random_normal_initializer())
            conv2 = tf.nn.conv2d(conv1, kernel_conv2, [1, 1, 1, 1], padding='SAME')
            biases_conv2 = variable_in_cpu('biases', [32], tf.constant_initializer())
            pre_activation = tf.nn.bias_add(conv2, biases_conv2)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
        
#         pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1],
#                                padding='SAME', name='pool1')
#         self.pool1 = pool1
        norm1 = tf.nn.lrn(conv2, bias=1.0, name='norm1')
#         self.norm1= norm1
#         
        with tf.variable_scope('full2') as scope:
            reshape_norm1 = tf.reshape(norm1, [-1, 8192])
            dim = reshape_norm1.get_shape()[1].value
            weights_full2 = variable_in_cpu('weights', shape=[dim, 128], 
                                      initializer=tf.random_normal_initializer(stddev=0.001))
            biases_full2 = variable_in_cpu('biases', [128], 
                                           initializer=tf.constant_initializer())
            full2 = tf.nn.relu(tf.matmul(reshape_norm1, weights_full2) + biases_full2,
                                name=scope.name)
#         
#         max1 = tf.reduce_max(local1)
# #         norm2 = tf.nn.l2_normalize(local1, 0, name='norm2')
#         local1 = local1/max1
#         self.local1 = local1
        with tf.variable_scope('softmax') as scope:
            weights_out = variable_in_cpu('weights', shape=[128, 3], 
                                      initializer=tf.random_normal_initializer(0.1))
            bias_out = variable_in_cpu('biases', [3], 
                                   initializer=tf.constant_initializer())
            self.temp = tf.matmul(full2, weights_out) + bias_out
            softmax = tf.nn.softmax(self.temp, name=scope.name)
#             
        self.prob = softmax
        self.cost = tf.nn.l2_loss(softmax - self.targets, 'Cost')

        self.reg_terms = 1e-3*( tf.nn.l2_loss(weights_full1) + tf.nn.l2_loss(weights_full2) +
                           tf.nn.l2_loss(weights_out) + tf.nn.l2_loss(kernel_conv1) + 
                            tf.nn.l2_loss(kernel_conv2) )
        
        self.loss = tf.add(self.cost, self.reg_terms, 'Loss') 
#         self.eval = softmax

        
    def get_move(self):
        """
        """
        possible_moves = self.get_possible_moves()
        if not possible_moves[1] and not possible_moves[2]:
            # Must pass
            print('Must pass!')
            move = (self, -1, [0,0])
        else:
            winning_probs = []
            for m in possible_moves[1]:
                self.game.play(m)
                winning_probs.append((1, m, self.eval_Qnet(self.game.game_state)))
            for m in possible_moves[2]:
                self.game.play(m)
                winning_probs.append((2, m, self.eval_Qnet(self.game.game_state)))
            best_move = max(winning_probs, key=lambda x : x[2])
            move = (self, best_move[0], best_move[1])
        
        return move
    
    
    @staticmethod
    def eval_Qnet(game_state):
        """
        TODO:
        """
        pass
    

def pop_random(L, batch_size):
    """
    """
    M = []
    try:
        for _ in range(batch_size):
            M.append(L.pop(np.random.randint(len(L))))
    except:
        pass
    
    return M


def get_batch(datafile, list_datasets, input_size=231):
    """
    """
    batch_size = len(list_datasets)
    x = np.zeros((batch_size, input_size))
    y = np.zeros((batch_size, 3))
    
    ctr = 0
    for name in list_datasets:
        data = datafile[name]

        board_state = data['board_state'][:,1:]
        num_dominoes = np.array(data['num_dominoes'][:])
        passes = data['passes'][:]
        player_hand = data['player_hand'][:]

        flat_input = np.expand_dims(np.concatenate((np.array(num_dominoes), player_hand.flatten(),
                                 passes.flatten(), board_state.flatten())), 0)
        
        win_probs = np.array(data['win_probs'][:])
        win_probs = np.append(win_probs, 1-np.sum(win_probs))
        
        x[ctr] = flat_input
        y[ctr] = win_probs
        ctr +=1
    
    return x, y
        

    
def train_DeepQPlayer(datafile='data/domino_main.db', batch_size=5, learning_rate=1e-5, 
                      num_epochs=1000, savedir='saves/nn_0.0/', network_version='0.0',
                      restore=False):
    """
    """
#     import os
#     if not os.path.exists(savedir): os.makedirs(savedir)

#     f = h5py.File(datafile,'r')
    import pickle
    db = pickle.load(open(datafile, 'r'))
    
    nsamps = len(db['input'])
    db_input_train, db_input_valid = db['input'][:-nsamps//5], db['input'][-nsamps//5:]
    db_output_train, db_output_valid = db['output'][:-nsamps//5], db['output'][-nsamps//5:]
    nsamps_valid = len(db_output_valid)
    
    game = Dominoes(num_players=NUM_PLAYERS, hand_size=HAND_SIZE, topnum=TOPNUM)
    player = DeepQDominoPlayer(1, game)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(player.loss)
    
    init_op = tf.global_variables_initializer()
    
    best_valid_cost = np.inf
    saver = tf.train.Saver() 
    with tf.Session() as sess:
        sess.run(init_op)
        for _ in range(num_epochs):
            cost_per_epoch = 0
            btc = 0
            data_batch = main_iterator(db_input_train, db_output_train, batch_size)
            for batch_x, batch_y in data_batch:
#                 game_ids_in_this_batch = pop_random(list_train, batch_size)
#                 batch_x, batch_y = get_batch(f, game_ids_in_this_batch)
#                 print(batch_x)
#                 print(batch_y)
                _, c = sess.run([optimizer, player.cost], 
                                feed_dict={player.input : batch_x,
                                           player.targets : batch_y} )
                
                cost_per_epoch += c
#                 print(btc,)
                btc += 1
            
            valid_x, valid_y = list(main_iterator(db_input_valid, db_output_valid, 1000))[0]
            valid_cost, l = sess.run([player.cost/nsamps_valid, player.loss], 
                                     feed_dict={player.input : valid_x,
                                           player.targets : valid_y})
            if valid_cost < best_valid_cost:
                print('Saving network...')
                saver.save(sess, savedir + 'model_'+network_version)
                best_valid_cost = valid_cost
                
            print('Cost per epoch(valid):', cost_per_epoch, valid_cost, l)
            



if __name__ == '__main__':
    f = h5py.File('data/set1.hdf5','a')

    d = f['game0']
#     print(d['win_probs'][:])
#     print(d.keys())
#     board_state = d['board_state'][:,1:]
#     num_dominoes = d['num_dominoes'][:]
#     passes = d['passes'][:]
#     player_hand = d['player_hand'][:]
#     print(board_state.shape, len(num_dominoes), passes.shape, player_hand.shape)
    
#     flat_input = np.expand_dims(np.concatenate((np.array(num_dominoes), player_hand.flatten(),
#                                  passes.flatten(), board_state.flatten())), 0)
#     print(flat_input)
#     print(len(flat_input))
    list_datasets = list(f.keys())
    game_ids_in_this_batch = pop_random(list_datasets, batch_size=10)
#     print(game_ids_in_this_batch)
#     print(game_ids_in_this_batch[0] in list_datasets, list_datasets)
    train_DeepQPlayer()
#     x, y = get_batch(f, game_ids_in_this_batch)
# #     print(x.shape, x)
# #     print(y)
#     with tf.Session() as sess:
#         game = Dominoes(num_players=NUM_PLAYERS, hand_size=HAND_SIZE, topnum=TOPNUM)
#         player = DeepQDominoPlayer(1, game)
# # #      
#         sess.run(tf.global_variables_initializer())
#         c = sess.run(player.prob, feed_dict={player.input : x})
#         d = sess.run(player.cost, feed_dict={player.input : x, player.targets : y})
# # #         x = sess.run(player.eval, feed_dict={player.action_space : np.random.randn(12, 12,1,1)})
# # #         y = sess.run(player.weights, feed_dict={player.action_space : np.random.randn(12,12,1,1)})
# # #         w = sess.run(player.biases, feed_dict={player.action_space : np.random.randn(12,12,1,1)})
# # #         v = sess.run(player.local1, feed_dict={player.action_space : np.random.randn(12,12,1,1)})
# #         z = sess.run(player.temp, feed_dict={player.input : flat_input})
#         print('cur', c)
#         print('cost', d)
#         print('temp', z)
#         print('zeros', x)
#         print('weights', y)
#         print('biases',w)
#         print('local1', v)
#         print('temp', z)
#         print(v.size, y.size)
#         print('mm',np.matmul(v, y))
#         a = np.exp(v)
#         print(a)
#         print(np.exp(v)/np.sum(a))
        
        
        

    
    
    
    