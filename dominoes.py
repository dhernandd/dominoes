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

class Dominoes():
    """
    The environment in a game of dominoes.
    """
    def __init__(self, num_players, hand_size, topnum=7, save_file=None):
        """
        """
        self.num_players = num_players
        self.hand_size = hand_size
        self.topnum = topnum
        self.save_file = save_file
        self.max_length_game = num_players*hand_size - num_players + 1
        
        
        
    def setup(self, players):
        """
        Sets up a domino match, possibly consisting of more than one game.
        """
        self.players = players
        self.players_dict = {player.num_player : player for player in players}
        
        
    def reset(self, game_state=None):
        """
        Sets up a single board state, possibly including an initial state for
        it. Setting up a game includes setting up a board, done here, and
        setting up the players, which include drawing initial hands, etc.
        
        Args:
            init_board: List of board moves with the format [player, head, dom]
            where player is an integer from [1:num_players], head is in [1,2]
            and dom is a domino.
        """
        if game_state is not None:
            init_board = game_state['board_state']
            self.num_dominoes_per_player = game_state['num_dominoes']
            self.one_hot_passes = game_state['passes']
            self.last_player = self.players_dict[game_state['player']]
            last_player_id = self.last_player.num_player
            
            # Sanity checks: Does a move correspond to a length 4 array in the board
            # state? Is the board size smaller than the maximal length of the game?
            assert len(init_board[0]) == 4, 'Board state is not properly formatted.'
            assert len(init_board) == 2*self.max_length_game, 'Board state is not properly formatted!'
            
#             l = len(init_board)
#             if l < 2*self.max_length_game:
#                 init_board = init_board + [0]*4*(2*self.max_length_game - l)
            self.board_state = init_board
            self.salida = init_board[0][2:]
            
            # At the end, make it a numpy array. TODO: Not sure about this decision.
            self.board_state = np.array(self.board_state)

#             last_player_id = self.board_state[self.num_moves-1][0]
            self.cur_player = self.players_dict[last_player_id % self.num_players + 1]
#             self.num_dominoes_per_player = [pl.num_dominoes for pl in self.players]
            
            self.num_moves = len(np.trim_zeros(init_board[:, 0]))
#             print('self.num_moves', self.num_moves)
            self.num_passes = self.num_moves - len(np.trim_zeros(init_board[:, 2]))

            self.game_started = True
        else:
            self.num_moves = 0
            self.board_state = np.array([[0]*4]*2*self.max_length_game)
            self.heads = [0,0]
            
            self.cur_player = self.players[0]
            self.num_dominoes_per_player = [self.hand_size]*self.num_players
            self.game_started = False
                
            self.one_hot_passes = [[0]*self.topnum for i in range(self.num_players)]
            self.num_passes = 0
        
        # self.game_board is convenient for visualization
        self.game_board = [[0, 0]]*self.max_length_game
        self.back_pos = 0
        num_passes = 0
        if init_board is not None:
            for i, state in enumerate(self.board_state):
                dom = state[2:].tolist()
                if i == 0:
                    self.game_board[0] = list(dom)
                    self.heads = list(dom)
                else:
                    if state[1] == 2:
                        # Sanity check:
                        assert self.heads[1] in dom, "Forro!"
                        
                        self.game_board[i-num_passes] = ( dom if self.heads[1] == dom[0] 
                                               else dom[::-1] )
                        self.back_pos += 1
                        self.heads[1] = self.game_board[self.back_pos][1]
                    elif state[1] == 1:
                        # Sanity check:
                        assert self.heads[0] in state[2:], "Forro!"
                        
                        self.game_board[1:i+1-num_passes] = self.game_board[:i-num_passes] 
                        self.game_board[0] = ( dom if self.heads[0] == dom[1] 
                                               else dom[::-1] )                        
                        self.back_pos += 1
                        self.heads[0] = self.game_board[0][0]
                    elif state[1] == -1:
                        for i in self.heads:
                            self.one_hot_passes[state[0]-1][i-1] = 1 
                        num_passes += 1
#                 self.num_moves += 1
                
#             self.game_board[0] = self.game_state[0][2:4].tolist()  
        self.finished = False
        
    
    def make_game_state_dict(self):
        """
        """
        self.game_state = {}
        self.game_state['board'] = self.board_state
        self.game_state['num_dominoes'] = self.num_dominoes_per_player
        self.game_state['one_hot_passes'] = self.one_hot_passes
        self.game_state['heads'] = self.heads
        
        
    def get_cur_player(self):
        """
        Looks at the game state and returns the player following the one that
        played last. For the moment, the player that played last
        """
        if not self.game_started:
            self.cur_player = self.players[0]

        return self.cur_player
        
        
    def get_player_from_id(self, plyr_id):
        """
        """
        return self.players_dict[plyr_id]
        
        
    def play(self, player, move, save=False):
        """
        Plays a move from player player on the board.
        """
        pos = move[0]
        dom = move[1]
#         self.cur_player = player
        num_player = player.num_player
        
#         print('Player', num_player, 'to play', dom, 'in pos', pos, '. Heads:', self.heads )
#         print('board:', self.game_board)
        # Sanity check: Can the current player be deduced from the board state?
        if self.game_started:
#             print('self.num_moves', self.num_moves)
#             print('num_player, self.board_state[self.num_moves-1][0] + 1:', 
#                   num_player, self.board_state[self.num_moves-1][0] % self.num_players + 1, self.num_moves)
            assert num_player == ( self.board_state[self.num_moves-1][0] 
                                                      % self.num_players + 1), \
                     "Something went wrong. It seems it is not this player's turn." 
                        
        # If new game, play first move
        if not self.game_started:
            self.board_state[0] = np.array([num_player, pos, dom[0], dom[1]])
            self.game_board[0] = list(dom)
            self.heads = list(dom)
            self.salida = list(dom)
            
            self.num_dominoes_per_player[0] -= 1
            player.hand_size -= 1

            self.game_started = True
        # Else, play!
        else:
            if pos != -1:
                # Sanity check: Can the domino be placed on the board at the
                # specified head?
                assert self.heads[pos-1] in dom, "Forro!"
                
                # Reset the passes counter if non-passing move
                self.num_passes = 0
                
#                 print('num_player', num_player)
                self.board_state[self.num_moves] = np.array([num_player, pos, dom[0], dom[1]])
                self.num_dominoes_per_player[num_player-1] -= 1
                player.hand_size -= 1
                
                # Sanity check:
                assert self.num_dominoes_per_player[num_player-1] == player.hand_size, \
                    "The game is reporting a number of dominoes left for the player \
                    that is different from those reported by the player itself."
                
                if pos == 2:
#                     print('self.game_board[self.back_pos], dom', self.game_board[self.back_pos], dom)
                    if self.game_board[self.back_pos][1] == dom[0]:
                        self.game_board[self.back_pos+1] = dom
                    elif self.game_board[self.back_pos][1] == dom[1]:
                        self.game_board[self.back_pos+1] = dom[::-1]
                    self.heads[1] = self.game_board[self.back_pos+1][1]
#                     print('self.game_board[self.back_pos], dom', self.game_board, dom)
                self.back_pos += 1
                if pos == 1:
#                     print('self.game_board[0], dom', self.game_board[0], dom)
                    self.game_board[1:self.back_pos+1] = self.game_board[:self.back_pos]
#                     print('self.game_board, dom', self.game_board, dom)
                    if self.game_board[1][0] == dom[1]:
                        self.game_board[0] = dom
                    elif self.game_board[1][0] == dom[0]:
                        self.game_board[0] = dom[::-1]
#                     print('self.game_board, dom', self.game_board, dom)
                    self.heads[0] = self.game_board[0][0]
                
                # After a non-passing move, end game if the player has no
                # dominoes left.
                if player.hand_size == 0:
                    self.num_moves += 1
                    self.end_game(player)
                             
            # Finally, if a player passes...
            if pos == -1:
#                 print('Player', num_player, 'passes on move', self.num_moves+1, '!')
#                 print('heads:', self.heads)
#                 print('board:', self.game_board)
                self.num_passes += 1
                self.board_state[self.num_moves] = np.array([num_player, pos, 0, 0])
                                
                # End game if the number of passes equals the number of players
                if self.num_passes == self.num_players:
                    self.num_moves += 1
                    self.end_game()

                for i in self.heads:
                    self.one_hot_passes[num_player-1][i-1] = 1 
#                 print(self.one_hot_passes)
            # Modify the state of ever-watching players after a move
            for observing_player in self.players:
                if observing_player.is_watching and observing_player.num_player != num_player:
                    observing_player.passive_update_after_move(dom)
            
#         last_player_id = self.board_state[self.num_moves-1][0]
#             print('self.board_state', self.board_state)
#             print('last_player_id, next_player', last_player_id, last_player_id % NUM_PLAYERS + 1)
            
        if self.save_file is not None:
            if self.num_moves == 0 or (np.random.randint(1,6) == 1 and save):
                self.save_afterstate(player)
        self.cur_player = self.players_dict[num_player % self.num_players + 1]

#         print('self.game_state', self.game_state)
        self.num_moves += 1


    def end_game(self, player=None):
        """
        End the game.
        """
        if player is not None:
            self.winner = player.num_player
#             print('Se pego player', self.winner, '!')
#             print('Winner: Player', self.winner)
        else:
#             print('Se tranco!')
            players_score = []
            scores = []
            for player in self.players:
                s = sum([sum(d) for d in player.dominoes])
                scores.append(s)
                players_score.append((player.num_player, s))
#                 print('sc', player.num_player, s, player.dominoes)
            best_score = min(scores)
            winners = [sc for sc in players_score if sc[1] == best_score]
            if len(winners) == 1:
                self.winner = winners[0][0]
#                 print('Scores:', players_score)
#                 print('Winner: Player', self.winner)
#                 print('GAME STATE:', self.game_state[:self.num_moves])
            else:
                self.winner = 0
#                 print("It's a draw")
#                 print('Scores:', players_score)
        
        for pl in self.players:
            pl.is_ready = False
            
        self.finished = True
#         self.display_game()
        
    
    def save_afterstate(self, player):
        """
        """
        print('Saving...')

        f = h5py.File(self.save_file,'a')
        
        num_states = len(f.keys())
        grp = f.create_group('game' + str(num_states))
        
        data_dict = {'player' : player.num_player, 'board_state' : self.board_state,
                    'num_dominoes' : self.num_dominoes_per_player, 
                    'player_hand' : player.hand_one_hot,
                    'passes' : self.one_hot_passes}
        
        for key, value in data_dict.items():
            grp.create_dataset(key, data=value)


    def display_game(self):
        """
        """
        print('Number of moves:', self.num_moves)
        print('Game Board:', self.game_board[:self.back_pos+1])
        
        
    def _print_game_state(self):
        """
        """
        print('Game State:', self.game_state[:self.num_moves])
