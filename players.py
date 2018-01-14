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

from dominoes import Dominoes
from utils import draw_mc, roll_game


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


    def reset(self, start_dominoes, one_hot=True):
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
                print('Must pass!')
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
            

class DeepQDominoPlayer():
    """
    """
    def __init__(self, num_player, game):
        """
        """
        DominoPlayer.__init__(self, num_player, game)


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