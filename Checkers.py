import pygame
from Board import Board
from Game import Game
import checkers_utils
from Path import path, checkers_cam_num

import os
import cv2

class Checkers:
    def __init__(self, screen):
        self.screen = screen
        self.running = True
        self.FPS = pygame.time.Clock()

    def _draw(self, board):
        board.draw(self.screen)
        pygame.display.update()

    def main(self, window_width, window_height):
        checkers_cam = cv2.VideoCapture(checkers_cam_num) 
        quit_flag = False #Ori added for debugging purposes only
        board_size = 8
        tile_width, tile_height = window_width // board_size, window_height // board_size
        board = Board(tile_width, tile_height, board_size)
        game = Game()
        board_bin = [                   
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0]
        ]                                   # initialization bord
        bord_color = [                  
            ['', 'bp', '', 'bp', '', 'bp', '', 'bp'],
            ['bp', '', 'bp', '', 'bp', '', 'bp', ''],
            ['', 'bp', '', 'bp', '', 'bp', '', 'bp'],
            ['', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', ''],
            ['rp', '', 'rp', '', 'rp', '', 'rp', ''],
            ['', 'rp', '', 'rp', '', 'rp', '', 'rp'],
            ['rp', '', 'rp', '', 'rp', '', 'rp', '']
        ]                                   # initialization bord
        new_board = [board_bin, bord_color]  # initialization bord
        curr_holo_mat = None
        reset_flag = 0
        while self.running:
            game.check_jump(board)
            if game.is_game_over(board):
                    game.message()
                    self.running = False

            else:
                old_board = new_board
                new_board, pos, curr_holo_mat, reset_flag = checkers_utils.cal_turn(old_board, curr_holo_mat, reset_flag,
                                                                               checkers_cam, verbose=False)
                if quit_flag:
                    break
                if pos[0] == 1  or pos[0] == 2:                # a player moved someting
                    if pos[0] == 1:
                        x_event = pos[1][0]
                        y_event = pos[1][1]
                        ip_event = (int(x_event)*80+5, int(y_event)*80+5)    # selecting a pawn
                        board.handle_click(ip_event)
                        self._draw(board)
                        self.FPS.tick(60)
                        game.check_jump(board)
                    x_event = pos[2][0]
                    y_event = pos[2][1]
                    ip_event = (int(x_event)*80+5, int(y_event)*80+5)    # moving the selected pawn
                    board.handle_click(ip_event)
            for self.event in pygame.event.get():                # checking if click Exit
                if self.event.type == pygame.QUIT:
                    self.running = False

            self._draw(board)
            self.FPS.tick(60)
            
        checkers_cam.release()
        os.remove(path + "checkers_images/red/player-pawn.png")
        os.remove(path + "checkers_images/black/player-pawn.png")
        cv2.destroyAllWindows