import pygame
from Board import Board
from Game import Game
import checkers_utils
import time
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
        checkers_cam = cv2.VideoCapture(0) 
        # if not checkers_cam.isOpened():
        #     checkers_cam.open(1)
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
            start_time = time.time()
            game.check_jump(board)
            if game.is_game_over(board):
                    game.message()
                    self.running = False

            else:
                old_board = new_board
                print(f"time it took: {time.time()-start_time}")
                new_board, pos, curr_holo_mat, reset_flag = checkers_utils.cal_turn(old_board, curr_holo_mat, reset_flag,
                                                                                    checkers_cam)
                start_time = time.time()
                if True and pos[0] == True:                # a player moved someting
                    x_event = pos[1][0]
                    y_event = pos[1][1]
                    ip_event = (int(x_event)*80+5, int(y_event)*80+5)    # selecting a pawn
                    board.handle_click(ip_event)
                    self._draw(board)
                    self.FPS.tick(60)
                    game.check_jump(board)
                    x_event = pos[2][0]
                    y_event = pos[2][1]
                    ip_event = (int(x_event)*80, int(y_event)*80)    # moving the selected pawn
                    board.handle_click(ip_event)
            for self.event in pygame.event.get():                # checking if click Exit
                if self.event.type == pygame.QUIT:
                    self.running = False

                #if not game.is_game_over(board):
                 #   if self.event.type == pygame.MOUSEBUTTONDOWN:
                  #      a = self.event.pos
                   #     print(a)
                   #     board.handle_click(self.event.pos)
                #if game.is_game_over(board):
                    #game.message()
                    #self.running = False

            self._draw(board)
            self.FPS.tick(60)
            
        checkers_cam.release()