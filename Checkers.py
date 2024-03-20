import pygame
from Board import Board
from Game import Game
import checkers_utils

class Checkers:
    def __init__(self, screen, camera_api):
        self.screen = screen
        self.running = True
        self.FPS = pygame.time.Clock()
        self.camera_api = camera_api

    def _draw(self, board):
        board.draw(self.screen)
        pygame.display.update()

    def main(self, window_width, window_height):
        self.camera_api.open_camera()
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

        while self.running:
            game.check_jump(board)
            if game.is_game_over(board):
                    game.message()
                    self.running = False
                    self.camera_api.close_cameras()
            else:
                old_board = new_board
                frame = self.camera_api.read_frame(camera="z")
                new_board, change, pos = checkers_utils.cal_turn_test(old_board, frame)
                if change == True and pos[0] == True:                # a player moved someting
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
                    self.camera_api.close_cameras()
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

