import pygame
from Board import Board
from Game import Game
import checkers_utils
import time

class Checkers:
    def __init__(self, screen, camera_api):
        self.screen = screen
        self.running = True
        self.FPS = pygame.time.Clock()
        self.camera_api = camera_api


    def display_illegal_move_message(self, message):
        font = pygame.font.SysFont(None, 36)
        text_surface = font.render(message, True, (255, 0, 0))  # Red text
        text_rect = text_surface.get_rect(center=(self.screen.get_width()/2, self.screen.get_height()/2))
         # Create a background rectangle slightly larger than the text
        background_rect = pygame.Rect(text_rect.left - 10, text_rect.top - 10, text_rect.width + 20, text_rect.height + 20)
        pygame.draw.rect(self.screen, (0, 0, 0), background_rect)  # Draw the background rectangle in black
        self.screen.blit(text_surface, text_rect)
        pygame.display.update()
        pygame.time.wait(2000)  # Display the message for 2 seconds


    def _draw(self, board):
        board.draw(self.screen)
        pygame.display.update()

    def main(self, window_width, window_height):
        #############elf.camera_api.open_camera_checkers_cam()
        board_size = 8
        tile_width, tile_height = window_width // board_size, window_height // board_size
        board = Board(tile_width, tile_height, board_size)
        game = Game()
        legal_turn = True
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
        curr_holo_met = None
        reset_flg = 0
        while self.running:
            start_time = time.time()
            game.check_jump(board)
            if game.is_game_over(board):
                    game.message()
                    self.running = False
                    self.camera_api.close_checkers_cam()
            else:
                old_board = new_board if legal_turn else old_board
                #old_board = new_board
                print(f"time it took: {time.time()-start_time}")
                new_board, change, pos = checkers_utils.cal_turn_test(old_board)
                start_time = time.time()
                if change == True and pos[0] == True:                       # if a player moved something
                    x_event = pos[1][0]
                    y_event = pos[1][1]
                    ip_event = (int(x_event)*80+5, int(y_event)*80+5)           # selecting a pawn
                    board.handle_click(ip_event, legal_turn)
                    #legal_turn, error_message = board.handle_click(ip_event, legal_turn)    # Update legal_turn based on the move legality
                    #if not legal_turn:
                    #    self.display_illegal_move_message(error_message)        # Display the message if the move was illegal
                    #    continue                                                # Optionally skip the rest of the loop if you want to wait for a legal move
                                                                            # If the move was legal, proceed with the game update
                    self._draw(board)
                    self.FPS.tick(60)
                    game.check_jump(board)
                    x_event = pos[2][0]
                    y_event = pos[2][1]
                    ip_event = (int(x_event)*80, int(y_event)*80)               # moving the selected pawn
                    board.handle_click(ip_event, legal_turn)
                    #legal_turn, error_message = board.handle_click(ip_event, legal_turn)    # Again, update legal_turn based on this move
                    #if not legal_turn:
                    #    self.display_illegal_move_message(error_message)        # Display the message if the move was illegal
                    #    continue                                                # Optionally skip the rest of the loop if you want to wait for a legal move
                                                                            # If the move was legal, proceed with the game update
            for self.event in pygame.event.get():                           # checking if click Exit
                if self.event.type == pygame.QUIT:
                    self.running = False
                    self.camera_api.close_checkers_cam()
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
            
