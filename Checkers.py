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


    def display_illegal_move_message(self, message):
        font = pygame.font.SysFont(None, 36)
        text_surface = font.render(message, True, (255, 0, 0))  # Red text
        text_rect = text_surface.get_rect(center=(self.screen.get_width()/2, self.screen.get_height()/2))
         # Create a background rectangle slightly larger than the text
        background_rect = pygame.Rect(text_rect.left - 10, text_rect.top - 10, text_rect.width + 20, text_rect.height + 20)
        pygame.draw.rect(self.screen, (0, 0, 0), background_rect)  # Draw the background rectangle in white
        self.screen.blit(text_surface, text_rect)
        pygame.display.update()
        pygame.time.wait(2000)  # Display the message for 2 seconds

    def display_move_message(self, from_pos, to_pos, player):
        message = f"{player} tile Moved from {from_pos} to {to_pos}"
        font = pygame.font.SysFont(None, 36)
        text_surface = font.render(message, True, (255, 255, 255))  # White text for better visibility
        text_rect = text_surface.get_rect(center=(self.screen.get_width()/2, self.screen.get_height() * 0.1))  # Display at the top
        background_rect = pygame.Rect(text_rect.left - 10, text_rect.top - 10, text_rect.width + 20, text_rect.height + 20)
        pygame.draw.rect(self.screen, (0, 0, 0), background_rect)  # Black background
        self.screen.blit(text_surface, text_rect)
        pygame.display.update()
        pygame.time.wait(2000)  # Show the message for 2 seconds before continuing
        # Clear the message by redrawing the screen
        #self._draw(board)
        #pygame.display.update()


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

    def display_move_message(self, from_pos, to_pos, player):
        message = f"{player} tile Moved from {from_pos} to {to_pos}"
        font = pygame.font.SysFont(None, 36)
        text_surface = font.render(message, True, (255, 255, 255))  # White text for better visibility
        text_rect = text_surface.get_rect(center=(self.screen.get_width()/2, self.screen.get_height() * 0.1))  # Display at the top
        background_rect = pygame.Rect(text_rect.left - 10, text_rect.top - 10, text_rect.width + 20, text_rect.height + 20)
        pygame.draw.rect(self.screen, (0, 0, 0), background_rect)  # Black background
        self.screen.blit(text_surface, text_rect)
        pygame.display.update()
        pygame.time.wait(2000)  # Show the message for 2 seconds before continuing
        # Clear the message by redrawing the screen
        #self._draw(board)
        #pygame.display.update()

    def _draw(self, board):
        board.draw(self.screen)
        pygame.display.update()

    def main(self, window_width, window_height):
        game_start = True
        pos = (0, None, None)
        error_dict = {"0": "waiting", "1":"", "2":"", "3":"two powns had moved in the same image", "4": "someting went wrong, pown added to the game", "5": "The board must be initialized", "6": "The pawn that was eaten was not taken off the board", "7": "Invalid move - return the board to the previous position"}
        checkers_cam = cv2.VideoCapture(checkers_cam_num) 
        quit_flag = False #Ori added for debugging purposes only
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
        curr_holo_mat = None
        reset_flag = 0
        while self.running:
            game.check_jump(board)
            if game.is_game_over(board):
                    game.message()
                    self.running = False

            else:
                old_board = new_board if legal_turn and (pos[0] == 1 or pos[0] == 2 or pos[0] == 0) else old_board
                new_board, pos, curr_holo_mat, reset_flag, game_start = checkers_utils.cal_turn(old_board, curr_holo_mat, reset_flag,
                                                                               checkers_cam, game_start, verbose=False)
                if quit_flag:
                    break
                if pos[0] == 1  or pos[0] == 2:                           # if a player moved something
                    if pos[0] == 1:
                        x_event = pos[1][0]
                        y_event = pos[1][1]
                        ip_event = (int(x_event)*80+5, int(y_event)*80+5)           # selecting a pawn
                        legal_turn, error_message = board.handle_click(ip_event, legal_turn)    # Update legal_turn based on the move legality
                        if not legal_turn:
                            self.display_illegal_move_message(error_message)        # Display the message if the move was illegal
                            continue                                                # Optionally skip the rest of the loop if you want to wait for a legal move
                                                                                # If the move was legal, proceed with the game update
                        self._draw(board)
                        self.FPS.tick(60)
                        game.check_jump(board)
                    x_event = pos[2][0]
                    y_event = pos[2][1]
                    ip_event = (int(x_event)*80+5, int(y_event)*80+5)               # moving the selected pawn
                    legal_turn, error_message = board.handle_click(ip_event, legal_turn)    # Again, update legal_turn based on this move
                    if not legal_turn:
                        self.display_illegal_move_message(error_message)        # Display the message if the move was illegal
                        continue                                                # Optionally skip the rest of the loop if you want to wait for a legal move
                                                                            
                                                                                # If the move was legal, proceed with the game update
                    if legal_turn:
                        from_pos,to_pos = [[pos[1][0],pos[1][1]],[pos[2][0],pos[2][1]]]
                        print(from_pos)
                        print(to_pos)
                        try:
                            self.display_move_message(from_pos, to_pos,board.turn)
                        except:
                            pass
                else:
                    legal_turn = False
                    if pos[0] == 0:
                        err_msg = f"Waiting for the {board.turn} player to play"
                    else:
                        err_msg = error_dict.get(str(pos[0]))
                    self.display_illegal_move_message(err_msg)

            for self.event in pygame.event.get():                               # checking if click Exit
                if self.event.type == pygame.QUIT:
                    self.running = False


            self._draw(board)
            self.FPS.tick(60)
            
        checkers_cam.release()
        file = path + "checkers_images/red/player-pawn.png"
        if os.path.isfile(file): 
            os.remove(file)
        file = path + "checkers_images/white/player-pawn.png"
        if os.path.isfile(file): 
            os.remove(file)
        cv2.destroyAllWindows