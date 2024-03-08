import pygame
from Board import Board
from Game import Game

pygame.init()

class Checkers:
    def __init__(self, screen):
        self.screen = screen
        self.running = True
        self.FPS = pygame.time.Clock()

    def _draw(self, board):
        board.draw(self.screen)
        pygame.display.update()

    def main(self, window_width, window_height):
        board_size = 8
        tile_width, tile_height = window_width // board_size, window_height // board_size
        board = Board(tile_width, tile_height, board_size)
        game = Game()
        while self.running:
            game.check_jump(board)
            is_change = input("is change?")
            if is_change == "yes":
                x_event = input("x: ")
                y_event = input("y: ")
                ip_event = (int(x_event)*80, int(y_event)*80)
                board.handle_click(ip_event)
            for self.event in pygame.event.get():
                if self.event.type == pygame.QUIT:
                    self.running = False
                if not game.is_game_over(board):
                    if self.event.type == pygame.MOUSEBUTTONDOWN:
                        a = self.event.pos
                        print(a)
                   #     board.handle_click(self.event.pos)
                #else:
                 #   game.message()
                  #  self.running = False

            self._draw(board)
            self.FPS.tick(60)



window_size = (640, 640)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Checkers")

checkers = Checkers(screen)
checkers.main(window_size[0], window_size[1])