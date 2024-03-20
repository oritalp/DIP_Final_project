import numpy as np
import matplotlib.pyplot as plt
import board_utils



def ip_to_matrix(intersections, pawns_location):
    board_bin = [[0 for _ in range(8)] for _ in range(8)]
    bord_color = [["" for _ in range(8)] for _ in range(8)]
    unique_x = []
    unique_y = []
    for i in range(7):
        x_avg = 0
        y_avg = 0
        for j in range(7):
            a = intersections[i+j*7][0]
            b = intersections[i*7+j][1]
            x_avg +=  intersections[i+j*7][0]/7
            y_avg +=  intersections[i*7+j][1]/7
        unique_x.append(round(x_avg))
        unique_y.append(round(y_avg))
    x_to_grid = {pixel: index for index, pixel in enumerate(unique_x)}
    y_to_grid = {pixel: index for index, pixel in enumerate(unique_y)}
    for x, y, color in pawns_location:
            grid_x = x_to_grid[min(unique_x, key=lambda k: 0>k-x)]
            grid_y = y_to_grid[min(unique_y, key=lambda k: 0>k-y)]
            board_bin[grid_y][grid_x] = 1  # Place a pawn on the board
            bord_color[grid_y][grid_x] = color
    board = [board_bin, bord_color]
    return board


def matrix_to_move(new_board, old_board):
    new_board_bin = np.array(new_board[0], dtype=int)
    new_board_color = np.array(new_board[1])
    old_board_bin = np.array(old_board[0], dtype=int)
    old_board_color = np.array(old_board[1])
    changes = np.subtract(new_board_bin, old_board_bin)
    move_from = np.where(changes == -1)
    move_to = np.where(changes == 1)
    move_occurred = (len(move_to[0]) != 0)
    pown_move_to = None
    pown_move_from = None
    if move_occurred:
        if len(move_to[0]) > 1:
            move_occurred = False
            print("two powns had moved in the same image")               # TODO: need to alert error
        if len(move_from[0]) == 1:                                  #pown moved
            pown_move_from = [move_from[1][0],move_from[0][0]]
            pown_move_to = [move_to[1][0],move_to[0][0]]
        elif len(move_from[0]) == 0:
            print("someting went wrong, pown added to the game")
        else:                                                       #pown moves and anoder eaten
            pown_move_to = [move_to[1][0],move_to[0][0]]
            turn_color = new_board_color[pown_move_to[1],pown_move_to[0]]
            for i in range(len(move_from[0])):
                pown_pos = [move_from[0][i],move_from[1][i]]
                pown_color = old_board_color[pown_pos[0]][pown_pos[1]]
                if turn_color == pown_color:
                    pown_move_from = pown_pos
                    break
    pos = (move_occurred, pown_move_from, pown_move_to)    
    return pos      # pos = (True/False, (x1,y1), (x2,y2))



def cal_turn(old_board, frame):
    ref_img = plt.imread("images_taken/ref_img.jpg")
    aligned_img,_ = board_utils.align_board(frame, ref_img, verbose=False)
    intersect = board_utils.get_intersections(aligned_img, verbose=True)
    if intersect[0] != None: # TODO: fix
        change = True
        pawns_location = board_utils.pawns_location()
        new_board = ip_to_matrix(intersect, pawns_location)
        pos = matrix_to_move(new_board, old_board)
    else:
        change = False
        pos = (False, None, None)
        new_board = old_board
    return new_board, change, pos   # pos = (True/False, (x1,y1), (x2,y2))

def cal_turn_test(old_board, frame):
    change = True
    bin_board = eval(input("enter bin_board: "))
    colore_board = eval(input("enter clore_board: "))
    new_board = [bin_board, colore_board]
    pos = matrix_to_move(new_board, old_board)
    return new_board, change, pos