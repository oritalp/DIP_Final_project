import numpy as np
import board_utils
from tkinter.filedialog import asksaveasfile
import cv2


def on_closing():
    exit()      # exit the script


def choose_red(computer_cam):
    ret, frame = computer_cam.read()
    if ret:
        img = frame
    else:
        print(f"Error: Failed to capture frame from camera.")
    B, G, R = cv2.split(img)
    # Strengthen the red component
    R_strengthened = np.clip(R.astype(np.float32) * 9, 0, 255).astype(np.uint8)  # Adjust the factor as needed
    # Create an alpha channel
    alpha_channel = np.ones(R.shape, dtype=R.dtype) * 255  # Fully opaque. Adjust if you want transparency.
    # Merge the B, G, R, and alpha channels into one BGRA image
    img = cv2.merge((B, G, R_strengthened, alpha_channel))
    cv2.imshow("window", img)
    cv2.waitKey(1300)
    cv2.destroyWindow("window")
    save_path = "checkers_images/red/"
    file_name = "player-pawn.png"
    hh, ww = img.shape[:2]
    hh2 = hh // 2
    ww2 = ww // 2
    # define circles
    radius1 = min(hh2,ww2)
    xc = 1000  
    yc = 500   
    # draw filled circles in white on black background as masks
    mask = np.zeros_like(img)
    mask = cv2.circle(mask, (xc,yc), radius1, (255,255,255), -1)
    # put mask into alpha channel of input
    img[:, :, 3] = mask[:,:,0]
    # save results
    cv2.imwrite(save_path+file_name, img)


def choose_white(computer_cam):
    ret, frame = computer_cam.read()
    if ret:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA) #adding an alpha channel
    else:
        print(f"Error: Failed to capture frame from camera.")
    cv2.imshow("window", img)
    cv2.waitKey(1300)
    cv2.destroyWindow("window")
    save_path = "checkers_images/black/"
    file_name = "player-pawn.png"
    hh, ww = img.shape[:2]
    hh2 = hh // 2
    ww2 = ww // 2
    # define circles
    radius1 = min(hh2,ww2)
    xc = 1000  
    yc = 500  
    # draw filled circles in white on black background as masks
    mask = np.zeros_like(img)
    mask = cv2.circle(mask, (xc,yc), radius1, (255,255,255), -1)
    # put mask into alpha channel of input
    img[:, :, 3] = mask[:,:,0]
    # save results
    cv2.imwrite(save_path+file_name, img)

def ip_to_matrix(intersections, pawns_location):
    # Initialize a 8x8 binary matrix for board presence (0 for empty, 1 for occupied)
    board_bin = [[0 for _ in range(8)] for _ in range(8)]
    # Initialize a 8x8 matrix for board colors ("" for empty, "rp" or "bp" for red or black pawns)
    board_color = [["" for _ in range(8)] for _ in range(8)]
    # Lists to hold the unique x and y coordinates of the intersections
    unique_x = []
    unique_y = []
    # Calculate the average position for each column and row based on intersections
    sorted_by_x = sorted(intersections, key=lambda x: x[0]) # Sorting by X-axis values
    sorted_by_y = sorted(intersections, key=lambda y: y[1]) # Sorting by Y-axis values
    for i in range(7):
        x_avg = 0
        y_avg = 0
        for j in range(7):
            a = sorted_by_x[i*7+j][0]
            b = sorted_by_y[i*7+j][1]
            x_avg +=  sorted_by_x[i*7+j][0]/7
            y_avg +=  sorted_by_y[i*7+j][1]/7
        unique_y.append(round(x_avg))
        unique_x.append(round(y_avg))
    # Place pawns on the board based on their locations
    for x, y, color, _ in pawns_location:
            grid_x = 0
            grid_y = 0
            for i in range(7):
                if x > unique_x[i]:
                    grid_x = i + 1
            for j in range(7):
                if y > unique_y[j]:
                    grid_y = j + 1
            # Update the board's binary and color matrices with the pawn's presence and color
            board_bin[grid_x][grid_y] = 1  # Place a pawn on the board
            if color == 1:
                board_color[grid_x][grid_y] = "rp"
            else:
                board_color[grid_x][grid_y] = "bp"
    return board_bin, board_color


def matrix_to_move(new_board, old_board):
    new_board_bin = np.array(new_board[0], dtype=int)
    new_board_color = np.array(new_board[1])
    old_board_bin = np.array(old_board[0], dtype=int)
    old_board_color = np.array(old_board[1])
    # Calculate the differences between the new and old presence/absence states to identify changes
    changes = np.subtract(new_board_bin, old_board_bin)
    # Locate the positions where pieces have been removed (-1) and added (1)
    move_from = np.where(changes == -1)
    move_to = np.where(changes == 1)
    move_occurred = (len(move_to[0]) != 0)
    # Initialize variables to track the type of move and positions involved
    move = 0
    pown_move_to = None
    pown_move_from = None
    # Process the move if one has occurred
    if move_occurred:
        if len(move_to[0]) > 1 and len(move_from[0]) == 0:    # probably hand is covering the bord
            pass
        elif len(move_to[0]) > 1: # Error: Two pieces moved simultaneously
            move = 3
            print("two powns had moved in the same image")               
        elif len(move_from[0]) == 1:   # A pown moved     
            move = 1
            # Convert positions from matrix indices to game positions                                  
            pown_move_from = [move_from[1][0],move_from[0][0]]
            pown_move_to = [move_to[1][0],move_to[0][0]]
        elif len(move_from[0]) == 0:   # Error: A piece was added unexpectedly
            move = 4
            print("someting went wrong, pown added to the game")
        else:  # A pawn was moved and another was eaten
            move = 2
            # Identify the position to which the pawn moved
            pown_move_to = [move_to[1][0],move_to[0][0]]
            turn_color = new_board_color[pown_move_to[1],pown_move_to[0]]
            # Identify the pawn that was moved based on color
            for i in range(len(move_from[0])):
                pown_pos = [move_from[0][i],move_from[1][i]]
                pown_color = old_board_color[pown_pos[0]][pown_pos[1]]
                if turn_color == pown_color:
                    pown_move_from = pown_pos
                    break
    pos = (move, pown_move_from, pown_move_to)    
    return pos  # pos = (move, (x1,y1), (x2,y2)) - move = 0 - no move, 1 - pawn had moved, 2 - a pawn had move and eat, 3 and 4 some kind of error



def cal_turn(old_board, curr_holo_mat, reset_flag, checkers_cam, verbose = False):
    num_of_vote = 5
    ref_img = cv2.imread("images_taken/new_alligned.jpg")
    break_flag = False
    res = 0
    locs_list = []
    quit_flag = False # for debug

    while not break_flag:
        ret, frame = checkers_cam.read()
        if not ret:
            raise Exception("Error: Failed to capture frame from checkers camera.")
        else:
            res, aligned_frame, curr_holo_mat, intersections, pawnas_locs = board_utils.get_locations(frame, ref_img, 
                                                                                                      curr_holo_mat, 
                                                                                                      reset_flag,
                                                                                                     verbose=False)
            print(res)
            if res != 0:
                locs_list = []
                reset_flag = 1
                if verbose:
                    cv2.putText(frame, "Couldn't find the board's inner corners", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow("Debgging cal_turn", frame)

            else:
                locs_list.append((intersections, pawnas_locs))  # pawns_locs is a 4-tuple list: (x, y, color, radius)
                reset_flag = 0
                if verbose:
                    cv2.imshow("Debgging cal_turn", aligned_frame)

            if len(locs_list) == num_of_vote:
                break_flag = True

            if cv2.waitKey(1) & 0xFF == ord('q'):
                quit_flag = True
                break

            if cv2.waitKey(1) & 0xFF == ord('r'):
                reset_flag = 1

    if quit_flag:
        return None, None, None, None, True

    if not quit_flag:
        assert(reset_flag == 0)
    # thake the last 5 good sampels and doing majority vote
    #board_list = []     # for debug
    vote_bin_matrix = [[0 for _ in range(8)] for _ in range(8)]
    new_board = [[[0 for _ in range(8)] for _ in range(8)], [["" for _ in range(8)] for _ in range(8)]]
    for vote_num in range(num_of_vote):
        board_bin, board_color = ip_to_matrix(locs_list[vote_num][0], locs_list[vote_num][1])
        #board_list.append([board_bin, board_color])
        for i in range(8):
            for j in range(8):
                if board_bin[i][j] == 1:
                    vote_bin_matrix[i][j] += 1
                    if vote_bin_matrix[i][j] > 2:
                        new_board[0][i][j] = 1
                        new_board[1][i][j] = board_color[i][j]

    pos  = matrix_to_move(new_board, old_board)
    # if pos[0] == 0:     # TODO: fix with nadav
    #     new_board = old_board
    return new_board, pos, curr_holo_mat, reset_flag   # pos = (move, (x1,y1), (x2,y2)) - move = 0 - no move, 1 - pawn had moved, 2 - a pawn had move and eat, 3 and 4 some kind of erre

def cal_turn_test(old_board):
    change = True
    bin_board = eval(input("enter bin_board: "))
    colore_board = eval(input("enter clore_board: "))
    new_board = [bin_board, colore_board]
    pos = matrix_to_move(new_board, old_board)
    return new_board, change, pos

