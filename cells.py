import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse
from matplotlib import colors
import numpy as np
import os



generations = 100
board_h = 600
board_w = 600

save_video = True
save_video_command = "ffmpeg -framerate 12 -i boards/%d_board.png output.mp4"

board = torch.randint(0, 1, (1, board_h, board_w)).float().unsqueeze(0) #[batch, chan, x, y]

#Initial generation
on_cells = torch.tensor([[0,1,1,0],
                         [1,1,1,1],
                         [1,1,1,1],
                         [0,1,1,0]]
                         ).float().unsqueeze(0).unsqueeze(0)



mid_2 = int(board.size(2)//2)
mid_3 = int(board.size(3)//2)

board[:, :, mid_2:mid_2 + on_cells.size(2), mid_3:mid_3 + on_cells.size(3)] = on_cells

conv = torch.nn.Conv2d(1, 1, 3, 1, 1, 1, 1, False)
conv.weight = nn.Parameter(torch.tensor([[1,1,1],[1,0,1],[1,1,1]]).float().unsqueeze(0).unsqueeze(0)) #neighbour count kernel

cycles = []

plt.imsave(f"boards/0000_board.png",board.squeeze(), cmap=colors.ListedColormap(['black', 'white']))


'''
Seeds -> In each time step, a cell turns on or is "born" if it was off or "dead" but had exactly two neighbors that were on; 

'''

for i_gen  in tqdm(range(0, generations)):
    cycles.append(board.clone())

    neigh_score = conv(board)
    
    #Rule for Seeds
    board_born = board==0
    next_gen = board_born & (neigh_score == 2)
    board[next_gen] = 1
    board[~next_gen] = 0

    plt.imsave(f"boards/{i_gen}_board.png",board.squeeze(), cmap=colors.ListedColormap(['black', 'white']))

if save_video:
    os.system(save_video_command)