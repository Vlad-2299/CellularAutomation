import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from matplotlib import colors
import numpy as np
import os
import shutil
import math


parser = argparse.ArgumentParser(description='Brian Silverman Cellular Automation')


parser.add_argument('--name',
                       default='seeds',
                       type=str,
                       help='Celular Automation: -seeds -bbrain')


parser.add_argument('--n_generations',
                       default=250,
                       type=int,
                       help='Number of generations')


parser.add_argument('--w_width',
                       default=500,
                       type=int,
                       help='Window width')


parser.add_argument('--w_hight',
                       default=500,
                       type=int,
                       help='Window hight')


parser.add_argument('--save_video',
                       default=True,
                       type=bool,
                       help='Save output as mp4')


args = parser.parse_args()



def seeds_logic(board_, neighbour):
    '''
    Rule for Seeds
    board_: current life state
    neighbour: conv layer output

    return: next life cycle
    '''
    board_born = board_ == 0
    next_gen = board_born & (neighbour == 2)
    board_[next_gen] = 1
    board_[~next_gen] = 0

    return board_


def brain_logic(board_, neighbour):
    '''
    Rule for Brian's Brain
    board_: current life state
    neighbour: conv layer output

    return: next life cycle
    '''
    board_born = board_ == 0
    board_dying = board_ == 1

    next_gen = board_born & (neighbour == 2) & ~board_dying


    board_[~next_gen] = 0
    board_[next_gen] = 1
    board_[board_dying] = 3

    return board_


seeds_cmap = colors.ListedColormap(['black', 'white'])
bbrain_cmap = colors.ListedColormap(['black', 'white', 'blue'])


game_name = args.name


conv = torch.nn.Conv2d(1, 1, 3, 1, 1, 1, 1, False)
conv.weight = nn.Parameter(torch.tensor([[1,1,1],[1,0,1],[1,1,1]]).float().unsqueeze(0).unsqueeze(0)) #neighbour count kernel


if os.path.exists(game_name):
    shutil.rmtree(game_name)
os.makedirs(game_name, exist_ok=True)


if game_name == 'seeds':
    board = torch.randint(0, 1, (1, args.w_hight, args.w_width)).float().unsqueeze(0) #[batch, chan, x, y]
    plt.imsave(f'{game_name}/0_board.png',board.squeeze(), cmap=seeds_cmap)
    

    #Initial generation of seeds
    on_cells = torch.tensor([[0,1,1,0],
                            [1,1,1,1],
                            [1,1,1,1],
                            [0,1,1,0]]
                            ).float().unsqueeze(0).unsqueeze(0)
    

    mid_2 = int(board.size(2)//2)
    mid_3 = int(board.size(3)//2)
    board[:, :, mid_2:mid_2 + on_cells.size(2), mid_3:mid_3 + on_cells.size(3)] = on_cells


    for i_gen  in tqdm(range(1, args.n_generations + 1)):
        neigh_score = conv(board)
        
        board = seeds_logic(board, neigh_score)

        plt.imsave(f"{game_name}/{i_gen}_board.png",board.squeeze(), cmap=seeds_cmap)


elif game_name == 'bbrain':
    board = torch.randint(0, 2, (1, args.w_hight, args.w_width)).float().unsqueeze(0) #[batch, chan, x, y]
    plt.imsave(f'{game_name}/0_board.png',board.squeeze(), cmap=seeds_cmap)

    for i_gen  in tqdm(range(1, args.n_generations + 1)):
        
        dying_ = board == 3
        board[dying_] = 0

        neigh_score = conv(board)
        board[dying_] = 3
        board = brain_logic(board_=board, neighbour=neigh_score)

        plt.imsave(f"{game_name}/{i_gen}_board.png",board.squeeze(), cmap=bbrain_cmap)


if args.save_video:
    os.system(f'ffmpeg -framerate 12 -i {game_name}/%d_board.png output_{game_name}.mp4')