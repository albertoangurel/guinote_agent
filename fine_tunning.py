'''
This script is used to apply the final stage of the learning started in trick_training.py and game_training.py: the fine tunning. It completes the creation of a bot that can play Guiñote automatically.

We only modify the model, to allow training on the trickBlock and the learning function, saving the files with the correct names. The remaining functions are imported from game_training.py
'''

import numpy as np
import math
import random
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from time import perf_counter
from collections import deque


import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import secrets


import deck
import trick_training
import game_training




class agent(game_training.agent):
    


    def learn(self,node=None):

        if node is not None:
            raise ValueError( "This parallel version starts each game from a fresh state.")
        self.model.eval()

        for iteration in range(self.args['first_iteration'],self.args['first_iteration']+self.args['iterations']):
            num_new_cards=0
            num_new_swaps=0
            sp=perf_counter()
            
    
            model_weights = {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.model.state_dict().items()
            }

            with ProcessPoolExecutor(
                max_workers=self.args.get("selfPlay_workers", 2),
                mp_context=mp.get_context("spawn"),
                initializer=initialize_worker,
                initargs=(self.args, model_weights),
                ) as pool:

                results = pool.map(
                    worker_self_play,
                    range(self.args["selfPlays"]),
                    chunksize=1,
                    )

                for new_cards, new_swaps in results:
                    self.replay_cards.extend(new_cards)
                    self.replay_swaps.extend(new_swaps)

                    num_new_cards += len(new_cards)
                    num_new_swaps += len(new_swaps)

            cards = list(self.replay_cards)
            swaps = list(self.replay_swaps)
            
            check=perf_counter()
            print(f"Self-play took {int(check-sp)} seconds")
            
            for _ in range(self.args['epochs']):
                sampled_cards = random.sample(cards, k=min(num_new_cards, len(cards)))
                sampled_swaps = random.sample(swaps, k=min(num_new_swaps, len(swaps)))
                self.train_play(sampled_cards)
                self.train_swap(sampled_swaps)

            
            torch.save(self.model.state_dict(), f"FineTunning_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"FineTunning_optimizer_{iteration}.pt")
            self.save_replay(path="FineTunning_replay.pt")
            
            tt=perf_counter()-check
            print(f"Training took {int(tt)} seconds")
            print(f"Iteration {iteration} complete")

    


class LongTermGate(nn.Module):
    def __init__(self,game,numBlocks=10,num_hidden=64):
        super().__init__()
        self.game=game
        self.encoding='long-term'
        self.input_channels=7
        self.output_channels=3

        self.trickBlock=trick_training.TrickBlock(self.game)

        
        

        self.startBlock=nn.Sequential(
            nn.Conv2d(self.input_channels,num_hidden,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        self.backBone=nn.ModuleList([trick_training.ResBlock2CNN(num_hidden) for i in range(numBlocks)])

        self.gateHead=nn.Sequential(
            nn.Conv2d(num_hidden,num_hidden,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_hidden),nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.game.num_cards*num_hidden,self.game.num_cards)
            )

        self.sevenHead=nn.Sequential(
            nn.Conv2d(num_hidden,3,kernel_size=3,padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),nn.Flatten(),
            nn.Linear(3*self.game.num_cards,1)
            )

        self.valueHead=nn.Sequential(
            nn.Conv2d(num_hidden,3,kernel_size=3,padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),nn.Flatten(),
            nn.Linear(3*self.game.num_cards,1))


    def forward(self,trick_code,long_code):
        trick_logits,_=self.trickBlock(trick_code)
        x=self.startBlock(long_code)
        for block in self.backBone:
            x=block(x)
        gate_logits=self.gateHead(x)
        card_logits = trick_logits+torch.nn.functional.logsigmoid(gate_logits)
        seven_logits=self.sevenHead(x)
        value=self.valueHead(x)
        return card_logits, seven_logits,value

    


_worker_agent = None


def initialize_worker(args, model_weights):
    global _worker_agent

    torch.set_num_threads(1)

    # Independent randomness in each worker.
    seed = secrets.randbits(32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    game = deck.Guinote()

    # Must match the architecture used by your main model.
    model = LongTermGate(game)
    model.load_state_dict(model_weights)
    model.eval()

    # Self-play does not use the optimizer.
    _worker_agent = agent(game, args, model, optimizer=None)


@torch.no_grad()
def worker_self_play(_):
    return _worker_agent.selfPlay()




if __name__=='__main__':
    start=perf_counter()
    game=deck.Guinote()
    torch.set_num_threads(4)
    args={'C':50,'mcts_searches':100,'iterations':25,'selfPlays':100,'epochs':5,'first_iteration':1,'max_cards':20000,'max_swaps':2000,'selfPlay_workers':4}
    network=LongTermGate(game)
    optimizer=torch.optim.Adam(network.parameters(), lr=1e-5, weight_decay=1e-4)
    
    
    try:
        network.load_state_dict(torch.load("FineTunning.pt",map_location="cpu",weights_only=True))
    except:
        raise AttributeError ("network cannot be loaded")
    
    try:
        optimizer.load_state_dict(torch.load("FineTunning_optimizer.pt"))
    except:
        raise AttributeError ("optimizer cannot be loaded")
    
    

    
    alpha=agent(game,args,network,optimizer)
    alpha.load_replay(path="FineTunning_replay.pt")
    alpha.learn()
    torch.save(network.state_dict(), f"FineTunning.pt")
    torch.save(optimizer.state_dict(), f"FineTunning_optimizer.pt")
    elapsed=perf_counter()-start
   
    print(f"Program took {int(elapsed)} seconds")
    


            
 