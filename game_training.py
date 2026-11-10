'''
This script transfers the knowledge adquired in trick_training.py to develop a model that plays guiñote with a strategy maximizing the amount of points obtained in the full game.

The approach consists of including a logical gate in the output probabilities given in trick_training.py for a given state. It represents the benefit of saving a good card to have it later in the game.
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


class Node:
    #This class defines the nodes used to develop the MCTS algorithm along with the functions needed for the algorithm
    def __init__(self, game,state,parent=None,policy=None,last=None):
        self.game=game
        self.state=state
        self.parent=parent
        self.children=[]
        self.visits=0
        self.values=0
        self.policy=policy
        if self.policy is None:
            self.policy=dict()
        self.last=last

    

    def select_child(self,C):
        #selects the best children according to the PUCT function defined below.
        return max(self.children,key=lambda child: self.PUCT(child,C))

    def PUCT_exploitation(self,child):
        exploitation= child.values / child.visits if child.visits else 0
        if self.state.player_to_play%2==1:
            exploitation*=-1
        if child.visits==0:
            exploitation=100
        return exploitation
    

    def PUCT(self,child,C): 
        #defines the PUCT function that is used to decide which child to explore.
        exploration = (C* self.policy[child]* math.sqrt(self.visits)/(1 + child.visits))
        return self.PUCT_exploitation(child) + exploration

    def expand(self,playable=None):
        #creates the children of a given node, in bijection with the legal actions from the parent state
        if self.state.next=='play':
            if playable is None:
                playable=self.game.playable_cards(self.state)
            for st,rk in np.argwhere(playable):
                card=deck.Card(self.state.deck.suits[st],self.state.deck.ranks[rk])
                child_state=self.state.copy()
                self.game.next(state=child_state,card=card)
                while not self.game.game_over(child_state) and child_state.next!='play' and child_state.next!='swap':
                    self.game.next(child_state)
                child=Node(game=self.game,state=child_state,parent=self,last=card)
                self.children.append(child)

        if self.state.next=='swap':
            can_swap = any(self.game.allowed_swapping(self.state, player) for player in self.game.team(self.state.player_to_play))
            decisions = ['No', 'Yes'] if can_swap else ['No']
            for decision in decisions:
                child_state=self.state.copy()
                self.game.next(state=child_state,swap=decision)
                while not self.game.game_over(child_state) and child_state.next!='play' and child_state.next!='swap':
                    self.game.next(state=child_state)
                child=Node(game=self.game,state=child_state,parent=self,last=decision)
                self.children.append(child)

        

    def backpropagate(self,value):
        #backpropagate, either the terminal result or the network value, of the newly selected node
        self.visits+=1
        self.values+=value
        if self.parent is not None:
            self.parent.backpropagate(value)

    def copy(self):
        return deepcopy(self)








class MCTS:
    #class that implements the Montecarlo Tree Search Algorithm, 
    def __init__(self,game,args,model=None):
        self.game=game
        self.args=args
        self.model=model

    @torch.inference_mode()
    def search(self,root):
        
        root.parent=None
        
        has_model=self.model is not None
 
        while root.visits<self.args['mcts_searches']:
            #node=Node(self.game,self.args,state)
            node=root

            while len(node.children)>0:
                node=node.select_child(self.args['C'])
                
            
  
            is_terminal=self.game.game_over(node.state)
            if not is_terminal:
                playable = (self.game.playable_cards(node.state) if node.state.next == 'play'else None)
                node.expand(playable=playable)
            
                if has_model:
                    policy_cards,policy_seven,value=self.model(self.game.encode_trick_perspective(node.state),self.game.encode_longterm_perspective(node.state))
                    if node.state.next=="play":
                        policy_cards=torch.softmax(policy_cards,axis=1).squeeze(0).cpu().numpy()
                        policy_cards=policy_cards.reshape((4,10))
                        policy_cards*=playable
                        if np.sum(policy_cards)>0:
                            policy_cards/=np.sum(policy_cards)
                            for child in node.children:
                                st,rk=node.state.deck.encode(child.last)
                                node.policy[child]=policy_cards[st,rk]
                        else:
                            for child in node.children:
                                node.policy[child]=1/len(node.children)
                    elif node.state.next=="swap":
                        if len(node.children) == 1:
                            node.policy[node.children[0]] = 1.0
                        else:
                            policy_seven=torch.sigmoid(policy_seven).item()
                            for child in node.children:
                                if child.last=='Yes':
                                    node.policy[child]=policy_seven
                                else: 
                                    node.policy[child]=1-policy_seven
                    else:
                        raise ValueError ("Expected play/swap decision")
                    # Acting-team perspective → global team-0 perspective
                    value = value.item()
                    if node.state.player_to_play % 2 == 1:
                        value = -value
                else:
                    for child in node.children:
                        node.policy[child]=1/len(node.children)
                    value=self.game.random_play(node.state.copy())
            else:
                value=self.game.team_points(node.state)[0]-self.game.team_points(node.state)[1]
            
            node.backpropagate(value)

        

    def mcts_policy(self,node):
        #returns the greedy policy obtained after developing the MCTS algorithm
        self.search(node)
        policy=dict()
        normalized_values=[node.PUCT_exploitation(child) if child.visits else -np.inf for child in node.children]
        best_value=max(normalized_values)
        num_max_values=np.sum([v==best_value for v in normalized_values]).astype(np.float64)
        return {child: 1.0 / num_max_values if v == best_value else 0.0 
                for child, v in zip(node.children, normalized_values)}

    

    

class agent:

    def __init__(self,game,args,model,optimizer):
        self.game=game
        self.args=args
        self.model=model
        self.optimizer=optimizer
        self.mcts=MCTS(game,args,model)
        self.replay_cards = deque(maxlen=args['max_cards'])
        self.replay_swaps = deque(maxlen=args['max_swaps'])     


    def selfPlay(self,node=None):
        memory_cards=[]
        memory_swaps=[]
        if node:
            node.visits=0
            node.values=0
            node.children=[]
            node.policy=dict()
        else:
            starter=np.random.choice(list(range(4)))
            state=self.game.initial_state(starting_player=starter)
            node=Node(self.game,state)


        while not node.game.game_over(node.state):
            policy=self.mcts.mcts_policy(node) 
            team_sign=1 if node.state.player_to_play%2==0 else -1
            if node.state.next=='play':
                encoded_policy=np.zeros((node.state.deck.num_suits,node.state.deck.num_ranks))
                for child in node.children:
                    st,rk=node.state.deck.encode(child.last)
                    encoded_policy[st,rk]=policy[child]
                encoded_policy=encoded_policy.reshape(-1)
                memory_cards.append([self.game.encode_trick_perspective(node.state),self.game.encode_longterm_perspective(node.state),encoded_policy,team_sign])
            elif node.state.next == 'swap':
                if len(node.children) > 1:
                    for child in node.children:
                        if child.last == 'Yes':
                            encoded_policy = policy[child]
                            memory_swaps.append([self.game.encode_trick_perspective(node.state),self.game.encode_longterm_perspective(node.state),encoded_policy,team_sign])
                            break
            else:
                raise ValueError("Expected play/swap action")
            node=np.random.choice(node.children,p=[policy[child] for child in node.children])

            
            
        score0,score1=self.game.team_points(node.state)
        difference=score0-score1

        for memory in (memory_cards,memory_swaps):
            for sample in memory:
                sample[3]*=difference
        
        return memory_cards,memory_swaps

    def train_play(self,memory_cards):
        if not memory_cards:
            return
        
        
        training_trick_states, training_long_states, target_policy, target_value=zip(*memory_cards)

        
        dataset=TensorDataset(torch.cat(training_trick_states, dim=0),torch.cat(training_long_states, dim=0),torch.tensor(np.stack(target_policy),dtype=torch.float32),torch.tensor(target_value,dtype=torch.float32))

        loader=DataLoader(dataset,self.args.get("batch_size", 64),shuffle=True)
        self.model.train()
        torch.set_num_threads(4)
        
        for trick_states,long_states,policies,values in loader:
            out_policies,_,out_values=self.model(trick_states,long_states)
            values=values.reshape_as(out_values)
        
            policy_loss=F.cross_entropy(out_policies,policies)
            value_loss=F.mse_loss(out_values,values)

            loss_fn=value_loss/300+policy_loss
            
            

            self.optimizer.zero_grad()
            loss_fn.backward()
            self.optimizer.step()

        

    def train_swap(self,memory_swaps):
            if not memory_swaps:
                return
            
        
            
            training_trick_states, training_long_states,target_policy, target_value=zip(*memory_swaps)
    
            
            dataset=TensorDataset(torch.cat(training_trick_states, dim=0),torch.cat(training_long_states, dim=0),torch.tensor(np.stack(target_policy),dtype=torch.float32),torch.tensor(target_value,dtype=torch.float32))
            
    
            loader=DataLoader(dataset,self.args.get("batch_size", 64),shuffle=True)
            self.model.train()
            torch.set_num_threads(4) 
            
            for trick_states,long_states,policies,values in loader:
                _,out_policies,out_values=self.model(trick_states,long_states)
                policies=policies.reshape_as(out_policies)
                values=values.reshape_as(out_values)
            
                policy_loss=F.binary_cross_entropy_with_logits(out_policies,policies)
                value_loss=F.mse_loss(out_values,values)
    
                loss_fn=value_loss/100+policy_loss
                
                
    
                self.optimizer.zero_grad()
                loss_fn.backward()
                self.optimizer.step()


    def learn(self,node=None):

        if node is not None:
            raise ValueError( "This parallel version starts each game from a fresh state.")
        self.model.eval()

        for iteration in range(self.args['first_iteration'],self.args['first_iteration']+self.args['iterations']):
            num_new_cards=0
            num_new_swaps=0
            sp=perf_counter()
            
           

            

            # A CPU snapshot of the current network.
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

            
            torch.save(self.model.state_dict(), f"GamePlay_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"GamePlay_optimizer_{iteration}.pt")
            self.save_replay(path="GamePlay_replay.pt")
            
            tt=perf_counter()-check
            print(f"Training took {int(tt)} seconds")
            print(f"Iteration {iteration} complete")

    def save_replay(self, path):
        data = {
            "cards": [
                (
                    trick_state.detach().cpu(),
                    long_state.detach().cpu(),
                    torch.as_tensor(policy, dtype=torch.float32).cpu(),
                    float(value),
                )
                for trick_state, long_state, policy, value in self.replay_cards
            ],
            "swaps": [
                (
                    trick_state.detach().cpu(),
                    long_state.detach().cpu(),
                    float(policy),
                    float(value),
                )
                for trick_state, long_state, policy, value in self.replay_swaps
            ],
        }
        torch.save(data, path)


    def load_replay(self, path):
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
        except FileNotFoundError:
            return

        cards = [
            (trick_state, long_state, policy.numpy(), float(value))
            for trick_state, long_state, policy, value in data["cards"]
        ]
        swaps = [
            (trick_state, long_state, float(policy), float(value))
            for trick_state, long_state, policy, value in data["swaps"]
        ]

        self.replay_cards.clear()
        self.replay_swaps.clear()

        self.replay_cards.extend(cards)
        self.replay_swaps.extend(swaps)


class LongTermGate(nn.Module):
    def __init__(self,game,numBlocks=10,num_hidden=64):
        super().__init__()
        self.game=game
        self.encoding='long-term'
        self.input_channels=7
        self.output_channels=3

        self.trickBlock=trick_training.TrickBlock(self.game)

        self.trickBlock.load_state_dict(torch.load("TrickPlay/TrickTraining_200.pt",map_location="cpu",weights_only=True))

        self.trickBlock.requires_grad_(False)
        self.trickBlock.eval()

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

    def train(self, mode=True):
        super().train(mode)
        self.trickBlock.eval()
        return self


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
    args={'C':50,'mcts_searches':100,'iterations':200,'selfPlays':100,'epochs':5,'first_iteration':1,'max_cards':20000,'max_swaps':2000,'selfPlay_workers':4}
    network=LongTermGate(game)
    optimizer=torch.optim.Adam(network.parameters(), lr=0.001, weight_decay=1e-4)

    total_params = sum(p.numel() for p in network.parameters())
    print(f"Total parameters: {total_params:,}")
    
    try:
        network.load_state_dict(torch.load("GamePlay.pt",map_location="cpu",weights_only=True))
    except:
        raise AttributeError ("network cannot be loaded")
    try:
        optimizer.load_state_dict(torch.load("GamePlay_optimizer.pt"))
    except:
        raise AttributeError ("optimizer cannot be loaded")
    
    
    
    alpha=agent(game,args,network,optimizer)
    alpha.load_replay(path="GamePlay_replay.pt")
    alpha.learn()
    torch.save(network.state_dict(), f"GamePlay.pt")
    torch.save(optimizer.state_dict(), f"GamePlay_optimizer.pt")
    elapsed=perf_counter()-start
   
    print(f"Program took {int(elapsed)} seconds")
    


            
 