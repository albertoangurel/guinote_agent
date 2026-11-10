'''This script is used for the first stage of the training of the neural network. 

It is based on a Montecarlo Tree Search in which players try to maximize the points obtained in the current trick.

It uses a neural network formed by 10 ResNET blocks

It is the first stage of the training. The knowledge learnt in this stage will be transfered (in game_training.py) to develop a general strategy. 


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


import deck


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
                playable = self.game.playable_cards(self.state)
            for st,rk in np.argwhere(playable):
                card=deck.Card(self.state.deck.suits[st],self.state.deck.ranks[rk])
                child_state=self.state.copy()
                self.game.next(state=child_state,card=card)
                while not self.game.game_over(child_state) and child_state.next!='play' and child_state.next!='swap' and child_state.next!='solve_trick':
                    self.game.next(child_state)
                child=Node(game=self.game,state=child_state,parent=self,last=card)
                self.children.append(child)

        if self.state.next=='swap':
            for decision in ['No','Yes']:
                child_state=self.state.copy()
                self.game.next(state=child_state,swap=decision)
                while not self.game.game_over(child_state) and child_state.next!='play' and child_state.next!='swap' and child_state.next!='solve_trick':
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

    @torch.no_grad()
    def search(self,root):
        #main function of the algorithm
        root.parent=None
        
        has_model=self.model is not None
 
        while root.visits<self.args['mcts_searches']:
            #node=Node(self.game,self.args,state)
            node=root

            while len(node.children)>0:
                node=node.select_child(self.args['C'])
                
            trick_terminated=node.state.next=='solve_trick'
            if not trick_terminated:
                playable = (self.game.playable_cards(node.state) if node.state.next == 'play'else None)
                node.expand(playable=playable)
                if has_model:
                    policy_cards,value=self.model(self.game.encode_trick_perspective(node.state))
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
                        #policy_seven=torch.sigmoid(policy_seven).item()
                        policy_seven=0.5
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
                    value=self.game.random_complete_trick(node.state.copy())
            else:
                winner, points=self.game.trick_winner_and_points(node.state.copy())
                value=points if winner%2==0 else -points

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
        memory=[]
        trick_memory=[]
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
            if node.state.next=='play':
                policy=self.mcts.mcts_policy(node) 
                team_sign=1 if node.state.player_to_play%2==0 else -1
                encoded_policy=np.zeros((4,10))
                for child in node.children:
                    st,rk=node.state.deck.encode(child.last)
                    encoded_policy[st,rk]=policy[child]
                encoded_policy=encoded_policy.reshape(-1)
                trick_memory.append([self.game.encode_trick_perspective(node.state),encoded_policy,team_sign])
                node=np.random.choice(node.children,p=[policy[child] for child in node.children])
            elif node.state.next=='swap':
                node.expand()
                node=np.random.choice(node.children)
            elif node.state.next=='solve_trick':
                new_state=node.state.copy()
                winner,points=self.game.trick_winner_and_points(node.state.copy())
                if winner%2==1:
                    points*=-1
                for item in trick_memory:
                    item[2]*=points
                memory+=trick_memory
                trick_memory=[]
                while not self.game.game_over(new_state) and new_state.next!='play' and new_state.next!='swap':
                    self.game.next(new_state)
                node=Node(self.game,new_state)
            else:
                raise ValueError("Expected play/swap/solve_trick action")
        
        return memory

    def train_play(self,memory_cards):
        if not memory_cards:
            return
        
        
        training_states, target_policy, target_value=zip(*memory_cards)

        
        dataset=TensorDataset(torch.cat(training_states, dim=0),torch.tensor(np.stack(target_policy),dtype=torch.float32),torch.tensor(target_value,dtype=torch.float32))

        loader=DataLoader(dataset,self.args.get("batch_size", 64),shuffle=True)
        torch.set_num_threads(4)
        self.model.train()

        
        for states,policies,values in loader:
            out_policies,out_values=self.model(states)
            values=values.reshape_as(out_values)
        
            policy_loss=F.cross_entropy(out_policies,policies)
            value_loss=F.mse_loss(out_values,values)

            loss_fn=value_loss/300+policy_loss
            
            

            self.optimizer.zero_grad()
            loss_fn.backward()
            self.optimizer.step()

        

    def learn(self,node=None):

        
        

        for iteration in range(self.args['first_iteration'],self.args['first_iteration']+self.args['iterations']):
            num_new_cards=0
            sp=perf_counter()
            
            with torch.no_grad():
                torch.set_num_threads(1)
                self.model.eval()
                for _ in range(self.args['selfPlays']):
                    new_cards=self.selfPlay(node)
                    self.replay_cards.extend(new_cards)
                    

                    num_new_cards+=len(new_cards)
                    

            cards = list(self.replay_cards)
            
            
            check=perf_counter()
            print(f"Self-play took {int(check-sp)} seconds")
            
            for _ in range(self.args['epochs']):
                sampled_cards = random.sample(cards, k=min(5*num_new_cards, len(cards)))
                
                self.train_play(sampled_cards)
                

            
            torch.save(self.model.state_dict(), f"TrickTraining_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"TrickTraining_optimizer_{iteration}.pt")
            self.save_replay(path="TrickTraining_replay.pt")
            
            tt=perf_counter()-check
            print(f"Training took {int(tt)} seconds")
            print(f"Iteration {iteration} complete")

    def save_replay(self, path):
        data = {
            "cards": [(state.detach().cpu(),torch.as_tensor(policy, dtype=torch.float32),float(value),) for state, policy, value in self.replay_cards],
            "swaps": [(state.detach().cpu(),float(policy),float(value),) for state, policy, value in self.replay_swaps],
        }

        torch.save(data, path)


    def load_replay(self, path):
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
        except FileNotFoundError:
            return  # First execution: start with empty buffers.

        self.replay_cards.clear()
        self.replay_swaps.clear()

        self.replay_cards.extend((state, policy.numpy(), value) for state, policy, value in data["cards"])
        self.replay_swaps.extend(data["swaps"]) 


class ResBlock2CNN (nn.Module):
    def __init__(self,num_hidden):
        super().__init__()
        self.conv1=nn.Conv2d(num_hidden,num_hidden,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(num_hidden)
        self.conv2=nn.Conv2d(num_hidden,num_hidden,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(num_hidden)
        

    def forward(self,x):
        residual=x
        x=F.relu(self.bn1(self.conv1(x)))
        x=self.bn2(self.conv2(x))
        x+=residual
        x=F.relu(x)
        return x

class TrickBlock(nn.Module):
    def __init__(self,game,num_blocks=20,num_hidden=64):
        super().__init__()
        self.game=game
        self.num_blocks=num_blocks
        self.num_hidden=num_hidden
        self.encoding='trick'
        self.input_channels=self.game.num_trick_images
        self.output_channels=2

        self.startBlock=nn.Sequential(
                    nn.Conv2d(self.input_channels,num_hidden,kernel_size=3,padding=1),
                    nn.BatchNorm2d(num_hidden),
                    nn.ReLU()
                )
        self.backBone=nn.ModuleList([ResBlock2CNN(num_hidden) for i in range(self.num_blocks)])

        self.policyHead=nn.Sequential(
            nn.Conv2d(num_hidden,num_hidden,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(game.num_cards*num_hidden,self.game.num_cards)
            )

        self.valueHead=nn.Sequential(
             nn.Conv2d(num_hidden,1,kernel_size=3,padding=1),
             nn.BatchNorm2d(1),
             nn.ReLU(),
             nn.Flatten(),
             nn.Linear(game.num_cards,1)
             )

    def forward(self,x):
        x=self.startBlock(x)
        for block in self.backBone:
            x=block(x)
        cards=self.policyHead(x)
        value=self.valueHead(x)
        return cards,value
        

        




if __name__=='__main__':
    start=perf_counter()
    game=deck.Guinote()
    
    args={'C':50,'mcts_searches':100,'iterations':200,'selfPlays':100,'epochs':5,'first_iteration':1,'max_cards':40000,'max_swaps':2000}
    network=TrickBlock(game)
    optimizer=torch.optim.Adam(network.parameters(), lr=1e-4,weight_decay=1e-4)
    
   
    
    try:
        network.load_state_dict(torch.load("TrickTraining.pt",map_location="cpu",weights_only=True))
    except:
        raise AttributeError ("network cannot be loaded")
    try:
        optimizer.load_state_dict(torch.load("TrickTraining_optimizer.pt"))
    except:
        raise AttributeError ("optimizer cannot be loaded")
    

    
    alpha=agent(game,args,network,optimizer)
    alpha.load_replay(path="TrickTraining_replay.pt")
    alpha.learn()
    torch.save(network.state_dict(), f"TrickTraining.pt")
    torch.save(optimizer.state_dict(), f"TrickTraining_optimizer.pt")
    elapsed=perf_counter()-start
   
    print(f"Program took {int(elapsed)} seconds")
    

    







