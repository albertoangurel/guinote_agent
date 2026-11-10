import numpy as np
import math
import torch
from time import perf_counter

import deck
import trick_training
import game_training

class Match:
    def __init__(self,game,models,starter=None):
        self.game=game
        self.models=models.copy() #given in a list between 1 and 4 elements
        if starter is None:
            starter=np.random.choice(list(range(self.game.num_players)))
        self.node=game_training.Node(self.game, self.game.initial_state(starting_player=starter))

    def tunning(self):
        if len(self.models)==1:
            self.models*=4
        elif len(self.models)==2:
             self.models*=2
        elif len(self.models)==3:
            self.models.append(self.models[1])
        elif len(self.models)!=4:
            raise ValueError ("Provide between 1 and 4 agents.")

    @torch.no_grad
    def play(self,stochastic=True):
        node=self.node.copy()
        self.tunning()
        
        while not node.game.game_over(node.state):
                if not node.children:
                    node.expand()
                player=node.state.player_to_play
                policy=dict()
                if self.models[player]!='random' and node.state.next!='solve_trick':
                    self.models[player].eval()
                    if self.models[player].encoding=='trick':
                        output=self.models[player](self.game.encode_trick_perspective(node.state))
                    elif self.models[player].encoding=='full':
                        output=self.models[player](self.game.encode_perspective(node.state))
                    elif self.models[player].encoding=='long-term':
                        output=self.models[player](self.game.encode_trick_perspective(node.state),self.game.encode_longterm_perspective(node.state))
                    else:
                        raise ValueError('Model encoding not conptemplated')
                    if self.models[player].output_channels==2:
                        policy_cards,_=output
                    elif self.models[player].output_channels==3:
                        policy_cards,policy_seven,_=output
                    else:
                        raise ValueError ('Unexpected number of output channels')
                    if node.state.next=='play':
                        policy_cards=torch.softmax(policy_cards,axis=1).detach().squeeze(0).cpu().numpy()
                        policy_cards = policy_cards.astype(np.float64) #avoid rounding errors with np.choice
                        policy_cards=policy_cards.reshape((4,10))
                        policy_cards*=self.game.playable_cards(node.state)
                        if np.sum(policy_cards)>0:
                            policy_cards/=np.sum(policy_cards)
                            for child in node.children:
                                st,rk=node.state.deck.encode(child.last)
                                policy[child]=policy_cards[st,rk]

                        else:
                            for child in node.children:
                                policy[child]=1/len(node.children)      
                    elif node.state.next=='swap':
                            if self.models[player].output_channels==3:
                                policy_seven=torch.sigmoid(policy_seven).item()
                            else:
                                policy_seven=0.5
                            for child in node.children:
                                if child.last=='Yes':
                                    policy[child]=policy_seven
                                else: 
                                    policy[child]=1-policy_seven
                    else:
                        raise ValueError ("Expected play/swap action")
                    
                else:
                    for child in node.children:
                        policy[child]=1/len(node.children)
                if stochastic:
                    node=np.random.choice(node.children,p=[policy[child] for child in node.children])
                else:
                    index=np.argmax([policy[child] for child in node.children])
                    node=node.children[index]
        score0,score1=node.game.team_points(node.state)
        return score0-score1
    



if __name__=='__main__':
    start=perf_counter()
    game=deck.Guinote()
    models=['random']
    network1=game_training.LongTermGate(game)
    network2=trick_training.TrickBlock(game)
    network3=game_training.LongTermGate(game)
    
    
    try:
        network1.load_state_dict(torch.load("FineTunning_25.pt",map_location="cpu",weights_only=True))
    except:
        raise FileNotFoundError ("Could not open the file 1")
    
    try:
        network2.load_state_dict(torch.load("TrickTraining/TrickTraining_200.pt",map_location="cpu",weights_only=True))
    except:
        raise FileNotFoundError ("Could not open the file 2")
    try:
        network3.load_state_dict(torch.load("GamePlay_100.pt",map_location="cpu",weights_only=True))
    except:
        raise FileNotFoundError ("Could not open the file 1")
    
    
    models=[network3,'random']

   
    result=[]
    n=3000
    sto=False
    for _ in range(n):
        match=Match(game,models)
        result.append(match.play(stochastic=sto))
    print(f"Mean: {np.mean(result)}")
    print(f"Std: {np.std(result)/math.sqrt(n):.2f}")
    print(f"The first results were {result[0:10]}")
    elapsed=perf_counter()-start
    print(f"Program took {int(elapsed)} seconds")
    
