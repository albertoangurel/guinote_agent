'''
This script includes all the classes and functions required to modelize the development of (the first stage) of the game Guiñote.
'''


import numpy as np
import torch
import random
from copy import copy, deepcopy





class Card:
    #object representing each card in the deck
    def __init__(self,suit,rank):
        self.suit=suit
        self.rank=rank

class Deck:
    #class representing a certain deck of cards
    def __init__(self,suits,ranks):
        self.suits=suits
        self.ranks=ranks
        self.num_suits=len(self.suits)
        self.num_ranks=len(self.ranks)
        self.num_cards=self.num_suits*self.num_ranks
        self.cards=[Card(suit,rank) for suit in self.suits for rank in self.ranks]
        random.shuffle(self.cards)

    def pop(self):
        item=self.cards.pop()
        return item

class SpanishDeck(Deck):
    #class representing the spanish deck of cards. It consists of 4 suits and 10 ranks.
    #the ranks of the cards are ordered according to Guiñote rules, which differ from the standard order.
    def __init__(self):
        self.suits=['Oros','Copas','Espadas','Bastos']
        self.ranks=['As','Three','King','Jack','Queen','Seven','Six','Five','Four','Two']
        super().__init__(suits=self.suits,ranks=self.ranks)
        
        super().__init__(self.suits,self.ranks)

class GuinoteDeck(SpanishDeck):
    #class used to represent the poins associated to each card, considering the Guiñote rules
    def __init__(self):
        super().__init__()
        self.suit_code={'Oros':0,'Copas':1,'Espadas':2,'Bastos':3}
        self.rank_code={'As':0,'Three':1,'King':2,'Jack':3,'Queen':4,'Seven':5,'Six':6,'Five':7,'Four':8,'Two':9}
        

    def points(self,card):
        points={'As':11,'Three':10,'King':4,'Jack':3,'Queen':2,'Seven':0,'Six':0,'Five':0,'Four':0,'Two':0}
        return points[card.rank]
    
    def encode(self,card):
        return self.suit_code[card.suit], self.rank_code[card.rank]

class Guinote_state:
   #this class represent each possible state that can be achieved in a Guiñote game. It will be used to create the Guiñote class below 
    def __init__(self,deck,hands,trick,seen_players,seen_rounds,player_to_play,trump,points,round,songs,next):
        self.deck=deck
        self.hands=hands
        self.trick=trick
        self.seen_players=seen_players
        self.seen_rounds=seen_rounds #represent cards already played, indicating the rounds
        #negative values indicate card still in the player hands 
        self.player_to_play=player_to_play
        self.trump=trump
        self.points=points
        self.round=round
        self.songs=songs
        self.next=next

    def copy(self):
        result = copy(self)

        result.deck = copy(self.deck)
        result.deck.cards = [Card(card.suit, card.rank) for card in self.deck.cards]

        result.hands = self.hands.copy()
        result.seen_players = self.seen_players.copy()
        result.seen_rounds = self.seen_rounds.copy()
        result.points = self.points.copy()

        result.trick = [Card(card.suit, card.rank) for card in self.trick]
        result.trump = Card(self.trump.suit, self.trump.rank)
        result.songs = self.songs.copy()

        return result

    

    
        

class Guinote:
    #class containing the functions needed to create a Guiñote game
    def __init__(self,num_players=4,hand_size=6,starting_player=0):
    
        self.num_players=num_players
        self.hand_size=hand_size
        self.starting_player=starting_player
        self.num_cards=40
        self.total_rounds=self.num_cards//self.num_players
        self.picking_rounds=self.total_rounds-self.hand_size
        self.num_trick_images=self.num_players+2
        
        
            
    def initial_state(self,points=None,starting_player=0):
        #creates the initial state of the game
        deck=GuinoteDeck()
        if self.hand_size*self.num_players>=deck.num_cards:
            raise ValueError ("Not enough cards.")
        if points is None:
            points=np.zeros(self.num_players)
        hands=np.zeros((self.num_players,deck.num_suits,deck.num_ranks))
        seen_players=np.zeros((deck.num_suits,deck.num_ranks))
        seen_rounds=np.zeros((deck.num_suits,deck.num_ranks))
        if self.hand_size*self.num_players>deck.num_cards:
            raise ValueError ("Not enough cards")
        for player in range(self.num_players):
            for _ in range(self.hand_size):
                item=deck.pop()
                s,r=deck.encode(item)
                hands[player,s,r]=1
        trump=deck.pop()
        songs=[suit for suit in deck.suits]
        return Guinote_state(deck=deck,hands=hands,trick=[],seen_players=seen_players,seen_rounds=seen_rounds,player_to_play=starting_player,trump=trump,points=points,round=1,songs=songs,next='play')

    def next(self,state,card=None,swap=None):
        #given a state, used to obtain the next state in the game, given the appropriate choice of card and swap, if applicable.
        #It will use the particular functions below, depending on which action is required next
        if state.round<=self.picking_rounds+1:
            if state.next=='play':
                self.play_card(state,card)
                if len(state.trick)>=4:
                    state.next='solve_trick'
            elif state.next=='solve_trick':
                self.solve_trick(state)
                if state.round<=self.picking_rounds+1:
                    state.next='swap'
                else:
                    state.next='sing'
            elif state.next=='swap':
                for player in self.team(state.player_to_play):
                    if swap=='Yes' and self.allowed_swapping(state,player):
                        self.swap_seven(state,player)
                state.next='collection'
            elif state.next=='collection':
                self.collection(state)
                state.next='sing'
            elif state.next=='sing':
                for player in self.team(state.player_to_play):
                    self.sing(state,player)
                state.next='play'
            else:
                raise ValueError ("Next do not describe any valid action.")
        else:
            if state.next=='play':
                self.play_card(state,card)
                if len(state.trick)>=4:
                    state.next='solve_trick'
            elif state.next=='solve_trick':
                self.solve_trick(state)
                state.next='sing'
            elif state.next=='sing':
                for player in self.team(state.player_to_play):
                    self.sing(state,player)
                if state.round<=self.picking_rounds+1:
                    state.next='swap'
                else:
                    state.next='play' #no cards left in the deck, so swapping not allowed
            elif state.next=='swap':
                for player in self.team(state.player_to_play):
                    if swap=='Yes' and self.allowed_swapping(state,player):
                        self.swap_seven(state,player)
                state.next='collection'
            elif state.next=='collection':
                self.collection(state)
                state.next='play'
            else:
                raise ValueError ("Next do not describe any valid action.")
        
    def play_card(self,state,card):
        #describe the action of plating a card and modifies the Guiñote state to the resulting state after haven played the cards
        s,r=state.deck.encode(card)
        if self.playable_cards(state)[s,r]==0:
            raise ValueError ("You cannot play that card.")
        if len(state.trick)>=self.num_players:
            raise ValueError ("Collect the cards before playing.")
        state.hands[state.player_to_play,s,r]=0
        state.seen_players[s,r]=state.player_to_play+1
        state.seen_rounds[s,r]=state.round
        state.trick.append(card)
        self.next_player(state)  

    def playable_cards(self,state):
        #returns the cards that can be legally played in a given guiñote state
        playable=state.hands[state.player_to_play,:,:].copy()
        if state.round<=self.picking_rounds or not state.trick:
            return playable
        #get the suit codes of the leading card and the trump
        ls,_=state.deck.encode(state.trick[0])
        ts,_=state.deck.encode(state.trump)

        #check the highest cards of leading suit and trump existing in the sut
        lr, tr=10,10
        for card in state.trick:
            if card.suit==state.trick[0].suit:
                lr=min(lr,state.deck.rank_code[card.rank])
            if card.suit==state.trump.suit:
                tr=min(tr,state.deck.rank_code[card.rank])

        #check if partner is winning the trick
        partner=len(state.trick)-2
        if partner>=0 and state.trick[partner].suit==state.trump.suit and state.deck.rank_code[state.trick[partner].rank]==tr:
            partner_winning=True
        elif partner>=0 and tr==10 and state.trick[partner].suit==state.trick[0].suit and state.deck.rank_code[state.trick[partner].rank]==lr:
            partner_winning=True
        else:
            partner_winning=False
            
        if playable[ls].any():
            playable[:ls, :] = 0
            playable[ls + 1:, :] = 0

            if (not partner_winning and (tr == 10 or ts == ls) and playable[ls, :lr].any()):
                playable[ls, lr:] = 0

        elif not partner_winning and ts != ls:
            if playable[ts, :tr].any():
                playable[:ts, :] = 0
                playable[ts + 1:, :] = 0
                playable[ts, tr:] = 0

        return playable
        


    

    def trick_winner_and_points(self,state):
        if len(state.trick)!=self.num_players:
            raise ValueError("The trick is not complete.")
        #once the trick is complete, the player_to_play of the state is the player who played the first card

        #compute the winner of the trick
        leading_card=state.trick[0]
        winner=state.player_to_play
        points=0
        for i in range(1,self.num_players):
            if state.trick[i].suit==state.trump.suit:
                if leading_card.suit!=state.trump.suit:
                    leading_card=state.trick[i]
                    winner=(state.player_to_play+i)%self.num_players
                    continue
            if state.trick[i].suit==leading_card.suit:
                if state.deck.rank_code[state.trick[i].rank]<state.deck.rank_code[leading_card.rank]:
                    leading_card=state.trick[i]
                    winner=(state.player_to_play+i)%self.num_players
        while state.trick:
            item=state.trick.pop()
            s,r=state.deck.encode(item)
            points+=state.deck.points(item)
        return winner, points

    def solve_trick(self,state):
        winner, trick_points=self.trick_winner_and_points(state)
        state.points[winner]+=trick_points
        state.player_to_play=winner
        if state.round==self.total_rounds: #last 10 points
            state.points[winner]+=10
        state.round+=1
        

    def sing(self,state,player):
         #after the trick, if a player of the winning team has in his hand the king and the jack of the same suit, they get 20 extra points, which become 40 if the suit it the trump
        if state.round==1:
            raise ValueError ("Singing is not allowed until one round is played")
        if player not in self.team(state.player_to_play):
            raise ValueError ("Your team needs to win a trick before singing.")
        for suit in state.songs.copy():
            code=state.deck.suit_code[suit]
            if state.hands[player,code,2]==1 and state.hands[player,code,3]==1:
                state.points[player]+=20
                state.seen_players[code,2]=-(player+1)
                state.seen_players[code,3]=-(player+1)
                state.seen_rounds[code,2]=-state.round+1
                state.seen_rounds[code,3]=-state.round+1
                if suit==state.trump.suit:
                    state.points[player]+=20
                state.songs.remove(suit)


    def collection(self,state):
        i=0
        while state.deck.cards and i<self.num_players:
            card=state.deck.pop()
            player=(state.player_to_play+i)%self.num_players
            st,rk=state.deck.encode(card)
            state.hands[player,st,rk]=1
            i+=1
        if state.round==self.picking_rounds+1: #after 4th trick, last player to pick up gets the trump
            player=(state.player_to_play+i)%self.num_players #i=num_players-1 here
            st,rk=state.deck.encode(state.trump)
            state.hands[player,st,rk]=1
            state.seen_players[st, rk] = -(player+1)
            state.seen_rounds[st, rk] = -state.round+1

    def swap_seven(self,state,player):
        if not self.allowed_swapping(state,player):
            raise ValueError("Swapping not allowed.")
        ts,tr=state.deck.encode(state.trump)
        seven=state.deck.rank_code['Seven']
        state.hands[player,ts,seven]=0
        state.hands[player,ts,tr]=1
        state.trump.rank='Seven'
        state.seen_players[ts,tr]=-(player+1)
        state.seen_rounds[ts,tr]=-state.round+1
  

    def allowed_swapping(self,state,player):
        seven=state.deck.rank_code['Seven']
        if state.round>self.picking_rounds+1:
            return False
        if state.trick:
            return False #swapping must be done befor the suit is initiated
        if np.sum(state.hands[player,:,:])>self.hand_size-1:
            return False #swapping must be done before picking the card
        ts,tr=state.deck.encode(state.trump)
        if tr>5:
            return False #swapping is not allowed for lower cards
        if state.hands[player,ts,seven]==0:
            return False #player need to possess the 7th of trump
        return True


    def next_player(self,state):
        #returns the next player to play after playing a card
        state.player_to_play=(state.player_to_play+1)%self.num_players

    def team(self, player):
        #returns the players that for the team to which a given player belongs
        return [player,(player+2)%4]

    def team_points(self,state):
        #returns the total team points achieved by two reams
        return (state.points[0]+state.points[2],state.points[1]+state.points[3])

    def game_over(self,state):
        #return whether, once reached certain state, the game is finished
        return not state.hands.any() and not state.trick and not state.deck.cards

    #below there are functions used to create the matrices representing the differenct states, which will be used in the neural networs

    def trump_code(self,state):
        #codes the trump card
        code=np.zeros((state.deck.num_suits,state.deck.num_ranks))
        st,rk=state.deck.encode(state.trump)
        code[st,rk]=1
        return code

    def trick_code(self,state):
        #codes the cards played in the current trick
        code=[]
        for card in state.trick:
            layer=np.zeros((4,10))
            st,rk=state.deck.encode(card)
            layer[st,rk]=1
            code.append(layer)
        for _ in range(self.num_players-1-len(state.trick)):
            layer=np.zeros((4,10))
            code.append(layer)
        return code

    def songs_code(self,state):
        #codes the songs that have already been played in the game
        code=np.ones((state.deck.num_suits,state.deck.num_ranks))
        for st in range(state.deck.num_suits):
            for rk in range(state.deck.num_ranks):
                if state.deck.suits[st] not in state.songs:
                    code[st,rk]=0
        return code

    



    def encode_state(self,state):
        #codes all the information describing a given state of the game
        lst_hands=[state.hands[(state.player_to_play+i)%self.num_players,:,:] for i in range(1,self.num_players+1)]
        lst_trump=[self.trump_code(state)]
        lst_trick=self.trick_code(state)
        lst_seen_players_hand=[state.seen_players==-((state.player_to_play+i)%self.num_players+1) for i in range(1,self.num_players+1)]
        lst_seen_players_discard=[state.seen_players==(state.player_to_play+i)%self.num_players+1 for i in range(self.num_players)]
        lst_seen_rounds_hand=[state.seen_rounds==-round for round in range(1,self.total_rounds+1)]
        lst_seen_rounds_discard=[state.seen_rounds==round for round in range(1,self.total_rounds+1)]
        lst_songs=[self.songs_code(state)]
        lst_scores=[self.team_points(state)[state.player_to_play%2]*np.ones((state.deck.num_suits,state.deck.num_ranks)),self.team_points(state)[(state.player_to_play+1)%2]*np.ones((state.deck.num_suits,state.deck.num_ranks))]

            
        lst=lst_hands+lst_trump+lst_trick+lst_seen_players_hand+lst_seen_players_discard+lst_seen_rounds_hand+lst_seen_rounds_discard+lst_songs+lst_scores
        #standard setting, return 39*4*10

        return torch.tensor(np.stack(lst,axis=0).astype(np.float32))

    def encode_perspective(self,state):
        #codes all the information available to the next player to play
        return self.encode_state(state)[3:,:,:].unsqueeze(0)

    def encode_trick_perspective(self,state):
        #code use to train the model seeing the trick perspective
        hand=[state.hands[state.player_to_play,:,:]]
        trump=[self.trump_code(state)]
        trick=self.trick_code(state)
        seen=[state.seen_players>0]
        lst=hand+trump+trick+seen
        return torch.tensor(np.stack(lst,axis=0).astype(np.float32)).unsqueeze(0)

    def encode_longterm_perspective(self,state):
        #code used to train the final model, including the long-term information
        hand=[state.hands[state.player_to_play,:,:]]
        trump=[self.trump_code(state)]
        seen=[state.seen_players>0]
        round=[state.round*np.ones((state.deck.num_suits,state.deck.num_ranks))]
        songs=[self.songs_code(state)]
        scores=[self.team_points(state)[state.player_to_play%2]*np.ones((state.deck.num_suits,state.deck.num_ranks)),self.team_points(state)[(state.player_to_play+1)%2]*np.ones((state.deck.num_suits,state.deck.num_ranks))]
        lst=hand+trump+seen+round+songs+scores
        return torch.tensor(np.stack(lst,axis=0).astype(np.float32)).unsqueeze(0) #1+7+4+10

    #the functions below are used to simulate random play 
    
    def random_complete_trick(self,state):
        #completes the trick with random card choices
        card=None
        swap=False
        while not self.game_over(state) and state.next!='solve_trick':
            if state.next=='play':
                positions = np.argwhere(self.playable_cards(state))
                index = np.random.choice(len(positions))
                st,rk=positions[index]
                card=Card(state.deck.suits[st],state.deck.ranks[rk])
            if state.next=='swap':
                swap=np.random.choice(['Yes','No'])
            self.next(state,card,swap)
        winner, points=self.trick_winner_and_points(state)
        if winner%2==0:
            return points
        else:
            return -points

    def random_play(self,state):
        #completes the full game with random choices for player actions
        card=None
        swap=False
        while not self.game_over(state):
                if state.next=='play':
                    positions = np.argwhere(self.playable_cards(state))
                    index = np.random.choice(len(positions))
                    st,rk=positions[index]
                    card=Card(state.deck.suits[st],state.deck.ranks[rk])
                if state.next=='swap':
                    swap=np.random.choice(['Yes','No'])
                self.next(state,card,swap)
        return self.team_points(state)[0]-self.team_points(state)[1]

    def agent_play(self,state,agent=['random']):
            #completes the game with the strategy given by a certain agent
            #when agent='random', the action choices are chosen randomly with equal probability
            if len(agent)==1:
                agent*=4
            elif len(agent)==2:
                agent*=2
            elif len(agent)==3:
                agent.append(agent[1])
            elif len(agent)!=4:
                raise ValueError ("Provide between 1 and 4 agents.")
            card=None
            swap=None
            while not self.game_over(state):
                    if state.next=='play':
                        player=state.player_to_play
                        if agent[player]=='random':
                            positions = np.argwhere(self.playable_cards(state))
                            index = np.random.choice(len(positions))
                            st,rk=positions[index]
                            card=Card(state.deck.suits[st],state.deck.ranks[rk])
                        else:
                            card=agent[player].choose_card(state) #define choice
                    if state.next=='swap':
                        player=state.player_to_play
                        if agent[player]=='random':
                            swap=np.random.choice(['Yes','No'])
                        else:
                            swap=agent[player].choose_swap(state)
                    self.next(state,card=card,swap=swap)
                    card=None
                    swap=None
            return self.team_points(state)[0]-self.team_points(state)[1]




