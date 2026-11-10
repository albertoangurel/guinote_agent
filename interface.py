#!/usr/bin/env python3
"""Native Python Guiñote table. Run: python guinote_gui.py

Select your saved .pt/.pth network with the Load network button, then play
South. Order around the table is anticlockwise: South -> East -> North -> West. Compare two policies before each move.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import queue
import threading

from PIL import Image, ImageOps, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import numpy as np
    import torch
    import deck
    from model_utils import load_network, network_decision, random_decision, action_label, singing_cards
except ImportError as exc:
    raise SystemExit(f'Missing dependency: {exc.name}. Install with: python -m pip install -r requirements.txt') from exc


SEAT_NAMES = {0:'South · You', 1:'East', 2:'North · Your partner', 3:'West'}
SUIT_COLORS = {'Oros':'#c08e31', 'Copas':'#ab5143', 'Espadas':'#426d86', 'Bastos':'#587b48'}
RANK_NUMBERS = {'As':'1', 'Three':'3', 'King':'12', 'Jack':'10', 'Queen':'11',
                'Seven':'7', 'Six':'6', 'Five':'5', 'Four':'4', 'Two':'2'}
NETWORK_COLORS = {'A': '#f1c75b', 'B': '#84d4ef'}
PAPER, INK, MUTED, FELT, GOLD = "#f5ebeb", '#24372d', '#7b8978', '#245b49', '#d1b573'


def seat_positions(width, height):
    """The game already increments player ids: give player 1 the EAST seat."""
    return {0:(width*.5, height-178), 1:(width-88, height*.22+20),
            2:(width*.5, 35), 3:(88, height*.22+20)}


class GuinoteApp:
    def __init__(self, root, checkpoint=None, checkpoint_b=None):
        self.root = root
        self._image_sources = {}
        self._card_images = {}
        self.asset_dir = Path(__file__).resolve().parent / 'card_assets'
        root.title('Guiñote ')
        root.geometry('1400x940')
        root.minsize(1180, 900)
        root.configure(bg=PAPER)
        self.game = deck.Guinote(starting_player=0)
        self.state = self.game.initial_state()
        self.networks = {'A': None, 'B': None}
        self.decisions = {}
        self.errors = {}
        self.comparison_ready = False
        self.busy = False
        self.job_kind = ''
        self.epoch = 0
        self.timer = None
        self.queue = queue.Queue()
        self.paused = True
        self.hint_card = None
        self.undo_stack = []
        self.song_history = []
        self.event_text = 'No announcements yet.'
        self.event_hold = False
        self.notes = ['A new hand. South leads; play continues anticlockwise.']
        if not (self.asset_dir / 'card_back.png').is_file():
            self.notes.append('Card artwork folder missing: extract card_assets beside guinote_gui.py. Using simple cards.')
        self.closed = False
        self.finished = False
        self.reveal = tk.BooleanVar(value=False)
        self.controller = tk.StringVar(value='')
        self.status = tk.StringVar()
        self.score_us = tk.StringVar()
        self.score_them = tk.StringVar()
        self.round_label = tk.StringVar()
        self.model_status = {slot: tk.StringVar(value='Not loaded') for slot in ('A', 'B')}
        self.control_status = tk.StringVar()
        self._startup_checkpoints = [(slot, path) for slot, path in
                                     [('A', checkpoint), ('B', checkpoint_b)] if path]
        self._build()
        self.root.protocol('WM_DELETE_WINDOW', self._close)
        self.root.after(60, self._poll)
        self.render()
        self.root.after(100, self._startup)

    def _build(self):
        style = ttk.Style()
        # Keep explicit foreground/background pairs even in macOS dark mode.
        style.theme_use('clam')
        style.configure('.', background=PAPER, foreground=INK)
        # Flat, high-contrast controls with consistent hover/pressed states.
        style.configure('TButton', background='#355c9a', foreground='#ffffff',
                        borderwidth=0, relief='flat', padding=(12, 8),
                        font=('Helvetica', 10, 'bold'), focusthickness=1,
                        focuscolor='#bfd4f5', bordercolor=PAPER,
                        lightcolor='#355c9a', darkcolor='#355c9a')
        style.map('TButton', foreground=[('disabled', '#f5f7f6'), ('active', '#ffffff')],
                  background=[('disabled', '#8aa6cc'), ('pressed', '#244472'), ('active', '#4775b8')],
                  lightcolor=[('disabled', '#8aa6cc'), ('pressed', '#244472'), ('active', '#4775b8')],
                  darkcolor=[('disabled', '#8aa6cc'), ('pressed', '#244472'), ('active', '#4775b8')],
                  relief=[('pressed', 'flat'), ('active', 'flat')])
        style.configure('TCheckbutton', background=PAPER, foreground=INK)
        style.map('TCheckbutton', background=[('active', PAPER)],
                  foreground=[('disabled', '#526254'), ('active', INK)])
        style.configure('TCombobox', foreground=INK, fieldbackground='#ffffff', background=PAPER)
        style.map('TCombobox', foreground=[('disabled', '#526254'), ('readonly', INK)],
                  fieldbackground=[('disabled', '#e7e7e2'), ('readonly', '#ffffff')],
                  selectforeground=[('readonly', INK)], selectbackground=[('readonly', '#ffffff')])
        self.root.option_add('*TCombobox*Listbox.background', '#ffffff')
        self.root.option_add('*TCombobox*Listbox.foreground', INK)
        self.root.option_add('*TCombobox*Listbox.selectBackground', FELT)
        self.root.option_add('*TCombobox*Listbox.selectForeground', '#ffffff')
        style.configure('Room.TFrame', background=PAPER)
        style.configure('Room.TLabel', background=PAPER, foreground=INK)
        style.configure('Subtle.TLabel', background=PAPER, foreground=MUTED)
        style.configure('Room.TLabelframe', background=PAPER)
        style.configure('Room.TLabelframe.Label', background=PAPER, foreground=INK)
        # Inherit the same modern appearance for the network loading controls.
        style.configure('NetworkLoad.TButton', padding=(14, 9))
        style.configure('Inactive.TButton', background='#557db7', foreground='#ffffff',
                        lightcolor='#557db7', darkcolor='#557db7')
        style.map('Inactive.TButton',
                  background=[('disabled', '#8aa6cc'), ('pressed', '#355c9a'), ('active', '#668fc9')],
                  lightcolor=[('disabled', '#8aa6cc'), ('pressed', '#355c9a'), ('active', '#668fc9')],
                  darkcolor=[('disabled', '#8aa6cc'), ('pressed', '#355c9a'), ('active', '#668fc9')],
                  foreground=[('disabled', '#edf3fc'), ('active', '#ffffff')])
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        header = ttk.Frame(self.root, style='Room.TFrame', padding=(20, 14))
        header.grid(row=0, column=0, sticky='ew')
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text='Guiñote.', style='Room.TLabel',
                  font=('Georgia', 25)).grid(row=0, column=0, sticky='w')
        self.new_button = ttk.Button(header, text='New hand', command=self.new_hand)
        self.new_button.grid(row=0, column=1)
        self.undo_button = ttk.Button(header, text='Undo', command=self.undo)
        self.undo_button.grid(row=0, column=2, padx=(8, 0))
        room = ttk.Frame(self.root, style='Room.TFrame', padding=(15, 0, 15, 15))
        room.grid(row=1, column=0, sticky='nsew')
        room.columnconfigure(0, weight=1)
        room.rowconfigure(1, weight=1)
        scores = ttk.Frame(room, style='Room.TFrame')
        scores.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        scores.columnconfigure(0, weight=1)
        ttk.Label(scores, textvariable=self.score_us, style='Room.TLabel', font=('Georgia', 18)).grid(row=0, column=0, sticky='w')
        ttk.Label(scores, textvariable=self.score_them, style='Room.TLabel', font=('Georgia', 18)).grid(row=0, column=1, sticky='e')
        ttk.Label(scores, textvariable=self.round_label, style='Subtle.TLabel').grid(row=1, column=0, columnspan=2)
        self.table = tk.Canvas(room, bg=FELT, highlightthickness=0, width=820, height=650)
        self.table.grid(row=1, column=0, sticky='nsew')
        self.table.bind('<Configure>', lambda event: self._draw_table())
        self.table.bind('<Leave>', lambda event: self.table.configure(cursor=''))
        bottom = ttk.Frame(room, style='Room.TFrame')
        bottom.grid(row=3, column=0, sticky='ew')
        bottom.columnconfigure(0, weight=1)
        side = ttk.Frame(room, style='Room.TFrame', padding=(15, 0, 0, 0))
        side.grid(row=0, column=1, rowspan=4, sticky='nsew')
        side.columnconfigure(0, weight=1)
        side.rowconfigure(4, weight=1)
        self.load_buttons = {}
        for i, slot in enumerate(('A', 'B')):
            frame = ttk.LabelFrame(side, text=f'Network {slot}', style='Room.TLabelframe', padding=8)
            frame.grid(row=i, column=0, sticky='ew', pady=(0, 7))
            frame.columnconfigure(0, weight=1)
            tk.Label(frame, textvariable=self.model_status[slot],
                     foreground='#ffffff', background=FELT,
                     font=('Helvetica', 11, 'bold'), anchor='w', justify='left',
                     wraplength=265, padx=8, pady=4).grid(row=0, column=0, sticky='ew')
            button = ttk.Button(frame, text='Load…', style='NetworkLoad.TButton', command=lambda slot=slot: self.open_network(slot=slot))
            button.grid(row=0, column=1, padx=(8, 0))
            self.load_buttons[slot] = button
        controls = ttk.LabelFrame(side, text='Computer turns', style='Room.TLabelframe', padding=8)
        controls.grid(row=2, column=0, sticky='ew', pady=(0, 7))
        controls.columnconfigure(0, weight=1)
        ttk.Label(controls, textvariable=self.control_status, style='Room.TLabel').grid(row=0, column=0, columnspan=2, sticky='w')
        self.controller_buttons = {}
        for column, slot in enumerate(('A', 'B')):
            button = ttk.Button(controls, text=f'Play with {slot}',
                                command=lambda slot=slot: self._select_controller(slot))
            button.grid(row=1, column=column, sticky='ew', pady=5,
                        padx=(0, 5) if column == 0 else (0, 0))
            self.controller_buttons[slot] = button
        self.pause_button = ttk.Button(controls, text='Resume automatic play', command=self.toggle_pause)
        self.pause_button.grid(row=2, column=0, sticky='ew')
        self.step_button = ttk.Button(controls, text='Next computer move', command=lambda: self._auto_turn(single=True))
        self.step_button.grid(row=2, column=1, padx=(5, 0), sticky='ew')
        ttk.Checkbutton(controls, text='Reveal the other hands', variable=self.reveal, command=self._visibility_changed).grid(row=3, column=0, columnspan=2, sticky='w', pady=(6, 0))
        self.comparison_title = tk.StringVar(value='Current decision')
        compare = ttk.LabelFrame(side, text='Policy comparison · before the move', style='Room.TLabelframe', padding=8)
        compare.grid(row=3, column=0, sticky='ew', pady=(0, 7))
        self.comparison_frame = compare
        tk.Label(compare, textvariable=self.comparison_title, background=FELT, foreground='#ffffff',
                 font=('Helvetica', 12, 'bold'), wraplength=350, justify='left', padx=8, pady=8
                 ).pack(fill='x')
        value_frame = ttk.Frame(compare, style='Room.TFrame')
        value_frame.pack(fill='x', pady=(7, 2))
        value_frame.columnconfigure(0, weight=1)
        value_frame.columnconfigure(1, weight=1)
        self.value_text = {slot: tk.StringVar(value=f'{slot}: no prediction') for slot in ('A', 'B')}
        self.value_buttons = {}
        for i, slot in enumerate(('A', 'B')):
            button = ttk.Button(value_frame, textvariable=self.value_text[slot],
                                command=lambda slot=slot: self._explain_value(slot))
            button.grid(row=0, column=i, sticky='ew', padx=(0, 5) if i == 0 else (5, 0))
            self.value_buttons[slot] = button
        self.probability_canvas = tk.Canvas(compare, width=370, height=265, bg='#fffdf6', highlightthickness=0)
        self.probability_canvas.pack(fill='x', pady=5)
        self.evaluation = tk.StringVar()
        ttk.Label(compare, textvariable=self.evaluation, style='Subtle.TLabel', wraplength=360).pack(anchor='w')
        self.swap_frame = ttk.Frame(bottom, style='Room.TFrame')
        self.swap_frame.grid(row=1, column=0, columnspan=2, sticky='w', pady=(6, 0))
        self.swap_frame.grid_remove()
        self.swap_yes = ttk.Button(self.swap_frame, text='Exchange seven', command=lambda: self._apply('Yes', manual=True))
        self.swap_yes.pack(side='left', padx=(0, 6))
        self.swap_no = ttk.Button(self.swap_frame, text='Keep seven', command=lambda: self._apply('No', manual=True))
        self.swap_no.pack(side='left')
        self.announcements = ttk.Frame(room, style='Room.TFrame')
        self.announcements.grid(row=2, column=0, sticky='ew', pady=(6, 0))
        self.announcements.columnconfigure(0, weight=1)
        self.song_canvas = tk.Canvas(self.announcements, width=350, height=56,
                                     background=PAPER, highlightthickness=0)
        self.song_canvas.grid(row=0, column=0, sticky='ew')
        self.song_canvas.bind('<Configure>', lambda event: self._draw_songs())

    def _round_rectangle(self,x0,y0,x1,y1,color,outline='',width=1,tags=()):
        r=7
        points=[x0+r,y0,x1-r,y0,x1,y0,x1,y0+r,x1,y1-r,x1,y1,x1-r,y1,x0+r,y1,x0,y1,x0,y1-r,x0,y0+r,x0,y0]
        return self.table.create_polygon(points,smooth=True,fill=color,outline=outline,width=width,tags=tags)

    def _suit(self,name,x,y,size,color,tags=(),canvas=None):
        c=self.table if canvas is None else canvas
        if name=='Oros':
            c.create_oval(x-size*.39,y-size*.39,x+size*.39,y+size*.39,outline=color,width=2,tags=tags)
            c.create_oval(x-size*.28,y-size*.28,x+size*.28,y+size*.28,outline=color,width=1,tags=tags)
            c.create_polygon(x,y-size*.18,x+size*.1,y,x,y+size*.18,x-size*.1,y,fill=color,tags=tags)
        elif name=='Copas':
            c.create_line(x-size*.32,y-size*.32,x-size*.26,y+size*.05,x,y+size*.22,x+size*.26,y+size*.05,x+size*.32,y-size*.32,fill=color,width=3,smooth=True,tags=tags)
            c.create_line(x-size*.33,y-size*.31,x+size*.33,y-size*.31,fill=color,width=2,tags=tags)
            c.create_line(x,y+size*.22,x,y+size*.40,fill=color,width=2,tags=tags)
            c.create_line(x-size*.19,y+size*.42,x+size*.19,y+size*.42,fill=color,width=2,tags=tags)
        elif name=='Espadas':
            c.create_polygon(x+size*.18,y-size*.44,x+size*.30,y-size*.30,x-size*.04,y+size*.18,x-size*.16,y+size*.1,fill='',outline=color,width=2,tags=tags)
            c.create_line(x-size*.26,y+size*.09,x+size*.11,y+size*.28,fill=color,width=3,tags=tags)
            c.create_line(x-size*.08,y+size*.20,x-size*.21,y+size*.43,fill=color,width=3,tags=tags)
        else:
            c.create_line(x-size*.17,y+size*.40,x+size*.18,y-size*.34,fill=color,width=max(5,int(size*.20)),capstyle='round',tags=tags)
            for k in [-.15,.08]:
                c.create_line(x+size*(k-.16),y-size*k,x+size*(k+.12),y-size*(k-.10),fill='#f5f1dd',width=1,tags=tags)

    def _card_image(self, card, width, height, back=False):
        suits = {'Oros': 'coins', 'Copas': 'cups', 'Espadas': 'swords', 'Bastos': 'clubs'}
        name = 'card_back' if back else f'card_{suits[card.suit]}_{int(RANK_NUMBERS[card.rank]):02d}'
        key = (name, int(width), int(height))
        if key not in self._card_images:
            if name not in self._image_sources:
                path = self.asset_dir / (name + '.png')
                if not path.is_file():
                    return None
                with Image.open(path) as source:
                    self._image_sources[name] = source.convert('RGBA')
            scaled = ImageOps.contain(self._image_sources[name], (int(width), int(height)), Image.Resampling.LANCZOS)
            self._card_images[key] = ImageTk.PhotoImage(scaled, master=self.root)
        return self._card_images[key]

    def _card(self,card,x,y,width=73,height=106,back=False,tag='',legal=False,hinted=False):
        c=self.table
        tags=(tag,) if tag else ()
        x0,x1,y0,y1=x-width/2,x+width/2,y-height/2,y+height/2
        self._round_rectangle(x0+2,y0+3,x1+2,y1+3,'#183f34',tags=tags)
        outline='#f0ce76' if hinted else '#d4caa6' if legal else '#e4dfc9'
        self._round_rectangle(x0,y0,x1,y1,'#faf7e9',outline,3 if hinted else 1.5,tags)
        artwork = self._card_image(card, width-4, height-4, back=back)
        if artwork is not None:
            c.create_image(x, y, image=artwork, tags=tags)
            return
        if back:
            c.create_rectangle(x0+4,y0+4,x1-4,y1-4,fill='#63816a',outline='#bbb98e',tags=tags)
            c.create_polygon(x,y-height*.22,x+width*.23,y,x,y+height*.22,x-width*.23,y,outline='#d8d4ad',fill='',tags=tags)
            return
        color=SUIT_COLORS[card.suit]
        number=RANK_NUMBERS[card.rank]
        c.create_rectangle(x0+4,y0+4,x1-4,y1-4,outline='#e7e1cc',tags=tags)
        font=max(10,int(width*.23))
        c.create_text(x0+9,y0+12,text=number,anchor='w',font=('Georgia',font,'bold'),fill=color,tags=tags)
        c.create_text(x1-8,y1-12,text=number,anchor='e',font=('Georgia',font,'bold'),fill=color,tags=tags)
        self._suit(card.suit,x,y,width*.60,color,tags)
        if width>=55:
            points=self.state.deck.points(card)
            c.create_text(x,y1-17,text=f'{points} pts' if points else card.suit.upper(),font=('Helvetica',7),fill=color,tags=tags)

    def _draw_table(self):
        if not hasattr(self,'table') or self.closed:return
        c=self.table;c.delete('all')
        w,h=max(c.winfo_width(),700),max(c.winfo_height(),480)
        c.create_oval(35,51,w-35,h-42,outline='#6b8b66',width=1)
        c.create_text(w/2,20,text='SOUTH → EAST → NORTH → WEST',fill='#bdcead',font=('Helvetica',9))
        if not self.state.trick:
            c.create_text(w/2,h*.44,text='guiñote.',fill='#ffffff',font=('Georgia',38,'italic'))
            c.create_text(w/2,h*.44+37,text='',fill='#e1e8cc',font=('Helvetica',8))
        positions=seat_positions(w,h)
        for p,(x,y) in positions.items():
            active=not self.game.game_over(self.state) and self.state.next=='play' and self.state.player_to_play==p
            label=SEAT_NAMES[p]
            c.create_text(x,y,text=('● ' if active else '')+label,fill='#f0d795' if active else '#e1e8cc',font=('Helvetica',11,'bold'))
            if p:
                hand=[deck.Card(self.state.deck.suits[st],self.state.deck.ranks[rk]) for st,rk in np.argwhere(self.state.hands[p])]
                for i,card in enumerate(hand):
                    if p==2:
                        card_width, card_height = 56, 86
                        px=x+(i-(len(hand)-1)/2)*61
                        py=y+61
                    else:
                        # Fit three full rows above the stock, without overlap.
                        card_height = min(93, max(40, (h-y-215)/3))
                        card_width = card_height * 207/319
                        px=x+(i%2-.5)*(card_width+10)
                        py=y+24+card_height/2+(i//2)*(card_height+10)
                    self._card(card,px,py,card_width,card_height,back=not self.reveal.get(),tag=f'opponent-{p}-{i}')
        # Never hide the trump: show its face beside the stock at every phase.
        tx,ty=w-100,h-87
        left=len(self.state.deck.cards)+(1 if self.state.deck.cards else 0)
        c.create_text(tx,ty-75,text='TRUMP · '+self.state.trump.suit.upper(),fill='#efd28b',font=('Helvetica',9,'bold'),tags=('trump-label',))
        if left:
            self._card(None,tx-25,ty+4,63,91,back=True,tag='stock')
            c.create_text(tx,ty+65,text=f'{left} cards to draw',fill='#c8d5b7',font=('Helvetica',8))
        else:
            c.create_text(tx,ty+65,text='Trump suit reference · stock closed',fill='#c8d5b7',font=('Helvetica',7))
        self._card(self.state.trump,tx+13,ty,63,91,tag='trump-face')
        # Keep played cards clear of both hands and the announcement tray.
        trick_positions={0:(w*.5,h-235),1:(w*.63,h*.47),2:(w*.5,190),3:(w*.37,h*.47)}
        start=(self.state.player_to_play-len(self.state.trick))%4
        for i,card in enumerate(self.state.trick):
            owner = int(self.state.seen_players[self.state.deck.encode(card)]) - 1
            p = owner if 0 <= owner < 4 else (start+i)%4
            x,y=trick_positions[p]
            self._card(card,x,y,64,93,tag=f'trick-{p}')
            c.create_text(x+43 if p in (0,2) else x, y if p in (0,2) else y+54,
                          anchor='w' if p in (0,2) else 'center',
                          text=SEAT_NAMES[p].split(' · ')[0],fill='#d3dfbf',font=('Helvetica',8))
        legal=self.game.playable_cards(self.state) if self.state.next=='play' else np.zeros((4,10))
        hands=[(st,rk,deck.Card(self.state.deck.suits[st],self.state.deck.ranks[rk])) for st,rk in np.argwhere(self.state.hands[0])]
        card_w=min(77,max(59,(w-245)/7))
        can_play=not self.busy and self.comparison_ready and self.state.next=='play' and self.state.player_to_play==0
        for i,(st,rk,card) in enumerate(hands):
            allowed=can_play and bool(legal[st,rk])
            tag=f'hand-{st}-{rk}'
            x=w/2+(i-(len(hands)-1)/2)*(card_w+5)
            y=h-90-(7 if allowed else 0)
            hinted=self.hint_card==(int(st),int(rk))
            self._card(card,x,y,card_w,104,tag=tag,legal=allowed,hinted=hinted)
            if self.state.next == 'play' and self.state.player_to_play == 0:
                for j, slot in enumerate(('A', 'B')):
                    prob = self._card_probability(slot, card)
                    label = f'{slot}: {prob:.1%}' if prob is not None else f'{slot}: —'
                    c.create_text(x, y+62+j*15, text=label,
                                  fill=NETWORK_COLORS[slot], font=('Helvetica', 9, 'bold'))
            if can_play and not allowed:
                c.create_rectangle(x-card_w/2,y-52,x+card_w/2,y+52,fill=FELT,stipple='gray50',outline='',tags=(tag,))
            if allowed:
                c.tag_bind(tag,'<Button-1>',lambda event,chosen=card:self.play_card(chosen))
                c.tag_bind(tag,'<Enter>',lambda event:c.configure(cursor='hand2'))
                c.tag_bind(tag,'<Leave>',lambda event:c.configure(cursor=''))

    def _draw_songs(self):
        """Compact public pairs: two cards, player, suit and points."""
        if self.closed or not hasattr(self, 'song_canvas'):
            return
        canvas = self.song_canvas
        canvas.delete('all')
        if not self.song_history:
            self.announcements.grid_remove()
            return
        self.announcements.grid()
        columns = max(1, canvas.winfo_width() // 190)
        rows = (len(self.song_history) + columns - 1) // columns
        canvas.configure(height=rows*56)
        for index, (player, suit, points, _trick) in enumerate(self.song_history):
            left, top = (index % columns)*190, (index // columns)*56
            for card, x in zip(singing_cards(suit), (left+17, left+50)):
                artwork = self._card_image(card, 30, 46)
                if artwork is not None:
                    canvas.create_image(x, top+26, image=artwork, tags=('song-card',))
                else:
                    canvas.create_text(x, top+26, text=RANK_NUMBERS[card.rank],
                                       fill=SUIT_COLORS[suit], tags=('song-card',))
            name = SEAT_NAMES[player].split(' · ')[0]
            canvas.create_text(left+74, top+17, text=name, anchor='w', fill=INK,
                               font=('Helvetica', 10, 'bold'), tags=('song-player',))
            canvas.create_text(left+74, top+36, text=f'{suit} · {points}', anchor='w',
                               fill=INK, font=('Helvetica', 10), tags=('song-suit',))

    def _announce(self, text):
        self.event_text = text
        self.event_hold = True
        self.notes.append(text)

    def _explain_value(self, slot):
        result = self.decisions.get(slot)
        if result is None or result.value is None:
            return
        scope = getattr(result, 'value_scope', 'hand')
        meaning = ('the current trick' if scope == 'trick' else 'the final whole-hand point difference')
        messagebox.showinfo(f'Network {slot} prediction',
            f'{result.value:+.1f} points for {meaning}.\n\n'
            'Positive favours South + North; negative favours East + West.\n'
            'Trick and whole-hand values predict different quantities and should not be compared directly.',
            parent=self.root)

    def _human_swap(self):
        return self.state.next=='swap' and self.state.player_to_play%2==0 and self.game.allowed_swapping(self.state,0)

    @staticmethod
    def _action_key(action):
        return action if isinstance(action, str) else (action.suit, action.rank)

    def _card_probability(self, slot, card):
        if not self.comparison_ready or slot not in self.decisions:
            return None
        key = self._action_key(card)
        return next((p for a, p in self.decisions[slot].probabilities
                     if self._action_key(a) == key), 0.0)

    def _visibility_changed(self):
        self.render()

    def _comparison_visible(self):
        return self.state.player_to_play == 0 or self.reveal.get()

    def _draw_comparison(self):
        canvas = self.probability_canvas
        canvas.delete('all')
        if not self._comparison_visible():
            self.comparison_frame.grid_remove()
            for slot in ('A', 'B'):
                self.value_text[slot].set(f'{slot} · Hidden hand')
                self.value_buttons[slot].configure(state='disabled')
            return
        self.comparison_frame.grid()
        live = not self.game.game_over(self.state) and self.state.next in ('play', 'swap')
        for slot in ('A', 'B'):
            result = self.decisions.get(slot) if live and self.comparison_ready else None
            if result is not None and result.value is not None:
                scope = getattr(result, 'value_scope', 'hand')
                self.value_text[slot].set(f'{slot} · {scope.title()} value\n{result.value:+.1f} pts')
                self.value_buttons[slot].configure(state='normal')
            else:
                label = 'not loaded' if self.networks[slot] is None else 'unavailable' if slot in self.errors else '—'
                self.value_text[slot].set(f'{slot} · Value\n{label}')
                self.value_buttons[slot].configure(state='disabled')
        if not live:
            self.comparison_title.set('Hand complete' if self.game.game_over(self.state) else 'Between decisions')
            self.evaluation.set('Probabilities refresh at the next play or swap decision.')
            return
        player = self.state.player_to_play
        self.comparison_title.set(('PLAYER TO PLAY: ' + SEAT_NAMES[player]) if self.state.next == 'play' else ('SEVEN EXCHANGE: ' + ('South · You' if self._human_swap() else SEAT_NAMES[player])))
        if not self.comparison_ready:
            canvas.create_text(12, 28, anchor='w', text='Evaluating both networks…', fill=INK)
            self.evaluation.set('')
            return
        if self.state.next == 'play':
            actions = [deck.Card(self.state.deck.suits[st], self.state.deck.ranks[rk])
                       for st, rk in np.argwhere(self.state.hands[player])]
            legal = self.game.playable_cards(self.state)
        else:
            actions, legal = ['No', 'Yes'], None
        maps = {slot: {self._action_key(a): p for a, p in decision.probabilities}
                for slot, decision in self.decisions.items()}
        if 'A' in maps:
            actions.sort(key=lambda action: -maps['A'].get(self._action_key(action), 0.0))
        canvas.configure(height=max(110, 30+len(actions)*28))
        canvas.create_text(12, 12, text='Card / action', anchor='w', fill=INK, font=('Helvetica', 10, 'bold'))
        for slot, x in [('A', 230), ('B', 320)]:
            canvas.create_text(x, 12, text=f'Network {slot}', fill=INK, font=('Helvetica', 10, 'bold'))
        for i, action in enumerate(actions):
            y = 35+i*28
            allowed = bool(legal[self.state.deck.encode(action)]) if legal is not None else (
                action == 'No' or any(self.game.allowed_swapping(self.state, p) for p in self.game.team(player)))
            label = action_label(action) + ('' if allowed else ' · illegal')
            if not isinstance(action, str):
                self._suit(action.suit, 20, y, 19, SUIT_COLORS[action.suit],
                           tags=('policy-suit',), canvas=canvas)
            canvas.create_text(36 if not isinstance(action, str) else 10, y, text=label,
                               anchor='w', fill=INK if allowed else MUTED, font=('Helvetica', 9),
                               tags=('policy-action',))
            key = self._action_key(action)
            for slot, x in [('A', 230), ('B', 320)]:
                prob = maps.get(slot, {}).get(key, 0.0)
                text = '—' if slot not in maps or not allowed else f'{prob:.1%}'
                canvas.create_rectangle(x-37, y+10, x+37, y+14, fill='#e9e8e0', outline='')
                if slot in maps and allowed:
                    canvas.create_rectangle(x-37, y+10, x-37+74*prob, y+14, fill=NETWORK_COLORS[slot], outline='')
                canvas.create_text(x, y, text=text, fill=INK, font=('Helvetica', 10, 'bold'))
        notes = ['Sorted by A · legal probabilities total 100%.' if 'A' in maps else 'Load network A to sort by its probabilities.']
        for slot in ('A', 'B'):
            if slot in self.errors:
                notes.append(f'{slot}: evaluation failed — {self.errors[slot]}')
            elif self.networks[slot] is None:
                notes.append(f'{slot}: not loaded.')
            elif slot in self.decisions:
                result = self.decisions[slot]
                if self.state.next == 'swap' and self.networks[slot].model.output_channels == 2:
                    notes.append(f'{slot}: no swap head; neutral probabilities.')
        notes.append('Values: South + North. Click for details.')
        self.evaluation.set('\n'.join(notes))

    def render(self):
        us, them = self.game.team_points(self.state)
        self.finished = self.game.game_over(self.state)
        self.score_us.set(f'You + North    {int(us)}')
        self.score_them.set(f'East + West    {int(them)}')
        self.round_label.set(f'Trick {min(self.state.round, 10)} / 10 · South → East → North → West')
        available = not self.busy
        for widget in [self.new_button, *self.load_buttons.values()]:
            widget.configure(state='normal' if available else 'disabled')
        slots = [slot for slot in ('A', 'B') if self.networks[slot] is not None]
        self.control_status.set('Computer moves use network:' if slots else 'Random legal moves · no network loaded')
        if slots and self.controller.get() not in slots:
            self.controller.set(slots[0])
        elif not slots:
            self.controller.set('')
        for slot, button in self.controller_buttons.items():
            selected = self.controller.get() == slot
            button.configure(text=f'{slot} · Playing' if selected else f'Play with {slot}',
                             style='TButton' if selected else 'Inactive.TButton',
                             state='normal' if available and slot in slots else 'disabled')
        self.pause_button.configure(text='Resume automatic play' if self.paused else 'Pause automatic play')
        human = (self.state.next == 'play' and self.state.player_to_play == 0) or self._human_swap()
        ready = self.comparison_ready or self.state.next not in ('play', 'swap')
        self.step_button.configure(state='normal' if available and ready and self.paused and not human and not self.finished else 'disabled')
        self.undo_button.configure(state='normal' if available and self.undo_stack else 'disabled')
        if self._human_swap() and not self.finished:
            self.swap_frame.grid()
        else:
            self.swap_frame.grid_remove()
        for button in (self.swap_yes, self.swap_no):
            button.configure(state='normal' if available and self.comparison_ready else 'disabled')
        if self.finished:
            who = 'Your team wins' if us > them else 'East + West win' if them > us else 'The hand is tied'
            self.status.set(f'{who}: {int(us)}–{int(them)}. Click New hand to play again.')
        elif self.busy:
            self.status.set('Loading network…' if self.job_kind.startswith('load:') else 'Evaluating both networks before the move…')
        elif human:
            self.status.set('Compare A and B, then ' + ('choose whether to exchange the seven.' if self._human_swap() else 'click a highlighted card.'))
        elif self.paused:
            self.status.set('Inspect both policies, then click Next computer move. Automatic play is paused.' if self._comparison_visible() else 'Hand hidden. Reveal the other hands to compare policies, or click Next computer move.')
        else:
            self.status.set('Automatic play · policies are displayed before each computer move.')
        self._draw_table()
        self._draw_comparison()
        self._draw_songs()

    def _cancel_timer(self):
        if self.timer is not None:
            self.root.after_cancel(self.timer)
            self.timer = None

    def _schedule(self):
        self._cancel_timer()
        if self.closed or self.busy or self.paused or self.finished or self._human_swap():
            return
        if self.state.next == 'play' and self.state.player_to_play == 0:
            return
        if self.state.next in ('play', 'swap') and not self.comparison_ready:
            return
        delay = 2200 if self.state.next in ('play', 'swap') else 1100 if self.state.next == 'solve_trick' else 250
        if self.event_hold:
            delay = max(delay, 2600)
        self.timer = self.root.after(delay, self._auto_turn)

    def _submit(self, kind, function):
        self._cancel_timer()
        self.busy = True
        self.job_kind = kind
        token = self.epoch
        self.render()
        def run():
            try:
                self.queue.put((token, kind, function(), None))
            except Exception as exc:
                self.queue.put((token, kind, None, str(exc)))
        threading.Thread(target=run, daemon=True, name=f'guinote-{kind}').start()

    def _poll(self):
        if self.closed:
            return
        try:
            while True:
                token, kind, result, err = self.queue.get_nowait()
                if token != self.epoch:
                    continue
                self.busy = False
                if err:
                    self.paused = True
                    self.render()
                    messagebox.showerror('Guiñote', err, parent=self.root)
                    if kind.startswith('load:'):
                        self._startup()
                    continue
                if kind.startswith('load:'):
                    slot = kind.split(':')[1]
                    self.networks[slot] = result
                    self.model_status[slot].set(f'{result.path.name}\n{result.model.encoding} · {result.blocks} blocks · {result.hidden} channels')
                    self.notes.append(f'Network {slot}: loaded {result.path.name}.')
                    self._invalidate()
                    self._startup()
                else:
                    self.decisions, self.errors = result
                    self.comparison_ready = True
                    if self.errors:
                        self.paused = True
                    self.render()
                    self._schedule()
        except queue.Empty:
            pass
        self.root.after(60, self._poll)

    def _startup(self):
        if self._startup_checkpoints:
            slot, path = self._startup_checkpoints.pop(0)
            self.open_network(path, slot)
        else:
            self._prepare_turn()

    def _invalidate(self):
        self.epoch += 1
        self.decisions = {}
        self.errors = {}
        self.comparison_ready = False
        self.hint_card = None

    def _prepare_turn(self):
        if self.busy or self.closed:
            return
        if self.game.game_over(self.state) or self.state.next not in ('play', 'swap'):
            self.render()
            self._schedule()
            return
        if self.comparison_ready:
            self.render()
            self._schedule()
            return
        networks = [(slot, item.model) for slot, item in self.networks.items() if item is not None]
        if not networks:
            self.comparison_ready = True
            self.render()
            self._schedule()
            return
        state = self.state.copy()
        def evaluate():
            decisions, errors = {}, {}
            for slot, model in networks:
                try:
                    decisions[slot] = network_decision(self.game, state.copy(), model)
                except Exception as exc:
                    errors[slot] = str(exc)
            return decisions, errors
        self._submit('compare', evaluate)

    def open_network(self, path=None, slot='A'):
        if self.busy:
            return
        self._cancel_timer()
        if not path:
            path = filedialog.askopenfilename(parent=self.root, title=f'Load network {slot}',
                filetypes=[('PyTorch weights', '*.pt *.pth *.ckpt'), ('All files', '*')])
        if not path:
            self._schedule()
            return
        self._submit('load:'+slot, lambda: load_network(path, self.game))

    def _select_controller(self, slot):
        if self.busy or self.networks[slot] is None:
            return
        self.controller.set(slot)
        self.render()
        self._schedule()

    def _auto_turn(self, single=False):
        if self.busy or self.finished or (self.paused and not single):
            return
        if self._human_swap() or (self.state.next == 'play' and self.state.player_to_play == 0):
            return
        if self.state.next not in ('play', 'swap'):
            self._apply()
            return
        if not self.comparison_ready:
            self._prepare_turn()
            return
        if not any(self.networks.values()):
            action = random_decision(self.game, self.state).action
        else:
            slot = self.controller.get()
            if slot not in self.decisions:
                self.paused = True
                self.render()
                messagebox.showerror('Network evaluation', self.errors.get(slot, 'The selected network has no decision.'), parent=self.root)
                return
            action = self.decisions[slot].action
        self._apply(action)

    def play_card(self, card):
        if self.busy or not self.comparison_ready or self.state.next != 'play' or self.state.player_to_play != 0:
            return
        if not self.game.playable_cards(self.state)[self.state.deck.encode(card)]:
            return
        self._apply(card, manual=True)

    def _apply(self, action=None, manual=False):
        self._cancel_timer()
        if self.busy or self.game.game_over(self.state):
            return
        if self.state.next in ('play', 'swap') and not self.comparison_ready:
            return
        if manual:
            self.undo_stack.append((self.state.copy(), list(self.notes), list(self.song_history), self.event_text))
        phase, p, r = self.state.next, self.state.player_to_play, self.state.round
        oldpoints = self.state.points.copy()
        oldtrick, oldsongs = list(self.state.trick), list(self.state.songs)
        oldtrump = (self.state.trump.suit, self.state.trump.rank)
        before = self.state.copy()
        try:
            self.game.next(self.state, card=action if phase == 'play' else None, swap=action if phase == 'swap' else None)
        except Exception as exc:
            self.state = before
            if manual:
                self.undo_stack.pop()
            self.paused = True
            self.render()
            messagebox.showerror('Cannot play', str(exc), parent=self.root)
            return
        self.event_hold = False
        if phase == 'play':
            self.notes.append(f'{SEAT_NAMES[p]} plays {action_label(action)}.')
        elif phase == 'solve_trick':
            winner = self.state.player_to_play
            points = int(self.state.points[winner]-oldpoints[winner])
            self._announce(f'Trick {r} collected by {SEAT_NAMES[winner]} · +{points} points')
            self.notes.append('Cards: ' + ', '.join(action_label(c) for c in oldtrick))
        elif phase == 'sing':
            announcements = []
            for suit in oldsongs:
                if suit not in self.state.songs:
                    st = before.deck.suit_code[suit]
                    # The engine exposes the king and sota with the singer's id.
                    king_card, sota_card = singing_cards(suit)
                    king = before.deck.rank_code[king_card.rank]
                    sota = before.deck.rank_code[sota_card.rank]
                    singer = next((player for player in self.game.team(p)
                                   if before.hands[player, st, king] and before.hands[player, st, sota]), None)
                    if singer is None:
                        announcements.append(f'{suit} song recorded')
                        continue
                    points = 40 if suit == before.trump.suit else 20
                    self.song_history.append((singer, suit, points, r-1))
                    announcements.append(f'{SEAT_NAMES[singer]} sang {suit} · +{points} points')
            if announcements:
                self._announce(' / '.join(announcements))
        elif phase == 'swap':
            if action == 'Yes' and oldtrump != (self.state.trump.suit, self.state.trump.rank):
                st, rk = before.deck.encode(before.trump)
                seven = before.deck.rank_code['Seven']
                players = [player for player in self.game.team(p)
                           if before.hands[player, st, seven] and self.state.hands[player, st, rk]]
                who = ', '.join(SEAT_NAMES[player] for player in players) or 'Winning team'
                self._announce(f'{who} exchanges Siete de {oldtrump[0]} for {action_label(deck.Card(*oldtrump))}')
            else:
                self._announce('Seven kept · no trump exchange')
        elif phase == 'collection':
            drawn = len(before.deck.cards) - len(self.state.deck.cards)
            # The exposed trump is also drawn when the stock empties.
            drawn += int(bool(before.deck.cards) and not self.state.deck.cards)
            if drawn:
                self._announce(f'Cards drawn from stock: {drawn} · {SEAT_NAMES[p]} draws first'
                               + (' · stock now closed' if not self.state.deck.cards else ''))
        self._invalidate()
        self._prepare_turn()

    def toggle_pause(self):
        self.paused = not self.paused
        self.render()
        self._schedule()

    def undo(self):
        if self.busy or not self.undo_stack:
            return
        self._cancel_timer()
        self.state, self.notes, self.song_history, self.event_text = self.undo_stack.pop()
        self.event_hold = False
        self._invalidate()
        self._prepare_turn()

    def new_hand(self):
        if self.busy:
            return
        self._cancel_timer()
        self.state = self.game.initial_state()
        self.undo_stack = []
        self.song_history = []
        self.event_text = 'No announcements yet.'
        self.event_hold = False
        self.notes = ['A new hand. South leads; play continues anticlockwise.']
        self._invalidate()
        self._prepare_turn()

    def help(self):
        messagebox.showinfo('Guiñote',
            'Load networks A and B to compare their direct policy probabilities on every decision, including your South turn.\n\n'
            'Each column totals 100% across legal actions (subject to rounding). Illegal cards are marked with a dash. '
            'South also has A/B percentages under each card.\n\n'
            'Choose A or B to control all computer players. If no network is loaded, computer moves are random. '
            'South always remains under your control.\n\n'
            'Automatic play starts paused so you can inspect every decision. Use Next computer move to advance, '
            'or resume for a 2.2-second preview before each computer decision.\n\n'
            'Only direct network evaluation is used. A model without a swap head uses neutral swap probabilities. '
            'Policies are sorted by A and shown only for visible hands. Reveal the other hands to compare computer turns.\n\n'
            'Announced pairs appear in their own panel, with player, suit, points, Rey and Sota. Scroll to view earlier pairs.\n\n'
            'Card artwork: Basquetteur, Wikimedia Commons, CC BY-SA 3.0. SVG collection by gjenkins20; rendered to PNG for this interface.\n'
            'github.com/gjenkins20/spanish-playing-cards-svg' , parent=self.root)

    def _close(self):
        self.closed = True
        self.epoch += 1
        self._cancel_timer()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, help='Network A weights')
    parser.add_argument('--checkpoint-b', type=Path, help='Network B weights')
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = tk.Tk()
    GuinoteApp(root, args.checkpoint, args.checkpoint_b)
    root.mainloop()


if __name__ == '__main__':
    main()
