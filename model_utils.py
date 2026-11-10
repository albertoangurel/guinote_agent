"""Checkpoint loading and decisions for the user's Guiñote network."""
from dataclasses import dataclass
from pathlib import Path
import re
import importlib

import numpy as np
import torch

import deck
import game_training


@dataclass
class LoadedNetwork:
    model: torch.nn.Module
    path: Path
    blocks: int
    hidden: int


@dataclass
class Decision:
    action: object
    value: float | None
    probabilities: list
    value_scope: str = 'hand'


def _dimensions(weights, prefix=''):
    first = weights.get(prefix + 'startBlock.0.weight')
    if first is None or first.ndim != 4:
        raise ValueError(f'Missing convolution: {prefix}startBlock.0.weight')
    pattern = re.escape(prefix) + r'backBone\.(\d+)\.'
    indices = {int(m.group(1)) for key in weights
               if (m := re.match(pattern, key))}
    blocks = max(indices) + 1 if indices else 0
    if indices != set(range(blocks)):
        raise ValueError('The residual block indices are incomplete.')
    return blocks, int(first.shape[0])


class _CheckpointGate(torch.nn.Module):
    """LongTermGate inference architecture; all weights come from one checkpoint.

    Unlike its training constructor, this does not load TrickTraining.pt.
    """
    def __init__(self, game, weights, blocks, hidden):
        super().__init__()
        nn = torch.nn
        module = importlib.import_module('trick_training')
        self.encoding = 'long-term'
        self.output_channels = 3
        self.input_channels = int(weights['startBlock.0.weight'].shape[1])
        self.game = game
        trick_blocks, trick_hidden = _dimensions(weights, 'trickBlock.')
        self.trickBlock = module.TrickBlock(
            game, num_blocks=trick_blocks, num_hidden=trick_hidden)
        self.startBlock = nn.Sequential(
            nn.Conv2d(self.input_channels, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden), nn.ReLU())
        self.backBone = nn.ModuleList(
            [module.ResBlock2CNN(hidden) for _ in range(blocks)])
        for name in ('gateHead', 'sevenHead', 'valueHead'):
            width = int(weights[name + '.0.weight'].shape[0])
            linear = weights[name + '.4.weight']
            setattr(self, name, nn.Sequential(
                nn.Conv2d(hidden, width, 3, padding=1),
                nn.BatchNorm2d(width), nn.ReLU(), nn.Flatten(),
                nn.Linear(int(linear.shape[1]), int(linear.shape[0]))))
        self.trickBlock.requires_grad_(False)

    def forward(self, trick_code, long_code):
        trick_logits, _ = self.trickBlock(trick_code)
        x = self.startBlock(long_code)
        for block in self.backBone:
            x = block(x)
        cards = trick_logits + torch.nn.functional.logsigmoid(self.gateHead(x))
        return cards, self.sevenHead(x), self.valueHead(x)


def _model_outputs(game, state, model):
    """Return card logits, swap logits (or None), and value."""
    encoding = model.encoding
    encoders = {
        'full': ('encode_perspective',),
        'trick': ('encode_trick_perspective',),
        'long-term': ('encode_trick_perspective', 'encode_longterm_perspective'),
    }
    if encoding not in encoders:
        raise ValueError(f'Unsupported model.encoding: {encoding!r}')
    device = next(model.parameters()).device
    inputs = []
    for name in encoders[encoding]:
        encoder = getattr(game, name, None)
        if encoder is None:
            raise ValueError(f'Your deck.py needs {name} for this network.')
        x = encoder(state).to(device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        inputs.append(x)
    outputs = model(*inputs)
    count = model.output_channels
    if count not in (2, 3) or not isinstance(outputs, (tuple, list)) or len(outputs) != count:
        raise ValueError('Expected output_channels=2 (cards,value) or 3 (cards,seven,value).')
    expected = [(1, game.num_cards)] + [(1, 1)] * (count - 1)
    if any(not torch.is_tensor(x) or tuple(x.shape) != shape
           for x, shape in zip(outputs, expected)):
        raise ValueError(f'Unexpected network output shapes; expected {expected}.')
    if not all(torch.isfinite(x).all().item() for x in outputs):
        raise ValueError('The network produced non-finite outputs.')
    if count == 2:
        return outputs[0], None, outputs[1]
    return tuple(outputs)


def load_network(path, game, model=None):
    """Load ResNet, TrickBlock, or LongTermGate weights on CPU.

    Plain state_dict files do not save encoding/output_channels attributes.
    Known architectures are identified by their weight keys. An already
    constructed model can be supplied for another compatible architecture.
    """
    path = Path(path).expanduser()
    payload = torch.load(path, map_location='cpu', weights_only=True)
    metadata = payload if isinstance(payload, dict) else {}
    if isinstance(payload, dict):
        for name in ('model_state_dict', 'state_dict'):
            if name in payload:
                payload = payload[name]
                break
    if not isinstance(payload, dict) or not payload:
        raise ValueError('Select a state_dict checkpoint, not an optimizer or replay file.')
    weights = {}
    for name, tensor in payload.items():
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise ValueError('This file is not a network state_dict.')
        while name.startswith(('module.', '_orig_mod.')):
            name = name.split('.', 1)[1]
        if name in weights:
            raise ValueError(f'Duplicate checkpoint key: {name}')
        weights[name] = tensor
    blocks, hidden = _dimensions(weights)
    if model is None:
        if any(key.startswith('trickBlock.') for key in weights):
            model = _CheckpointGate(game, weights, blocks, hidden)
        elif not any(key.startswith('sevenHead.') for key in weights):
            module = importlib.import_module('trick_training')
            model = module.TrickBlock(game, num_blocks=blocks, num_hidden=hidden)
            # Compatibility with older TrickBlock definitions.
            if not hasattr(model, 'encoding'): model.encoding = 'trick'
            if not hasattr(model, 'output_channels'): model.output_channels = 2
        else:
            factory = getattr(game_training, 'ResNet', None)
            if factory is None:
                raise ValueError('This full-encoding checkpoint needs a ResNet class in game_training.py. '
                                 'TrickPlay and GamePlay checkpoints are supported without train.py.')
            model = factory(game, numBlocks=blocks, num_hidden=hidden)
            if not hasattr(model, 'encoding'): model.encoding = 'full'
            if not hasattr(model, 'output_channels'): model.output_channels = 3
    if not hasattr(model, 'encoding') or not hasattr(model, 'output_channels'):
        raise ValueError('The model must define encoding and output_channels.')
    if metadata is not payload:
        for name in ('encoding', 'output_channels'):
            if name in metadata and metadata[name] != getattr(model, name):
                raise ValueError(f'Checkpoint {name} does not match the constructed model.')
    model.to('cpu')
    if hasattr(model, 'device'): model.device = torch.device('cpu')
    model.load_state_dict(weights, strict=True)
    model.eval()
    with torch.inference_mode():
        _model_outputs(game, game.initial_state(), model)
    return LoadedNetwork(model, path, blocks, hidden)


def action_label(action):
    if isinstance(action, str):
        return 'Exchange seven' if action == 'Yes' else 'Keep seven'
    ranks = {'As':'As', 'Three':'Tres', 'King':'Rey', 'Jack':'Sota', 'Queen':'Caballo',
             'Seven':'Siete', 'Six':'Seis', 'Five':'Cinco', 'Four':'Cuatro', 'Two':'Dos'}
    return f'{ranks[action.rank]} de {action.suit}'


def singing_cards(suit):
    """The public announcement pair: Rey (12) and Sota (10) of one suit."""
    return deck.Card(suit, 'King'), deck.Card(suit, 'Jack')


def random_decision(game, state):
    if state.next == 'swap':
        return Decision(str(np.random.choice(['No', 'Yes'])), None, [])
    legal = np.argwhere(game.playable_cards(state))
    st, rk = legal[np.random.randint(len(legal))]
    return Decision(deck.Card(state.deck.suits[st], state.deck.ranks[rk]), None, [])


@torch.inference_mode()
def network_decision(game, state, model):
    """One forward pass, with probabilities normalized over legal actions only."""
    if state.next not in ('play', 'swap') or game.game_over(state):
        raise ValueError('A network decision requires a live play or swap phase.')
    model.eval()
    cards, seven, value = _model_outputs(game, state, model)
    if state.next == 'play':
        logits = cards[0].cpu().numpy().reshape(4, 10)
        legal = np.argwhere(game.playable_cards(state))
        if not len(legal):
            raise ValueError('There are no legal cards in this position.')
        valid_logits = np.array([logits[st, rk] for st, rk in legal], dtype=float)
        probs = np.exp(valid_logits - valid_logits.max())
        probs /= probs.sum()
        actions = [deck.Card(state.deck.suits[st], state.deck.ranks[rk]) for st, rk in legal]
    else:
        can_swap = any(game.allowed_swapping(state, p)
                       for p in game.team(state.player_to_play))
        if not can_swap:
            actions, probs = ['No'], np.array([1.0])
        else:
            yes = 0.5 if seven is None else float(torch.sigmoid(seven).item())
            actions, probs = ['No', 'Yes'], np.array([1-yes, yes])
    ordered = sorted(zip(actions, probs.tolist()), key=lambda x: -x[1])
    estimate = float(value.item())
    if state.player_to_play % 2:
        estimate = -estimate
    return Decision(ordered[0][0], estimate, ordered,
                    value_scope='trick' if model.encoding == 'trick' else 'hand')
