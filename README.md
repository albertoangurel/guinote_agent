Guiñote AI

**Learning to play a four-player partnership card game with neural networks, tree search, and self-play.**

This project explores how an agent can combine tactical card-play knowledge with longer-term team strategy in Guiñote, a Spanish trick-taking game. It brings together a Python game environment, PyTorch models, Monte Carlo Tree Search (MCTS), parallel self-play. The game can be visualized wit a desktop interface for playing against and inspecting trained networks, created with ChatGPT Work.


The central experiment is a transfer learning approach: reuse a pretrained trick-level policy, then learn how to adjust it to improve the outcome of a complete hand.

It is important to note we are using a teacher approach, in which the model is trained using an MCTS algorithm which uses information that is not available to the player during a real game. However, only the available information is given to the neural network, meaning that it will be able to play a real game.

## Project highlights

- **Guiñote interface**: Python implementation of the first stage of game Guiñote. Note that the second round has not been implemented yet. The rules of the game can be found in https://www.nhfournier.es/en/como-jugar/guinote/
- **Structured decision-making:** card selection and trump-seven exchanges have separate policy outputs, alongside a team-score value estimate.
- **Transfer learning:** a pretrained `TrickBlock` supplies tactical card logits to a larger `LongTermGate` model.
- **Search-guided self-play:** neural-network priors and value estimates guide MCTS, which generates training targets.
- **Interactive inspection:** a Tkinter interface supports playing, loading two networks, inspecting legal-action probabilities, and revealing hands for analysis.


## Model architecture

The full-hand model combines two branches:

| Component | Role |
| --- | --- |
| `TrickBlock` | Produces pretrained card-play logits from the trick-perspective encoding. |
| Long-term residual network | Processes the long-term state encoding; defaults to 10 residual blocks and 64 hidden channels. |
| Gate head | Adjusts the trick model's logits using longer-term context. |
| Seven head | Produces a logit for exchanging the trump seven when the action is available. |
| Value head | Estimates the final team-score difference from the acting team's perspective. |

The card-policy branches are combined as:

```python
card_logits = trick_logits + F.logsigmoid(gate_logits)
```

After softmax, this is equivalent to multiplying the trick policy by a learned sigmoid gate and renormalizing. Legal-action masking restricts the resulting policy to playable cards.

In the current training implementation, `TrickBlock` is frozen while the long-term branch learns. Joint fine-tuning with a smaller learning rate for the pretrained branch is a planned extension.

## Training pipeline

1. Copy the current model weights into independent self-play workers.
2. Generate games with  MCTS-guided decisions.
3. Store state encodings and search-generated policy targets in separate card and swap replay buffers.
4. Assign the final score difference to each sample from its acting team's perspective.
5. Train the policy and value heads on sampled replay data.
6. Save model weights, optimizer state, and replay data.

Card policies use cross-entropy loss; swap policies use binary cross-entropy with logits. The value head uses mean squared error, scaled separately in the card and swap training steps.

The search uses a PUCT-style selection score. Its current policy target is the greedy approach that puts equal probability on the visited children tied for the highest estimated action value.




## Training

The current training entry point is configured to resume an existing experiment. Each stage of the training has been done with a different scipt:
-Trick_training.py: trick-level training.
-Game_training.py: transfer learning
-Fine_tunning.py: final fine tunning of the model

## Interface 
The interface can be displayed running the script interface.py. This script, and model_utils.py, have been coded by ChatGPT Work.



## Evaluation and current limitations

This is an experimental agent; a reproducible benchmark suite and human-strength claims are not yet established here.

A useful evaluation protocol is to compare successive checkpoints against themselves a random legal-action baseline and a fixed reference model, and report mean score differences with confidence intervals.

As a keen guiñote player, the author can compare the strategies of the trained models with common strategies played by human experts. The main difference between strategies is that human players tend to keep cards with the trump suit until the final phase of the game. This difference is hugely noticeable in the trick training, as expected, and it is breifly reduced in second stage of the training process. The author cannot confidently say which strategy is statistically better.

However, it has been detected that the agent took provably bad decisions in some specific states of the game, what suggests that there is room for improvement. In each stage, the improvement was monitored by comparing the performances of the newest models with the partially trained models, and stopped when there was no evicence of improvement, since further training might lead to overfitting. 

The author believes that a trained, bigger neural network (current one has 2.5 M parameters) might lead to better performance. Especially, one should use more parameters to build the logical gate. 

To illustrate the improvement of the model with training, we have done the following statistical experiment. We let each pair of following agents play 3000 games and, using matches.py, compute the expected value of the difference of points between teams, with confidence intervals of 95%
Agent 0 ('random'): random player.
Agent 1 (TrickTraining_100.pt): trained with 10 000 self play games maximizing the points in each trick
Agent 2 (TrickTraining_200.pt): trained with 20 000 self play games maximizing the points in each trick
Agent 3 (GamePlay_100.pt): agent 2 further trained with 10 000self play games maximizing the total points.
Agent 4 (GamePlay_200.pt): agent 2 further trained with 20 000 self play games maximizing the total points.
Agent 5 (Fine_Tunning_25.pt): agent 4 fine tunned with 2 500 self play games.

The obtained results are described below. Each measurement has a 95% confidence interval of, approximately, 1.7:

Agent 5 vs. Agent 4: 0.50 
Agent 5 vs. Agent 3: 2.63 
Agent 5 vs. Agent 2: 9.33 
Agent 5 vs. Agent 1: 11.11 
Agent 5 vs. Agent 0: 28.62
Agent 4 vs. Agent 3: 3.20
Agent 4 vs. Agent 2: 10.13
Agent 4 vs. Agent 1: 11.32
Agent 4 vs. Agent 0: 27.52
Agent 3 vs. Agent 2: 7.70
Agent 3 vs. Agent 1: 9.70
Agent 3 vs. Agent 0: 25.18
Agent 2 vs. Agent 1: 11.51
Agent 2 vs. Agent 0: 24.92
Agent 1 vs. Agent 0: 11.50

Seeing the results, we can see that the first two stages of the training produced an improvement of the training. However, there is no significant statistical evidence that the fine tunning produced an improvement in the performance.








## Author and acknowledgements

**Alberto Angurel Andres

The interface uses Spanish playing-card artwork from the [gjenkins20 SVG collection](https://github.com/gjenkins20/spanish-playing-cards-svg). The interface credits the artwork to Basquetteur / Wikimedia Commons under CC BY-SA 3.0. Retain the upstream attribution and license notices when redistributing the assets; the artwork's license is separate from any license chosen for the project code.
