# Snake AI – Genetic Algorithm + Neural Network

This project trains a Snake-playing agent using a simple neural network evolved with a genetic algorithm (no backprop, just evolution).

## Quick Demo

**Iteration 50 (early training)**
![Iteration 50](/docs/iter50.gif)

**Iteration 350 (after first breakthrough)**
![Iteration 350](/docs/iter350.gif)

**Iteration 480 (after second breakthrough)**
![Iteration 480](/docs/iter480.gif)

**Training Progress**

![Training chart](/docs/training_chart.png)

### explanation of the chart

We can see that initially the best score was stagnant / slightly increasing, meaning the snakes were not learning much as they were hitting walls quickly (see iteration 50 gif).
Around iteration 150 we see a first breakthrough where the best score jumps significantly (from 13 to 150), indicating that the snakes have learned to survive longer and navigate the grid more effectively (see iteration 350 gif).
Then we can identify a second breakthrough around iteration 380 where the best score increases again (from 150 to 261), showing further improvement in the snakes' ability to play the game (see iteration 480 gif).

We do not see any other significant breakthroughs after iteration 400, suggesting that the snakes have reached a plateau in their learning curve. This indicates that while they have improved their gameplay, further enhancements may require changes in the training approach or model architecture.

The snakes' performance is i believe limited by the simplicity of the neural network and the input features provided to it.
Currently the features are quite basic ( is there an obstacle in each direction, is the food on top/below/left/right, and the direction the snake is moving). More complex features or a larger network might help improve performance further.

Features that i believe could help:

- Distance to food in each direction
- The length of the snake
- More hidden layers / neurons in the neural network

We could also experiment with a "memory" mechanism, allowing the snake to remember past states or actions, which could help it make better decisions like avoiding its own tail.

This could also be achieved by having a CNN that for input has the entire grid state (for each cell: empty, food, snake body, snake head) instead of just a few features.
But this method would prevent the model to generalize to different grid sizes.
To keep generalization we could change the game to move the snake to the other side of the grid when it goes out of bounds instead of ending the game.
But a better solution would be to have the CNN with a fixed input size (e.g.: 10x10 grid around the snake head) and have the snake always in the center of the input grid.
This would alow generalization to different grid sizes while still providing the model with a full view of the snake's immediate surroundings.

## What It Does

- Plays the classic Snake game on a small grid.
- Uses a tiny neural network to decide the next move.
- Learns by evolving many snakes over generations (selection, crossover, mutation).
- Saves the best model to `model.txt` so you can replay it later.

## Main Files

- `main.py` – Run training and/or play with the best model.
- `genetic.py` – Genetic algorithm (population, selection, evolution).
- `NN_numpy.py` – Very small neural net implementation (NumPy only).
- `snake.py` – Game rules and state.
- `vue.py` – Pygame window to visualize the snake.

## How to Run

Requirements:

- Python 3.10+
- `numpy`, `pygame`

Install deps:

```bash
pip install numpy pygame
```

Train from scratch (and then watch the agent):

```bash
python main.py
```

If a trained model (`model.txt`) already exists and you just want to watch it play:

```bash
python main.py --load
```

## Notes

- Training can take a while (many games are simulated).
- You can tweak population size, number of generations, and mutation rate directly in the code to trade off speed vs. performance.

Feel free to fork, tweak, and experiment.
