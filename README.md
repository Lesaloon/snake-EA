# Snake AI – Genetic Algorithm + Neural Network

This project trains a Snake-playing agent using a simple neural network evolved with a genetic algorithm (no backprop, just evolution).

## Quick Demo

**Iteration 50 (early training)**
![Iteration 50](iter_50.gif)

**Iteration 350 (later training)**
![Iteration 350](iter_350.gif)

**Training Progress**
![Training chart](training_chart.png)

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
