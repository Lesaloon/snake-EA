# Snake AI – Genetic Algorithm + Simple Neural Network

This project trains a neural network to play the classic Snake game using a genetic algorithm (no gradient backpropagation). The agent learns by evolving weights and biases across generations, selecting and recombining the most successful networks.

## Overview
- **Goal**: Automatically learn a policy (choose movement direction) that grows the snake and avoids dying.
- **Approach**: A fully-connected neural network is evolved via a genetic algorithm (selection, crossover, mutation). Fitness is based on both apples eaten (snake length) and survival steps.
- **Game Board**: 10×10 grid during training (configurable via `gameParams`).
- **Architecture**: Default network layers: `8 → 24 → 4` (inputs, hidden, actions).
  - 8 features (see below)
  - 24 hidden neurons (sigmoid activation)
  - 4 outputs = action scores for: Up, Down, Left, Right (argmax picked)
- **Inference**: Once trained, the best network is saved to `model.txt` and can be reloaded.

## Input Features (8)
Extracted each timestep from `snake.Game.getFeatures()`:
1. Obstacle directly above (1/0)
2. Obstacle directly below (1/0)
3. Obstacle directly left (1/0)
4. Obstacle directly right (1/0)
5. Fruit vertical relation: 1 (fruit above), 0 (same row), -1 (fruit below)
6. Fruit horizontal relation: 1 (fruit right), 0 (same column), -1 (fruit left)
7. Current direction (0=Up,1=Down,2=Left,3=Right)
8. Distance to border continuing straight (scalar)

## Genetic Algorithm Parameters (in `main.py` / `genetic.optimize`)
- Population size: `400`
- Elitist selection size: `50` (top individuals carried forward)
- Crossover probability (`pc`): `0.8` (else mutation-only reproduction)
- Mutation rate (`mr`): `2.0` (% chance per weight/bias to perturb via Gaussian noise)
- Iterations (generations): `1000`

### Fitness Evaluation
For each individual:
- Plays `nbGames` (default 10) games.
- At each game end: score combines snake length (`game.score`) and survival steps (`game.steps`).
- Normalized by grid size and a scaling constant to keep values comparable.

### Variation Operators
- **Crossover**: For each weight and bias, randomly inherit from parent1 or parent2.
- **Mutation**: With probability `mr/100` per parameter, add Gaussian noise (`random.gauss(0,1)`).

## Files
| File | Purpose |
|------|---------|
| `main.py` | Entry point: trains or loads a model, then runs live visualization. |
| `genetic.py` | Genetic algorithm logic (population handling, selection, evolution). |
| `NN_numpy.py` | Lightweight neural network implementation (sigmoid activations, save/load). |
| `snake.py` | Game mechanics: grid, serpent movement, feature extraction. |
| `vue.py` | Pygame-based rendering of the Snake game and agent behavior. |
| `model.txt` | Serialized trained model (layer sizes, then biases and weights). |
| `data.csv` | Example progress log (likely iteration metrics: id, best/average score, % progress). |

## Running the Project
Prerequisites:
- Python 3.10+
- Packages: `numpy`, `pygame`

Install dependencies:
```bash
pip install numpy pygame
```

### Train a New Model
```bash
python main.py
```
This will:
- Run the genetic optimization (can take time: 400×1000 evaluations × 10 games each).
- Save the best network to `model.txt`.
- Launch the visualization window and continuously play with the trained agent.

### Load an Existing Model
If `model.txt` exists:
```bash
python main.py --load
```
Skips training and directly loads weights.

## Controls & Visualization
- A Pygame window displays apple (fruit), snake head, body, and tail sprites.
- Window title updates with current score (snake length).
- Close window (standard window controls) to exit; otherwise loop runs indefinitely generating new games.

## Model Format (`model.txt`)
Structure:
1. First line: space-separated layer sizes (e.g., `8 24 4`).
2. For each non-input layer:
   - One line of biases (space-separated floats).
   - Then one line per neuron containing its incoming weights.

You can swap in another trained model by replacing this file (ensure matching architecture).

## Performance Logging (`data.csv`)
Sample numeric progression (not automatically written by current code, but represents possible logged outputs):
- Columns appear to be: iteration id, best score, average score, progress percent.
- Useful for plotting evolution curves.

## Customization Tips
- Change grid size via `gameParams['height']` / `['width']` (keep consistent for training & inference).
- Adjust difficulty by lowering mutation rate (more refinement) or increasing population size (diversity, longer runtime).
- Add new features in `snake.Game.getFeatures()` and update `nbFeatures` constant + architecture.
- Replace sigmoid with another activation (e.g., tanh) by modifying `Layer.compute`.

## Limitations
- No pathfinding or lookahead; only reactive single-step perception.
- Genetic training is computationally expensive (many game simulations). Consider parallelization or reducing generations for quicker experiments.
- Fitness heuristic may favor survival over aggressive fruit collection; can be tuned.

## Possible Extensions
- Add stagnation penalty for long loops without apples.
- Introduce multi-objective scoring (length vs. efficiency).
- Log evolution automatically to CSV for analysis (currently implicit).
- Implement replay buffer or hybrid evolutionary + gradient refinement.

## License
No explicit license file included. Treat as educational material unless specified otherwise.

## Acknowledgments
Simple educational implementation combining classic Snake mechanics with evolutionary neural optimization for demonstration and experimentation.
