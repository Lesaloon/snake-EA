import numpy
import random
from NN_numpy import *
import concurrent.futures
from snake import *
import datetime


class Individu:
    def __init__(self, nn, id):
        self.nn = nn
        self.id = id
        self.score = 0.0

    def clone(self, cible):
        for idx, layer in enumerate(cible.nn.layers[1:]):
            layer.bias = self.nn.layers[idx + 1].bias.copy()
            layer.weights = self.nn.layers[idx + 1].weights.copy()


def _select_weighted(pool):
    total = sum(ind.score for ind in pool)
    if total <= 0:
        return random.choice(pool)
    r = random.random() * total
    s = 0.0
    for ind in pool:
        s += ind.score
        if s >= r:
            return ind
    return pool[-1]


def eval(sol: Individu, gameParams):
    score = 0.0
    for _ in range(gameParams["nbGames"]):
        game = Game(gameParams["height"], gameParams["width"])
        while game.enCours:
            features = game.getFeatures()
            pred = numpy.argmax(sol.nn.compute(features))
            game.direction = int(pred)
            game.refresh()
        score += 1000 * (game.score - 4) + game.steps
    sol.score = score / (
        gameParams["nbGames"] * gameParams["width"] * gameParams["height"]
    )
    return sol.score


def optimize(
    taillePopulation,
    tailleSelection,
    pc,
    mr,
    arch,
    gameParams,
    nbIterations,
    nn=None,
    skip_iterations=0,
):
    if nn is not None:
        print(
            "Démarrage de l'optimisation génétique à partir d'un modèle pré-entraîné..."
        )
        print(
            f"Paramètres : population={taillePopulation}, sélection={tailleSelection}, pc={pc}, mr={mr}, architecture={arch}, nbIterations={nbIterations}"
        )
        population = [Individu(nn, 0)]
        for i in range(1, taillePopulation):
            population.append(Individu(NeuralNet(arch), i))
    else:
        print("Démarrage de l'optimisation génétique...")
        print(
            f"Paramètres : population={taillePopulation}, sélection={tailleSelection}, pc={pc}, mr={mr}, architecture={arch}, nbIterations={nbIterations}"
        )
        population = [Individu(NeuralNet(arch), i) for i in range(taillePopulation)]

    for iteration in range(nbIterations):
        iteration += skip_iterations
        datetime_start = datetime.datetime.now()
        # Evaluation
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(eval, individu, gameParams) for individu in population
            ]
            concurrent.futures.wait(futures)

        population.sort(key=lambda x: x.score, reverse=True)
        best = population[0].score
        avg = sum(ind.score for ind in population) / len(population)
        log(iteration, best, avg, datetime_start, nbIterations)
        # Elitism: retain the top 'tailleSelection' individuals directly
        nextPopulation = population[:tailleSelection]

        # Reproduction
        # temp que la prochaine population n'est pas aussi grande que la population initiale
        while len(nextPopulation) < taillePopulation:
            # on effectue un crossover avec probabilité pc, sinon une mutation (Pc permet de moduler le taux de mutation)
            if random.random() < pc:
                # on choisit deux parents pondérés par leur score ( les meilleurs ont plus de chances d'être choisis )
                parent1 = _select_weighted(nextPopulation)
                parent2 = _select_weighted(nextPopulation)
                # on crée un enfant en croisant les poids et biais des deux parents
                child = Individu(NeuralNet(arch), len(nextPopulation))
                # pour chaque couche du réseau exceptée la couche d'entrée
                for idx, layer in enumerate(child.nn.layers[1:]):
                    for j in range(layer.size):
                        # bias crossover
                        if random.random() < 0.5:
                            layer.bias[j] = parent1.nn.layers[idx + 1].bias[j]
                        else:
                            layer.bias[j] = parent2.nn.layers[idx + 1].bias[j]
                        # weights crossover
                        for k in range(layer.weights.shape[1]):
                            if random.random() < 0.5:
                                layer.weights[j][k] = parent1.nn.layers[
                                    idx + 1
                                ].weights[j][k]
                            else:
                                layer.weights[j][k] = parent2.nn.layers[
                                    idx + 1
                                ].weights[j][k]
                nextPopulation.append(child)
            else:
                # on choisit un  parent pondéré par son score
                parent = _select_weighted(nextPopulation)
                # l'enfant est une copie du parent avec des mutations aléatoires
                child = Individu(NeuralNet(arch), len(nextPopulation))
                parent.clone(child)
                for idx, layer in enumerate(child.nn.layers[1:]):
                    for j in range(layer.size):
                        # mutation du bias
                        if random.random() < mr / 100.0:
                            layer.bias[j] += random.gauss(0, 1)
                        # mutation des poids
                        for k in range(layer.weights.shape[1]):
                            if random.random() < mr / 100.0:
                                layer.weights[j][k] += random.gauss(0, 1)
                nextPopulation.append(child)
        # on remplace la population par la nouvelle génération
        population = nextPopulation

        if (iteration + 1) % 10 == 0:
            # Sauvegarde du meilleur individu tous les 10 itérations
            population[0].nn.save(f"checkpoints/model_iter_{iteration+1}.txt")

    # Final evaluation of last generation to ensure returned best is up-to-date
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(eval, individu, gameParams) for individu in population
        ]
        concurrent.futures.wait(futures)
    population.sort(key=lambda x: x.score, reverse=True)
    print(
        f"Optimisation terminée. Score final meilleur individu = {population[0].score:.2f}"
    )
    return population[0].nn


def log(iteration, best, avg, time, nbIterations):
    datetime_now = datetime.datetime.now()
    time_delta = datetime_now - time
    print(
        f"Iteration {iteration+1:04d}/{nbIterations} | Best={best:.2f} | Avg={avg:.2f} | Progress={(iteration+1)/nbIterations*100:.2f}% | Time={time_delta.total_seconds():.2f}s"
    )
    # send to file
    with open("genetic_log.csv", "a") as file:
        file.write(
            f"{datetime_now}, {iteration+1}, {nbIterations}, {best:.2f}, {avg:.2f}\n"
        )
        file.close()
