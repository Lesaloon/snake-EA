from snake import *
from vue import *
import genetic
import sys


# les paramètres d'évaluation : les grilles sont en 10 par 10, et on joue 10 parties
gameParams = {"nbGames": 10, "height": 10, "width": 10}

# check if we have a flag to load a pre-trained model

if len(sys.argv) > 1 and sys.argv[1] == "--load":
    nn = genetic.NeuralNet([8, 24, 4])
    nn.load(sys.argv[2])
# relaod from checkpoint if specified
elif len(sys.argv) > 2 and sys.argv[1] == "--load-checkpoint":
    nn = genetic.NeuralNet([8, 24, 4])
    nn.load(sys.argv[2])
    # get the iteration number from the filename
    iteration_number = int(sys.argv[2].split("_")[-1].split(".")[0])
    print(f"Loaded checkpoint from iteration {iteration_number}")
    # resume optimization from checkpoint
    nn = genetic.optimize(
        taillePopulation=400,
        tailleSelection=50,
        pc=0.8,
        mr=2.0,
        arch=[8, 24, 4],
        gameParams=gameParams,
        nbIterations=1000,
        nn=nn,
        skip_iterations=iteration_number,
    )
else:
    # on procède à l'optimisation
    nn = genetic.optimize(
        taillePopulation=400,
        tailleSelection=50,
        pc=0.8,
        mr=2.0,
        arch=[nbFeatures, 24, nbActions],
        gameParams=gameParams,
        nbIterations=1000,
    )

    # on enregistre le modèle obtenu, on pourra le recharger dans un autre code pour l'utiliser en inférence
    nn.save("model.txt")

# on initialise l'interface graphique -> on va pouvoir observer les performances de notre réseau de neurones en direct
vue = SnakeVue(gameParams["height"], gameParams["width"], 64)
fps = pygame.time.Clock()
gameSpeed = 20

# Tant que l'on ne fait pas à Ctrl-C
while True:
    # On créé une partie avec les mêmes dimensions que lors de l'apprentissage
    game = Game(gameParams["height"], gameParams["width"])
    # Tant que la partie n'est pas finie, on joue (enfin pas nous, le réseau de neurones)
    while game.enCours:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        pred = np.argmax(
            nn.compute(game.getFeatures())
        )  # on extrait les features de la partie, et on demande au réseau quelle direction choisir
        game.direction = int(pred)  # on joue la direction choisie par le réseau
        game.refresh()
        if not game.enCours:
            break
        vue.displayGame(game)
        fps.tick(gameSpeed)
