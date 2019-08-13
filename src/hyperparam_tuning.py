from src.neural_network import nn_constants as nnc
from src.main import main
from src.neural_network import nn_model


def hyperparam_search():
    hyperparams = {
        "lr" : [0.1, 0.01],
        "optimizer": [adam, rmsprop]
    }

    for params in hyperparams.items():
        nnc.OPTIMIZER = params["optimzer"]
        nnc.LOSS = params["loss"]
        nnc.MODEL_SAVE_PATH = r"fdsafs"
        main()

    print("Complete search for hyperparams")


if __name__ == "__main__":
    hyperparam_search()

