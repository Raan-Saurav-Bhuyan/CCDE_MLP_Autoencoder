# Import local modules: --->
from train import train_model
# from train_strategy import train_model
from test import test_model

def main():
    # print("==== Starting Training ====")
    train_model()

    print("\n==== Starting Evaluation ====")
    test_model()

if __name__ == "__main__":
    main()
