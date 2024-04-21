from pathlib import Path
import pickle
import pandas as pd

from archive.language_archive import LanguageModel

def genz() -> pd.DataFrame:
    filepath = Path("./data/input/genz/genz.csv")
    df = pd.read_csv(filepath, encoding="utf-8")
    return df["phrase"].values


if __name__ == "__main__":
    from pprint import pprint

    zoomer = LanguageModel(order=4)

    try:
        with open("./models/Language_Model-Character-GenZ.pkl", mode="rb") as f:
            model = pickle.load(f)
        print("Loaded model")
        zoomer.load_model(model)
    except FileNotFoundError as err:
        print(err)

        print("Training...")
        zoomer.batch_train(genz(), initial_history="")
        with open("./models/Language_Model-Character-GenZ.pkl", mode="wb") as f:
            pickle.dump(zoomer.model, f)

        print("Saved model")

    print()

    print("Generating...\n")
    for s in zoomer.generate(initial_text="You", k=1, length=100):
        print(s)