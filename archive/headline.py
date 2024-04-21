import pandas as pd

from pathlib import Path
from zasta.language import Markov
import markovify

if __name__ == "__main__":

    try:
        # Parquet loads faster than CSV, so try if it exists
        file_path = Path("./data/clickbait_headlines.parquet")
        df = pd.read_parquet(file_path, engine="pyarrow")
    except FileNotFoundError as err:
        print(err)
        file_path = Path("./data/clickbait_headlines.csv")
        df = pd.read_csv(file_path)
        df.to_parquet(file_path.parent/f"{file_path.stem}.parquet", engine="pyarrow")

    

    headlines = df["headline_text"].dropna().values
    headlines = "\n".join(headlines)
    
    # mark = Markov(headlines)
    num_sentences = 5
    # for _ in range(num_sentences):
    #     print(mark.new_sentence())

    print("-"*20)

    print("Markovify")
    text_model = markovify.NewlineText(headlines)
    for i in range(num_sentences):
        print(text_model.make_sentence())
