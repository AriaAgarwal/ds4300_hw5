import pandas as pd

def load_data():
    spotify_df = pd.read_csv("spotify.csv")
    print(spotify_df)

def main():
    load_data()

main()