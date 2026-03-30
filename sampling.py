from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


def load_data(features):
    """
    load the data and normalize the features

    args:
        features: list of features to normalize

    returns:
        dataframe with normalized features
    """
    df = pd.read_csv("spotify.csv")
    # normalize the features
    df[features] = MinMaxScaler().fit_transform(df[features])
    return df


def cluster_sample(pool, n, n_clusters, random_state, features):
    """
    sample n songs from a pool using KMeans clustering per genre

    args:
        pool: dataframe with the songs to sample from
        n: number of songs to sample
        n_clusters: number of KMeans clusters per genre
        random_state: for reproducibility
        features: list of features to use for clustering

    returns:
        dataframe with the sampled songs from outside seed artists'genres
    """
    # return empty dataframe if n is less than or equal to 0 or pool is empty
    if n <= 0 or pool.empty:
        return pd.DataFrame()

    # initialize list to store sampled songs
    samples = []
    # calculate number of songs to sample per genre
    per_genre = max(1, n // pool["track_genre"].nunique())

    # group songs by genre and sample from each cluster
    for genre, genre_df in pool.groupby("track_genre"):
        # calculate number of clusters to use
        k = min(n_clusters, len(genre_df))
        # copy genre dataframe and add cluster column
        genre_df = genre_df.copy()
        genre_df["cluster"] = KMeans(k, random_state=random_state, n_init="auto").fit_predict(genre_df[features])

        # calculate number of songs to sample per cluster
        per_cluster = max(1, per_genre // k)
        # sample from each cluster
        for _, cluster_df in genre_df.groupby("cluster"):
            samples.append(cluster_df.sample(min(len(cluster_df), per_cluster), random_state=random_state))

    # concatenate sampled songs and drop duplicates
    result = pd.concat(samples).drop_duplicates("track_id")

    # if more songs than needed, sample n songs
    if len(result) > n:
        return result.sample(n, random_state=random_state)
    # if less songs than needed, sample from extra pool
    if len(result) < n:
        extra_pool = pool[~pool["track_id"].isin(result["track_id"])]
        extra = extra_pool.sample(min(n - len(result), len(extra_pool)), random_state=random_state)
        result = pd.concat([result, extra], ignore_index=True)
    return result


def sample_songs(df, seed_artists, total_size, random_frac, n_clusters, random_state, features):
    """
    sample songs from the dataframe

    args:
        df: dataframe with the songs to sample from
        seed_artists: list of seed artists
        total_size: total number of songs to sample
        random_frac: fraction of songs to sample randomly
        n_clusters: number of KMeans clusters per genre
        random_state: for reproducibility
        features: list of features to use for clustering

    returns:
        dataframe with the sampled songs
    """
    # filter for seed artists
    pattern = "|".join(seed_artists)
    seeds = df[df["artists"].str.contains(pattern, na=False, case=False)]
    # get seed genres
    seed_genres = set(seeds["track_genre"].dropna())
    # get pool of songs not in seed artists
    pool = df[~df.index.isin(seeds.index)]

    # sample random fraction of pool
    random_sample = pool.sample(min(int(total_size * random_frac), len(pool)), random_state=random_state)
    pool = pool[~pool.index.isin(random_sample.index)]

    # calculate number of songs remaining
    n_remaining = total_size - len(seeds) - len(random_sample)
    # sample from seed genres
    seed_genre_sample  = cluster_sample(pool[pool["track_genre"].isin(seed_genres)],  n_remaining // 2, n_clusters, random_state, features)
    # sample from other genres
    other_genre_sample = cluster_sample(pool[~pool["track_genre"].isin(seed_genres)], n_remaining - n_remaining // 2, n_clusters, random_state, features)

    # concatenate sampled songs and drop duplicates
    result = (
        pd.concat([seeds, random_sample, seed_genre_sample, other_genre_sample])
        .drop_duplicates("track_id")
        .reset_index(drop=True)
    )

    # add sample type column
    result["sample_type"] = "other_genre"
    result.loc[result["track_genre"].isin(seed_genres), "sample_type"] = "seed_genre"
    result.loc[result["artists"].str.contains(pattern, na=False, case=False), "sample_type"] = "seed"
    result.loc[result["track_id"].isin(random_sample["track_id"]), "sample_type"] = "random"

    return result


def main():

    features = [
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    ]
    seed_artists = ["The Strokes", "Regina Spektor"]
    total_size = 1000
    random_frac = 0.10
    n_clusters = 10
    random_state = 42

    df = load_data(features)
    sampled = sample_songs(df, seed_artists, total_size, random_frac, n_clusters, random_state, features)
    sampled.to_csv("spotify_sampled.csv", index=False)


if __name__ == "__main__":
    main()