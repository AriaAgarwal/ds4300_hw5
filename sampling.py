from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


def load_data(features):
    df = pd.read_csv("spotify.csv")
    df[features] = MinMaxScaler().fit_transform(df[features])
    return df


def cluster_sample(pool, n, n_clusters, random_state, features):
    """Sample n songs from a pool using KMeans clustering per genre."""
    if n <= 0 or pool.empty:
        return pd.DataFrame()

    samples = []
    per_genre = max(1, n // pool["track_genre"].nunique())

    for genre, genre_df in pool.groupby("track_genre"):
        k = min(n_clusters, len(genre_df))
        genre_df = genre_df.copy()
        genre_df["cluster"] = KMeans(k, random_state=random_state, n_init="auto").fit_predict(genre_df[features])

        per_cluster = max(1, per_genre // k)
        for _, cluster_df in genre_df.groupby("cluster"):
            samples.append(cluster_df.sample(min(len(cluster_df), per_cluster), random_state=random_state))

    result = pd.concat(samples).drop_duplicates("track_id")

    if len(result) > n:
        return result.sample(n, random_state=random_state)
    if len(result) < n:
        extra_pool = pool[~pool["track_id"].isin(result["track_id"])]
        extra = extra_pool.sample(min(n - len(result), len(extra_pool)), random_state=random_state)
        result = pd.concat([result, extra], ignore_index=True)

    return result


def sample_songs(df, seed_artists, total_size, random_frac, n_clusters, random_state, features):
    pattern = "|".join(seed_artists)
    seeds = df[df["artists"].str.contains(pattern, na=False, case=False)]
    seed_genres = set(seeds["track_genre"].dropna())
    pool = df[~df.index.isin(seeds.index)]

    random_sample = pool.sample(min(int(total_size * random_frac), len(pool)), random_state=random_state)
    pool = pool[~pool.index.isin(random_sample.index)]

    n_remaining = total_size - len(seeds) - len(random_sample)
    seed_genre_sample  = cluster_sample(pool[pool["track_genre"].isin(seed_genres)],  n_remaining // 2, n_clusters, random_state, features)
    other_genre_sample = cluster_sample(pool[~pool["track_genre"].isin(seed_genres)], n_remaining - n_remaining // 2, n_clusters, random_state, features)

    result = (
        pd.concat([seeds, random_sample, seed_genre_sample, other_genre_sample])
        .drop_duplicates("track_id")
        .reset_index(drop=True)
    )

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