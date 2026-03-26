from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


def load_data(filepath: str) -> pd.DataFrame:
    """load the Spotify dataset"""
    return pd.read_csv(filepath)


def normalize_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """min-max normalization on features"""
    df = df.copy()
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df


def cluster_sample(
    pool: pd.DataFrame,
    n_total: int,
    *,
    n_clusters: int,
    random_state: int,
    features: list[str],
) -> pd.DataFrame:
    """
    sample songs from each genre using KMeans cluster sampling

    args:
        pool: DataFrame with the songs to sample from
        n_total: total number of songs to sample
        n_clusters: number of KMeans clusters per genre
        random_state: for reproducibility
        features: list of features to use for clustering

    returns:
        DataFrame with the sampled songs from outside seed artists'genres
    """
    if n_total <= 0 or len(pool) == 0:
        return pd.DataFrame()

    genres = pool["track_genre"].unique()
    per_genre = max(1, n_total // max(1, len(genres)))

    samples: list[pd.DataFrame] = []
    for _, genre_df in pool.groupby("track_genre"):
        if len(genre_df) == 0:
            continue

        if len(genre_df) < n_clusters:
            samples.append(genre_df)
            continue

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
        )

        genre_df = genre_df.copy()
        genre_df["cluster"] = kmeans.fit_predict(genre_df[features])

        # approximate allocation per cluster.
        per_cluster = max(1, per_genre // n_clusters)
        cluster_sample_df = (
            genre_df.groupby("cluster", group_keys=False)
            .apply(
                lambda x: x.sample(
                    min(len(x), per_cluster),
                    random_state=random_state,
                )
            )
            .reset_index(drop=True)
        )
        samples.append(cluster_sample_df)

    result = pd.concat(samples).drop_duplicates("track_id")
    if len(result) > n_total:
        result = result.sample(n=n_total, random_state=random_state)
    elif len(result) < n_total:
        already = set(result["track_id"])
        gap_pool = pool[~pool["track_id"].isin(already)]
        gap = min(n_total - len(result), len(gap_pool))
        if gap > 0:
            result = pd.concat(
                [result, gap_pool.sample(n=gap, random_state=random_state)],
                ignore_index=True,
            )

    return result


def sample_songs(
    df: pd.DataFrame,
    seed_artists: list[str],
    total_size: int = 2000,
    random_frac: float = 0.10,
    n_clusters: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    builds a sample of songs for the recommendation graph
      - 50% of songs from seed artists' genres
      - 50% of songs equally across all other genres
      - 10% purely random

    args:
        filepath: path to the Spotify dataset
        seed_artists: list of artist names to guarantee in the sample
        total_size: total number of songs in the final sample
        random_frac: fraction of total_size to allocate to random sampling
        n_clusters: number of KMeans clusters per genre
        random_state: for reproducibility

    returns:
        final_sample: DataFrame with the sampled songs
    """
    # normalize the features used for clustering
    df = normalize_features(df, features)

    # ensure seed songs are included
    pattern = "|".join(seed_artists)
    seeds = df[df["artists"].str.contains(pattern, na=False, case=False)]
    print(f"Seed songs found: {len(seeds)} across {len(seed_artists)} artist(s)")

    if len(seeds) == 0:
        raise ValueError(f"None of the seed artists were found: {seed_artists}")

    seed_genres = set(seeds["track_genre"].dropna().unique())
    print(f"Seed artist genres: {seed_genres}")

    remaining = df[~df.index.isin(seeds.index)].copy()

    # randomly sample 10% of the remaining songs
    n_random = int(total_size * random_frac)
    n_random = min(n_random, len(remaining))
    random_sample = (
        remaining.sample(n=n_random, random_state=random_state)
        if n_random > 0
        else pd.DataFrame(columns=df.columns)
    )
    remaining = remaining[~remaining.index.isin(random_sample.index)]

    # split the remaining songs 50/50
    if len(seeds) > total_size:
        # If the seed songs already exceed the total budget, trim them.
        seeds = seeds.sample(n=total_size, random_state=random_state)
        random_sample = pd.DataFrame(columns=df.columns)
        seed_genre_sample = pd.DataFrame()
        other_genre_sample = pd.DataFrame()
        final_sample = seeds.copy().reset_index(drop=True)
        final_sample["sample_type"] = "seed"
        print(f"\nFinal sample size: {len(final_sample)}")
        print(final_sample["sample_type"].value_counts().to_string())
        return final_sample

    n_hybrid = total_size - len(seeds) - len(random_sample)
    n_seed_genres = n_hybrid // 2
    n_other_genres = n_hybrid - n_seed_genres

    seed_genre_pool = remaining[remaining["track_genre"].isin(seed_genres)]
    other_genre_pool = remaining[~remaining["track_genre"].isin(seed_genres)]

    print(f"Seed genre pool:  {len(seed_genre_pool)} available → sampling {n_seed_genres}")
    print(f"Other genre pool: {len(other_genre_pool)} available → sampling {n_other_genres}")

    seed_genre_sample = cluster_sample(
        seed_genre_pool,
        n_seed_genres,
        n_clusters=n_clusters,
        random_state=random_state,
        features=features,
    )
    other_genre_sample = cluster_sample(
        other_genre_pool,
        n_other_genres,
        n_clusters=n_clusters,
        random_state=random_state,
        features=features,
    )

    # combine the samples
    final_sample = (
        pd.concat([seeds, random_sample, seed_genre_sample, other_genre_sample])
        .drop_duplicates("track_id")
        .reset_index(drop=True)
    )

    final_sample["sample_type"] = "other_genre"
    final_sample.loc[
        final_sample["track_genre"].isin(seed_genres),
        "sample_type",
    ] = "seed_genre"
    final_sample.loc[
        final_sample["artists"].str.contains(pattern, na=False, case=False),
        "sample_type",
    ] = "seed"
    final_sample.loc[
        final_sample["track_id"].isin(
            random_sample["track_id"] if len(random_sample) else []
        ),
        "sample_type",
    ] = "random"

    print(f"\nFinal sample size: {len(final_sample)}")
    print(final_sample["sample_type"].value_counts().to_string())

    return final_sample


def main():
    filepath = "spotify.csv"
    seed_artists = ["The Strokes", "Regina Spektor"]

    df = load_data(filepath)

    sampled = sample_songs(
        df,
        seed_artists=seed_artists,
        total_size=2000,
        random_frac=0.10,
        n_clusters=10,
        random_state=42,
    )

    sampled.to_csv("spotify_sampled.csv", index=False)
    return sampled


if __name__ == "__main__":
    main()