from __future__ import annotations
import numpy as np
import pandas as pd

def assign_mood(df):
    """
    assign a mood label to each song based on energy/valence
      - happy: high energy, high valence
      - intense: high energy, low valence
      - chill: low energy, high valence
      - melancholy: low energy, low valence

    args:
        df: DataFrame with energy and valence columns

    returns:
        DataFrame with mood column
    """
    df = df.copy()
    # assign mood labels based on energy/valence quadrants
    conditions = [
        (df["energy"] >= 0.5) & (df["valence"] >= 0.5),
        (df["energy"] >= 0.5) & (df["valence"] <  0.5),
        (df["energy"] <  0.5) & (df["valence"] >= 0.5),
        (df["energy"] <  0.5) & (df["valence"] <  0.5),
    ]
    labels = ["happy", "intense", "chill", "melancholy"]
    df["mood"] = np.select(conditions, labels, default="unknown")
    return df


def generate_edges(df, features, similarity_threshold):
    """
    compute pairwise similarity between all songs using numpy
    vectorization and return edges where similarity_score >= threshold.

    args:
        df: sampled songs DataFrame (features already normalized)

    returns:
        DataFrame where each row is one edge with full node info
        for both songs plus all edge properties
    """
    df = assign_mood(df).reset_index(drop=True)

    # vectorized pairwise audio distance 
    X = df[features].values.astype(float)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2) / np.sqrt(len(features))

    # vectorized same_genre and same_mood matrices
    genres = df["track_genre"].to_numpy()
    moods = df["mood"].to_numpy()
    genre_matrix = (genres[:, np.newaxis] == genres[np.newaxis, :])
    mood_matrix  = (moods[:, np.newaxis]  == moods[np.newaxis, :])

    # vectorized similarity score matrix
    score_matrix = (
        0.5 * (1 - dist_matrix)
        + 0.3 * genre_matrix.astype(float)
        + 0.2 * mood_matrix.astype(float)
    )

    # extract upper triangle only of score matrix
    i_idx, j_idx = np.triu_indices(len(df), k=1)
    scores = score_matrix[i_idx, j_idx]

    # apply threshold 
    mask   = scores >= similarity_threshold
    i_idx  = i_idx[mask]
    j_idx  = j_idx[mask]
    scores = scores[mask]

    # build edge dataframe
    a = df.iloc[i_idx].reset_index(drop=True)
    b = df.iloc[j_idx].reset_index(drop=True)

    edges_df = pd.DataFrame({
        # node A properties
        "track_id_a": a["track_id"],
        "track_name_a": a["track_name"],
        "artists_a": a["artists"],
        "album_name_a": a["album_name"],
        "track_genre_a": a["track_genre"],
        "popularity_a": a["popularity"],
        "mood_a": a["mood"],
        "sample_type_a": a["sample_type"],
        # node B properties
        "track_id_b": b["track_id"],
        "track_name_b": b["track_name"],
        "artists_b": b["artists"],
        "album_name_b": b["album_name"],
        "track_genre_b": b["track_genre"],
        "popularity_b": b["popularity"],
        "mood_b": b["mood"],
        "sample_type_b": b["sample_type"],
        # edge properties
        "audio_distance": np.round(dist_matrix[i_idx, j_idx], 6),
        "same_genre": genre_matrix[i_idx, j_idx],
        "same_mood": mood_matrix[i_idx, j_idx],
        "similarity_score": np.round(scores, 6),
    })

    return edges_df


def main():
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
    similarity_threshold = 0.7

    df = pd.read_csv("spotify_sampled.csv")
    edges_df = generate_edges(df, features, similarity_threshold)
    edges_df.to_csv("spotify_edges.csv", index=False)


if __name__ == "__main__":
    main()