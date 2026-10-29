"""
Command line runner for the Music Recommender Simulation.

Loads the song catalog, scores every song against a user profile, and
prints the top recommendations with the reasons behind each score.

Scoring (max 9.0): genre 4.0, mood 2.0, energy 2.0, acousticness 1.0 (optional).
"""

try:
    from src.recommender import load_songs, recommend_songs
except ModuleNotFoundError:
    from recommender import load_songs, recommend_songs

MAX_SCORE = 9.0


def main() -> None:
    songs = load_songs("data/songs.csv")

    # Example 1: Happy pop, high energy, doesn't care about acousticness
    user_prefs = {
        "favorite_genres": ["pop"],   # up to 3 genres
        "favorite_mood": "happy",
        "target_energy": 0.8,         # 0.0 to 1.0
        "likes_acoustic": False,      # True, False, or None to ignore acousticness
    }

    # # Example 2: Sad rock, low energy, likes acoustic
    # user_prefs = {
    #     "favorite_genres": ["rock"],
    #     "favorite_mood": "sad",
    #     "target_energy": 0.3,
    #     "likes_acoustic": True,
    # }

    # # Example 3: Mixed genres, medium energy, doesn't care about acousticness
    # user_prefs = {
    #     "favorite_genres": ["jazz", "classical", "blues"],
    #     "favorite_mood": "relaxed",
    #     "target_energy": 0.5,
    #     "likes_acoustic": None,
    # }

    # # Example 4: Lofi and Rock, low energy, likes acoustic
    # user_prefs = {
    #     "favorite_genres": ["lofi", "rock"],
    #     "favorite_mood": "relaxed",
    #     "target_energy": 0.4,
    #     "likes_acoustic": True,
    # }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print(
        f"\nUser profile: genres={', '.join(user_prefs['favorite_genres'])}, "
        f"mood={user_prefs['favorite_mood']}, energy={user_prefs['target_energy']:.2f}, "
        f"likes_acoustic={user_prefs['likes_acoustic']}"
    )
    print("\nTop recommendations:\n")
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"{rank}. {song['title']} by {song['artist']} - Score: {score:.2f}/{MAX_SCORE:.2f}")
        for reason in explanation.split("; "):
            print(f"     - {reason}")
        print()


if __name__ == "__main__":
    main()
