# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

In this small music recommendation system, it will prioritize User preferences against genres (and genres associated), mood, energy, and acoustics through a weighted scale, ranking the top scored songs in a currated list. This model will assume the user does not know the songs and artists in the database, so the user will not be reccomended songs based on those two categories like other music recommenders.

---

## How The System Works

Each `Song` has four things used in the system: genre, mood, energy, and acousticness. Title and artist is for display only.

A `UserProfile` stores: 
- a list of favorite genres of 1 to 3 songs
- a favorite mood
- a preferred amount of energy from 0 to 1
- a toggle for if user likes acoustics or not

The scoring rule for each song goes as follows:

1| Favorite genres are given the highest weight of 4.0 points. There are no extra rankings between a list of genres Genres that are an extension of a favorite are also weighed the same. 
- Example: Favorite genre of `rock` == `rock` and `indie rock`
- Example: Favorite genre of `indie rock` == `indie rock` (since it is a specific genre)

2| Mood has a weighted score of 2.0 points. If the mood matches the one in the favorites, it is awarded those points.

3| Energy is also weighted at 2.0 points. If the energy is closer to the target energy in `UserProfile`, it is given more points.
- Example: Target energy of 0.20 will award more points to songs with 0.12 energy compared to songs with 0.78 energy

4| Acousticness is the lowest weight of 1.0 point, and is an optional preference. If the user likes acoustics, then each song will be scored higher the more acoustics it has. Same goes for if the user does not want acoustics. Songs with less will be scored higher in that category. If acoustics are left out by the user, than it will not be considered during the final score.

Every song will go through these scoring rules, and the songs with the highest scores will be added to the recommendation list first.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

```bash
pytest
```
................................................................         [100%]
64 passed in 1.47s

---

## Sample Recommendation Output
Example 1
```
User profile: genres=pop, mood=happy, energy=0.80, likes_acoustic=False

Top recommendations:

1. Satellite Heart by The Dayglow Kids - Score: 8.80/9.00
     - genre match: "pop" (+4.0)
     - mood match: "happy" (+2.0)
     - energy 0.80 vs target 0.80 (+2.00)
     - acousticness 0.20, prefers non-acoustic (+0.80)

2. Sunrise City by Neon Echo - Score: 8.78/9.00
     - genre match: "pop" (+4.0)
     - mood match: "happy" (+2.0)
     - energy 0.82 vs target 0.80 (+1.96)
     - acousticness 0.18, prefers non-acoustic (+0.82)

3. Bright Weather by Mira Sol - Score: 8.64/9.00
     - genre match: "pop" (+4.0)
     - mood match: "happy" (+2.0)
     - energy 0.74 vs target 0.80 (+1.88)
     - acousticness 0.24, prefers non-acoustic (+0.76)

4. Rooftop Lights by Indigo Parade - Score: 8.57/9.00
     - genre match: "indie pop" contains "pop" (+4.0)
     - mood match: "happy" (+2.0)
     - energy 0.76 vs target 0.80 (+1.92)
     - acousticness 0.35, prefers non-acoustic (+0.65)

5. Polaroid Summer by Violet Transit - Score: 8.51/9.00
     - genre match: "indie pop" contains "pop" (+4.0)
     - mood match: "happy" (+2.0)
     - energy 0.71 vs target 0.80 (+1.82)
     - acousticness 0.31, prefers non-acoustic (+0.69)
```

Example 2
```
User profile: genres=rock, mood=sad, energy=0.30, likes_acoustic=True

Top recommendations:

1. Wild Horizon by Westbound Riot - Score: 4.99/9.00
     - genre match: "rock" (+4.0)
     - energy 0.86 vs target 0.30 (+0.88)
     - acousticness 0.11, likes acoustic (+0.11)

2. Signal Fire by Granite Avenue - Score: 4.96/9.00
     - genre match: "rock" (+4.0)
     - energy 0.88 vs target 0.30 (+0.84)
     - acousticness 0.12, likes acoustic (+0.12)

3. Storm Runner by Voltline - Score: 4.88/9.00
     - genre match: "rock" (+4.0)
     - energy 0.91 vs target 0.30 (+0.78)
     - acousticness 0.10, likes acoustic (+0.10)

4. Spacewalk Thoughts by Orbit Bloom - Score: 2.88/9.00
     - energy 0.28 vs target 0.30 (+1.96)
     - acousticness 0.92, likes acoustic (+0.92)

5. Winter Nocturne by Rowan Ensemble - Score: 2.81/9.00
     - energy 0.25 vs target 0.30 (+1.90)
     - acousticness 0.91, likes acoustic (+0.91)
```

Example 3
```
User profile: genres=jazz, classical, blues, mood=relaxed, energy=0.50, likes_acoustic=None

Top recommendations:

1. Blue Hour Cafe by Ellis Row - Score: 7.88/9.00
     - genre match: "jazz" (+4.0)
     - mood match: "relaxed" (+2.0)
     - energy 0.44 vs target 0.50 (+1.88)

2. Blue Window by Miles Arden - Score: 7.86/9.00
     - genre match: "jazz" (+4.0)
     - mood match: "relaxed" (+2.0)
     - energy 0.43 vs target 0.50 (+1.86)

3. Coffee Shop Stories by Slow Stereo - Score: 7.74/9.00
     - genre match: "jazz" (+4.0)
     - mood match: "relaxed" (+2.0)
     - energy 0.37 vs target 0.50 (+1.74)

4. Winter Nocturne by Rowan Ensemble - Score: 5.50/9.00
     - genre match: "classical" (+4.0)
     - energy 0.25 vs target 0.50 (+1.50)

5. Glass Cathedral by Aster Quartet - Score: 5.44/9.00
     - genre match: "classical" (+4.0)
     - energy 0.22 vs target 0.50 (+1.44)
```

Example 4
```
User profile: genres=lofi, rock, mood=relaxed, energy=0.40, likes_acoustic=True

Top recommendations:

1. Focus Flow by LoRoom - Score: 6.78/9.00
     - genre match: "lofi" (+4.0)
     - energy 0.40 vs target 0.40 (+2.00)
     - acousticness 0.78, likes acoustic (+0.78)

2. Library Rain by Paper Lanterns - Score: 6.76/9.00
     - genre match: "lofi" (+4.0)
     - energy 0.35 vs target 0.40 (+1.90)
     - acousticness 0.86, likes acoustic (+0.86)

3. Window Seat by Tape Garden - Score: 6.75/9.00
     - genre match: "lofi" (+4.0)
     - energy 0.38 vs target 0.40 (+1.96)
     - acousticness 0.79, likes acoustic (+0.79)

4. Soft Transit by Lowline - Score: 6.68/9.00
     - genre match: "lofi" (+4.0)
     - energy 0.33 vs target 0.40 (+1.86)
     - acousticness 0.82, likes acoustic (+0.82)

5. Midnight Coding by LoRoom - Score: 6.67/9.00
     - genre match: "lofi" (+4.0)
     - energy 0.42 vs target 0.40 (+1.96)
     - acousticness 0.71, likes acoustic (+0.71)
```

**Screenshot or video** *(optional)*: <!-- Insert a screenshot or demo video link here -->

---

## Experiments You Tried

- I flipped the weights of genre and energy, so genre was worth 2.0, and energy 4.0. This had the biggest effect on Example 2 in `Sample Recommendation Output`, where no rock songs were present in the recommendations. An unfavorable change if the User wishes for a genre more than energy. A sample of this change is provided below.
Experiment on Example 2
```
User profile: genres=rock, mood=sad, energy=0.30, likes_acoustic=True

Top recommendations:

1. Spacewalk Thoughts by Orbit Bloom - Score: 4.84/9.00
     - energy 0.28 vs target 0.30 (+3.92)
     - acousticness 0.92, likes acoustic (+0.92)

2. Winter Nocturne by Rowan Ensemble - Score: 4.71/9.00
     - energy 0.25 vs target 0.30 (+3.80)
     - acousticness 0.91, likes acoustic (+0.91)

3. Soft Transit by Lowline - Score: 4.70/9.00
     - energy 0.33 vs target 0.30 (+3.88)
     - acousticness 0.82, likes acoustic (+0.82)

4. Library Rain by Paper Lanterns - Score: 4.66/9.00
     - energy 0.35 vs target 0.30 (+3.80)
     - acousticness 0.86, likes acoustic (+0.86)

5. Meadow Lines by The Alder Sisters - Score: 4.63/9.00
     - energy 0.36 vs target 0.30 (+3.76)
     - acousticness 0.87, likes acoustic (+0.87)

```

---

## Limitations and Risks

- If the number of songs asked for is bigger than the amount of songs in the genre, than genre will no longer matter to the score. It will only add points from energy, mood, and acoustics. While working as intended, since genre has the most weight in my system, it may make the recommendation score looked scewed. Reference Example 2 in `Sample Recommendation Output`

- User can choose up to 3 genres. These genres cannot be ranked, and therefore don't hold more weight than the other. So whatever comes first in the list is what will be sought out first.

- Multiple genre matches, like in Example 3 in `Sample Recommendation Output`, won't show all genres choosen if the number of songs asked for does not accommodate all of them. Since the user in Example 3 only asked for 5 songs, blues was not considered in the roster. Good for matching the best songs based on all categories, but bad if the User wanted songs from all genres.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this



