# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

**MusicMatchMaker** *MMM for short*

---

## 2. Intended Use  

This system takes the user's music preferences and pulls songs from the dataset that matches it the most. It assumes that the user prefers the genre of a song over anything else. This was made for classroom experimentations.  

---

## 3. How the Model Works    

A user can add their preferences to their profile. Up to 3 genres, their preferred mood and energy, and if they want acoustics or not. A list of songs are then scored by these preferecnes. Genre weighs the most (4.0 points), mood and energy weigh the same (2.0 points), and acoustics weigh the least (1.0 points). Once every song is scored out of 9.0 points, the recommender will give the top number of songs the user asks for. If your preferences change, your recommendations will change. If the user's genre is the base of other genres in the list, the recommender will recommend those songs higher. A favorite of "pop" will also give points to "indie pop", not vice versa.

---

## 4. Data  

The dataset holds 50 songs, including title, artist, genre, mood, energy, tempo, valence, and acoustics. 17 different genres show up in the dataset, pop songs (and their equivalents) being in there the most. Of the songs selected, energy and tempo match in scale, so a high energy song has a high tempo as well. 

---

## 5. Strengths  

The system works well with a genre preference for pop or lofi. The system works with multiple genres prefered. It recognizes sub genres as belonging to main genre.

---

## 6. Limitations and Bias  

- System struggles with genres outside of pop due to the number of pop songs in the dataset.

- If the number of songs asked for is bigger than the amount of songs in the genre, then genre will no longer matter to the score. It will only add points from energy, mood, and acoustics. While working as intended, since genre has the most weight in my system, it may make the recommendation score looked scewed. And because some songs only show up 3 or 4 times in the list, anything higher will recommend a different genre.

- User can choose up to 3 genres. These genres cannot be ranked, and therefore don't hold more weight than the other. So whatever comes first in the list is what will be sought out first. If the user does not ask for a large amount of songs, than the chances of all the genres showing up is small.

- While 'indie pop' is recognized as a type of 'pop', 'metal' is not recognized as a type of 'rock'. 

---

## 7. Evaluation  

A sad rock profile with low energy makes the scores of the songs lower than usual, meaning none of the rock songs are considered sad or low energy. This did not stop the rock songs from showing up in the list, since only 3 'rock' songs exist in the list. The main thing the evaluation proved was how little of each genre I put into the dataset. And of the genres inside, many match in energy, mood, and tempo. The lists came out as diverse as the data.

---

## 8. Future Work  

- Adding more songs to the dataset, at least 100. Make sure the genres are repeated at least 5 times, as well as more sub categories of the main genres. Make sure the categories of each song don't follow a pattern.

- Add a way for the user to choose what matters more to their preferences. Maybe a slider between each category to choose the weight, or a way to list their top categories in descending order of points.

- Adding tempo and valence to the scoring.

- Make sub category matching work without matching exact words. So rock can recognize metal or screamo as similar.

- Let the user choose how to weight their preferred genres so one can show up before the other.

- (Possibly) Changing the target energy to a range instead of a single point.

---

## 9. Personal Reflection  

The biggest thing I learned is how much data is needed for substantial results. I thought adding 40 songs to the default dataset would be more than enough for testing, and it only scratched the surface of the possible test cases. Having at least 200 songs in the database with more diversity would have given me different results. Considering what should be important to weigh in a song recommendation system is also difficult. Especially when those weights are set in stone. But for what I have to work with, it is expected. Other music recommenders match multiple users for comparison, weighs likes and dislikes, holds memory of multiple sessions, keeps in mind the songs you put in playlist, etc.
