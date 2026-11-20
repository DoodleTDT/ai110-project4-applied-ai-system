## 1. Model Name  
**MusicMatchMaker** *MMM for short*

---
## 2. AI Collaboration 
While working with the AI, I found planning for the app took more time than actually creating it. And that's because the AI needed a good base. I have tried projects in the past before where I didn't fully check the plan of a program before running it, and it always made cleanup take longer. It's harder to find the main problem when so many little ones are in the way. So that's what was prioritized during this project.

The pacing at which the AI worked was also something I had to adapt to. No matter the AI model, it tends to try to do everything in one go, giving very little breathing room for review until after the fact. It was important, once the plan for the project was finalized, that I then asked Claude to create a step by step implementation plan before execution. This meant that the memory of the AI would always remind itself to one thing at a time and check back with me when it completed.

---
## 3. Surprises
I keep underestimating just how much Claude fills in the gaps of a project. When I first wrote out the idea of the model, it proceeded to layout a list of ways to go about it, with features didn't realize I would need. For how much the AI tends to go overboard or fast paced, it's also very thourough.

---
## 4. Suggestions and Reviews
The major thing that I kept from the plan it first created was the mmd layout. Every relationship was the correct one, so it didn't need to be tweaked in that way. What I did have to review was a guardrail I felt wasn't implemented at all, and that was the credit limit and AI model used in the design. It reccommended a model that only pro subcriptions can have, which I found unneccesary. Claude is used for two things: interpreting the request and citing a reason for each songs addition to the playlist. This does not take a higher model to achieve, so the model was downgraded to something anyone could use without a subscription to the service. It also did not have a limit to the credits used per request, so I added that as a guardrail.

Another thing that I had to double check was how it was going to handle the original set of songs in the csv file. It had first planned to place all those songs into the library, without verification. This would not be good since every song in the file was made up. So I changed the plan so now the songs pass through Deezer verification and testing just like any new song, which allowed them to be rejected by the system and keep the library free of bad data.

---
## 5. Limitations and Bias  
1) Single source of song collection
All songs pulled for the playlist and library come from LastFM, which hosts a number of popular songs on the internet. This is a good source for a generic playlist request. But if someone were to try and find more obscure or smaller artists/songs to add to the playlist, it would be difficult. LastFM only holds so much, and I did not provide a second or third source of songs to check for.

2) Requires two outside accounts to function
With the way the model works, a person would need to create two accounts to use it themselves. They would also have to pay a minimum of five dollars for one of those accounts, since the model requires a Claude key.

3) User must approve every song in the pending list
This was a decision that keeps a human in the loop of the program, but may become tedious if a backlog of pending songs are waiting to be approved, regardless if the person knows it's real or not.

---


