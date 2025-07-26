# Try It Yourself: Thinking Like an LLM

This exercise is designed to give you a better intuition for what it feels like to be a "next-word predictor."

---

### Exercise 1: The Human Autocomplete Game

You can simulate the core task of an LLM with your own brain.

1.  **Read the following sentence beginnings.** For each one, think of the very first word that pops into your head to continue the sentence. Don't overthink it—just go with your first instinct.

    *   "The sky is..."
    *   "For breakfast, I had a cup of..."
    *   "The best movie I ever saw was..."
    *   "To be, or not to..."
    *   "The quick brown fox jumps over the lazy..."

2.  **Analyze your answers.**
    *   For "The sky is...", you probably thought "blue."
    *   For the last two, you almost certainly thought "be" and "dog."
    *   Why were these so easy? Because, like an LLM, your brain has been trained on a massive dataset of language throughout your life. You have learned these common patterns.

3.  **Now, try a harder one.**
    *   "The ancient manuscript, discovered in the forgotten library, was written in a strange, indecipherable..."

    What word did you pick? "Language"? "Script"? "Code"? "Handwriting"? There are several good options here, but they all fall within a narrow range of possibilities. An LLM does the same thing: it calculates a probability score for all possible next words and usually picks the one with the highest score.

---

### Exercise 2: Guiding the Prediction with Context

This exercise shows how changing the initial text (the "prompt") dramatically changes the next-word prediction.

1.  **Go to your favorite AI chatbot.**
2.  **Type the following incomplete sentence, but DO NOT press enter yet:**
    > "The patient was rushed to the hospital. The doctor ran into the room and shouted,"

3.  **Think about it:** What do you expect the next word to be? Probably something like "Clear!", "Stat!", or "Get me the..."

4.  **Now, complete the prompt and send it:**
    > "The patient was rushed to the hospital. The doctor ran into the room and shouted, **'This is a library!'**"

5.  **Ask the chatbot to complete the story.** The chatbot is now forced to reconcile two conflicting contexts: a medical emergency and a library. Its next-word predictions will be completely different than in the first scenario. It has to find a creative path forward that makes sense of the new, strange input.

6.  **Try another one.**
    *   **Prompt 1:** "The detective drew his weapon and said," -> Ask the AI to complete this.
    *   **Prompt 2:** "The comedian drew his water pistol and said," -> Ask the AI to complete this.

7.  **Reflect:** Notice how the entire tone and direction of the story changes based on these initial words. This is the essence of **prompt engineering**, which we will cover in Part 3. By carefully choosing the starting words, you are guiding the LLM's powerful next-word prediction engine toward the output you want.
