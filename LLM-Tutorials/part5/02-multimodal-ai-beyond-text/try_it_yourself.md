# Try It Yourself: Having a Multimodal Conversation

This exercise will let you experience the power of a multimodal model directly. You will give the AI an image and then ask it questions, forcing it to combine its visual understanding with its language capabilities.

---

### The Scenario

We will use a powerful, natively multimodal model like GPT-4o or Google's Gemini to act as a "culinary assistant." We'll give it a picture of food ingredients and ask it to create a recipe.

### Your Toolkit

You will need a chatbot that supports image uploads. The best options are:
*   **ChatGPT** (using the GPT-4o model, which is available to free users)
*   **Google Gemini** ([https://gemini.google.com/](https://gemini.google.com/))
*   **Claude.ai** ([https://claude.ai/](https://claude.ai/))

You will also need a picture. Find a picture on the internet of several common food ingredients sitting on a kitchen counter. For example, search for "ingredients on a counter" and find a picture that has things like tomatoes, onions, garlic, pasta, and herbs. Save it to your computer.

---

### Exercise: The AI Sous-Chef

1.  **Start a new chat** in ChatGPT, Gemini, or Claude.

2.  **Upload the image.** Look for a paperclip or image icon next to the text input box. Upload the picture of the ingredients you saved.

3.  **Ask your first question (Visual Identification).** Don't add any other text yet. Just upload the image and ask:
    > "What ingredients do you see in this picture?"

4.  **Analyze the response.** The AI should be able to identify most, if not all, of the ingredients correctly. It has successfully converted the image into a set of concepts.

5.  **Ask a follow-up question (Reasoning and Creativity).** Now, in the same chat, ask a follow-up question that requires the AI to use the information it just extracted from the image.
    > "Great. Using only the ingredients you see in the picture, what is a simple recipe I could make for dinner tonight?"

6.  **Analyze the response.** The AI should now generate a recipe that logically follows from the ingredients it identified. It's combining its visual understanding with its vast knowledge of cooking and recipes.

7.  **Ask an even more complex follow-up (Constraint-based Reasoning).** Let's add a constraint.
    > "That sounds good, but I'm trying to eat healthy. Can you modify that recipe to be lower in calories? And what wine would you pair with it?"

8.  **Analyze the final response.** The model now has to perform several tasks at once:
    *   It must **remember** the ingredients from the image.
    *   It must **recall** the recipe it just generated.
    *   It must **modify** that recipe based on the new "healthy" constraint.
    *   It must access a completely different domain of knowledge (wine pairing) and relate it to the recipe.

**Reflection:**
This exercise demonstrates a true multimodal conversation. The AI isn't just identifying objects; it's holding a dialogue where the core context is an image. This seamless blending of vision and language is the hallmark of modern frontier models and unlocks a huge range of new possibilities, from helping with homework by looking at a diagram to describing what's happening on a live video feed.
