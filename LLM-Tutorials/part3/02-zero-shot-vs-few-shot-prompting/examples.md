# Examples in Action: Zero-Shot vs. Few-Shot

The difference between asking and showing is one of the most practical concepts in prompt engineering. Let's see it in action.

### Example 1: Extracting Structured Data

*   **Goal:** Pull specific pieces of information from a block of text and structure it as JSON. This is a common task in data processing.

*   **Zero-Shot Prompt:**
    > Extract the name, company, and job title from the following text. Format the output as a JSON object.
    >
    > Text: "After a long career at Apple, where she was the Senior Vice President of Retail, Angela Ahrendts joined the board of Airbnb in 2019."

*   **Result (Good, but maybe inconsistent):** The model will likely succeed, producing:
    ```json
    {
      "name": "Angela Ahrendts",
      "company": "Airbnb",
      "job_title": "board member" 
    }
    ```
    The problem is consistency. If you give it another text, it might use `position` instead of `job_title`, or `organization` instead of `company`.

*   **Few-Shot Prompt (Better for consistency):**
    > Extract the name, company, and job title from the following texts. Format the output as a JSON object with the keys "personName", "companyName", and "title".
    >
    > **Text 1:** "Satya Nadella is the CEO of Microsoft."
    > **JSON 1:**
    > ```json
    > {
    >   "personName": "Satya Nadella",
    >   "companyName": "Microsoft",
    >   "title": "CEO"
    > }
    > ```
    > ---
    > **Text 2:** "After a long career at Apple, where she was the Senior Vice President of Retail, Angela Ahrendts joined the board of Airbnb in 2019."
    > **JSON 2:**

*   **Result (More reliable):** By providing an example, you've shown the model the *exact* key names you want (`personName`, `companyName`, `title`). The model will now follow this pattern reliably, which is critical for building a real application that expects a consistent data structure.

---

### Example 2: Matching a Specific Tone and Style

*   **Goal:** Generate a product description that matches a quirky, fun brand voice.

*   **Zero-Shot Prompt:**
    > Write a product description for a new coffee mug that has a picture of a cat on it. Make it fun and quirky.

*   **Result (Generic "fun"):** The model will write something friendly, but it might not match your specific style.
    > "Start your day with a smile! Our new Cat Mug is the purr-fect companion for your morning coffee. Made from high-quality ceramic, it's sure to be your new favorite!"

*   **Few-Shot Prompt (Teaching the specific style):**
    > Write a product description in our brand's quirky and slightly sarcastic voice.
    >
    > **Product 1:** A notebook with a blank cover.
    > **Description 1:** "It's a notebook. For your thoughts. Or your grocery lists. Or your secret plans for world domination. We're not judging. It's just paper, after all."
    > ---
    > **Product 2:** A new coffee mug that has a picture of a cat on it.
    > **Description 2:**

*   **Result (Specific, stylized):** The model now understands your brand's voice.
    > "This mug has a cat on it. It holds coffee, tea, or your tears of existential dread. The cat doesn't care. It's judging you either way. You should probably buy it."

*   **Analysis:** The few-shot prompt didn't just tell the model to be "fun and quirky"; it *showed* the model a perfect example of that style. The model learned the pattern of short sentences, sarcasm, and direct address, and then applied it to the new product.
