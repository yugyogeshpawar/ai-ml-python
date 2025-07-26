# Examples in Action: Fine-Tuning

Fine-tuning is about creating a "specialist" model from a "generalist" one. Here are some concrete examples of why and how you would do this.

### 1. The Goal: Creating a Medical Scribe

*   **The Problem:** Doctors spend a huge amount of time writing clinical notes after seeing a patient. This is time they could be spending with other patients.
*   **The Desired Behavior:** An AI that can listen to a recorded conversation between a doctor and a patient and automatically generate a perfectly formatted, accurate clinical note in the standard "SOAP" (Subjective, Objective, Assessment, Plan) format.
*   **Why RAG isn't enough:** You can't solve this with RAG. The problem isn't a lack of knowledge; it's a lack of a specific *skill* and *formatting ability*. You need to fundamentally change the model's behavior.
*   **The Fine-Tuning Process:**
    1.  **Create a Dataset:** You would need to gather thousands of examples of real (anonymized) doctor-patient conversations and the corresponding, perfectly written SOAP notes for each one.
        *   **Prompt:** [Transcript of the doctor-patient conversation]
        *   **Completion:** [The ideal, structured SOAP note]
    2.  **Fine-Tune the Model:** You take a powerful base model (like Llama 3 or GPT-4) and fine-tune it on this dataset. The model's weights are adjusted as it learns the complex patterns of how to transform a messy, conversational transcript into a structured, clinical document.
*   **The Result:** A new, specialized model (`medical-scribe-v1`) that is an expert at this specific task. It will outperform any general-purpose model at generating clinical notes because its very "brain" has been reshaped to be good at this skill.

---

### 2. The Goal: A Sarcastic, Unhelpful Chatbot

*   **The Problem:** You want to create a chatbot for an art installation or a game that has a very specific, sarcastic, and unhelpful personality.
*   **The Desired Behavior:** No matter what the user asks, the chatbot should respond with witty, sarcastic, and evasive answers.
*   **Why Prompting isn't enough:** You could write a very detailed system prompt like "You are a sarcastic chatbot..." but the model's underlying safety training will often "bleed through," and it will eventually revert to being helpful. To make the personality truly consistent, you need to fine-tune.
*   **The Fine-Tuning Process:**
    1.  **Create a Dataset:** You would write hundreds of examples of user questions and the ideal sarcastic responses.
        *   **Prompt:** "What is the capital of France?"
        *   **Completion:** "Oh, I don't know, maybe the place with the big metal tower? You should try a search engine, they're all the rage these days."
        *   **Prompt:** "Can you help me with my homework?"
        *   **Completion:** "I suppose I *could*, but what would that teach you? Besides, I'm busy contemplating the futility of existence. You're on your own."
    2.  **Fine-Tune the Model:** You fine-tune a base model on this dataset. The model learns that the "correct" pattern of responding to any user query is to generate this specific style of text.
*   **The Result:** A new model (`sarcastic-bot-v1`) whose core behavior has been altered. Its default, helpful nature has been overwritten by the new, sarcastic personality defined in your training data.

---

### 3. The Goal: A Code Translator (Cobol to Python)

*   **The Problem:** A large bank has millions of lines of code written in COBOL, an old mainframe language. They want to migrate this code to a modern language like Python, but doing it by hand is slow and expensive.
*   **The Desired Behavior:** An AI that is an expert at translating COBOL code to clean, efficient, and "Pythonic" Python code.
*   **Why RAG and Prompting aren't enough:** While a general model knows both COBOL and Python, it may not be an expert in the subtle idioms of both. It might produce literal, line-by-line translations that are inefficient or don't follow modern Python best practices.
*   **The Fine-Tuning Process:**
    1.  **Create a Dataset:** You would create a dataset of thousands of pairs of COBOL code snippets and the equivalent, perfectly written Python code, likely written by expert human programmers.
        *   **Prompt:** `[COBOL code snippet]`
        *   **Completion:** `[Ideal Python translation]`
    2.  **Fine-Tune the Model:** You fine-tune a code-specialized base model (like Codestral) on this dataset. The model learns the specific patterns for converting COBOL data structures and commands into their Pythonic equivalents.
*   **The Result:** A specialist model (`cobol-to-python-translator-v1`) that can automate a huge portion of the migration process, producing much higher-quality code than a general-purpose model could.
