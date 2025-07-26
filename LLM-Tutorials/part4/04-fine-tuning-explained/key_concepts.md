# Key Concepts: Fine-Tuning

Here are the most important terms related to fine-tuning.

### 1. Pre-trained Model
-   **What it is:** A massive, general-purpose model (like GPT-4 or Llama 3) that has already been trained on a huge dataset from the internet.
-   **Analogy:** A doctor who has just graduated from medical school (a General Practitioner). They have a broad base of knowledge about all areas of medicine but are not yet a specialist in any single one.
-   **Why it matters:** Pre-trained models are the starting point for fine-tuning. They provide a powerful foundation of language understanding and reasoning that we can then adapt to our specific needs.

### 2. Fine-Tuning
-   **What it is:** The process of taking a pre-trained model and continuing to train it on a smaller, specific dataset to make it an expert at a particular task or style.
-   **Analogy:** A GP deciding to become a Cardiologist. They go through several more years of focused training (residency and fellowship) where they only study the heart. This makes them a specialist.
-   **Why it matters:** It's how you change the model's core behavior. While RAG gives the model new *knowledge*, fine-tuning teaches it a new *skill* or *personality*.

### 3. RAG vs. Fine-Tuning
-   **The Core Difference:** RAG is for knowledge, Fine-tuning is for skill.
-   **RAG (Retrieval-Augmented Generation):**
    -   **Goal:** To answer questions about information the model wasn't trained on.
    -   **Method:** Looks up relevant information from an external source (like an open book) and adds it to the prompt.
    -   **Does it change the model?** No. The model's internal weights remain the same.
-   **Fine-Tuning:**
    -   **Goal:** To change the model's behavior, style, or make it an expert in a specific domain.
    -   **Method:** Continues the training process with a new, specialized dataset, which permanently updates the model's internal weights.
    -   **Does it change the model?** Yes. It creates a new, specialized model.
