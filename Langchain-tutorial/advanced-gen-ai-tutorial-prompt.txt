# Prompt for Generating an Advanced, Comprehensive Generative AI Tutorial

## 1. Overall Objective
Generate a complete, multi-part tutorial on Generative AI, suitable for hosting on GitHub. The tutorial should guide a user from foundational concepts to advanced, state-of-the-art topics, providing both theoretical knowledge and practical implementation skills.

## 2. Target Audience
The tutorial is for developers, researchers, and students who want a deep and comprehensive understanding of Generative AI. It assumes a basic knowledge of Python and a willingness to learn complex concepts.

## 3. Core Philosophy & Style
- **Conceptual Depth:** Go beyond surface-level explanations to provide a deep understanding of the underlying mechanisms.
- **Practical and Modern:** Use up-to-date libraries and techniques.
- **Structured and Progressive:** The tutorial should build on itself, with each part providing the foundation for the next.

## 4. High-Level Structure
The tutorial will be divided into four main parts.

- **Part 1: Foundations of Generative AI**
- **Part 2: Core Generative Models**
- **Part 3: Building Generative AI Applications**
- **Part 4: Advanced Topics and Frontiers**

## 5. Detailed Content Structure (For Each Topic)
For every topic, generate the following files inside a dedicated topic directory:
1.  **`README.md`**: The main lesson file with detailed explanations.
2.  **`[topic]_example.py`**: A runnable Python script (where applicable).
3.  **`key_concepts.md`**: A file summarizing the key takeaways and vocabulary.
4.  **`interview_questions.md`**: 3 relevant interview questions and detailed answers.

---

### **Part 1: Foundations of Generative AI**
- **Goal:** Introduce the basic concepts and building blocks.
- **Topics:**
    1.  `01-what-is-generative-ai`: Generative vs. Discriminative AI.
    2.  `02-how-llms-work`: Tokens, transformers, next-token prediction.
    3.  `03-prompt-engineering`: Basic and advanced prompting techniques.
    4.  `04-introduction-to-embeddings`: Representing data as vectors.

### **Part 2: Core Generative Models**
- **Goal:** Explore the different types of generative models.
- **Topics:**
    1.  `01-text-generation-models`: GPT, LLaMA, etc.
    2.  `02-image-generation-models`: Stable Diffusion, Midjourney, DALL-E.
    3.  `03-code-generation-models`: Codex, Code Llama.
    4.  `04-multimodal-models`: Models that handle multiple data types.

### **Part 3: Building Generative AI Applications**
- **Goal:** Cover the practical aspects of building applications.
- **Topics:**
    1.  `01-using-apis-vs-local-models`: Trade-offs and best practices.
    2.  `02-fine-tuning-models`: Full fine-tuning vs. parameter-efficient methods.
    3.  `03-retrieval-augmented-generation-rag`: The complete RAG pipeline.
    4.  `04-ethics-and-safety`: Bias, misinformation, and safety guardrails.

### **Part 4: Advanced Topics and Frontiers**
- **Goal:** Dive into state-of-the-art concepts and future directions.
- **Topics:**
    1.  `01-model-internals-transformers`: A deeper dive into the Transformer architecture, self-attention, and positional encodings.
    2.  `02-advanced-fine-tuning-peft`: Techniques like LoRA and QLoRA for efficient fine-tuning.
    3.  `03-ai-safety-and-alignment`: Advanced techniques for making models safer, including constitutional AI and red teaming.
    4.  `04-the-future-of-gen-ai`: AI agents, the path to AGI, and the future of multimodal models.

---

## 6. Content Generation Guidelines
- **For `README.md` files:**
    - Explain concepts clearly and concisely. Use analogies and diagrams.
- **For `[topic]_example.py` files:**
    - Use popular libraries like `transformers`, `diffusers`, `peft`, etc.
    - Keep the code simple and focused on the lesson's concept.
- **For `key_concepts.md` and `interview_questions.md` files:**
    - Ensure the content is accurate and reflects a deep understanding of the topic.

## 7. Final Output Format
The final output should be a series of file creation commands, each containing the full path and the complete, formatted content for that file. The file structure must be strictly followed.
