# Prompt for Generating a Comprehensive Generative AI Tutorial

## 1. Overall Objective
Generate a complete, multi-part tutorial on the fundamentals of Generative AI. The tutorial should be suitable for hosting on GitHub and should cover the core concepts, models, and applications of Generative AI, from foundational theory to practical implementation.

## 2. Target Audience
The tutorial is for developers, students, and tech enthusiasts who want to build a strong foundational understanding of Generative AI. No prior AI/ML experience is required, but a basic understanding of Python is assumed.

## 3. Core Philosophy & Style
- **Conceptual Clarity:** Focus on explaining the core concepts in simple, intuitive terms. Use analogies and real-world examples.
- **Broad Scope:** Cover the key pillars of Generative AI, including text, images, and code generation.
- **Practical and Hands-On:** Each lesson should be accompanied by simple, runnable code examples that demonstrate the concepts.

## 4. High-Level Structure
The tutorial will be divided into three main parts.

- **Part 1: Foundations of Generative AI**
- **Part 2: Core Generative Models**
- **Part 3: Building Generative AI Applications**

## 5. Detailed Content Structure (For Each Topic)
For every topic, generate the following files inside a dedicated topic directory:
1.  **`README.md`**: The main lesson file with detailed explanations.
2.  **`[topic]_example.py`**: A runnable Python script (where applicable).
3.  **`key_concepts.md`**: A file summarizing the key takeaways and vocabulary for the lesson.
4.  **`interview_questions.md`**: 3 relevant interview questions and detailed answers.

---

### **Part 1: Foundations of Generative AI**
- **Goal:** Introduce the basic concepts and building blocks of AI and machine learning.
- **Topics:**
    1.  `01-what-is-generative-ai`: Defining Generative AI vs. Discriminative AI. Real-world examples.
    2.  `02-how-llms-work`: The concept of tokens, transformers, and next-token prediction.
    3.  `03-prompt-engineering`: The art and science of crafting effective prompts.
    4.  `04-introduction-to-embeddings`: The concept of representing data as vectors.

- **Project:** A simple command-line application that uses a pre-trained model (like GPT-2) to generate text based on a user's prompt.

### **Part 2: Core Generative Models**
- **Goal:** Explore the different types of generative models and their applications.
- **Topics:**
    1.  `01-text-generation-models`: GPT, LLaMA, etc. How they are used for summarization, translation, and creative writing.
    2.  `02-image-generation-models`: Stable Diffusion, Midjourney, DALL-E. The concept of text-to-image generation.
    3.  `03-code-generation-models`: Codex, Code Llama. How they are used for code completion and generation.
    4.  `04-multimodal-models`: Models that can understand and generate multiple types of data (e.g., text and images).

- **Project:** A simple web application (using Gradio or Streamlit) that allows a user to enter a text prompt and see a generated image from a pre-trained model.

### **Part 3: Building Generative AI Applications**
- **Goal:** Cover the practical aspects of building and deploying Generative AI applications.
- **Topics:**
    1.  `01-using-apis-vs-local-models`: The trade-offs between using a hosted API (like OpenAI's) and running a model locally.
    2.  `02-fine-tuning-models`: The concept of fine-tuning a pre-trained model on your own data for a specific task.
    3.  `03-retrieval-augmented-generation-rag`: A high-level overview of RAG as a technique to reduce hallucinations and use external data.
    4.  `04-ethics-and-safety`: The importance of responsible AI, including topics like bias, misinformation, and safety guardrails.

- **Project:** A simple RAG application that answers questions based on a small text file, demonstrating how to ground a model's responses in factual data.

---

## 6. Content Generation Guidelines
- **For `README.md` files:**
    - Explain concepts clearly and concisely.
    - Use analogies and diagrams where helpful.
- **For `[topic]_example.py` files:**
    - Use popular libraries like `transformers`, `diffusers`, `gradio`, etc.
    - Keep the code simple and focused on the lesson's concept.
- **For `key_concepts.md` and `interview_questions.md` files:**
    - Ensure the content is accurate and reflects a solid understanding of the topic.

## 7. Final Output Format
The final output should be a series of file creation commands, each containing the full path and the complete, formatted content for that file. The file structure must be strictly followed.
