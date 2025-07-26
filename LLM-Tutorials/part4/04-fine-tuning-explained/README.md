# Part 4: Building with AI: Your First Projects
## Topic 4: Fine-Tuning Explained

We've seen how RAG can give a model access to external knowledge. But what if we want to change the model's core behavior? What if we want to teach it a new skill, a specific style, or make it an expert in a niche domain?

For this, we can use a technique called **fine-tuning**.

---

### Generalist vs. Specialist: The Doctor Analogy

Let's think about doctors.

*   **A pre-trained LLM (like GPT-4) is like a General Practitioner (GP).** They have a vast amount of general medical knowledge. They can answer questions about thousands of different topics, from the common cold to basic anatomy. They are incredibly useful and versatile.

*   **A fine-tuned model is like a Specialist (e.g., a Cardiologist).** This is a doctor who, after completing their general medical degree, went through several more years of specialized training focused only on the heart. They have a much deeper knowledge of that one specific area than any GP. You wouldn't ask a cardiologist about a skin rash, but for a complex heart problem, they are the expert you need.

> **Simple Definition:** Fine-tuning is the process of taking a pre-trained, general-purpose LLM and continuing its training on a smaller, specialized dataset. This adapts the model to become an expert at a specific task, style, or domain.

Unlike prompting, fine-tuning actually **changes the internal weights** of the model. You are permanently altering its "brain" to make it better at a specific job.

---

### How Does Fine-Tuning Work?

The process involves creating a high-quality dataset of examples that demonstrate the desired behavior.

1.  **Create a Dataset:** You assemble a dataset of prompt-completion pairs. Each pair is an example of the input you would give the model and the *perfect* output you would want it to produce. For example, if you want to fine-tune a model to be a helpful SQL query generator, your dataset might look like this:

    *   **Prompt:** "Find all users from California."
    *   **Completion:** `SELECT * FROM users WHERE state = 'CA';`
    *   **Prompt:** "Count the number of orders placed last month."
    *   **Completion:** `SELECT COUNT(*) FROM orders WHERE order_date >= '2023-10-01' AND order_date < '2023-11-01';`

    You would need hundreds or even thousands of these high-quality examples.

2.  **Continue the Training:** You then use this specialized dataset to continue the model's training process. The model starts with the weights of the pre-trained generalist model (e.g., Llama 3) and uses Gradient Descent to adjust those weights to get better and better at predicting the "completion" based on the "prompt" in your dataset.

3.  **A New, Specialist Model:** The result is a new model file. You now have a specialist model that is an expert at converting natural language questions into SQL queries.

### RAG vs. Fine-Tuning

These are two different tools for two different jobs, and they are often used together.

| Feature              | RAG (Retrieval-Augmented Generation)                               | Fine-Tuning                                                      |
| -------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------- |
| **Analogy**          | Open-book test                                                     | Going to medical school to specialize                          |
| **Purpose**          | To provide the model with **knowledge**.                           | To teach the model a new **skill** or **style**.                 |
| **How it works**     | Retrieves relevant info and puts it in the prompt (context).       | Updates the model's internal weights through further training.   |
| **When to use it**   | When you need to answer questions about specific, changing documents. | When you need to change the model's fundamental behavior or format. |
| **Example Use Case** | A chatbot that answers questions about your company's internal HR policies. | A chatbot that always responds in the voice of Shakespeare.      |

**The Power of Combining Them:**
You can create a RAG system that uses a fine-tuned model. For example, you could fine-tune a model to be an expert at summarizing legal documents. Then, you could use that specialist model in a RAG system that retrieves specific legal contracts and asks the model to summarize them. This gives you the best of both worlds: a model with a specialized skill operating on specific, up-to-date knowledge.
