# Key Concepts: Mixture of Experts (MoE)

Here are the key terms for understanding this efficient model architecture.

### 1. Dense Model
-   **What it is:** A standard LLM architecture where every parameter in the model is used to process every single token.
-   **Analogy:** A "full-company meeting." To answer any question, no matter how simple, you have to get every single employee (parameter) in the room. This is powerful but very slow and expensive.
-   **Why it matters:** This is the traditional way of building LLMs. As models get bigger, the cost and slowness of dense models become a major problem.

### 2. Mixture of Experts (MoE) Model
-   **What it is:** A more efficient LLM architecture that uses a collection of smaller "expert" networks and a "router" that sends each token only to the most relevant experts.
-   **Analogy:** A company with specialized departments. When you have a legal question, a smart receptionist (the router) sends you directly to the legal department (the experts), ignoring everyone else. This is much faster and more efficient.
-   **Why it matters:** MoE allows for the creation of models with a massive number of total parameters (making them very knowledgeable) while keeping the cost of running them much lower. It's a key technology for building next-generation, highly capable yet efficient models.

### 3. Router (or Gating Network)
-   **What it is:** The small neural network inside an MoE model that is responsible for deciding which "experts" should be activated for a given token.
-   **Analogy:** The smart receptionist or dispatcher. Its only job is to look at an incoming request (a token) and quickly direct it to the right specialist.
-   **Why it matters:** The router is the key component that makes the MoE system work. A good router will learn to send tokens to the experts that are best equipped to handle them, leading to better performance and efficiency.
