# Interview Questions: Text Splitters

### Q1: Why are Text Splitters necessary when working with LLMs, even after loading documents?

**Answer:**

Text Splitters are necessary primarily because of the **context window limit** of Large Language Models. LLMs can only process a finite amount of text (tokens) in a single input. If a document is larger than this limit, it cannot be fed directly to the LLM.

Text splitters break down large documents into smaller, manageable chunks that fit within the LLM's context window. This allows LLMs to process and reason over extensive bodies of text, which is crucial for applications like RAG (Retrieval-Augmented Generation) where you need to query large knowledge bases.

### Q2: What is the purpose of `chunk_size` and `chunk_overlap` in a text splitter?

**Answer:**

*   **`chunk_size`**: This parameter defines the **maximum size of each text chunk** that the splitter will create. It's typically measured in characters or tokens, depending on the specific splitter. The goal is to ensure that each chunk is small enough to fit within the LLM's context window.

*   **`chunk_overlap`**: This parameter specifies the **number of characters (or tokens) that will overlap** between consecutive chunks.
    *   **Purpose:** Overlap is crucial for maintaining context. When a document is split, information relevant to a sentence or idea might be at the boundary of two chunks. By adding overlap, you ensure that some surrounding context is present in both chunks, reducing the chance of losing critical information when the LLM processes individual chunks.
    *   **Impact:** A larger overlap provides more context but also increases the total number of tokens processed (and thus cost/latency). A smaller overlap might lead to fragmented context.

### Q3: Explain the strategy of `RecursiveCharacterTextSplitter` and why it's often preferred.

**Answer:**

The `RecursiveCharacterTextSplitter` is a widely used and highly effective text splitter because of its intelligent splitting strategy. It attempts to split text using a list of separators, trying them in order of preference.

**Strategy:**
1.  It first tries to split on the largest, most semantically meaningful separator (e.g., `"\n\n"` for paragraphs).
2.  If a chunk is still too large, it then tries the next separator in the list (e.g., `"\n"` for lines).
3.  This process continues down the list (e.g., ` " "` for words, `""` for characters) until the chunks are small enough.

**Why it's preferred:**
*   **Semantic Coherence:** By prioritizing larger separators, it tries to keep related sentences and paragraphs together, minimizing the disruption of semantic meaning. This results in more coherent chunks.
*   **Flexibility:** The ability to define a custom list of separators makes it adaptable to various document structures (e.g., markdown, code, prose).
*   **Robustness:** It gracefully handles cases where larger separators don't exist or don't result in small enough chunks, falling back to smaller units until the `chunk_size` is met.
