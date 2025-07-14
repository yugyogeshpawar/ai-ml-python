# Exercises: LangChain Expression Language (LCEL)

These exercises will help you get comfortable with building chains using LCEL.

### Exercise 1: Add a Length Check

1.  Copy the `lcel_example.py` script.
2.  Add a step to the chain that checks if the generated company name is longer than 10 characters. If it is, return an error message instead.
    *   **Hint:** You'll need to create a function that takes the output of the LLM (a string) and returns either the original string or an error message. You can then pipe this function into the chain using `|`.

This exercise demonstrates how to add custom logic and validation steps to your LCEL chains.

### Exercise 2: Create a Multi-Step Chain

Let's build a chain that first generates a company name and then creates a slogan for it, all using LCEL.

1.  Create a `ChatPromptTemplate` for generating a company name (same as in the lesson).
2.  Create a `ChatPromptTemplate` for generating a slogan, taking the company name as input.
3.  Initialize a `ChatOpenAI` model.
4.  Create a `StrOutputParser`.
5.  Build the chain using LCEL: `prompt_name | model | output_parser | prompt_slogan | model | output_parser`.
6.  Run the chain with a product like "organic dog food" and print the final output (which should be the slogan).

This exercise shows how to compose multiple steps into a single LCEL chain.

### Exercise 3: Add a Fallback

LCEL makes it easy to add fallbacks in case a step fails.

1.  Research how to use `chain.with_fallbacks()` in LCEL.
2.  Create a chain that uses an unreliable LLM (e.g., a smaller, faster model that might sometimes fail).
3.  Create a fallback chain that uses a more reliable (but potentially slower/more expensive) LLM.
4.  Use `chain.with_fallbacks([fallback_chain])` to create a chain that will automatically try the fallback if the first chain fails.
5.  Run the chain multiple times and observe how it uses the fallback when the first chain fails.

This exercise demonstrates how to build more robust and resilient LCEL chains.
