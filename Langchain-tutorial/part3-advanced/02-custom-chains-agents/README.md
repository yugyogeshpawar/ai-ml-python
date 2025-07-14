# 2. Custom Chains and Agents: Building Specialized Solutions

While LangChain provides a rich set of pre-built chains and agents, you'll often encounter situations where you need to create your own custom components to solve specific problems. This lesson explores how to build custom chains and agents, giving you the power to tailor LangChain to your exact needs.

## Why Build Custom Components?

*   **Unique Functionality:** You might need to integrate with a proprietary API, implement a specific business rule, or perform a custom data transformation that isn't covered by existing components.
*   **Optimization:** You can optimize a chain or agent for a particular task, potentially improving performance, reducing cost, or enhancing accuracy.
*   **Modularity and Reusability:** By encapsulating custom logic into reusable components, you can make your code more organized, maintainable, and testable.

## Custom Chains

There are two main ways to create custom chains:

1.  **Extending `Chain` Directly:** This gives you the most control over the chain's behavior but requires more code.
2.  **Using `SequentialChain` or LCEL for Simple Composition:** If your custom chain is simply a sequence of existing components, you can often use `SequentialChain` (for older code) or LCEL (recommended) to compose them without writing a new class.

## Step-by-Step Code Tutorial: Creating a Custom Chain with LCEL

Let's build a custom chain that generates a company name and then checks if that name is available as a domain name.

### 1. Define the Components

We'll need a Chat Model, a Prompt Template, an Output Parser, and a function to check domain name availability.

### 2. Create the Script

The `custom_chain_example.py` script demonstrates how to build this chain with LCEL.

### Key Concepts in the Code

1.  **`def check_domain(company_name: str) -> str:`**: This is our custom function that checks if a domain name is available. In a real application, this would involve making an API call to a domain registrar. For this example, we'll just use a placeholder.

2.  **`prompt_template = ChatPromptTemplate.from_template(...)`**: We define a prompt template to generate the company name.

3.  **`chain = prompt | model | StrOutputParser() | check_domain`**: This is the LCEL chain. It pipes the output of the LLM to our custom `check_domain` function.

4.  **`response = chain.invoke({"product": "colorful socks"})`**: We run the chain as usual. The `check_domain` function will automatically be called with the generated company name.

## Custom Agents

Creating custom agents is more complex than creating custom chains, as it involves defining the agent's reasoning logic and how it interacts with tools.

The general process is:
1.  **Define the Tools:** Create the functions that your agent will use to interact with the outside world.
2.  **Create a Custom Prompt Template:** This template will guide the LLM in its decision-making process. It needs to include:
    *   Instructions on how to use the tools.
    *   A description of each tool.
    *   A placeholder for the agent's "thoughts" and "actions."
    *   A placeholder for the "observation" (the result of the tool execution).
3.  **Define the Agent Logic:** This involves creating a function that takes the LLM's output and parses it into an action to take and an action input.
4.  **Create the Agent:** Use the `AgentExecutor` class to combine the LLM, tools, and agent logic into a runnable agent.

## Next Steps

You've now learned how to create custom chains and agents, giving you the power to tailor LangChain to your specific needs. The final advanced topic is to learn how to deploy these applications to production using LangServe.

**Practice:** [Exercises](./exercises.md)

**Next Lesson:** [LangServe](./../03-langserve/README.md)
