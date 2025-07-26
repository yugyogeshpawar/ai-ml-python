# Part 1 Project: Blog Post Generator

In this project, we will use the concepts we've learned in Part 1 to build a simple command-line application that generates a short blog post on a given topic.

## Objective

The goal is to create a chain that takes a topic from the user and generates a blog post with a title and content. We will use:

*   An `LLM` (OpenAI).
*   A `PromptTemplate` to structure our request to the LLM.
*   An `LLMChain` to tie them together.
*   A `PydanticOutputParser` to get the output in a structured format.

## Step-by-Step Implementation

### 1. Define the Desired Output Structure

First, let's define the structure of our blog post using a Pydantic model. This will help us get a clean, predictable output from the LLM.

```python
from pydantic import BaseModel, Field

class BlogPost(BaseModel):
    title: str = Field(description="The title of the blog post")
    content: str = Field(description="The content of the blog post")
```

### 2. Create the Main Script

Now, let's create the main Python script for our application.

**`blog_post_generator.py`**

```python
import os
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# --- 1. Define the Output Structure ---
class BlogPost(BaseModel):
    title: str = Field(description="The title of the blog post")
    content: str = Field(description="The content of the blog post, written in a markdown format")

# --- 2. Set up the Parser ---
parser = PydanticOutputParser(pydantic_object=BlogPost)

# --- 3. Create the Prompt Template ---
prompt_template = PromptTemplate(
    template="Generate a short blog post about {topic}.\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# --- 4. Initialize the LLM and build the Chain with LCEL ---
llm = OpenAI(temperature=0.7)
chain = prompt_template | llm | parser

# --- 5. Get User Input and Run the Chain ---
if __name__ == "__main__":
    topic = input("Enter a topic for the blog post: ")

    # Run the chain
    parsed_output = chain.invoke({"topic": topic})

    # Print the result
    print("\n--- Generated Blog Post ---\n")
    print(f"Title: {parsed_output.title}")
    print(f"Content:\n{parsed_output.content}")

```

### 3. How to Run the Project

1.  Make sure you have your `OPENAI_API_KEY` environment variable set.
2.  Navigate to the `project` directory in your terminal.
3.  Run the script:

    ```bash
    python blog_post_generator.py
    ```

4.  The script will prompt you to enter a topic. Type a topic and press Enter.

### Example Usage

```
Enter a topic for the blog post: the benefits of learning a new programming language

--- Generated Blog Post ---

Title: Unlock Your Potential: The Many Benefits of Learning a New Programming Language
Content:
Learning a new programming language can be one of the most rewarding investments you make in your professional and personal development. It's not just about adding another skill to your resume; it's about expanding your problem-solving abilities, boosting your creativity, and opening up new career opportunities.

One of the most significant benefits is the ability to think in new ways. Each language has its own paradigms and philosophies, and learning a new one forces you to approach problems from a different perspective. This can make you a more versatile and effective developer, even in your primary language.

Furthermore, learning a new language can lead to exciting career opportunities. Many specialized fields, such as data science (Python), mobile development (Swift/Kotlin), or systems programming (Rust), require specific language skills. By diversifying your knowledge, you make yourself a more attractive candidate in a competitive job market.

Finally, it's a great way to keep your mind sharp and stay passionate about technology. The challenge of mastering a new syntax and ecosystem can be incredibly stimulating and fun. So, what are you waiting for? Pick a language that interests you and start your learning journey today!
```

This project demonstrates how to combine the core components of LangChain to build a practical and useful application.
