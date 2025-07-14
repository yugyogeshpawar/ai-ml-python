# Exercises: Output Parsers

These exercises will help you practice structuring the output from LLMs.

### Exercise 1: Create a Simple List Parser

1.  Create a Python script.
2.  Your goal is to ask the LLM for a list of ingredients for a "Mojito" cocktail.
3.  Use the `CommaSeparatedListOutputParser` to parse the output.
4.  Create a `PromptTemplate` that includes the format instructions from the parser.
5.  Run your chain and print both the raw string output from the LLM and the final parsed list.

### Exercise 2: Build a Pydantic Parser for a Player Profile

You want to extract structured information about a famous athlete.

1.  Define a Pydantic model named `PlayerProfile` with the following fields:
    *   `name` (string)
    *   `sport` (string)
    *   `achievements` (a list of strings)
2.  Create a `PydanticOutputParser` for this model.
3.  Create a `PromptTemplate` that asks the LLM to generate a profile for a given athlete and includes the format instructions from the parser.
4.  Create an `LLMChain` and run it with the input "Michael Jordan".
5.  Print the resulting `PlayerProfile` object. You should be able to access its attributes like `parsed_output.name` and `parsed_output.achievements`.

### Exercise 3: Handle Parsing Errors

Sometimes, the LLM might not follow the format instructions perfectly, leading to a `ParseException`. Your task is to handle this gracefully.

1.  Take your script from Exercise 2.
2.  Instead of calling `parser.parse()`, use `parser.parse_with_prompt()`. This method is useful for debugging as it can sometimes correct minor formatting errors.
3.  Wrap the parsing step in a `try...except` block to catch a potential ` langchain.schema.OutputParserException`.
4.  In the `except` block, print a user-friendly error message and the raw output from the LLM that failed to parse.

This exercise demonstrates a crucial aspect of building robust LLM applications: gracefully handling cases where the model's output is not in the expected format.
