# 2. Setup and Installation: Building a Solid Foundation

Before we can start building exciting applications, we need to set up a clean, professional, and reproducible development environment. This lesson will guide you through the essential steps, explaining the "why" behind each one.

## Step 1: Install Python

LangChain is a Python library, so the first requirement is to have Python installed on your system. LangChain requires **Python 3.8 or newer**.

To check if you have Python and what version it is, open your terminal or command prompt and run:
```bash
python --version
```
If you get a version number like `Python 3.10.4`, you're good to go. If not, you can download the latest version from the [official Python website](https://www.python.org/downloads/).

## Step 2: Create a Virtual Environment (A Crucial Best Practice)

This is one of the most important habits for a Python developer. A virtual environment is an isolated directory that contains a specific version of Python and all the packages required for a particular project.

**Why is this so important?**
Imagine you have two projects:
-   Project A requires `langchain` version `0.1.0`.
-   Project B requires the newer `langchain` version `0.2.0`.

If you install these packages globally (on your main system), you'll have a conflict. Installing version `0.2.0` for Project B would overwrite the version needed for Project A, potentially breaking it.

A virtual environment solves this by creating a self-contained "bubble" for each project.

**How to do it:**
1.  Navigate to your project's root directory in the terminal.
2.  Create the environment (we'll call it `venv` by convention):
    ```bash
    python -m venv venv
    ```
3.  **Activate** the environment. This tells your terminal to use the Python and packages inside this `venv` directory instead of the global ones.
    *   **On macOS and Linux:**
        ```bash
        source venv/bin/activate
        ```
    *   **On Windows:**
        ```bash
        venv\Scripts\activate
        ```
    You'll know it's active because your terminal prompt will change to show `(venv)`.

## Step 3: Install LangChain and Dependencies

With your virtual environment active, you can now safely install the packages for our project. We'll start with the core `langchain` library and the `openai` package, which allows us to connect to OpenAI's models.

```bash
pip install langchain openai
```
This command downloads and installs the libraries only inside your `venv` bubble, leaving your global Python installation untouched.

## Step 4: Securely Manage Your API Keys

To use a service like OpenAI, you need to provide an API key. This key is a secret credential that proves you have permission to use their service.

**CRITICAL: Never, ever hard-code your API keys directly in your code.**
If you write `llm = OpenAI(api_key="sk-...")` and commit this code to a public GitHub repository, your key will be stolen and abused within minutes.

The industry-standard best practice is to use **environment variables**. An environment variable is a variable that is part of your operating system's environment, and it can be accessed by any program running in that environment.

**How to do it:**
1.  **Get your key:** Visit the [OpenAI platform website](https://platform.openai.com/), create an account, and generate a new secret key.
2.  **Set the variable:** In your terminal, set the environment variable. LangChain specifically looks for the name `OPENAI_API_KEY`.
    *   **On macOS and Linux (for the current session):**
        ```bash
        export OPENAI_API_KEY="your-api-key-here"
        ```
    *   **On Windows (for the current session):**
        ```bash
        set OPENAI_API_KEY="your-api-key-here"
        ```
    *   **For permanence:** To avoid typing this every time you open a new terminal, you should add the `export` command to your shell's startup file (e.g., `~/.zshrc` or `~/.bashrc`) and then restart your terminal or run `source ~/.zshrc`.

## Step 5: Verify Your Complete Installation

Now, let's confirm that all the pieces are working together. The `verify_setup.py` script in this directory provides a simple test.

**What the script does:**
1.  It imports `os` to check for the environment variable.
2.  It imports `OpenAI` from `langchain.llms`.
3.  When `OpenAI()` is called without any arguments, LangChain automatically and securely looks for the `OPENAI_API_KEY` in the environment.
4.  It sends a simple prompt, "Tell me a joke.", to the model.
5.  It prints the model's response.

Run the script from your terminal:
```bash
python verify_setup.py
```
If you see a joke printed to your console, your setup is successful! You have a clean, secure, and professional environment ready for building.

## Next Steps

With a solid foundation in place, we are now ready to dive into the core components of LangChain, starting with the models themselves.

**Practice:** [Exercises](./exercises.md)

**Next Lesson:** [LLMs](./../03-llms/README.md)
