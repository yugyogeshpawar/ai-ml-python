# Exercises: Setup and Installation

These exercises will help you practice the setup process and ensure your environment is ready for the rest of the tutorial.

### Exercise 1: Create a Project Directory and Virtual Environment

1.  Create a new directory for a test project called `my-langchain-test`.
2.  Navigate into this directory.
3.  Create a new Python virtual environment named `test-env`.
4.  Activate the virtual environment.

This exercise reinforces the best practice of using virtual environments for Python projects.

### Exercise 2: Install Packages and Create `requirements.txt`

1.  With your `test-env` virtual environment activated, install the `langchain` and `openai` packages using `pip`.
2.  Generate a `requirements.txt` file that lists the installed packages and their versions. The command `pip freeze > requirements.txt` is useful for this.
3.  Deactivate and then reactivate your virtual environment. Use the `requirements.txt` file to install the packages again (`pip install -r requirements.txt`). This simulates setting up the project on a new machine.

### Exercise 3: Securely Store and Access Your API Key

1.  If you haven't already, set your `OPENAI_API_KEY` as an environment variable for your current terminal session.
2.  Write a small Python script named `check_key.py` that imports the `os` module and prints the value of `os.environ.get("OPENAI_API_KEY")`.
3.  Run the script to confirm that your key is accessible from within your Python script.
4.  (Optional) For a more advanced exercise, research and implement a way to load environment variables from a `.env` file using a library like `python-dotenv`. This is a very common pattern in modern application development.
