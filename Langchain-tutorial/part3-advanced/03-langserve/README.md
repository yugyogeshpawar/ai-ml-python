# 3. LangServe: Deploying Your Chains as APIs

You've built a powerful LangChain application. Now, how do you share it with the world? How do you integrate it into a web application or make it accessible to other services? The answer is **LangServe**.

## What is LangServe?

LangServe is a library that helps you deploy your LangChain chains and agents as production-ready REST APIs. It takes care of the boilerplate code for creating a web server, defining API endpoints, and handling requests and responses.

**Key features of LangServe:**
*   **Automatic API Generation:** LangServe can automatically generate API endpoints for your LCEL chains, including input and output schemas.
*   **Interactive Playground:** It provides a built-in web interface (at `/docs`) where you can interact with your API, test different inputs, and see the results.
*   **Streaming and Async Support:** It fully supports streaming and asynchronous operations, allowing you to build responsive, real-time applications.
*   **Production-Ready:** LangServe is built on top of FastAPI and Pydantic, two popular and robust Python libraries for building high-performance APIs.

## Step-by-Step Code Tutorial

Let's take our simple LCEL chain from the first lesson and deploy it as a REST API using LangServe.

### 1. Install Necessary Libraries

We'll need `fastapi` (the web framework), `uvicorn` (the web server), and `langserve`.

```bash
pip install fastapi uvicorn langserve
```

### 2. Create the Server Script

The `langserve_example.py` script shows how to create a simple LangServe application.

### Key Concepts in the Code

1.  **`from fastapi import FastAPI`**: We import the FastAPI framework.
2.  **`from langserve import add_routes`**: This is the key function from LangServe.
3.  **`app = FastAPI(...)`**: We create a FastAPI application instance.
4.  **`add_routes(app, chain, path="/company-name")`**: This is where the magic happens.
    *   `app`: The FastAPI application to add the routes to.
    *   `chain`: The LCEL chain you want to deploy.
    *   `path`: The URL path for your API endpoint.
5.  **`if __name__ == "__main__": ...`**: This block allows you to run the server directly from the command line using `uvicorn`.

### 3. How to Run the Server

1.  Navigate to this directory in your terminal.
2.  Run the server using `uvicorn`:

    ```bash
    uvicorn langserve_example:app --reload
    ```
    *   `langserve_example`: The name of your Python file.
    *   `app`: The name of your FastAPI application instance.
    *   `--reload`: This tells the server to automatically restart when you make changes to the code.

3.  The server will start, usually at `http://127.0.0.1:8000`.

### 4. Interact with Your API

*   **Interactive Playground:** Open your web browser and go to `http://127.0.0.1:8000/docs`. You'll see an interactive API documentation page where you can test your `/company-name` endpoint.
*   **Using `curl`:** You can also interact with your API from the command line using `curl`:

    ```bash
    curl -X POST http://127.0.0.1:8000/company-name/invoke \
    -H "Content-Type: application/json" \
    -d '{"input": {"product": "eco-friendly water bottles"}}'
    ```

## Why is LangServe a Game-Changer?

LangServe dramatically simplifies the process of deploying LangChain applications. It bridges the gap between development and production, allowing you to focus on building your chains and agents without worrying about the complexities of web server development.

## Next Steps

You've learned how to build and deploy LangChain applications. The final piece of the puzzle is to explore how to make your applications more responsive and user-friendly by streaming responses in real-time.

**Practice:** [Exercises](./exercises.md)

**Next Lesson:** [Streaming](./../04-streaming/README.md)
