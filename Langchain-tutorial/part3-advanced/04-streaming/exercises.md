# Exercises: Streaming

These exercises will help you get comfortable with streaming responses from your LangChain applications.

### Exercise 1: Stream a Multi-Step Chain

1.  Take the multi-step chain you created in the LCEL exercises (the one that generates a company name and then a slogan).
2.  Use the `.stream()` method to run this chain.
3.  Print the chunks as they arrive. What do you observe? Does the output of the first step (the company name) appear before the second step (the slogan) starts generating?

This exercise demonstrates how streaming works with more complex, multi-step chains.

### Exercise 2: Build a Simple Streaming Web App

1.  Combine your knowledge of LangServe and streaming to create a simple web application that streams responses.
2.  Create a FastAPI application with a streaming endpoint.
    ```python
    from fastapi.responses import StreamingResponse

    @app.post("/stream-joke")
    async def stream_joke(data: dict):
        async def generate():
            async for chunk in chain.astream({"topic": data["topic"]}):
                yield chunk
        return StreamingResponse(generate(), media_type="text/plain")
    ```
3.  Use `chain.astream()` for asynchronous streaming within your FastAPI endpoint.
4.  Test this endpoint using `curl` or a simple Python script with the `requests` library.

This exercise shows how to use streaming in a real-world web application context.

### Exercise 3: Stream Intermediate Steps

LCEL allows you to stream not just the final output, but also the intermediate steps of your chain.

1.  Research how to use `chain.stream_log()` in LangChain.
2.  Use `stream_log()` on a multi-step chain (like the name and slogan generator).
3.  Iterate through the log and print the chunks. What information is available in the log? Can you see the inputs and outputs of each individual component in the chain as they execute?

This exercise is incredibly useful for debugging complex chains, as it gives you a real-time view of the data flowing through your application.
