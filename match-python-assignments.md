# Python `match` Statement Assignment Questions

## Beginner Level

1. **Basic Match Case**
   Write a Python function that takes a day of the week as input and uses a `match` statement to print whether it is a weekday or weekend.

2. **Match with Integers**
   Use the `match` statement to create a simple calculator. Take two integers and an operator (`+`, `-`, `*`, `/`) as input, and return the result.

3. **Match with Default Case**
   Create a `match` statement that categorizes a given number into:
   - "Negative"
   - "Zero"
   - "Positive"
   Provide a default case if the input is not a number.

## Intermediate Level

4. **Match with Multiple Patterns**
   Write a function that takes a grade (A, B, C, D, F) and returns a message:
   - A or B: "Well done!"
   - C or D: "Needs improvement"
   - F: "Failed"
   Use a single case for multiple patterns.

5. **Match with Tuples**
   Given a coordinate point `(x, y)`, use a `match` statement to determine:
   - If it’s on the X-axis
   - If it’s on the Y-axis
   - If it’s at the origin
   - If it’s in a quadrant

6. **Match with Class Objects**
   Define a class `Animal` with subclasses `Dog`, `Cat`, and `Bird`. Write a function that takes an instance and uses `match` to identify the type of animal.

## Advanced Level

7. **Nested Match Statements**
   Write a function that takes a dictionary with keys `type` and `value`, and uses nested `match` statements to process:
   - If `type` is "number", double the value
   - If `type` is "string", capitalize the string
   - If `type` is "list", return the length of the list

8. **Match with Guards**
   Create a program that matches a student's test score and prints the grade, using guards (`if` clauses) for score ranges:
   - 90 and above: A
   - 80–89: B
   - 70–79: C
   - 60–69: D
   - Below 60: F

9. **Destructuring with Match**
   Use the `match` statement to unpack a list of exactly three elements and print each item. If the list doesn't have exactly three items, print "Invalid input".

10. **Pattern Matching JSON-like Data**
   Write a function that uses pattern matching to extract user data from a dictionary that may contain:
   - `{"type": "user", "name": ..., "age": ...}`
   - `{"type": "admin", "name": ..., "permissions": [...]}`  
   Match the `type` and print a different message for users and admins.

