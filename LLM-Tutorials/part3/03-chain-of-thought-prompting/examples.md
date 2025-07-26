# Examples in Action: Chain-of-Thought Prompting

Forcing the model to "show its work" is a surprisingly effective way to improve its reasoning. Here are some practical examples of where Chain-of-Thought (CoT) prompting makes a huge difference.

### Example 1: Multi-Step Math Problems

This is the classic use case for CoT. LLMs are not calculators and often fail at multi-step arithmetic when asked for a direct answer.

*   **Standard Prompt (Likely to Fail):**
    > A farmer has 5 baskets of apples. Each basket contains 12 apples. He sells half of all the apples. Then, he gives 5 apples to his neighbor. How many apples does the farmer have left?

*   **Result (Often Incorrect):** The model might do one step correctly but fail on the next, or just guess a number. It might say `25` or `30`.

*   **Chain-of-Thought Prompt (Likely to Succeed):**
    > A farmer has 5 baskets of apples. Each basket contains 12 apples. He sells half of all the apples. Then, he gives 5 apples to his neighbor. How many apples does the farmer have left?
    >
    > Let's think step by step.

*   **Result (Showing the Work):**
    > 1.  First, calculate the total number of apples. The farmer has 5 baskets * 12 apples/basket = 60 apples.
    > 2.  Next, he sells half of the apples. Half of 60 is 60 / 2 = 30 apples. He has 30 apples left after selling.
    > 3.  Then, he gives 5 apples to his neighbor. So, we subtract 5 from the remaining apples: 30 - 5 = 25 apples.
    >
    > The farmer has 25 apples left.

*   **Analysis:** By breaking the problem down, the model can focus on one simple calculation at a time. The output of each step becomes part of the context for the next step, creating a logical chain that leads to the correct answer.

---

### Example 2: Logic Puzzles

LLMs can get confused by logic puzzles that require tracking multiple constraints or relationships.

*   **Standard Prompt (Can be Unreliable):**
    > There are three friends: Alex, Ben, and Clara. One is a doctor, one is a lawyer, and one is an engineer. Alex is not the lawyer. Ben is not the doctor. Clara is the lawyer. Who is the engineer?

*   **Result (Might be correct, but might get confused):** The model might just guess or get tangled in the negatives.

*   **Few-Shot Chain-of-Thought Prompt (More Robust):**
    > [This is a great case for a **few-shot** CoT, where you show it how to reason.]
    >
    > **Question:** At a party, there are three people: Tom, who is wearing a red shirt; Jane, who is wearing a blue shirt; and Sam, who is wearing a green shirt. Their favorite fruits are apples, bananas, and cherries. The person in the red shirt likes apples. Jane does not like bananas. Who likes cherries?
    > **Answer:**
    > - Tom is in the red shirt, so Tom likes apples.
    > - Jane does not like bananas. Since Tom likes apples, Jane must like cherries.
    > - Therefore, Sam must like bananas.
    > The final answer is: Jane
    >
    > ---
    >
    > **Question:** There are three friends: Alex, Ben, and Clara. One is a doctor, one is a lawyer, and one is an engineer. Alex is not the lawyer. Ben is not the doctor. Clara is the lawyer. Who is the engineer?
    > **Answer:**

*   **Result (Following the Pattern):**
    > - Clara is the lawyer. This leaves doctor and engineer for Alex and Ben.
    > - Alex is not the lawyer (which we already know).
    > - Ben is not the doctor. Since Clara is the lawyer, and Ben is not the doctor, Ben must be the engineer.
    > - This leaves Alex to be the doctor.
    > The final answer is: Ben

*   **Analysis:** By providing an example of how to use a process of elimination, you teach the model a logical framework for solving this kind of puzzle. It learns the *pattern* of reasoning, which is more powerful than just asking it to solve the problem from scratch.
