from first_agent import run_agent

# question 1
#  1. Simple case → single tool call
# Agent should directly use "add"

# run_agent("What is 42 + 58?")

# 2. Complex reasoning → planning required
# Agent must:
#   step 1: calculate area
#   step 2: calculate square root
run_agent(
    "I have a rectangle with width 12 and height 7. "
    "What is its area, and what is the square root of that area?"
)
# print(x)