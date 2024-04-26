import re

text = "RuntimeError: The size of tensor a (179) must match the size of tensor b (375) at non-singleton dimension 1"

# Use regex to extract the first number
match = re.search(r'\d+', text)
if match:
    first_number = int(match.group())
    print("First number:", type(first_number))
else:
    print("No number found in the text.")
