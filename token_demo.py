import tiktoken

encoding = tiktoken.get_encoding("o200k_base")
text = "I am learning artificial intelligence."

tokens = encoding.encode(text)

print(tokens)
print("Total tokens:", len(tokens))
decoded_text = encoding.decode(tokens)
print("Decoded text:", decoded_text)
