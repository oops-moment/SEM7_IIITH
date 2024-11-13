import re

# Function to remove special characters from a string
def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

# Function to remove specific words from a string
def remove_specific_words(text, words_to_remove):
    for word in words_to_remove:
        text = re.sub(rf'\b{word}\b', '', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()

# Example input text
sample_text = "The quick brown fox jumps over an amazing, lazy dog!"

# Remove special characters
text_no_special_characters = remove_special_characters(sample_text)
print("After removing special characters:")
print(text_no_special_characters)

# Remove specific words ("an", "the")
words_to_remove = ["an", "the"]
text_no_specific_words = remove_specific_words(text_no_special_characters, words_to_remove)
print("\nAfter removing specific words:")
print(text_no_specific_words)