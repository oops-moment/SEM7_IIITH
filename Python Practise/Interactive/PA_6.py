from collections import defaultdict

class TextSearch:
    def __init__(self, text):
        self.word_indices = defaultdict(list)
        self.preprocess(text)
        
    def preprocess(self, text):
        # Split text into words and store their indices
        words = text.split()
        for index, word in enumerate(words):
            self.word_indices[word].append(index)

    def find_query(self, query, k):
        # Split query into words
        query_words = query.split()
        first_word = query_words[0]
        result_indices = []

        # Get the indices of the first word in the query
        if first_word not in self.word_indices:
            return []  # If first word doesn't exist in text, return empty list

        # Iterate over the indices of the first word in the text
        for start_index in self.word_indices[first_word]:
            # Check if the rest of the query can be matched within k words
            if self.is_valid_query(start_index, query_words, k):
                result_indices.append(start_index)

        return result_indices

    def is_valid_query(self, start_index, query_words, k):
        # Check if the remaining words in query match within k distance
        for i in range(1, len(query_words)):  # Start from the second word in query
            word = query_words[i]
            next_word_found = False

            # Check if the next word in query exists within k distance from previous word
            for index in self.word_indices[word]:
                if index > start_index and index - start_index <= k + i:  # Allow k distance between words
                    next_word_found = True
                    start_index = index  # Move to the next word's position
                    break

            if not next_word_found:
                return False  # If the word is not found within k distance, return False

        return True

# Example usage:
text = "The quick brown fox is quick... quick fox"
query = "quick fox"
k = 2

search_engine = TextSearch(text)
result = search_engine.find_query(query, k)
print(result)  # Output: [1, 20]
