from collections import Counter

# Step 1: Input words and their frequencies
word_freq = {
    "low": 5,
    "lower": 2,
    "newest": 3,
    "widest": 6
}


# Step 2: Initialize words into characters
def initialize_tokens(word_freq):

    tokenized_words = {}

    for word in word_freq:

        tokens = [word[0]]

        for char in word[1:]:
            tokens.append("##" + char)

        tokenized_words[word] = tokens

    return tokenized_words


# Step 3: Calculate token and pair frequencies
def calculate_frequencies(tokenized_words, word_freq):

    token_freq = Counter()
    pair_freq = Counter()

    for word, tokens in tokenized_words.items():

        frequency = word_freq[word]

        for token in tokens:
            token_freq[token] += frequency

        for i in range(len(tokens) - 1):

            pair = (tokens[i], tokens[i + 1])

            pair_freq[pair] += frequency

    return token_freq, pair_freq


# Step 4: Calculate WordPiece scores
def calculate_scores(token_freq, pair_freq):

    scores = {}

    for pair, frequency in pair_freq.items():

        first, second = pair

        score = frequency / (
            token_freq[first] * token_freq[second]
        )

        scores[pair] = score

    return scores


# Step 5: Merge selected tokens
def merge_tokens(tokenized_words, best_pair):

    first, second = best_pair

    # Remove ## from the continuation token
    if second.startswith("##"):
        merged_token = first + second[2:]
    else:
        merged_token = first + second

    for word, tokens in tokenized_words.items():

        new_tokens = []

        i = 0

        while i < len(tokens):

            if (
                i < len(tokens) - 1
                and tokens[i] == first
                and tokens[i + 1] == second
            ):

                new_tokens.append(merged_token)
                i += 2

            else:

                new_tokens.append(tokens[i])
                i += 1

        tokenized_words[word] = new_tokens

    return merged_token


# Step 6: Complete WordPiece training
def wordpiece_training(word_freq):

    tokenized_words = initialize_tokens(word_freq)

    print("INITIAL TOKENIZATION")

    for word, tokens in tokenized_words.items():
        print(word, ":", tokens)

    iteration = 1

    while True:

        token_freq, pair_freq = calculate_frequencies(
            tokenized_words, word_freq
        )

        if not pair_freq:
            break

        scores = calculate_scores(token_freq, pair_freq)

        # Select pair with highest score
        best_pair = max(scores, key=scores.get)

        best_score = scores[best_pair]

        print("\nIteration:", iteration)

        print("Selected Pair:", best_pair)

        print("Pair Frequency:", pair_freq[best_pair])

        print("WordPiece Score:", round(best_score, 6))

        merged_token = merge_tokens(
            tokenized_words, best_pair
        )

        print("Merged Token:", merged_token)

        iteration += 1

    print("\nFINAL TOKENIZATION")

    for word, tokens in tokenized_words.items():

        print(word, ":", tokens)

    return tokenized_words


# Step 7: Execute
wordpiece_training(word_freq)
