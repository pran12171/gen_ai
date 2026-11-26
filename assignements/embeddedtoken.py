import numpy as np

# 1. Define 3-dimensional token embeddings
# D1 = Animal-related
# D2 = Living-being-related
# D3 = Object-related

token_embeddings = {
    "the":     [0.10, 0.10, 0.10],
    "dog":     [0.95, 0.95, 0.02],
    "cat":     [0.90, 0.90, 0.05],
    "chased":  [0.05, 0.05, 0.05]
}

# 2. Define positional embeddings
positional_embeddings = [
    [0.01, 0.02, 0.03],
    [0.02, 0.03, 0.04],
    [0.03, 0.04, 0.05],
    [0.04, 0.05, 0.06],
    [0.05, 0.06, 0.07]
]

# 3. Define two sentences
sentences = {
    "Sentence 1": "the dog chased the cat",
    "Sentence 2": "the cat chased the dog"
}

# 4. Process each sentence
for sentence_name, sentence in sentences.items():

    print("\n", "=" * 60)
    print(sentence_name, ":", sentence)
    print("=" * 60)

    tokens = sentence.split()

    for position, token in enumerate(tokens):

        token_vector = np.array(token_embeddings[token])
        position_vector = np.array(positional_embeddings[position])

        # Add token and positional embeddings
        final_embedding = token_vector + position_vector

        print(f"\nPosition: {position + 1}")
        print(f"Token: {token}")
        print(f"Token Embedding: {token_vector}")
        print(f"Position Embedding: {position_vector}")
        print(f"Final Embedding: {final_embedding}")


# 5. Compare dog and cat embeddings
print("\n\nTOKEN EMBEDDING COMPARISON")
print("-" * 40)

for word in ["dog", "cat"]:
    print(f"{word}: {token_embeddings[word]}")


# 6. Calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


dog = np.array(token_embeddings["dog"])
cat = np.array(token_embeddings["cat"])

similarity = cosine_similarity(dog, cat)

print("\nCosine Similarity between Dog and Cat:")
print(round(similarity, 4))


# 7. Compare the two occurrences of dog
print("\nDOG POSITION COMPARISON")
print("-" * 40)

dog_position_2 = (
    np.array(token_embeddings["dog"])
    + np.array(positional_embeddings[1])
)

dog_position_5 = (
    np.array(token_embeddings["dog"])
    + np.array(positional_embeddings[4])
)

print("Dog at Position 2:", dog_position_2)
print("Dog at Position 5:", dog_position_5)
