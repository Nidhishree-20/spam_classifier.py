# Sample dataset
data = [
    ("Congratulations you won a free lottery", "spam"),
    ("Call me when you reach home", "ham"),
    ("Win money now click this link", "spam"),
    ("Let's meet for lunch tomorrow", "ham"),
    ("Free entry in a prize contest", "spam"),
    ("Are you coming to class today", "ham")
]

# Step 1: Prepare data
spam_words = {}
ham_words = {}
spam_count = 0
ham_count = 0

# Step 2: Count word frequency
for message, label in data:
    words = message.lower().split()
    
    if label == "spam":
        spam_count += 1
        for word in words:
            spam_words[word] = spam_words.get(word, 0) + 1
    else:
        ham_count += 1
        for word in words:
            ham_words[word] = ham_words.get(word, 0) + 1

# Step 3: Total words
total_spam_words = sum(spam_words.values())
total_ham_words = sum(ham_words.values())

# Step 4: Vocabulary
vocab = set(list(spam_words.keys()) + list(ham_words.keys()))

# Step 5: Prediction function
def predict(message):
    words = message.lower().split()
    
    spam_prob = spam_count / (spam_count + ham_count)
    ham_prob = ham_count / (spam_count + ham_count)
    
    for word in words:
        # Laplace smoothing
        spam_word_prob = (spam_words.get(word, 0) + 1) / (total_spam_words + len(vocab))
        ham_word_prob = (ham_words.get(word, 0) + 1) / (total_ham_words + len(vocab))
        
        spam_prob *= spam_word_prob
        ham_prob *= ham_word_prob
    
    return "spam" if spam_prob > ham_prob else "ham"

# Step 6: Test
test_msg = "Congratulations you have won a prize"
result = predict(test_msg)

print("Message:", test_msg)
print("Prediction:", result)
