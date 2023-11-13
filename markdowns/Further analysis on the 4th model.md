```python
!pip install nltk
```

    [33mDEPRECATION: Loading egg at /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages/huggingface_hub-0.19.0-py3.8.egg is deprecated. pip 24.3 will enforce this behaviour change. A possible replacement is to use pip for package installation.. Discussion can be found at https://github.com/pypa/pip/issues/12330[0m[33m
    [0mRequirement already satisfied: nltk in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (3.8.1)
    Requirement already satisfied: click in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (from nltk) (8.1.7)
    Requirement already satisfied: joblib in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (from nltk) (1.3.2)
    Requirement already satisfied: regex>=2021.8.3 in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (from nltk) (2023.10.3)
    Requirement already satisfied: tqdm in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (from nltk) (4.65.0)



```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

    [nltk_data] Downloading package punkt to /Users/ching/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package stopwords to /Users/ching/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!





    True



## Excluding Stopwords


```python
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Load the CSV file
file_path = "4_sohan_ai_sentiment_analysis_processed_less.csv"
df = pd.read_csv(file_path)

# Filter positive and negative reviews
positive_reviews = df[df['Sentiment_Label'] == 'positive']['Preprocessed_Body']
negative_reviews = df[df['Sentiment_Label'] == 'negative']['Preprocessed_Body']

# Tokenization and stopwords removal
def process_reviews(reviews):
    tokens = [word.lower() for review in reviews for word in word_tokenize(str(review)) if word.isalpha()]
    filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
    return filtered_tokens

# Process positive and negative reviews
positive_tokens = process_reviews(positive_reviews)
negative_tokens = process_reviews(negative_reviews)

# Frequency analysis
positive_freq = Counter(positive_tokens)
negative_freq = Counter(negative_tokens)

# Identify top keywords
top_positive_keywords = positive_freq.most_common(10)
top_negative_keywords = negative_freq.most_common(10)

# Print Top Keywords
print("Top Positive Keywords:")
for keyword, frequency in top_positive_keywords:
    print(f"{keyword}: {frequency}")

print("\nTop Negative Keywords:")
for keyword, frequency in top_negative_keywords:
    print(f"{keyword}: {frequency}")

```

    Top Positive Keywords:
    stars: 224
    great: 113
    phone: 92
    sound: 88
    good: 85
    speaker: 74
    gift: 66
    works: 49
    use: 47
    stand: 42
    
    Top Negative Keywords:
    stars: 153
    phone: 67
    sound: 49
    speaker: 44
    work: 35
    would: 32
    working: 30
    product: 27
    money: 26
    use: 25



```python
from nltk.util import ngrams
from nltk.corpus import stopwords
from collections import Counter

# Function to process reviews
def process_reviews(reviews, n=1):
    tokens = [word.lower() for review in reviews for word in word_tokenize(str(review)) if word.isalpha()]
    filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
    n_grams = list(ngrams(filtered_tokens, n))
    return n_grams

# Process positive and negative reviews
positive_tokens = process_reviews(positive_reviews, n=2)  # Change n to the desired n-gram size
negative_tokens = process_reviews(negative_reviews, n=2)  # Change n to the desired n-gram size

# Frequency analysis
positive_freq = Counter(positive_tokens)
negative_freq = Counter(negative_tokens)

# Display top positive and negative keywords
print("Top Positive Keywords:")
for word, freq in positive_freq.most_common(10):
    print(f"{word}: {freq}")

print("\nTop Negative Keywords:")
for word, freq in negative_freq.most_common(10):
    print(f"{word}: {freq}")

```

    Top Positive Keywords:
    ('stars', 'great'): 34
    ('stars', 'good'): 21
    ('sound', 'quality'): 18
    ('good', 'sound'): 17
    ('works', 'great'): 15
    ('stars', 'works'): 13
    ('great', 'sound'): 13
    ('easy', 'use'): 11
    ('battery', 'life'): 11
    ('works', 'well'): 11
    
    Top Negative Keywords:
    ('stopped', 'working'): 19
    ('waste', 'money'): 12
    ('phone', 'stand'): 11
    ('money', 'stars'): 11
    ('sound', 'quality'): 9
    ('stars', 'stopped'): 7
    ('stars', 'broke'): 7
    ('worth', 'money'): 6
    ('stars', 'work'): 6
    ('cell', 'phone'): 6



```python
from nltk.util import ngrams
from nltk.corpus import stopwords
from collections import Counter

# Function to process reviews
def process_reviews(reviews, n=1):
    tokens = [word.lower() for review in reviews for word in word_tokenize(str(review)) if word.isalpha()]
    filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
    n_grams = list(ngrams(filtered_tokens, n))
    return n_grams

# Process positive and negative reviews
positive_tokens = process_reviews(positive_reviews, n=3)  # Change n to the desired n-gram size
negative_tokens = process_reviews(negative_reviews, n=3)  # Change n to the desired n-gram size

# Frequency analysis
positive_freq = Counter(positive_tokens)
negative_freq = Counter(negative_tokens)

# Display top positive and negative keywords
print("Top Positive Keywords:")
for word, freq in positive_freq.most_common(20):
    print(f"{word}: {freq}")

print("\nTop Negative Keywords:")
for word, freq in negative_freq.most_common(20):
    print(f"{word}: {freq}")

```

    Top Positive Keywords:
    ('good', 'sound', 'quality'): 6
    ('stars', 'great', 'sound'): 6
    ('stars', 'good', 'product'): 5
    ('stars', 'works', 'great'): 5
    ('stars', 'great', 'product'): 5
    ('stars', 'great', 'gift'): 4
    ('stars', 'good', 'value'): 4
    ('stars', 'loud', 'enough'): 4
    ('nice', 'gift', 'stars'): 3
    ('cell', 'phone', 'stand'): 3
    ('father', 'day', 'gift'): 3
    ('buy', 'stars', 'works'): 3
    ('stars', 'husband', 'loves'): 3
    ('stars', 'great', 'speaker'): 3
    ('stars', 'bought', 'gift'): 3
    ('product', 'would', 'recommend'): 3
    ('stars', 'good', 'sound'): 3
    ('easy', 'use', 'stars'): 3
    ('good', 'value', 'money'): 3
    ('stars', 'nice', 'little'): 2
    
    Top Negative Keywords:
    ('waste', 'money', 'stars'): 7
    ('stars', 'stopped', 'working'): 7
    ('stopped', 'working', 'months'): 6
    ('cell', 'phone', 'stand'): 4
    ('worth', 'money', 'stars'): 3
    ('speaker', 'stopped', 'working'): 3
    ('poor', 'sound', 'quality'): 3
    ('stars', 'worth', 'money'): 2
    ('stars', 'disappointed', 'thought'): 2
    ('type', 'c', 'cable'): 2
    ('unit', 'died', 'next'): 2
    ('died', 'next', 'day'): 2
    ('next', 'day', 'last'): 2
    ('day', 'last', 'day'): 2
    ('last', 'day', 'return'): 2
    ('loud', 'enough', 'stars'): 2
    ('stars', 'good', 'good'): 2
    ('phone', 'stars', 'sound'): 2
    ('stars', 'sound', 'quality'): 2
    ('stars', 'save', 'money'): 2


## Including Stopwords


```python
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from collections import Counter


# Function to process reviews
def process_reviews(reviews, n=1):
    tokens = [word.lower() for review in reviews for word in word_tokenize(str(review)) if word.isalpha()]
    n_grams = list(ngrams(tokens, n))
    return n_grams

# Process positive and negative reviews
positive_tokens = process_reviews(positive_reviews, n=3)  # Change n to the desired n-gram size
negative_tokens = process_reviews(negative_reviews, n=3)  # Change n to the desired n-gram size

# Frequency analysis
positive_freq = Counter(positive_tokens)
negative_freq = Counter(negative_tokens)

# Display top positive and negative keywords
print("Top Positive Keywords:")
for word, freq in positive_freq.most_common(20):
    print(f"{word}: {freq}")

print("\nTop Negative Keywords:")
for word, freq in negative_freq.most_common(20):
    print(f"{word}: {freq}")

```

    Top Positive Keywords:
    ('for', 'the', 'price'): 16
    ('as', 'a', 'gift'): 16
    ('i', 'bought', 'this'): 12
    ('this', 'for', 'my'): 12
    ('the', 'sound', 'is'): 11
    ('easy', 'to', 'use'): 11
    ('this', 'as', 'a'): 11
    ('a', 'gift', 'for'): 11
    ('gift', 'for', 'my'): 10
    ('for', 'my', 'husband'): 10
    ('this', 'was', 'a'): 9
    ('bought', 'this', 'for'): 9
    ('in', 'the', 'kitchen'): 9
    ('i', 'use', 'it'): 8
    ('the', 'sound', 'quality'): 8
    ('he', 'loves', 'it'): 7
    ('bought', 'this', 'as'): 7
    ('this', 'is', 'a'): 6
    ('use', 'it', 'for'): 6
    ('sound', 'quality', 'is'): 6
    
    Top Negative Keywords:
    ('waste', 'of', 'money'): 9
    ('stopped', 'working', 'after'): 9
    ('it', 'stopped', 'working'): 7
    ('worth', 'the', 'money'): 6
    ('did', 'not', 'work'): 6
    ('it', 'doesn', 't'): 6
    ('doesn', 't', 'work'): 6
    ('not', 'worth', 'it'): 6
    ('working', 'after', 'months'): 6
    ('stars', 'stopped', 'working'): 6
    ('not', 'worth', 'the'): 5
    ('to', 'my', 'phone'): 5
    ('of', 'money', 'stars'): 5
    ('to', 'use', 'it'): 5
    ('will', 'not', 'pair'): 5
    ('cell', 'phone', 'stand'): 4
    ('i', 'had', 'to'): 4
    ('bought', 'this', 'for'): 4
    ('this', 'for', 'my'): 4
    ('it', 'worked', 'great'): 4

