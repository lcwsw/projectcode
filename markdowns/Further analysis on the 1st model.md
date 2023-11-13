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
file_path = "1_sentiment_analysis_results_processed_less.csv"
df = pd.read_csv(file_path)

# Filter positive and negative reviews
positive_reviews = df[df['Sentiment_Label'] > 2]['Preprocessed_Body']
negative_reviews = df[df['Sentiment_Label'] < 2]['Preprocessed_Body']

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
    stars: 200
    great: 109
    phone: 100
    good: 83
    sound: 82
    speaker: 70
    gift: 64
    use: 48
    works: 46
    stand: 41
    
    Top Negative Keywords:
    stars: 137
    phone: 53
    sound: 43
    speaker: 37
    work: 30
    working: 30
    would: 26
    money: 24
    stopped: 24
    use: 22



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
    ('stars', 'good'): 22
    ('sound', 'quality'): 18
    ('good', 'sound'): 15
    ('works', 'great'): 13
    ('great', 'sound'): 13
    ('stars', 'works'): 12
    ('works', 'well'): 12
    ('easy', 'use'): 11
    ('battery', 'life'): 11
    
    Top Negative Keywords:
    ('stopped', 'working'): 19
    ('money', 'stars'): 11
    ('waste', 'money'): 11
    ('sound', 'quality'): 8
    ('phone', 'stand'): 7
    ('stars', 'stopped'): 7
    ('stars', 'broke'): 7
    ('stars', 'sound'): 6
    ('working', 'months'): 6
    ('stars', 'work'): 5



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
for word, freq in positive_freq.most_common(10):
    print(f"{word}: {freq}")

print("\nTop Negative Keywords:")
for word, freq in negative_freq.most_common(10):
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
    ('nice', 'gift', 'stars'): 3
    ('cell', 'phone', 'stand'): 3
    ('father', 'day', 'gift'): 3
    
    Top Negative Keywords:
    ('waste', 'money', 'stars'): 7
    ('stars', 'stopped', 'working'): 7
    ('stopped', 'working', 'months'): 6
    ('worth', 'money', 'stars'): 3
    ('cell', 'phone', 'stand'): 3
    ('speaker', 'stopped', 'working'): 3
    ('poor', 'sound', 'quality'): 3
    ('loud', 'enough', 'stars'): 2
    ('stars', 'good', 'good'): 2
    ('phone', 'stars', 'sound'): 2


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
    ('as', 'a', 'gift'): 14
    ('this', 'for', 'my'): 12
    ('easy', 'to', 'use'): 11
    ('i', 'bought', 'this'): 11
    ('the', 'sound', 'is'): 10
    ('for', 'my', 'husband'): 10
    ('this', 'was', 'a'): 9
    ('gift', 'for', 'my'): 9
    ('bought', 'this', 'for'): 9
    ('in', 'the', 'kitchen'): 9
    ('this', 'as', 'a'): 9
    ('a', 'gift', 'for'): 9
    ('the', 'sound', 'quality'): 8
    ('this', 'is', 'a'): 7
    ('i', 'use', 'it'): 7
    ('he', 'loves', 'it'): 7
    ('use', 'it', 'for'): 6
    ('stars', 'great', 'for'): 6
    ('sound', 'quality', 'is'): 6
    
    Top Negative Keywords:
    ('waste', 'of', 'money'): 9
    ('stopped', 'working', 'after'): 9
    ('it', 'stopped', 'working'): 7
    ('doesn', 't', 'work'): 6
    ('not', 'worth', 'it'): 6
    ('working', 'after', 'months'): 6
    ('stars', 'stopped', 'working'): 6
    ('it', 'doesn', 't'): 5
    ('worth', 'the', 'money'): 5
    ('of', 'money', 'stars'): 5
    ('to', 'use', 'it'): 5
    ('will', 'not', 'pair'): 5
    ('not', 'worth', 'the'): 4
    ('to', 'listen', 'to'): 4
    ('it', 'would', 'be'): 4
    ('as', 'loud', 'as'): 4
    ('to', 'return', 'it'): 4
    ('not', 'pair', 'with'): 4
    ('pair', 'with', 'my'): 4
    ('i', 'have', 'no'): 4

