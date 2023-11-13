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
file_path = "3_lxyuan_distilbert_processed_less.csv"
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
    stars: 235
    phone: 111
    great: 105
    good: 96
    sound: 94
    speaker: 77
    gift: 68
    use: 52
    works: 46
    product: 41
    
    Top Negative Keywords:
    stars: 123
    phone: 58
    speaker: 44
    sound: 40
    would: 29
    great: 27
    work: 26
    stand: 25
    use: 24
    product: 22



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
    ('stars', 'great'): 33
    ('stars', 'good'): 24
    ('sound', 'quality'): 21
    ('good', 'sound'): 17
    ('works', 'great'): 14
    ('easy', 'use'): 13
    ('phone', 'stand'): 12
    ('stars', 'works'): 12
    ('works', 'well'): 12
    ('great', 'sound'): 12
    
    Top Negative Keywords:
    ('stopped', 'working'): 14
    ('waste', 'money'): 10
    ('money', 'stars'): 9
    ('phone', 'stand'): 8
    ('cell', 'phone'): 7
    ('worth', 'money'): 6
    ('stars', 'work'): 6
    ('sound', 'quality'): 6
    ('stars', 'stopped'): 6
    ('work', 'stars'): 5



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
    ('stars', 'good', 'product'): 5
    ('stars', 'works', 'great'): 5
    ('stars', 'great', 'product'): 5
    ('stars', 'great', 'sound'): 5
    ('stars', 'great', 'gift'): 4
    ('stars', 'good', 'value'): 4
    ('nice', 'gift', 'stars'): 3
    ('cell', 'phone', 'stand'): 3
    ('father', 'day', 'gift'): 3
    ('stars', 'gift', 'gift'): 3
    ('stars', 'husband', 'loves'): 3
    ('stars', 'bought', 'gift'): 3
    ('product', 'would', 'recommend'): 3
    ('stars', 'works', 'well'): 3
    ('stars', 'good', 'sound'): 3
    ('easy', 'use', 'stars'): 3
    ('worth', 'money', 'stars'): 3
    ('good', 'value', 'money'): 3
    ('stars', 'nice', 'little'): 2
    
    Top Negative Keywords:
    ('waste', 'money', 'stars'): 6
    ('stars', 'stopped', 'working'): 6
    ('cell', 'phone', 'stand'): 4
    ('stopped', 'working', 'months'): 4
    ('poor', 'sound', 'quality'): 3
    ('volume', 'waste', 'money'): 2
    ('phone', 'stand', 'speaker'): 2
    ('thin', 'case', 'sure'): 2
    ('case', 'sure', 'would'): 2
    ('stars', 'worth', 'money'): 2
    ('stars', 'disappointed', 'thought'): 2
    ('could', 'barely', 'hear'): 2
    ('speaker', 'takes', 'space'): 2
    ('takes', 'space', 'cell'): 2
    ('latency', 'bluetooth', 'speakers'): 2
    ('type', 'c', 'cable'): 2
    ('unit', 'died', 'next'): 2
    ('died', 'next', 'day'): 2
    ('next', 'day', 'last'): 2
    ('day', 'last', 'day'): 2


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
    ('for', 'the', 'price'): 15
    ('as', 'a', 'gift'): 15
    ('easy', 'to', 'use'): 13
    ('the', 'sound', 'is'): 11
    ('i', 'bought', 'this'): 11
    ('gift', 'for', 'my'): 10
    ('this', 'for', 'my'): 10
    ('this', 'as', 'a'): 10
    ('a', 'gift', 'for'): 10
    ('this', 'was', 'a'): 9
    ('the', 'sound', 'quality'): 9
    ('bought', 'this', 'for'): 9
    ('for', 'my', 'husband'): 9
    ('in', 'the', 'kitchen'): 8
    ('this', 'is', 'a'): 7
    ('i', 'use', 'it'): 7
    ('sound', 'quality', 'is'): 7
    ('he', 'loves', 'it'): 7
    ('bought', 'this', 'as'): 7
    ('stars', 'great', 'for'): 6
    
    Top Negative Keywords:
    ('waste', 'of', 'money'): 7
    ('stopped', 'working', 'after'): 7
    ('this', 'for', 'my'): 6
    ('worth', 'the', 'money'): 6
    ('to', 'use', 'it'): 6
    ('did', 'not', 'work'): 6
    ('not', 'worth', 'it'): 6
    ('it', 'stopped', 'working'): 6
    ('it', 'doesn', 't'): 5
    ('not', 'worth', 'the'): 5
    ('to', 'my', 'phone'): 5
    ('stars', 'stopped', 'working'): 5
    ('cell', 'phone', 'stand'): 4
    ('as', 'loud', 'as'): 4
    ('if', 'you', 'are'): 4
    ('i', 'thought', 'it'): 4
    ('it', 's', 'not'): 4
    ('my', 'phone', 'and'): 4
    ('bought', 'this', 'for'): 4
    ('doesn', 't', 'work'): 4



```python

```
