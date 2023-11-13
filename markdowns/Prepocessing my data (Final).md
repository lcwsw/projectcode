## Import Library


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
nltk.download()
```

    showing info https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml





    True




```python
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Assuming 'Amazon_Product_Review_B08GC1G4Y9.csv' is your actual file path
file_path = 'Amazon_Product_Review_B08GC1G4Y9.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load stopwords
stop_words = set(stopwords.words('english'))

# Define a more lenient preprocessing function
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Remove punctuation
    tokens = [token.lower() for token in tokens if token.lower() not in string.punctuation]

    # Reassemble the text
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

# Apply preprocessing to the 'Body' column
df['Preprocessed_Body'] = df['Body with star'].apply(preprocess_text)

# Save the preprocessed data to a new CSV file
df.to_csv('preprocessed_data_veryless.csv', index=False)

```

    [nltk_data] Downloading package punkt to /Users/ching/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package stopwords to /Users/ching/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!

