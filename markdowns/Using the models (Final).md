# Different Models on preprocessed data


```python
!echo "y" | conda install -c huggingface transformers -y
!pip install huggingface-hub
```

    Retrieving notices: ...working... done
    Channels:
     - huggingface
     - pytorch
     - defaults
    Platform: osx-arm64
    Collecting package metadata (repodata.json): done
    Solving environment: done
    
    # All requested packages already installed.
    
    [33mDEPRECATION: Loading egg at /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages/huggingface_hub-0.19.0-py3.8.egg is deprecated. pip 24.3 will enforce this behaviour change. A possible replacement is to use pip for package installation.. Discussion can be found at https://github.com/pypa/pip/issues/12330[0m[33m
    [0mRequirement already satisfied: huggingface-hub in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages/huggingface_hub-0.19.0-py3.8.egg (0.19.0)
    Requirement already satisfied: filelock in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (from huggingface-hub) (3.9.0)
    Requirement already satisfied: fsspec>=2023.5.0 in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (from huggingface-hub) (2023.9.2)
    Requirement already satisfied: requests in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (from huggingface-hub) (2.31.0)
    Requirement already satisfied: tqdm>=4.42.1 in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (from huggingface-hub) (4.65.0)
    Requirement already satisfied: pyyaml>=5.1 in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (from huggingface-hub) (6.0.1)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (from huggingface-hub) (4.7.1)
    Requirement already satisfied: packaging>=20.9 in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (from huggingface-hub) (23.1)
    Requirement already satisfied: charset-normalizer<4,>=2 in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (from requests->huggingface-hub) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (from requests->huggingface-hub) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (from requests->huggingface-hub) (1.26.18)
    Requirement already satisfied: certifi>=2017.4.17 in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (from requests->huggingface-hub) (2023.7.22)



```python
# Install packages using Conda
!echo "y" |conda install pandas
```

    Channels:
     - pytorch
     - defaults
     - huggingface
    Platform: osx-arm64
    Collecting package metadata (repodata.json): done
    Solving environment: done
    
    # All requested packages already installed.
    



```python
!pip install torch 
!pip install chardet
```

    [33mDEPRECATION: Loading egg at /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages/huggingface_hub-0.19.0-py3.8.egg is deprecated. pip 24.3 will enforce this behaviour change. A possible replacement is to use pip for package installation.. Discussion can be found at https://github.com/pypa/pip/issues/12330[0m[33m
    [0mRequirement already satisfied: torch in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (2.1.0)
    Requirement already satisfied: filelock in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (from torch) (3.9.0)
    Requirement already satisfied: typing-extensions in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (from torch) (4.7.1)
    Requirement already satisfied: sympy in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (from torch) (1.12)
    Requirement already satisfied: networkx in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (from torch) (3.2.1)
    Requirement already satisfied: jinja2 in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (from torch) (3.1.2)
    Requirement already satisfied: fsspec in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (from torch) (2023.9.2)
    Requirement already satisfied: MarkupSafe>=2.0 in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (from jinja2->torch) (2.1.1)
    Requirement already satisfied: mpmath>=0.19 in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (from sympy->torch) (1.3.0)
    [33mDEPRECATION: Loading egg at /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages/huggingface_hub-0.19.0-py3.8.egg is deprecated. pip 24.3 will enforce this behaviour change. A possible replacement is to use pip for package installation.. Discussion can be found at https://github.com/pypa/pip/issues/12330[0m[33m
    [0mRequirement already satisfied: chardet in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (5.2.0)



```python
!pip install openpyxl
```

    [33mDEPRECATION: Loading egg at /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages/huggingface_hub-0.19.0-py3.8.egg is deprecated. pip 24.3 will enforce this behaviour change. A possible replacement is to use pip for package installation.. Discussion can be found at https://github.com/pypa/pip/issues/12330[0m[33m
    [0mRequirement already satisfied: openpyxl in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (3.1.2)
    Requirement already satisfied: et-xmlfile in /Users/ching/miniconda3/envs/myenv/lib/python3.11/site-packages (from openpyxl) (1.1.0)


## 1st LiYuan/amazon-review-sentiment-analysis


```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# Load the pre-trained model and tokenizer
model_name = "LiYuan/amazon-review-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load your preprocessed data from the CSV file
df = pd.read_csv('preprocessed_data_veryless.csv')

# Function to predict sentiment using the model
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    sentiment_score = torch.softmax(logits, dim=1).tolist()[0]  # Get the sentiment score
    return predicted_class, max(sentiment_score)  # Keep only the maximum sentiment score

# Apply the sentiment prediction to the 'Preprocessed_Body' column
df[['Sentiment_Label', 'Max_Sentiment_Score']] = df['Preprocessed_Body'].apply(predict_sentiment).apply(pd.Series)

# Display the DataFrame with the new 'Sentiment_Label' and 'Max_Sentiment_Score' columns
print(df[['Preprocessed_Body', 'Sentiment_Label', 'Max_Sentiment_Score']])

# Save the DataFrame with sentiment labels and scores to a new CSV file
df.to_csv('1_sentiment_analysis_results_processed_less.csv', index=False)

```

                                         Preprocessed_Body  Sentiment_Label  \
    0    5 stars nice little bluetooth box bought for a...              4.0   
    1    5 stars easy to operate .. the product was ama...              4.0   
    2    5 stars love my jteman cell phone holder i sim...              4.0   
    3    5 stars great buy on sale solid easy to use an...              4.0   
    4    5 stars this is a gift he loves listening to m...              4.0   
    ..                                                 ...              ...   
    370  1 stars chinese spy device my wife got me this...              0.0   
    371  1 stars received as a gift dead out of the box...              0.0   
    372  1 stars horrible im beyond late on this review...              0.0   
    373  1 stars defective audio issue i got this as a ...              0.0   
    374  1 stars will not charge i thought this was sup...              0.0   
    
         Max_Sentiment_Score  
    0               0.975006  
    1               0.987549  
    2               0.989188  
    3               0.993481  
    4               0.990091  
    ..                   ...  
    370             0.967510  
    371             0.980290  
    372             0.984328  
    373             0.967772  
    374             0.979149  
    
    [375 rows x 3 columns]


## 2nd nlptown/bert-base-multilingual-uncased-sentiment


```python
import torch
import openpyxl
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# Load the pre-trained model and tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load your preprocessed data from the CSV file
df = pd.read_csv('preprocessed_data_veryless.csv')

# Function to predict sentiment using the model
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    sentiment_score = torch.softmax(logits, dim=1).max().item()  # Get the highest sentiment score
    return predicted_class, sentiment_score

# Apply the sentiment prediction to the 'Preprocessed_Body' column
df[['Sentiment_Label', 'Sentiment_Score']] = df['Preprocessed_Body'].apply(predict_sentiment).apply(pd.Series)

# Display the DataFrame with the new 'Sentiment_Label' and 'Sentiment_Score' columns
print(df[['Preprocessed_Body', 'Sentiment_Label', 'Sentiment_Score']])

# Save the DataFrame with sentiment labels and scores to a new CSV file
df.to_csv('2_bert-base-multilingual-uncased-sentiment_processed_less.csv', index=False)

```

                                         Preprocessed_Body  Sentiment_Label  \
    0    5 stars nice little bluetooth box bought for a...              4.0   
    1    5 stars easy to operate .. the product was ama...              4.0   
    2    5 stars love my jteman cell phone holder i sim...              4.0   
    3    5 stars great buy on sale solid easy to use an...              4.0   
    4    5 stars this is a gift he loves listening to m...              4.0   
    ..                                                 ...              ...   
    370  1 stars chinese spy device my wife got me this...              0.0   
    371  1 stars received as a gift dead out of the box...              0.0   
    372  1 stars horrible im beyond late on this review...              0.0   
    373  1 stars defective audio issue i got this as a ...              0.0   
    374  1 stars will not charge i thought this was sup...              0.0   
    
         Sentiment_Score  
    0           0.889987  
    1           0.973634  
    2           0.985940  
    3           0.991592  
    4           0.985782  
    ..               ...  
    370         0.787746  
    371         0.963464  
    372         0.984079  
    373         0.920723  
    374         0.963809  
    
    [375 rows x 3 columns]


## 3rd lxyuan/distilbert-base-multilingual-cased-sentiments-student


```python
from transformers import pipeline
import pandas as pd

# Load the sentiment analysis pipeline with the specified model
classifier_distilbert = pipeline("sentiment-analysis", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")

# Load your CSV file into a DataFrame (replace 'your_file.csv' with your actual file path)
df = pd.read_csv('preprocessed_data_veryless.csv')

# Create empty lists to store labels and scores
labels = []
scores = []

# Process each review and store the results
for index, row in df.iterrows():
    text = row['Preprocessed_Body']  # Assuming 'Body' is the column containing the reviews
    try:
        # Truncate the text to the maximum sequence length
        truncated_text = text[:512]
        result = classifier_distilbert(truncated_text)[0]  # Assuming you only want the first result if multiple are returned
        labels.append(result['label'])
        scores.append(result['score'])
    except Exception as e:
        print(f"Skipped review at index {index} due to error: {e}")
        labels.append('unknown')
        scores.append('unknown')

# Add new columns to the DataFrame
df['Sentiment_Label'] = labels
df['Sentiment_Score'] = scores

# Save the processed DataFrame to a new CSV file
df.to_csv('3_lxyuan_distilbert_processed_less.csv', index=False)

```

## 4th sohan-ai/sentiment-analysis-model-amazon-reviews


```python
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the fine-tuned model from Hugging Face
model_name = "sohan-ai/sentiment-analysis-model-amazon-reviews"
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# Load your CSV file into a DataFrame (replace 'your_file.csv' with your actual file path)
df = pd.read_csv('preprocessed_data_veryless.csv')

# Create empty lists to store labels and scores
labels = []
scores = []

# Process each review and store the results
for index, row in df.iterrows():
    text = row['Preprocessed_Body']  # Assuming 'Body' is the column containing the reviews
    try:
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        predicted_label = "positive" if outputs.logits.argmax().item() == 1 else "negative"
        labels.append(predicted_label)
        scores.append(outputs.logits.softmax(dim=1).max().item())
    except Exception as e:
        print(f"Skipped review at index {index} due to error: {e}")
        labels.append('unknown')
        scores.append('unknown')

# Add new columns to the DataFrame
df['Sentiment_Label'] = labels
df['Sentiment_Score'] = scores

# Save the processed DataFrame to a new CSV file
df.to_csv('4_sohan_ai_sentiment_analysis_processed_less.csv', index=False)

```

    Token indices sequence length is longer than the specified maximum sequence length for this model (689 > 512). Running this sequence through the model will result in indexing errors


    Skipped review at index 101 due to error: The size of tensor a (689) must match the size of tensor b (512) at non-singleton dimension 1
    Skipped review at index 116 due to error: The size of tensor a (559) must match the size of tensor b (512) at non-singleton dimension 1



```python
import pandas as pd
! echo "y" | conda install -c conda-forge xlsxwriter

# Read CSV files into DataFrames
df1 = pd.read_csv('1_sentiment_analysis_results_processed_less.csv')
df2 = pd.read_csv('2_bert-base-multilingual-uncased-sentiment_processed_less.csv')
df3 = pd.read_csv('3_lxyuan_distilbert_processed_less.csv')
df4 = pd.read_csv('4_sohan_ai_sentiment_analysis_processed_less.csv')

# Create an Excel writer object
with pd.ExcelWriter('4_models_results_veryless.xlsx', engine='xlsxwriter') as writer:
    # Write each DataFrame to a different sheet
    df1.to_excel(writer, sheet_name='Model_1', index=False)
    df2.to_excel(writer, sheet_name='Model_2', index=False)
    df3.to_excel(writer, sheet_name='Model_3', index=False)
    df4.to_excel(writer, sheet_name='Model_4', index=False)

```

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    Channels:
     - conda-forge
     - pytorch
     - defaults
     - huggingface
    Platform: osx-arm64
    Collecting package metadata (repodata.json): done
    Solving environment: done
    
    ## Package Plan ##
    
      environment location: /Users/ching/miniconda3/envs/myenv
    
      added / updated specs:
        - xlsxwriter
    
    
    The following packages will be downloaded:
    
        package                    |            build
        ---------------------------|-----------------
        certifi-2023.7.22          |     pyhd8ed1ab_0         150 KB  conda-forge
        openssl-3.1.4              |       h0d3ecfb_0         2.0 MB  conda-forge
        xlsxwriter-3.1.9           |     pyhd8ed1ab_0         118 KB  conda-forge
        ------------------------------------------------------------
                                               Total:         2.3 MB
    
    The following NEW packages will be INSTALLED:
    
      xlsxwriter         conda-forge/noarch::xlsxwriter-3.1.9-pyhd8ed1ab_0 
    
    The following packages will be UPDATED:
    
      openssl              pkgs/main::openssl-3.0.12-h1a28f6b_0 --> conda-forge::openssl-3.1.4-h0d3ecfb_0 
    
    The following packages will be SUPERSEDED by a higher-priority channel:
    
      certifi            pkgs/main/osx-arm64::certifi-2023.7.2~ --> conda-forge/noarch::certifi-2023.7.22-pyhd8ed1ab_0 
    
    
    Proceed ([y]/n)? 
    
    Downloading and Extracting Packages:
    xlsxwriter-3.1.9     | 118 KB    |                                       |   0% 
    certifi-2023.7.22    | 150 KB    |                                       |   0% [A
    
    openssl-3.1.4        | 2.0 MB    |                                       |   0% [A[A
    certifi-2023.7.22    | 150 KB    | ###9                                  |  11% [A
    
    xlsxwriter-3.1.9     | 118 KB    | #####                                 |  14% [A[A
    xlsxwriter-3.1.9     | 118 KB    | ##################################### | 100% [A
    
    openssl-3.1.4        | 2.0 MB    | #####9                                |  16% [A[A
    
    openssl-3.1.4        | 2.0 MB    | ##################################### | 100% [A[A
    
                                                                                    [A[A
                                                                                    [A
    
                                                                                    [A[A
    Preparing transaction: done
    Verifying transaction: done
    Executing transaction: done

