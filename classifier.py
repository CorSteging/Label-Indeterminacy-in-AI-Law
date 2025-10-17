from transformers import LongformerTokenizer, LongformerForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import torch
import re
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import expit as sigmoid 

def build_training_set(all_datasets, split_id, method='wcorr'):
    '''
    Constructs a training dataset based on various relabeling or weighting strategies.

    method: 
        - 'obs': use only Grand Chamber cases
        - 'obs_ip': Grand Chamber + Inverse Propensity-Weighted Chamber cases
        - 'wcorr': naive observed labels from both sources
        - others: placeholders for future methods
    '''
    chamber_df = all_datasets['train']['chamber'][split_id].copy()
    grand_chamber_df = all_datasets['train']['grand_chamber'][split_id].copy()

    if method == 'wcorr':
        # Use all data as-is
        pass

    elif method == 'obs':
        # Use only Grand Chamber data
        chamber_df = chamber_df.head(0)

    elif method == 'obs_ip':
        # Load grand chamber data with precomputed propensities and inverse propensities
        prop_path = f"datasets/propensities/grand_chamber_with_propensity_split_{split_id}.pkl"
        return pd.read_pickle(prop_path)
        
    elif method == 'nn':
        # Impute the labels using nearest neigbour
        prop_path = f"datasets/nearest_neighbor/chamber_split_{split_id}.pkl"
        chamber_df = pd.read_pickle(prop_path)
        chamber_df['violation'] = chamber_df['label_nn']

    elif method.startswith('mexp'):
        # For this we need the votes, and sometimes also the 
        chamber_df = pd.read_csv(f"datasets/votes/chamber_split_{split_id}.csv")
        chamber_df['votes_for'] = chamber_df['votes_for'].astype(float)

        if method == 'mexpall':
            # If there are dissenting opinions, treat each opinion as a seperate case
            # with weight 1/7
            chamber_df = mexpall_converter(chamber_df)
            
        elif method == 'mexpavg':
            # Average the dissenting votes into a single label 
            chamber_df['violation'] = chamber_df.apply(
                lambda row: row['votes_for'] / 7 if pd.notnull(row['votes_for']) 
                                                 else row['violation'],
                axis=1
            ).astype(float)

        elif method == 'mexpmax':
            # Label violation if at least 1 judge voted for
            chamber_df['violation'] = chamber_df.apply(
                lambda row: 1 if pd.notnull(row['votes_for']) and row['votes_for'] > 0 
                              else row['violation'],
                axis=1
            )
        elif method == 'mexpmin':
            # Label non-violation if at least 1 judge voted against
            chamber_df['violation'] = chamber_df.apply(
                lambda row: 1 if pd.notnull(row['votes_for']) and row['votes_for'] == 7 else (
                    0 if pd.notnull(row['votes_for']) else row['violation']
                ),
                axis=1
            )

        elif method == 'mexpagr':
            # Keep only cases with unanimous agreement
            chamber_df = chamber_df[(chamber_df['votes_for'].isin([0, 7])) | (chamber_df['votes_for'].isna())]

        else:
            raise NotImplementedError(f"Unknown mexp method: {method}")

    else:
        raise NotImplementedError(f"Method {method} is not implemented.")

    # Merge chamber and grand chamber training data
    training_df = pd.concat([chamber_df, grand_chamber_df], ignore_index=True)
    return training_df

def mexpall_converter(df):
    """
    Expands each row into multiple rows based on number of votes.
    - Unanimous cases (0 or 7 votes for violation) are kept as-is.
    - Non-unanimous cases are expanded into 7 rows: 
      'votes_for' with violation=1, and the rest with violation=0.
    Each row gets a sample_weight of 1/7.
    """
    expanded_rows = []
    for _, row in df.iterrows():
        if np.isnan(row['votes_for']): 
            expanded_rows.append(row)
            continue
            
        votes_for = int(row['votes_for'])

        if votes_for == 0 or votes_for == 7:
            expanded_rows.append(row)
        else:
            base_row = row.drop(['violation', 'sample_weight'])
            for _ in range(votes_for):
                r = base_row.copy()
                r['violation'] = 1
                r['sample_weight'] = 1 / 7
                expanded_rows.append(r)
            for _ in range(7 - votes_for):
                r = base_row.copy()
                r['violation'] = 0
                r['sample_weight'] = 1 / 7
                expanded_rows.append(r)

    return pd.DataFrame(expanded_rows)
    
class WeightedTrainer(Trainer):
    def __init__(self, *args, method=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        sample_weights = inputs.pop("sample_weight", None)

        outputs = model(**inputs)
        logits = outputs.logits

        if self.method == "mexpavg":
            logits = logits[:, 1].unsqueeze(1)  # shape: (B, 1)
            labels = labels.float().unsqueeze(1)

            loss_fct = torch.nn.BCEWithLogitsLoss(reduction='none')
        else:
            labels = labels.long()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

        loss = loss_fct(logits, labels)

        if sample_weights is not None:
            sample_weights = sample_weights.to(loss.device).unsqueeze(1 if self.method == "mexpavg" else 0)
            loss = loss * sample_weights

        loss = loss.mean()
        return (loss, outputs) if return_outputs else loss



def impute_nn_labels_from_split(split_id: int, base_path="datasets/balanced/embeddings"):
    """
    Impute 'violation' labels for chamber cases using nearest-neighbor labels from grand chamber cases.

    Parameters:
    - split_id: int, identifier for the data split
    - base_path: str, base directory where the embedding dataframes are stored

    Returns:
    - chamber_df_imputed: pd.DataFrame with imputed 'violation' labels
    - grand_chamber_df: pd.DataFrame (unchanged)
    """

    chamber_df = pd.read_pickle(os.path.join(base_path, f"chamber_split_{split_id}.pkl"))
    grand_chamber_df = pd.read_pickle(os.path.join(base_path, f"grand_chamber_split_{split_id}.pkl"))

    chamber_embeddings = np.vstack(chamber_df['embedding'].values)
    grand_embeddings = np.vstack(grand_chamber_df['embedding'].values)
    grand_labels = grand_chamber_df['violation'].values

    sim_matrix = cosine_similarity(chamber_embeddings, grand_embeddings)
    nn_indices = sim_matrix.argmax(axis=1)
    imputed_labels = grand_labels[nn_indices]

    chamber_df_imputed = chamber_df.copy()
    chamber_df_imputed['violation'] = imputed_labels

    return chamber_df_imputed

    
# Prepare datasets for HuggingFace Trainer
def prepare_dataset(df):
    return Dataset.from_pandas(df)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    if labels.dtype == np.float32 or labels.dtype == np.float64:
        labels = (labels >= 0.5).astype(int)

    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds),
        'mcc': matthews_corrcoef(labels, preds)
    }


def print_metrics(title, prediction_output):
    print(f"\n📊 {title} Results:")
    for key, value in prediction_output.metrics.items():
        if isinstance(value, float):
            print(f"  {key:<20}: {value:.4f}")
        else:
            print(f"  {key:<20}: {value}")

# Handle both (B, 1) and (B, 2) shape logits
def get_class1_probs(logits):
    if logits.shape[1] == 1:
        # Logits for class 1 directly
        return sigmoid(logits[:, 0])
    else:
        # Logits for class 1 in position 1
        return sigmoid(logits[:, 1])

    
def clean_text(text):
    """
    Cleans ECtHR rulings by removing structural noise and unnecessary tokens.
    """
    text = re.sub(r'\n\d+\.', '>', text)
    remove_patterns = ['\n', '\xa0', '\t', '•', 'THE FACTS', 
                       'THE CIRCUMSTANCES OF THE CASE', 'I.']
    for pattern in remove_patterns:
        text = text.replace(pattern, ' ')
    return " ".join(text.split())

    
def head_tail_truncate(text, tokenizer, max_tokens=4094, head_ratio=0.5):
    """
    Truncates text by preserving head and tail of the tokenized input.
    """
    text = clean_text(text)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) <= max_tokens:
        return text
    
    head_len = int(max_tokens * head_ratio)
    tail_len = max_tokens - head_len
    
    head_text = tokenizer.decode(tokens[:head_len], skip_special_tokens=True)
    tail_text = tokenizer.decode(tokens[-tail_len:], skip_special_tokens=True)
    
    return head_text.strip() + " " + tail_text.strip()



def preprocess_and_tokenize(examples, tokenizer):
    processed_texts = [head_tail_truncate(t, tokenizer) for t in examples['text']]
    return tokenizer(
        processed_texts,
        padding='max_length',
        truncation=True,
        max_length=4096,
        return_tensors='pt'
    )
