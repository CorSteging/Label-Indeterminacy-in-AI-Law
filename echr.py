from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
import glob,re, os, sys, random
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, matthews_corrcoef, f1_score
from nltk.corpus import stopwords
from random import shuffle
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import pickle
import ast 
import pandas as pd
import numpy as np
import re
import math



def generate_balanced_subsets(df, label_col='violation', id_col='unique_id', n=7, random_seed=42):
    '''
    Generate n balanced subsets, each with:
    - All non-violation cases
    - Exactly as many violation cases (same count), 
      using disjoint sets when possible, and filling with sampled tail otherwise.
    '''

    np.random.seed(random_seed)
    
    # Separate the data
    df_non_violation = df[df[label_col] == 0]
    df_violation = df[df[label_col] == 1]

    N = len(df_non_violation)
    V = len(df_violation)

    # Shuffle violations
    violations_shuffled = df_violation.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    if V >= n * N:
        # Case A: enough violations for n disjoint full-sized groups
        size_per_group = N
        tail_sample_size = 0
    else:
        # Case B: not enough violations
        size_per_group = V // n
        tail_sample_size = N - size_per_group

    # First part: disjoint violation groups
    violation_groups = []
    for i in range(n):
        start_idx = i * size_per_group
        end_idx = start_idx + size_per_group
        violation_groups.append(violations_shuffled.iloc[start_idx:end_idx])

    # Add tail samples if needed
    if tail_sample_size > 0:
        tail_samples = df_violation.sample(n=n * tail_sample_size, replace=True, random_state=random_seed + 99).reset_index(drop=True)
        for i in range(n):
            tail = tail_samples.iloc[i * tail_sample_size : (i + 1) * tail_sample_size]
            violation_groups[i] = pd.concat([violation_groups[i], tail], ignore_index=True)

    # Build final balanced subsets
    balanced_subsets = []
    for i in range(n):
        subset = pd.concat([df_non_violation, violation_groups[i]], ignore_index=True)
        subset = subset.sample(frac=1, random_state=random_seed + i).reset_index(drop=True)
        balanced_subsets.append(subset)

    return balanced_subsets

def balance_by_downsampling(df, label_column='violation', majority_label=1, seed=42):
    """
    Downsamples the majority class (violation=1) to match the size of the minority class (violation=0),
    yielding a 50/50 split.
    
    Args:
        df (pd.DataFrame): Input dataframe containing a `violation` column with values 0 or 1.
        label_column (str): Name of the binary label column. Defaults to 'violation'.
        majority_label (int): The label value of the majority class. Defaults to 1.
        seed (int): Random seed for reproducibility.
        
    Returns:
        pd.DataFrame: A balanced dataframe with equal numbers of 0s and 1s.
    """
    # Separate minority and majority
    maj_df = df[df[label_column] == majority_label]
    min_df = df[df[label_column] != majority_label]
    
    # Downsample majority to match minority count
    maj_down = maj_df.sample(n=len(min_df), random_state=seed)
    
    # Concatenate and reshuffle
    balanced = pd.concat([min_df, maj_down], axis=0)
    balanced = balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return balanced


def create_dataset(path, article, part, body=None):
    if 'json' in path:
        return create_dataset_echrod(path, article, part, body)
    else:
        return create_dataset_med(path, article, part)
    
def json_to_text(doc, part='facts'):
    '''
    Extracts relevant parts of the case text from a json case
    :param doc: list of dictionaries (json format) representing the content of a case
    :param part: the relevant part to be extracted
    :return: string, the relevant text from the case    
    '''
    part_dict = {
        'procedure': 0,
        'facts': 1,
        'law': 2,
        'conclusion': 3,
    }
    if len(doc)-1 < part_dict[part]:
        return None
    doc = doc[part_dict[part]]
    
    def json_to_text_(doc):
        res = []
        if not len(doc['elements']):  # Remove this condition to add subsection titles 
            res.append(doc['content'])
        for e in doc['elements']:
            res.extend(json_to_text_(e))
        return res
    return '\n'.join(json_to_text_(doc))

def map_originating_body(name):
    if 'Section Committee' in name or 'Committee' in name or name == 'Plenary':
        return 'Misc'
    elif 'Section' in name or name == 'Chamber':
        return 'Chamber'
    elif name == 'Grand Chamber':
        return 'Grand Chamber'
    else:
        return 'Misc'

def create_dataset_echrod(path, article, part, body=None):
    '''
    Returns the desired text and labels from a json file.
    :param path: path to the JSON file
    :param article: string, the relevant article
    :param part: string, the relevant parts to use
    :param body: string or None, one of ['Chamber', 'Grand Chamber', 'Misc'], or None to include all
    :return: pd.DataFrame with columns ['id', 'text', 'year', 'violation']
    '''

    
    if article == 'All': 
        return return_all_cases(path, part)
    if article == 'multi':
        return create_multilabel_dataset(path, part)

    doc = pd.read_json(path)

    # Map and optionally filter by originating body
    doc['mapped_body'] = doc['originatingbody_name'].apply(map_originating_body)
    if body is not None:
        doc = doc[doc['mapped_body'] == body]

    # Add judgement info
    doc['full_text'] = doc['content'].apply(extract_full_text)
    doc['judgment_info'] = doc['full_text'].apply(extract_final_judgment)

    X, y, years, case_ids, judgement_info = [], [], [], [], []

    relevant_cases = doc[doc['article'].apply(lambda arts: article in arts)]

    for _, case in relevant_cases.iterrows():
        for conclusions in case['conclusion']:
            if 'base_article' in conclusions and conclusions['base_article'] == article:
                label = conclusions['type']
                if '+' in part:
                    text = ''
                    for p in part.split('+'):
                        t = json_to_text(list(case['content'].values())[0], p)
                        if isinstance(t, str):
                            text += t
                else:
                    text = json_to_text(list(case['content'].values())[0], part)

                if text:
                    case_ids.append(case['itemid'])
                    judgement_info.append(case['judgment_info'])
                    y.append(1 if label == 'violation' else 0)
                    X.append(rmc(text))
                    years.append(int(case['judgementdate'].split('/')[2].split(' ')[0]))

    return drop_empty(pd.DataFrame({
        'unique_id': list(range(0, len(case_ids))),
        'id': case_ids,
        'body': body,
        'text': X,
        'year': years,
        'violation': y, 
        'sample_weight': 1.0,
        'judgment_info': judgement_info,
    }))

def drop_empty(df):
    df = df.dropna(subset=["text"])
    df = df[~df["text"].str.strip().eq("")]
    return df

    
def get_article_index():
    return {'2':0, '3':1, '5':2, '6':3, '8':4, '10':5, '11':6, '13':7, '14':8}

def create_multilabel_dataset(path, part):
    doc = pd.read_json(path)
    X = []
    y = []
    years = []

    article_numbers = ['2', '3', '5', '6', '8', '10', '11', '13', '14']
    
    # Iterate through all relevant cases
    for idx, case in doc.iterrows():
        labels = {}
        for conclusion in case['conclusion']: # A case has a conclusion for each article
            if 'base_article' in conclusion.keys() and conclusion['base_article'] in article_numbers:
                article = conclusion['base_article']
                label = conclusion['type']
            labels[article] = label
        if '+' in part:
            text = ''
            for p in part.split('+'):
                t = json_to_text(list(case['content'].values())[0], p)
                if isinstance(t, str):
                    text += t
        else:
            text = json_to_text(list(case['content'].values())[0], part) # Extract the relevant parts (text) from the case
        if text:
            y.append(labels)
            X.append(rmc(text))
            years.append(int(case['judgementdate'].split('/')[2].split(' ')[0]))
    
    def conv_y(lbl):
        y = '000000000'
        for key, val in lbl.items():
            if val == 'violation':
                y = y[:get_article_index()[key]] + '1' + y[get_article_index()[key]+1:]
        return y
    
    y = [conv_y(lbl) for lbl in y]
    return pd.DataFrame({
        'unique_id': list(range(0, len(case_ids))),
        'text': X,
        'year': years,
        'violation': y
    })

def return_all_cases(path, part):
    doc = pd.read_json(path)
    X = []
    y = []
    years = []
    case_ids = []

    article_numbers = ['2', '3', '5', '6', '8', '10', '11', '13', '14']
    
    # Iterate through all relevant cases
    for idx, case in doc.iterrows():
        label = 0
        for conclusion in case['conclusion']: # A case has a conclusion for each article
            if 'base_article' in conclusion.keys() and conclusion['base_article'] in article_numbers:
                if conclusion['type'] == 'violation':
                    label = 1
        if '+' in part:
            text = ''
            for p in part.split('+'):
                t = json_to_text(list(case['content'].values())[0], p)
                if isinstance(t, str):
                    text += t
        else:
            text = json_to_text(list(case['content'].values())[0], part) # Extract the relevant parts (text) from the case
        if text:
            y.append(label)
            X.append(rmc(text))
            case_ids.append(case['itemid'])
            years.append(int(case['judgementdate'].split('/')[2].split(' ')[0]))
    
    return pd.DataFrame({
        'unique_id': list(range(0, len(case_ids))),
        'id': case_ids,
        'text': X,
        'year': years,
        'violation': y
    })

def rmc(text):
    '''
    Removes unnecessary characters if needed
    '''
    remove_words = ['\n', 'THE FACTS', 'THE CIRCUMSTANCES OF THE CASE', 'I.',  '\xa0', '\t', '•'] # Note that these are applied in order

    text = re.sub(r'\n\d\.', '>', text)
 # Removes numbering of facts and turns the numbers into into >
    text = text.replace('\n', ' ') # Removes \n marks and replaces them with white spaces
    for word in remove_words:
        text = text.replace(word, '')
    text = " ".join(text.split()) # Removes additional white spaces
    return text

def create_dataset_med(path, article, part):
    if article != 'All':
        article = 'Article'+article
    v = extract_parts(path+'train/'+article+'/violation/*.txt', 'violation', part)
    nv = extract_parts(path+'train/'+article+'/non-violation/*.txt', 'non-violation', part)
    
    df = pd.DataFrame([{'text': rmc(c[0]), 'year' : c[2], 'violation': 1} for c in v] + 
                      [{'text': rmc(c[0]), 'year' : c[2], 'violation': 0} for c in nv])
    return df

def balance_dataset(df, label='violation'):
    if df['violation'].mean() < 0.5: # too many non-violation cases
        new_df = df[df['violation']==1]
        nv_df = df[df['violation']==0].sample(n=len(new_df))
        new_df = pd.concat([new_df, nv_df])
    else: # Too may violation cases
        new_df = df[df['violation']==0]
        v_df = df[df['violation']==1].sample(n=len(new_df))
        new_df = pd.concat([new_df, v_df])
    return new_df


# Functions
def extract_text(starts, ends, cases, violation):
    facts = []
    D = []
    years = []
    for case in cases:
        contline = ''
        year = 0
        with open(case, encoding="utf8") as f:
            for line in f:
                dat = re.search(r'^([0-9]{1,2}\s\w+\s([0-9]{4}))', line)
                if dat != None:
                    year = int(dat.group(2))
                    break
            if year>0:
                years.append(year)
                wr = 0
                for line in f:
                    if wr == 0:
                        if re.search(starts, line) != None:
                            wr = 1
                    if wr == 1 and re.search(ends, line) == None:
                        contline += line
                        contline += '\n'
                    elif re.search(ends, line) != None:
                        break
                facts.append(contline)
    for i in range(len(facts)):
        D.append((facts[i], violation, years[i])) 
    return D

def extract_parts(train_path, violation, part): #extract text from different parts
    cases = glob.glob(train_path)

    facts = []
    D = []
    years = []
    
    if part == 'relevant_law': #seprarte extraction for relevant law
        for case in cases:
            year = 0
            contline = ''
            with open(case, 'r') as f:
                for line in f:
                    dat = re.search(r'^([0-9]{1,2}\s\w+\s([0-9]{4}))', line)
                    if dat != None:
                        year = int(dat.group(2))
                        break
                if year> 0:
                    years.append(year)
                    wr = 0
                    for line in f:
                        if wr == 0:
                            if re.search('RELEVANT', line) != None:
                                wr = 1
                        if wr == 1 and re.search('THE LAW', line) == None and re.search('PROCEEDINGS', line) == None:
                            contline += line
                            contline += '\n'
                        elif re.search('THE LAW', line) != None or re.search('PROCEEDINGS', line) != None:
                            break
                    facts.append(contline)
        for i in range(len(facts)):
            D.append((facts[i], violation, years[i]))
        
    if part == 'facts':
        starts = 'THE FACTS'
        ends ='THE LAW'
        D = extract_text(starts, ends, cases, violation)
    if part == 'circumstances':
        starts = 'CIRCUMSTANCES'
        ends ='RELEVANT'
        D = extract_text(starts, ends, cases, violation)
    if part == 'procedure':
        starts = 'PROCEDURE'
        ends ='THE FACTS'
        D = extract_text(starts, ends, cases, violation)
    if part == 'procedure+facts':
        starts = 'PROCEDURE'
        ends ='THE LAW'
        D = extract_text(starts, ends, cases, violation)
    return D

### Functions for running individual articles
def train_model_cross_val(Xtrain, Ytrain, vec, clf, debug=False, cv=10, n_jobs=-1): #Linear SVC model cross-validation
    if debug: print('***10-fold cross-validation***')
    pipeline = Pipeline([
        ('features', FeatureUnion(
            [vec],
        )),
        ('classifier', clf)
        ])
    Ypredict = cross_val_predict(pipeline, Xtrain, Ytrain, cv=cv, n_jobs=n_jobs) #10-fold cross-validation
    # return evaluate(Ytrain, Ypredict, debug=debug)
    return return_metrics(Ytrain, Ypredict)

def evaluate(Ytest, Ypredict, debug=False): #evaluate the model (accuracy, precision, recall, f-score, confusion matrix)
        acc = accuracy_score(Ytest, Ypredict)
        if debug:
            print('Accuracy:', acc)
            print('\nClassification report:\n', classification_report(Ytest, Ypredict))
            print('\nCR:', precision_recall_fscore_support(Ytest, Ypredict, average='macro'))
            print('\nConfusion matrix:\n', confusion_matrix(Ytest, Ypredict), '\n\n_______________________\n\n')
        return acc

def extract_full_text(case_dict):
    def collect_content(elements):
        texts = []
        for elem in elements:
            if isinstance(elem, dict):
                if 'content' in elem:
                    texts.append(elem['content'])
                if 'elements' in elem:
                    texts.extend(collect_content(elem['elements']))
        return texts

    # Assuming only one key in the case_dict (like '001-116027.docx')
    all_sections = list(case_dict.values())[0]
    return "\n".join(collect_content(all_sections))

def extract_final_judgment(text):
    # Normalize
    text = text.upper().replace("\xa0", " ")

    # Extract from "FOR THESE REASONS" until end
    match = re.search(r"(FOR THESE REASONS,? THE COURT[\s\S]+?)(DONE IN|\Z)", text)
    if not match:
        return None
    judgment_block = match.group(1)
    return judgment_block

    
