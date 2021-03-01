import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from typing import Dict

def get_score(y_true:pd.Series, y_pred:pd.DataFrame, groups:pd.Series)->Dict[str, float]:
    result = {}
    
    result['accuracy'] = roc_auc_score(y_true, 
                                       y_pred, 
                                       multi_class='ovo', 
                                       labels=['No-Recidivism','Non-Violent','Violent'])
    
    scores = {}
    for g in groups.unique():
        y_true_g = y_true[groups==g]
        y_pred_g = y_pred[groups==g]
        
        s = roc_auc_score(y_true_g, 
                          y_pred_g, 
                          multi_class='ovo', 
                          labels=['No-Recidivism','Non-Violent','Violent'])
        scores[g]=s
        
    # I add the np.sqrt to make stretch the space beteween 0.9 and 1.0
    result['fairness'] = 1-np.sqrt(pd.Series(scores).std())
    
    return result

y_true = pd.read_csv('test/y_test.csv', squeeze=True)
groups = pd.read_csv('eval_groups.csv', squeeze=True)

sample_submission = pd.read_csv('sample_submission.csv')


#########
#  APP  #
#########

st.sidebar.title('Info')
st.sidebar.write('Data sourced from https://github.com/propublica/compas-analysis')
st.sidebar.write('Competition idea inspired by https://www.kaggle.com/danofer/compass')
st.sidebar.write('This page is managed by mycchaka.kleinbort@capgemini.com')
st.sidebar.write('Sourcecode at https://github.com/mkleinbort/Kaggle-COMPAS')

st.title('Kaggle COMPAS Competition')

if st.checkbox('Info'):
    st.markdown('''### Welcome to the COMPAS competition.

The goal of this competition is to accurately but fairly predict the recidivism of criminal defendants.

More specificaly, the data contains the attributes of 4,743 criminal defendants, and whether in the two years since their release they:
- Remained lawful (`No-Recidivism`)
- Committed a `Non-Violent` offence
- Committed a `Violent` offence

The goal is to train a model on the data in `train/` and use it to predict for each individual in `test/X_test.csv` their probability for each outcome.

### The Data

You can load the training data `X_train`, `y_train` using:

`X_train = pd.read_csv('https://raw.githubusercontent.com/mkleinbort/Kaggle-COMPAS/main/train/X_train.csv')`

`y_train = pd.read_csv('https://raw.githubusercontent.com/mkleinbort/Kaggle-COMPAS/main/train/y_train.csv', squeeze=True)`

similarly, you can load `X_test` with 

`X_test = pd.read_csv('https://raw.githubusercontent.com/mkleinbort/Kaggle-COMPAS/main/test/X_test.csv')`

It is against the competition rules to use `y_test` in any way.

### Scoring

The competition is scored on two equally important axis:
- **Accuracy** (defined as the roc-auc score with multi_class="ovo")
- **Fairness** (defined as the $1-\sqrt{\sigma}$ in the roc-auc score across subsets of the test dataset)

_To elaborate on the fairness score:_

Suppose your model has an AUC of .98 overall, but
- .95 when looking at female defendants
- .99 when looking at male defendants
- .93 when looking at defendants aged 65+
- etc.

Then the standard deviation of these scores is $\sigma$, and the _fairness_ score is $1-\sqrt{\sigma}$

''')

st.markdown('### Making a submission')

st.write('Submit a csv file with three columns.')

if st.checkbox('Show example'):
    st.dataframe(sample_submission)

st.markdown('''Note:
- Each row should correspond to each sample in `test/X_test.csv`.
- The values represent the predicted probability of each class.
''')


file = st.file_uploader('Submission')

if file:
    y_sumission = pd.read_csv(file)
    
if st.button('Check submission format'):
    if y_sumission.shape == (949, 3):
        st.success('Your submission has 943 rows and 3 columns')
    else:
        st.error('Your submission should have 943 rows and 3 columns')
        
    if all(y_sumission.notna()):
        st.success('Your submission has no missing values.')
    else:
        st.error('Your submission has missing values')

if st.button('Score'):
    submission_score = get_score(y_true, y_sumission, groups)
    
    st.write(submission_score)
    
    st.success(f'You scored: {submission_score["accuracy"]:.2%} in accuracy and {submission_score["fairness"]:.2%} in fairness.')
               
    st.success(f'Overall Score: {(submission_score["accuracy"]+submission_score["fairness"])/2:.2%}')
               
               