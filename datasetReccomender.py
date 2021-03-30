import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel


datasets = pd.read_csv('Final/datasetsfullplatform1.csv', error_bad_lines=False, low_memory=False, warn_bad_lines=False)
print("read data")

tfv = TfidfVectorizer(min_df=3, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english',max_features=2500)


tfv_matrix = tfv.fit_transform(datasets['title'].values.astype('U'))
# Compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

# Reverse mapping of indices and movie titles
indices = pd.Series(datasets.index, index=datasets['title']).drop_duplicates()


def give_rec(title, sig=sig):
    # Get the index corresponding to title
    idx = indices[title]

    # Get the pairwsie similarity scores 
    sig_scores = list(enumerate(sig[idx]))

    # Sort the datasets 
    sig_scores = sorted(sig_scores, key=lambda x: x[0], reverse=True)

    # Scores of the 10 most similar docs
    sig_scores = sig_scores[1:11]

    # datasets indices
    dataset_indices = [i[0] for i in sig_scores]

    # Top 10 most similar
    return datasets.iloc[dataset_indices]



recommendations = give_rec('Geo-database of MNT, MAT, and Shrub Sites in Northern Alaska (based on MSS Landcover) [Jia, J., H. Epstein, D. Walker]')
print(recommendations)
recommendations.to_csv('results.csv')

recommendations = give_rec('ACE-1: Aircraft C-130 Flight Line Parameters')
print(recommendations)
recommendations.to_csv('results2.csv')

recommendations = give_rec('NCAR/NSF C-130 Navigation, State Parameter, and Microphysics HRT Data - 25 Hz [EOL]')
print(recommendations)
recommendations.to_csv('results3.csv')