from sklearn.utils import compute_sample_weight
import pandas as pd

def get_weights(path):
    df = pd.read_csv(path)
    print("loading weights", len(df))
    return compute_sample_weight("balanced", df.sort_values('filename')["score"])