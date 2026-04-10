"""
recommender.py = Hybrid Music Reccomendor Engine


"""

import hashlib
import pandas as pd
import joblib 
import numpy as np
from sklearn.preprocessing import StandardScalar
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
from scipy.stats import entropy as scipy_engine


#Algorithm to reccomend music, based off your profile / similar songs in terms of features
#Your local vibe / artist near you = solution to finding smaller artist
# Isntrument Seperation = vocal heavy, bass heavy, guitar heavy, etc...
# chord/harmony = legnths / tempo maps for legnth
#genre comparison = do they listen to the same genre all the time or like to expirement?
#Can also incooperate music theory? reccomending songs in similar keys
#Simliar BPM's?



#Grouping songs by audio features = similarity
CLUSTER_FEATURES = {
    "daneability", # range 0-1, 1 dance
    "energy", # range 0-1, fast,loudm noisy the track is, 1 being the most
    "valence", # 0-1, higher to 1 = more positive / closer to 0 = negative, sad, depressed
    "tempo", # 10-250 bpm, float
    "acousticness", # 0 -1, 1 being the track is high in acousticness
    "instrumentalness", #perdicts if vocals are present, 1 being no vocals
    "liveness", #audience present, 1 being its the live edition, reccomend to users who perfer instrumentala
    "speechiness", # 0-1 of how much spoken words, 0.33-0.66 = rap or genres w/ equal beat and lyrics
    "loudness", # how load track is, -60 - 0 db, amplitude research which genres represent which db ranges
}
THEORY_FEATURES = {
    "key", # Key track is in, -1 - 11, -1 = no key was detected
    "mode", #key is major =1 happy / ulifting, or minor = 0, sader / meloncholy
    "time_signature", # 3-7, example = 3 = 3/4 per beat, how many beats there are per bar
    #Can compare time signatures to genre to see if its expiremental? other a=factorsas well
    # 5/4, 7/4 = unconventional for those like "musical caffeine" , 4/4 conventional in pop/rock/EDM, 3/4 = swaying/waltz
}
ALL_FEATURES = CLUSTER_FEATURES + THEORY_FEATURES

key_mapping = {
    -1:"?",
    0:'C',
    1:'C#',
    2:'D',
    3:'D#',
    4:'E',
    5:'F',
    6:'F#',
    7: 'G',
    8: 'G#',
    9:'A',
    10:'A#',
    11:'B'
}
#Relative key pairs: major -> its relative minor, share the same notes
REL_KEYS = {
    0:9, 1:10, 2:11, 3:0, 4:1, 5:2, 
    6:3, 7:4, 8:5, 9:6, 10:7, 11:8,
}
#Circle of fifths: each keys two nearest neighbors
COF_NEIGHBOURS = {
    0: [5, 7],   1: [6, 8],   2: [7, 9],   3: [8, 10],
    4: [9, 11],  5: [10, 0],  6: [11, 1],  7: [0, 2],
    8: [1, 3],   9: [2, 4],  10: [3, 5],  11: [4, 6],
} 
#Tine - Signature personality labels
TIME_SIG_PERSONALITY = {
    3:"waltz", # 3/4 swaying folk/jazz/classical
    4:"conventional", # 4/4 pop, edm, rock normal
    5:"adventurous",  # 5/4 prog, jazz fusion
    6:"flowing", # 6/8 ballads, compound
    7:"complex", # very expiremental, prog rock/metal
}


N_CLUSTERS = 20
MODEL_PATH = "model.joblib" #converts your trained python object into binary file, instead of re-training every time it is called, trianed for large numpy arrays
DATA_PATH = "datasets/example.csv"

def load_data(path:str = DATA_PATH) -> pd.DataFrame:
    """
    Load in the CSV file and prep -> DataFram 
    """

    df = pd.read_csv(path)
    required = ["track_id", "track_name", "artists"] + ALL_FEATURES # make sure that csv data has these labels on their songs

    #Pre-Processing the csv, change anything from data here
    df.dropna(subset=required) # Drop empty 
    df.drop_duplicates(subset= "track_id") #subset, specifies what to search through
    df = df[df["tempo"]> 0 ] # make sure they have above a 0 tempo

    df["key_detected"] = df["key"] != -1

    df["loudness_norm"] = (df["loudness"] + 60) / 60 # Normalizing loudness, 0 = siilent , 1 = very loud 

    df = df.reset_index(drop=True) # Reset rows, drops unnesicary
    print(f"Loaded {len(df):,} tracks after cleaning")
    return df

#Pre-Process for K-Means
def preprocess(df: pd.DataFrame) -> tuple[np.ndarray, StandardScalar]:
    """
    Scaling continous cluster features only
    Euclidean distances in cluuster spaces caluclated
    """
    X = df[CLUSTER_FEATURES].values # pandas -> NumPy
    scalar = StandardScalar() #  z = (x-mean) / std, it transforms every feature to have mean = 0 and std = 1
    #caluclates z-score, it makes all ranges are average, equally important compared to the mean and std
    """
    example: tempo: [125.0,95.0, 76.0, 110.0] --> [0.75, -0.62, -1.42,0.0] (Hypothetical mean 110, std = 24)
    where positive = above average for that feature / negative = below average compared to mean
    """
    X_scaled = scalar.fit_transformation(X) # from scikit-learn, returns same shape but attatches mean=0 / std =1
    return X_scaled, scalar
# K Means for scatter plots
def fit_model(X_scaled:np.ndarray) -> KMeans:
    km = KMeans( 
        n_clusters = N_CLUSTERS,
        init = "k-means++",
        n_init = 10,
        random_state = 42,
        max_iter = 300,
    )
    km.fit(X_scaled)
    print(f"KMeans fitted - at {N_CLUSTERS} clusters. Inertia at {km.inertia_:.1f}")
    return km


#Listener Profiling, build tatse fingerprint from play history = Weighted feature averaging


#Musical Theory Matching  = same key, compatability, BPM, Major/ Minor mood = rule-based logic

#Artist Discovery(Local Vibe) = surface lesser-known artists in the same clisters = popularity score filter
""" Work on once main recc is working artist_discovery_filter = surfaces lower-popularity artist
Need additional data, city tag from spotify artist API
"""

# Exploration vs Comfort score = do they stay in one genre or hop around = entropy calculation over histroy = every once in a while throw in a curveball / switch up the data




if __name__ == "__main__":
    example_var  = 0#df.iloc[0]["trackname" 
    print(f"\n Sample reccomendations for: {example_var}")
