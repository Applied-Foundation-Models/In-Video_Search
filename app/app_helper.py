import pickle


# Function to return the file mapping
def get_file_mapping():
    return {
        "Biology 1": "bio_3_3_th5.pickle",
        "Biology 3": "biology3.pkl",
        "ML - 1": "ml1.pkl",
    }


# Function to load a pickle file
def load_pickle(file_name):
    with open(file_name, "rb") as file:
        data = pickle.load(file)
    return data
