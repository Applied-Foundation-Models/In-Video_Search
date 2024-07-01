import pickle

import streamlit as st
import torch


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


def load_embeddings(embedding_dict, key):
    # Extract values associated with the key
    values = [
        embedding_dict[k][key] for k in embedding_dict if key in embedding_dict[k]
    ]

    # Check if there are values to concatenate
    if not values:
        raise ValueError(
            f"No values found for key '{key}' in the embedding dictionary."
        )

    # Concatenate values into a single tensor
    concatenated_tensor = torch.cat(values, dim=0)

    return concatenated_tensor


def query_video_data(dict, keyframe, key):
    # Extract values associated with the key
    print(dict)
    print(key)
    print(keyframe)
    try:
        extracted_datapoint = dict[keyframe][key]
    except KeyError:
        raise ValueError(
            f"No values found for key '{key}' in the embedding dictionary."
        )
    return extracted_datapoint


def update_selection():
    selected_result = st.session_state.selected_result
    st.session_state.start_time = query_video_data(
        st.session_state["data"], selected_result, "timestamps"
    )[1]
    st.session_state.keyframe_summary = query_video_data(
        st.session_state["data"], selected_result, "llava_result"
    )
    # Rerun app
    st.rerun()
