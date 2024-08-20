import pickle

import streamlit as st
import torch
from loguru import logger
from PIL import Image


# Function to return the file mapping
def get_file_mapping():
    return {
        "Biology 1": {
            "pickle_file": "app/embeddings/biology_chapter_3_3_treshhold_5.pickle",
            "video_url": "data/raw/biology_chapter_3_3_treshhold_5/biology_chapter_3_3_treshhold_5.mp4",
        },
        "Biology 2": {
            "pickle_file": "app/embeddings/ch7_soil_agriculture.pickle",
            "video_url": "data/raw/ch7_soil_agriculture/ch7_soil_agriculture.mp4",
        },
        "Math 1": {
            "pickle_file": "app/embeddings/math_1.pickle",
            "video_url": "data/raw/math_1/math_1.mp4",
        },
    }


# Function to load a pickle file
def load_pickle(file_name):
    with open(file_name, "rb") as file:
        data = pickle.load(file)
    return data


def create_tensor_and_mapping(embedding_dict, key):
    # Initialize a list to store the embeddings and a dictionary to store the mapping
    embeddings = []
    key_to_index_mapping = {}
    # Iterate through the embedding dictionary
    for idx, k in enumerate(sorted(embedding_dict.keys())):
        if key in embedding_dict[k]:
            # Append the embedding to the list
            embeddings.append(embedding_dict[k][key])
            # Store the mapping from the original key to the index
            key_to_index_mapping[k] = idx

    # Check if there are embeddings to concatenate
    if not embeddings:
        raise ValueError(
            f"No values found for key '{key}' in the embedding dictionary."
        )

    # Concatenate the embeddings into a single tensor
    concatenated_tensor = torch.cat(embeddings, dim=0)

    # Calculate the number of embeddings
    k = len(embeddings)

    # Reshape the concatenated tensor to [k, embedding_length]
    reshaped_tensor = concatenated_tensor.view(k, -1)

    return reshaped_tensor, key_to_index_mapping


def query_video_data(dict, keyframe, key):
    # Extract values associated with the key
    try:
        extracted_datapoint = dict[keyframe][key]
    except KeyError:
        raise ValueError(
            f"No values found for key '{key}' in the embedding dictionary."
        )
    return extracted_datapoint


def update_selection():
    selected_result = st.session_state.selected_result
    logger.info("Querying data for selected result")
    st.session_state.start_time = query_video_data(
        st.session_state["data"], selected_result, "timestamps"
    )[1]
    logger.info("Timestamp querying - DONE")
    st.session_state.keyframe_summary = query_video_data(
        st.session_state["data"], selected_result, "llava_result"
    )
    logger.info("Video querying - DONE")
    # Rerun app
    st.rerun()


def display_results(selected_results):
    for i, result in enumerate(selected_results):
        path_to_image = query_video_data(
            st.session_state["data"], result[i], "img_path"
        )
        open_image(path_to_image)


def open_image(path):
    opened_image = Image.open(path)
    return opened_image


def retrieve_keys_from_indices(list_retrieval, key_to_index_mapping):
    # Create a reverse mapping from index to key
    index_to_key_mapping = {v: k for k, v in key_to_index_mapping.items()}

    # Retrieve the keys corresponding to the indices in list_retrieval
    keys_retrieved = [index_to_key_mapping[idx] for idx in list_retrieval]

    return keys_retrieved
