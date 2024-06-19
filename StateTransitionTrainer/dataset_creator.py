import numpy
import pandas as pd
from adlr_environments.constants import DATASET_PATH, TRAININGS_DATA_PATH, OPTIONS
import numpy as np
import os
from train import environment_creation


def create_dataset(transition_number: int = 10):
    """
    Sample State transitions from the dynamic world
    """
    options = OPTIONS
    env = environment_creation(num_workers=1, options=options, vector_environment=False)
    observation, info = env.reset(seed=42)
    transition_data = []
    for i in range(transition_number):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        transition_data.append(np.hstack((observation, terminated)))
    df_transitions = pd.DataFrame(transition_data)
    csv_path = os.path.join(os.pardir, DATASET_PATH)
    df_transitions.to_csv(csv_path, index=False, header=False)
    print(str(transition_number) + " State Transitions were created in " + DATASET_PATH)


def create_samples():
    """
    Clean the collected State transitions and create samples that contain the data and the label
    """
    data_directory = os.path.join(os.pardir, TRAININGS_DATA_PATH, "data/")
    label_directory = os.path.join(os.pardir, TRAININGS_DATA_PATH, "label/")

    os.makedirs(data_directory, exist_ok=True)
    os.makedirs(label_directory, exist_ok=True)

    # read data
    csv_path = os.path.join(os.pardir, DATASET_PATH)
    df_state_transitions = pd.read_csv(csv_path)

    sample_list = []
    sample_counter = 0
    for i in range(len(df_state_transitions)-3):
        # add array to list
        # if list has 3 elements without state transition everything is fine
        # if sample has a state transition, doesnt save list
        for l in range(i, i + 3):
            if np.array(df_state_transitions.loc[l])[-1] == 1:
                sample_list = []
                continue
            else:
                sample_list.append(np.array(df_state_transitions.loc[l][0:-1]))
        if len(sample_list) < 3:
            sample_list = []
        else:
            data = np.array(sample_list)
            label = np.array(df_state_transitions.loc[i + 3][0:-1])
            data_path = os.path.join(data_directory, f"data{sample_counter}.npy")
            label_path = os.path.join(label_directory, f"label{sample_counter}.npy")
            numpy.save(file=data_path, arr=data)
            numpy.save(file=label_path, arr=label)
            print(f"sample counter{sample_counter}")
            sample_counter += 1
            sample_list = []
    print(str(sample_counter) + " Samples were created in " + TRAININGS_DATA_PATH)


if __name__ == "__main__":
    create_dataset(transition_number=10)
    create_samples()
