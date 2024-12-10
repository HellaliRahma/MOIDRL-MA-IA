import pandas as pd
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.mixture import GaussianMixture

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input
import psutil
from sklearn.model_selection import train_test_split
import random

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score ,roc_auc_score
from Trainers import Trainer
import time
import csv
from sklearn.preprocessing import LabelBinarizer
from xgboost import XGBClassifier
from sklearn.feature_selection import mutual_info_classif

file_path ='/gpfswork/rech/bvj/uub66jn/MOIDRL-MA-IA-journal-experiments/Experiment1/Input_data/MLL.csv'
# file_path ='/home/rhellali/public/rhellali/MOIDRL-MA-IA-journal-experiments/Experiment1/Input_data/MLL.csv'

# parent_path=os.path.dirname(os.getcwd())
# file_path = parent_path +'/Input_data/MLL.csv'

data = pd.read_csv(file_path, sep=',')

# 2. Encoding Categorical Variables:
class_mapping = {'ALL': 0, 'AML': 1, 'MLL': 2}
data['class'] = data['class'].map(class_mapping)

Y = data['class']
columns_to_exclude = ['class']
X = data.drop(columns=columns_to_exclude)
X.columns = range(len(X.columns))

# Apply SMOTE
smote = SMOTE(random_state=4)
X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X, Y)

# Scale the features
scaler = preprocessing.StandardScaler().fit(X_train_SMOTE)
X_train_SMOTE_scaled_array = scaler.transform(X_train_SMOTE)
X_train_SMOTE_scaled = pd.DataFrame(X_train_SMOTE_scaled_array, columns=X_train_SMOTE.columns)

#Data splitting
X_train, X_test, y_train,  y_test = train_test_split(X_train_SMOTE_scaled, y_train_SMOTE, test_size = 0.3)

#Number of features
F = len(X.columns)
print(F)

# Configuration paramaters for the whole setup
gamma = 0.99 # Discount factor for past rewards
epsilon = 0.99 # Epsilon greedy parameter
batch_size = 64  # Size of batch taken from replay buffer

max_steps_per_episode = 1000
trainers_steps = int(max_steps_per_episode/2)

num_actions = 2
state_size = 49
terminal = False
hyperparameter = 0.001
num_agents = F

# Experience replay buffers
action_history = []
agents = list(range(num_agents))
agent_lists = {agent_id: [] for agent_id in agents}
state_history = []
rewards_list = [[0, 0, 0 , 0, 0, 0, 0, 0, 0]]* max_steps_per_episode
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
# Maximum replay length
max_memory_length = 2000
update_after_actions = 10
# How often to update the target network
update_target_network = 100
# Using huber loss for stability
loss_function = keras.losses.Huber()


from scipy.stats import entropy
from sklearn.preprocessing import LabelEncoder

def calculate_entropy(data):
    """ Calculate the entropy of the data. """
    value, counts = np.unique(data, return_counts=True)
    return entropy(counts, base=2)

def calculate_redundancy(data):
    """ Calculate the redundancy of the dataset. """
    data = pd.Series(data)
    entropy_value = calculate_entropy(data)
    unique_values = len(data.unique())
    if unique_values > 1:
        max_entropy = np.log2(unique_values)
        redundancy = 1 - (entropy_value / max_entropy)
    else:
        redundancy = 0  # No redundancy if there's only one unique value
    return redundancy

def calculate_redundancy_for_dataframe(features1):
    df = X_train[features1]
    """ Calculate redundancy for each column in a DataFrame. """
    redundancies = {}
    for column in df.columns:
        le = LabelEncoder()
        encoded_data = le.fit_transform(df[column])
        redundancies[column] = calculate_redundancy(encoded_data)

    total_redundancy = sum(redundancies.values())
    return total_redundancy

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01, clipnorm=1.0)
def show_RAM_usage():
    py = psutil.Process(os.getpid())
    print("**********************************************************")
    print('RAM usage: {} GB'.format(py.memory_info()[0]/2. ** 30))

def create_q_model():
    x = Input(shape=(state_size,))

    # a series of fully connected layer for estimating Q(s,a)
    y1 = tf.keras.layers.Dense(64, activation='relu')(x)
    y2 = tf.keras.layers.Dense(8, activation='relu')(y1)
    z = tf.keras.layers.Dense(num_actions, activation="linear")(y2)
    model = keras.Model(inputs=x, outputs=z)
    # model.compile(optimizer=optimizer)
    return model


def reset(agent_num):  # initialize the world
    M = random.randint(1, agent_num)
    initial_state = random.sample(range(agent_num), M)
    return initial_state

def state_representation(St):
    x_train = X_train[St]
    if x_train.empty:
        return [0] * 49
    else : 
        S = x_train.values  
        D = pd.DataFrame(data=S).describe().T
        D_meta = D.describe().T
        D_meta_new = D_meta.loc[:, ['mean', 'std', 'min', '25%' , '50%', '75%', 'max']]
        D_meta_new = D_meta_new.drop(D_meta_new.index[0])
        s = D_meta_new.values.flatten()
        return s


def relevance_score(input):
    x_train = X_train[input]
    information_gain = mutual_info_classif(x_train, y_train)
    final_relevance = 0
    for _, ig in zip(input, information_gain):
        final_relevance = final_relevance + ig
    return final_relevance

def accuracy(input1):
    x_train = X_train[input1]
    x_test = X_test[input1]

    rf_classifier = XGBClassifier(objective="binary:logistic", random_state=42)
    rf_classifier.fit(x_train, y_train)    
    y_pred = rf_classifier.predict(x_test)
    accur = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')
    label_binarizer = LabelBinarizer()
    y_test_encoded = label_binarizer.fit_transform(y_test)
    y_pred_encoded = label_binarizer.transform(y_pred)
    aucScore = roc_auc_score(y_test_encoded, y_pred_encoded, multi_class='ovr')
    
    relevance = relevance_score(input1)
    redundancy = calculate_redundancy_for_dataframe(input1)

    print("Feature number :",len(input1), "AUC :",round((aucScore*100), 1), "Acc :",round((accur*100), 1), "relevance : ", round(relevance, 3), "redundancy : ", round(redundancy,3))
    return aucScore ,accur, precision, recall, f1, relevance, redundancy

def get_reward(features, prv_scores):
    if len(features) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  
    else: 
        auc, acc, precision, recall, f1, relev, redun= accuracy(features)
        tot_f = len(features)
        
        ref_values = {
            'auc': 1,
            'relevance': 1
        }

        objectives = {
            'auc': ref_values['auc'] - auc,  # Invert the maximizing objective
            'features': tot_f,
            'redundancy': redun,
            'relevance': ref_values['relevance'] - relev, 

        }

        previous_objectives = {
            'auc': ref_values['auc'] - prv_scores['auc'],  # Invert the previous maximizing objective
            'features': prv_scores['features'],
            'redundancy': prv_scores['redundancy'],
            'relevance': ref_values['relevance'] - prv_scores['relevance']
        }

        
        valeurs = {
            'auc': auc,
            'features': tot_f,
            'redundancy': redun,
            'relevance' : relev
        }
        reward = 0
        for key in objectives:
            if key in previous_objectives:
                if objectives[key] < previous_objectives[key]:
                    #REWARD
                    reward += valeurs[key] * (previous_objectives[key] - objectives[key])

                elif objectives[key] == previous_objectives[key]:
                    #REWARD
                    reward += valeurs[key] * (previous_objectives[key] - objectives[key])

                else:
                    #PENALTY
                    reward -= valeurs[key] * (objectives[key] - previous_objectives[key])

        # reward = reward * 100
        print('Reward :', reward)
        return reward, auc, acc, precision, recall, f1, tot_f, relev, redun

def step(agents_actions, prv_obj_val):        

    reward, auc_score, accuracyScore, precisionScore, recallScore, f1Score ,feat_number, relevance, redundancy = get_reward(agents_actions, prv_obj_val)
    next_state = agents_actions
    next_state_vector = state_representation(next_state)
    if reward == 0:
        terminal = True
    else:
        terminal = False
    return [next_state, next_state_vector, reward, auc_score, accuracyScore , precisionScore, recallScore , f1Score, feat_number, relevance, redundancy, terminal]

def reset_previous_actions():     
    initial_previous_actions = [1] * num_agents
    return initial_previous_actions

def selected_features(list_actions):       
    features = []
    for i, act in enumerate(list_actions):
        if act == 1:
            features.append(i)
    return features

def clear_session():
    tf.keras.backend.clear_session()


# Function definitions for GMM-based sampling
def rank_and_select_top_samples(samples, rewards, top_p):
    sorted_indices = np.argsort(rewards)[::-1]  # Sort rewards in descending order
    top_count = max(1, int(len(samples) * top_p))  # Ensure at least one sample is selected
    top_indices = sorted_indices[:top_count]
    return samples[top_indices]

def train_gmm(samples, n_components,  sample_fraction=0.1):
    # print("len(samples)", len(samples))
    if len(samples) > 10000:
        sample_size = int(len(samples) * sample_fraction)
        samples = np.random.choice(samples, sample_size, replace=False)
    # Ensure n_components does not exceed number of samples
    n_components = min(n_components, len(samples))
    if len(samples) < 2:
        # Handle case where there are not enough samples to train GMM
        return None
    # gmm = GaussianMixture(n_components=n_components, covariance_type='full', max_iter=100)
    gmm = GaussianMixture(
        n_components=n_components, 
        covariance_type='diag',   # Simplified covariance type
        max_iter=20,              # Fewer iterations
        init_params='kmeans'  # Efficient initialization
    )
    gmm.fit(samples)
    return gmm

def generate_samples(gmm, n_samples):
    if gmm is None:
        return np.array([])  # Return an empty array if GMM is not trained
    return gmm.sample(n_samples)[0]

def gmm_based_generative_rectified_sampling(memory_dataset, top_p, n_components):
    T0 = memory_dataset[memory_dataset[:, 0] == 0]
    T1 = memory_dataset[memory_dataset[:, 0] == 1]
    
    high_quality_datasets = []
    
    for T in [T0, T1]:
        Ni = len(T)
        rewards = np.array([item[1] for item in T], dtype=float)
        # samples = np.array([item[2:] for item in T], dtype=float)  # Use the flattened state and next state
        
        high_quality_samples = rank_and_select_top_samples(T, rewards, top_p)
        
        gmm = train_gmm(high_quality_samples, n_components)
        
        n_generate = Ni - len(high_quality_samples)
        generated_samples = generate_samples(gmm, n_generate)
        
        T_prime = np.vstack((high_quality_samples, generated_samples)) if len(generated_samples) > 0 else high_quality_samples
        
        high_quality_datasets.append(T_prime)
    
    T_prime_combined = np.vstack(high_quality_datasets)
    
    mini_batch_size = min(batch_size, T_prime_combined.shape[0])  # Adjust mini-batch size if necessary
    mini_batch = T_prime_combined[np.random.choice(T_prime_combined.shape[0], size=mini_batch_size, replace=False)]
    
    return mini_batch

models = [create_q_model() for _ in range(num_agents)]
models_target = [create_q_model() for _ in range(num_agents)]


start_time = time.time()
state = reset(F)
episode_reward = 0
previous_actions = [i for i in range(num_agents)]
previous_aucScore,_, _, _, _, previous_relevance, previous_redundancy= accuracy(previous_actions)
prv_objectives_values = {
    'auc': previous_aucScore,
    'features' : len(previous_actions),
    'redundancy': previous_redundancy, 
    'relevance' : previous_relevance

}

for timestep in range(0, max_steps_per_episode):
    print("********** timestep **********", timestep)
    state = np.array(state)
    state = state.ravel()
    frame_count += 1
    state_vectors = state_representation(state)
    actions = []
    #  for actions list
    for i in range(num_agents):
        # For each agent, choose an action using epsilon-greedy
        if epsilon > np.random.rand(1)[0]:
            action = np.random.choice(num_actions)
        else:
            state_tensor = tf.convert_to_tensor(state_vectors)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = models[i](state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()
        agent_lists[i].append(action)
        actions.append(action)

    participated_agents = previous_actions                   
    initial_actions_feat = selected_features(actions)
    agent_to_train = []
    for j in range(len(participated_agents)):
        if participated_agents[j] not in initial_actions_feat:
            agent_to_train.append(participated_agents[j])

    if timestep <= trainers_steps:        

        trainer = Trainer(participated_agents)
        selected_features1, k, assertive_agents,hesitant_agents= trainer.Warm_up(participated_agents, initial_actions_feat)
        f1_score_kbest, advice_k_best = trainer.k_best_score(participated_agents, int(k),X_train, X_test, y_train,  y_test)
        f1_score_DT, advice_DT = trainer.decision_tree_score_MultiClass(participated_agents, assertive_agents, hesitant_agents,X_train, X_test, y_train,  y_test)

        if f1_score_kbest > f1_score_DT:          
            for l in range(len(hesitant_agents)):
                if hesitant_agents[l] in advice_k_best:
                    selected_features1.append(hesitant_agents[l])
        else:
            for o in range(len(hesitant_agents)):
                if hesitant_agents[o] in advice_DT:
                    selected_features1.append(hesitant_agents[o])
        for i in range(len(assertive_agents)):
            selected_features1.append(assertive_agents[i])
        if len(selected_features1) == 0:
            selected_features1 = reset(num_agents)

    else:
        selected_features1 = list(set(participated_agents).intersection(set(initial_actions_feat)))
        if len(selected_features1) == 0:
            selected_features1 = reset(num_agents)

    # Apply the sampled action in the environment for each agent
    state_next, state_next_vector, reward, aucScore, accScore, precisionScore, recallScore, f1Score, feat_numb,relevance, redundancy, done = step(selected_features1, prv_objectives_values)

    state_next = np.array(state_next).ravel()
    state_next_vector = np.array(state_next_vector).ravel()

    episode_reward += reward

    # Save actions and states in replay buffer
    state_history.append(state_vectors.tolist())
    state_next_history.append(state_next_vector.tolist())
    done_history.append(done)
    rewards_history.append(reward)

    rewards_list[timestep] = [round((accScore*100), 1), round((reward),1), round((aucScore*100), 1) , feat_numb, round((precisionScore*100), 1), round((recallScore*100), 1), round((f1Score*100), 1), round (relevance,2), round(redundancy,2), selected_features1]

    excel_file_path1 = '/gpfswork/rech/bvj/uub66jn/MOIDRL-MA-IA-journal-experiments/Experiment1/MLL/PredictionsMLL.txt'
    # excel_file_path1 = '/home/rhellali/public/rhellali/MOIDRL-MA-IA-journal-experiments/Experiment1/MLL/PredictionsMLL.txt'
    # excel_file_path1 = 'PredictionsMLL.txt'
    with open(excel_file_path1, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['AccuracyScore', 'Reward', 'AUCScore' , 'FeatureNumber', 'PrecisionScore', 'RecallScore', 'f1Score','relevance', 'redundancy' ,'selected_features1'])
        for values in rewards_list:
            csv_writer.writerow(values)

    state = state_next
    previous_actions = selected_features1
    previous_aucScore = aucScore    
    print("frame_count", frame_count)

    prv_objectives_values = {
        'auc': aucScore,
        'features' : feat_numb,
        'redundancy': redundancy, 
        'relevance' : relevance

    }

    # Update every fourth frame and once batch size is over 32
    if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
        for agent_idx in range(len(agent_to_train)):      
            index_agent_train = agent_to_train[agent_idx]  
            # Assume memory_dataset is a numpy array with columns [a, r, s, s']

            # Flatten state and next state arrays
            flattened_data = []
            for a, r, s, s_prime in zip(agent_lists[index_agent_train], rewards_history, state_history, state_next_history):
                flattened_data.append([a, r] + s + s_prime)

            # Convert to numpy array
            memory_dataset = np.array(flattened_data, dtype=object)
            # Define parameters
            top_p = 0.2  # Proportion of high-quality samples
            n_components = 2 # Number of GMM components

            # Get a mini-batch of samples
            mini_batch = gmm_based_generative_rectified_sampling(memory_dataset, top_p, n_components)
            # print("mini_batch : ", mini_batch)

            indices = np.random.choice(range(len(done_history)), size=batch_size)
            # Use the respective Q-network model for each agent
            state_sample = np.array([mini_batch[i][2:2+49] for i in range(len(mini_batch))], dtype=np.float32)
            state_next_sample = np.array([mini_batch[i][51:] for i in range(len(mini_batch))], dtype=np.float32)
            rewards_sample = [mini_batch[i][1] for i in range(len(mini_batch))]
            action_sample = [int(mini_batch[i][0]) for i in range(len(mini_batch))]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = models_target[index_agent_train].predict(state_next_sample)

            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )           

             # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample          

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = models[index_agent_train](state_sample)
                
                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)
            # Backpropagation
            grads = tape.gradient(loss, models[index_agent_train].trainable_variables)
            optimizer.apply_gradients(zip(grads, models[index_agent_train].trainable_variables))

            if frame_count % update_target_network == 0:
                print("update the the target network with new weights")
                # update the the target network with new weights
                models_target[index_agent_train].set_weights(models[index_agent_train].get_weights())
            if len(agent_lists[index_agent_train]) > max_memory_length:
                agent_lists[index_agent_train] = agent_lists[index_agent_train][-max_memory_length:]
            clear_session()
        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            rewards_history = rewards_history[-max_memory_length:]
            state_history = state_history[-max_memory_length:]
            state_next_history = state_next_history[-max_memory_length:]
            done_history = done_history[-max_memory_length:]
    show_RAM_usage()

end_time = (time.time()-start_time)/60
print("time spent :", end_time)
show_RAM_usage()