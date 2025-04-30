import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from keras.layers import Dense, Input
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


def csv_to_data(csv_80s, csv_90s, csv_00s, csv_10s, features, save_path="data"):

    # Step 1: Load CSVs (no sampling)
    df_80s = pd.read_csv(csv_80s, low_memory=False).dropna(subset=['mbid'])
    df_90s = pd.read_csv(csv_90s, low_memory=False).dropna(subset=['mbid'])
    df_00s = pd.read_csv(csv_00s, low_memory=False).dropna(subset=['mbid'])
    df_10s = pd.read_csv(csv_10s, low_memory=False).dropna(subset=['mbid'])

    # Step 2: Deduplicate MBIDs (priority: 80s > 90s > 00s > 10s)
    mbids_80s = set(df_80s['mbid'])
    df_90s = df_90s[~df_90s['mbid'].isin(mbids_80s)]
    mbids_90s = set(df_90s['mbid'])
    df_00s = df_00s[~df_00s['mbid'].isin(mbids_80s.union(mbids_90s))]
    mbids_00s = set(df_00s['mbid'])
    df_10s = df_10s[~df_10s['mbid'].isin(mbids_80s.union(mbids_90s).union(mbids_00s))]

    # Step 3: Add decade labels
    df_80s["label"] = 0
    df_90s["label"] = 1
    df_00s["label"] = 2 
    df_10s["label"] = 3

    # Step 4: Combine all decades
    df_all = pd.concat([df_80s, df_90s, df_00s, df_10s], ignore_index=True)

    # Step 5: Normalize features
    df_all = normalize_features(df_all, features)

    # --- Keep only expanded/normalized features + label ---
    keep_cols = []
    for col in df_all.columns:
        for feat in features:
            if col == feat or col.startswith(feat + "_"):
                keep_cols.append(col)
    keep_cols.append("label")
    df_all = df_all[keep_cols]

    # Print Columns
    # print("\n=== Columns after normalization and expansion ===\n")
    # for col in df_all.columns:
    #     print(col)
    # print("\n=== Total columns:", len(df_all.columns), "===\n")

    # Step 6: Prepare X and Y
    X = df_all.drop(columns=["label"]).values
    Y = df_all["label"].values

    # Display
    pd.set_option('display.max_columns', None)
    print(df_all.head())
    print(f"\n✅ Prepared {X.shape[0]} samples with {X.shape[1]} features each.")

    return X, Y


def normalize_features(df, features):
    df_normalized = df.copy()
    scaler = MinMaxScaler()

    for col in features:
        if col not in df_normalized.columns:
            continue
        
        series = df_normalized[col]

        # --- 1. Direct mapping ---
        if col in [
            "hl.highlevel.mood_acoustic.value",
            "hl.highlevel.mood_aggressive.value",
            "hl.highlevel.mood_electronic.value",
            "hl.highlevel.mood_happy.value",
            "hl.highlevel.mood_party.value",
            "hl.highlevel.mood_relaxed.value",
            "hl.highlevel.mood_sad.value",
            "hl.highlevel.timbre.value",
            "hl.highlevel.tonal_atonal.value",
            "hl.highlevel.voice_instrumental.value",
            "hl.highlevel.danceability.value"
        ]:
            mapping = {
                "acoustic": 1.0, "not_acoustic": 0.0,
                "aggressive": 1.0, "not_aggressive": 0.0,
                "electronic": 1.0, "not_electronic": 0.0,
                "happy": 1.0, "not_happy": 0.0,
                "party": 1.0, "not_party": 0.0,
                "relaxed": 1.0, "not_relaxed": 0.0,
                "sad": 1.0, "not_sad": 0.0,
                "bright": 1.0, "dark": 0.0,
                "tonal": 1.0, "atonal": 0.0,
                "voice": 1.0, "instrumental": 0.0,
                "danceable": 1.0, "not_danceable": 0.0,
            }
            df_normalized[col] = series.map(mapping).fillna(0.0)

        # --- 2. Special handling for ll.tonal.key_key ---
        elif col == "ll.tonal.key_key":
            all_keys = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
            series = series.astype(pd.CategoricalDtype(categories=all_keys))
            dummies = pd.get_dummies(series, prefix="ll.tonal.key_key", drop_first=False)
            dummies = dummies.astype(float)
            df_normalized = df_normalized.drop(columns=[col])
            df_normalized = pd.concat([df_normalized, dummies], axis=1)

        # --- 3. Normal one-hot encoding ---
        elif col in [
            "hl.highlevel.genre_dortmund.value",
            "hl.highlevel.genre_rosamerica.value",
            "hl.highlevel.genre_tzanetakis.value",
            "hl.highlevel.moods_mirex.value",
            "hl.highlevel.ismir04_rhythm.value"
        ]:
            dummies = pd.get_dummies(series, prefix=col, drop_first=True)
            dummies = dummies.astype(float)
            df_normalized = df_normalized.drop(columns=[col])
            df_normalized = pd.concat([df_normalized, dummies], axis=1)

        # --- 4. Major/Minor mapping ---
        elif col == "ll.tonal.key_scale":
            df_normalized[col] = series.map({"major": 1.0, "minor": 0.0}).fillna(0.0)

        # --- 5. Expand arrays ---
        else:
            try:
                sample_value = series.dropna().iloc[0]
                if isinstance(sample_value, str) and sample_value.strip().startswith('[') and sample_value.strip().endswith(']'):
                    parsed_array = series.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [0])
                    parsed_array = parsed_array.apply(lambda x: np.array(x))
                    array_df = pd.DataFrame(parsed_array.tolist(), index=df_normalized.index)
                    array_df = array_df.add_prefix(col + "_")
                    df_normalized = df_normalized.drop(columns=[col])
                    df_normalized = pd.concat([df_normalized, array_df], axis=1)
                else:
                    df_normalized[col] = pd.to_numeric(series, errors='coerce')
            except Exception:
                df_normalized[col] = pd.to_numeric(series, errors='coerce')

    # --- Drop bpm_histogram columns ---
    histogram_cols = [col for col in df_normalized.columns if "bpm_histogram" in col]
    if histogram_cols:
        df_normalized = df_normalized.drop(columns=histogram_cols)

    # --- Fill NaNs and Infs ---
    df_normalized = df_normalized.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # --- Scale ---
    numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns
    if "label" in numeric_cols:
        numeric_cols = numeric_cols.drop("label")
    if len(numeric_cols) > 0:
        df_normalized[numeric_cols] = scaler.fit_transform(df_normalized[numeric_cols])

    return df_normalized
    

def split_data_per_class(X, Y, val_ratio=0.1, test_ratio=0.1, num_classes=4):

    # 1. Combine X and Y into a DataFrame
    df = pd.DataFrame(X.tolist())  # if X is array of arrays
    df['label'] = Y

    # 2. Drop duplicate rows
    df = df.drop_duplicates()

    # 3. Recreate X and Y
    X = df.drop(columns=['label']).values
    Y = df['label'].values

    # 4. Now split per class
    X_train_list, Y_train_list = [], []
    X_val_list, Y_val_list = [], []
    X_test_list, Y_test_list = [], []

    for class_idx in range(num_classes):
        class_mask = (Y == class_idx)
        X_class = X[class_mask]
        Y_class = Y[class_mask]

        n_samples = len(X_class)
        n_val = int(n_samples * val_ratio)
        n_test = int(n_samples * test_ratio)
        n_train = n_samples - n_val - n_test

        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_class = X_class[indices]
        Y_class = Y_class[indices]

        X_train_list.append(X_class[:n_train])
        Y_train_list.append(Y_class[:n_train])

        X_val_list.append(X_class[n_train:n_train + n_val])
        Y_val_list.append(Y_class[n_train:n_train + n_val])

        X_test_list.append(X_class[n_train + n_val:])
        Y_test_list.append(Y_class[n_train + n_val:])

    X_train = np.concatenate(X_train_list)
    Y_train = np.concatenate(Y_train_list)
    X_val = np.concatenate(X_val_list)
    Y_val = np.concatenate(Y_val_list)
    X_test = np.concatenate(X_test_list)
    Y_test = np.concatenate(Y_test_list)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def evaluate_model(model, X_test, Y_test):
    """
    Evaluate a trained multiclass classification model on a test set and return the accuracy.

    Parameters
    ----------
    model : keras.models.Model
        The trained Keras model to evaluate.
    X_test : numpy.ndarray
        The test set features.
    Y_test : numpy.ndarray
        The true labels.

    Returns
    -------
    accuracy : float
        The accuracy of the model on the test set.
    """
    # Get raw model outputs
    predictions = model.predict(X_test)
    
    #predictions_label = (predictions > 0.5).astype(int).reshape(-1)
    predictions_label = np.argmax(predictions, axis=1)

    # Compute accuracy
    accuracy = np.mean(predictions_label == Y_test)
    return accuracy
    
    
def explore_data(X_train, Y_train, Y_test, Y_val):
  """
  Plots the distribution of classes in the training, validation, and test sets.

  Parameters
  ----------
  X_train : np.ndarray
      A numpy array containing the features of the training set.
  Y_train : np.ndarray
      A numpy array containing the labels of the training set.
  Y_test : np.ndarray
      A numpy array containing the labels of the test set.
  Y_val : np.ndarray
      A numpy array containing the labels of the validation set.

  Returns
  -------
  None
  """

  # Class names
  class_names = ["80s", "90s", "2000s", "2010s"]

  # Plot the distribution of classes in the training, validation, and test sets
  fig, ax = plt.subplots(1, 3, figsize=(10, 5))

  # Plot the distribution of classes in the training set
  train_class_counts = np.bincount(Y_train.astype(int))
  ax[0].bar(range(len(class_names)), train_class_counts)
  ax[0].set_xticks(range(len(class_names)))
  ax[0].set_xticklabels(class_names, rotation=45)
  ax[0].set_title('Training set')

  # Plot the distribution of classes in the test set
  test_class_counts = np.bincount(Y_val.astype(int))
  ax[1].bar(range(len(class_names)), test_class_counts)
  ax[1].set_xticks(range(len(class_names)))
  ax[1].set_xticklabels(class_names, rotation=45)
  ax[1].set_title('Validation set')

  # Plot the distribution of classes in the test set
  test_class_counts = np.bincount(Y_test.astype(int))
  ax[2].bar(range(len(class_names)), test_class_counts)
  ax[2].set_xticks(range(len(class_names)))
  ax[2].set_xticklabels(class_names, rotation=45)
  ax[2].set_title('Test set')

  plt.show()


def plot_loss(history):
  """
  Plot the training and validation loss and accuracy.

  Parameters
  ----------
  history : keras.callbacks.History
      The history object returned by the `fit` method of a Keras model.

  Returns
  -------
  None
  """

  # Plot the training and validation loss side by side
  fig, ax = plt.subplots(1, 2, figsize=(10, 5))

  # Plot the training and validation loss
  ax[0].plot(history.history['loss'], label='train')
  ax[0].plot(history.history['val_loss'], label='val')
  ax[0].set_xlabel('Epoch')
  ax[0].set_ylabel('Loss')
  ax[0].legend()

  # Plot the training and validation accuracy
  ax[1].plot(history.history['accuracy'], label='train')
  ax[1].plot(history.history['val_accuracy'], label='val')
  ax[1].set_xlabel('Epoch')
  ax[1].set_ylabel('Accuracy')
  ax[1].legend()

