from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import numpy as np

df = pd.read_csv(file_path)
label_columns = df.select_dtypes(include=['object']).columns

def manual_SMOTE(X, y, target_label, k_neighbors=5, random_state=None):
    np.random.seed(random_state)

    # Filter minority class samples
    minority_samples = X[y == target_label]

    # Create a nearest neighbors model
    nn_model = NearestNeighbors(n_neighbors=k_neighbors + 1)
    nn_model.fit(minority_samples)

    # Find k neighbors for each sample in the minority class
    neighbors = nn_model.kneighbors(minority_samples, return_distance=False)[:, 1:]

    # Generate synthetic samples
    num_synthetic_samples = X[y != target_label].shape[0] - minority_samples.shape[0]
    synthetic_samples = np.zeros((num_synthetic_samples, X.shape[1]))

    for i in range(num_synthetic_samples):
        # Randomly choose a sample from minority class
        sample_idx = np.random.randint(0, minority_samples.shape[0])

        # Randomly choose one of its k neighbors
        neighbor_idx = np.random.choice(neighbors[sample_idx])

        # Compute the difference between the sample and its chosen neighbor
        diff = minority_samples.iloc[neighbor_idx] - minority_samples.iloc[sample_idx]

        # Generate a random number between 0 and 1
        random_weight = np.random.random()

        # Generate synthetic sample
        synthetic_samples[i] = minority_samples.iloc[sample_idx] + random_weight * diff

    # Combine the original features with synthetic samples
    X_resampled = pd.concat([X, pd.DataFrame(synthetic_samples, columns=X.columns)], ignore_index=True)
    y_resampled = pd.concat([y, pd.Series([target_label] * num_synthetic_samples)], ignore_index=True)

    return X_resampled, y_resampled

label_encoder = LabelEncoder()
for column in label_columns:
    df[column] = label_encoder.fit_transform(df[column])

X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Apply manual SMOTE
target_label = 1  # 'Yes' label is encoded as 1
X_resampled, y_resampled = manual_SMOTE(X, y, target_label, random_state=42)

# Check the distribution of the target variable after SMOTE
y_resampled.value_counts()