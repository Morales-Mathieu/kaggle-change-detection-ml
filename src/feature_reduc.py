import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def apply_pca_on_colorimetry_features_both(
    train_df, test_df, colorimetry_cols, variance_threshold=0.85
):
    """
    Applies PCA only on the colorimetry features for both train and test DataFrames.
    The continuous colorimetry features are standardized and reduced to a set of principal components
    that explain a specified percentage of the variance (default is 85%).
    The PCA components replace the original colorimetry columns in both datasets, while all other features remain intact.

    Parameters
    ----------
    train_df : pandas.DataFrame
        The training DataFrame containing both colorimetry and other features.
    test_df : pandas.DataFrame
        The test DataFrame containing both colorimetry and other features.
    colorimetry_cols : list of str
        List of column names corresponding to the colorimetry statistics.
    variance_threshold : float, optional
        The fraction of variance to retain in PCA (default is 0.85 for 85%).

    Returns
    -------
    train_df_final : pandas.DataFrame
        The training DataFrame with PCA components replacing the original colorimetry features.
    test_df_final : pandas.DataFrame
        The test DataFrame with PCA components replacing the original colorimetry features.
    pca : sklearn.decomposition.PCA
        The fitted PCA object (useful for further transformations or interpretation).
    scaler : sklearn.preprocessing.StandardScaler
        The fitted StandardScaler object used to standardize the colorimetry features.
    """
    # --- TRAINING SET ---
    # Standardize the colorimetry features from train_df.
    scaler = StandardScaler()
    train_colorimetry = train_df[colorimetry_cols]
    train_colorimetry_scaled = scaler.fit_transform(train_colorimetry)

    # Fit PCA on the scaled training data to retain the desired variance.
    pca = PCA(n_components=variance_threshold, random_state=42)
    train_pca_components = pca.fit_transform(train_colorimetry_scaled)

    # Create a DataFrame for the PCA components.
    n_components = train_pca_components.shape[1]
    pca_cols = [f"pca_{i+1}" for i in range(n_components)]
    df_train_pca = pd.DataFrame(
        train_pca_components, columns=pca_cols, index=train_df.index
    )

    # Retain other features from train_df that are not in colorimetry_cols.
    df_train_other = train_df.drop(columns=colorimetry_cols)

    # Merge PCA components with the rest of the training features.
    train_df_final = pd.concat([df_train_other, df_train_pca], axis=1)

    # --- TEST SET ---
    # Standardize the colorimetry features in test_df using the scaler fitted on train_df.
    test_colorimetry = test_df[colorimetry_cols]
    test_colorimetry_scaled = scaler.transform(test_colorimetry)

    # Apply the PCA transformation using the model fitted on train data.
    test_pca_components = pca.transform(test_colorimetry_scaled)

    # Create a DataFrame for the PCA components in the test set.
    df_test_pca = pd.DataFrame(
        test_pca_components, columns=pca_cols, index=test_df.index
    )

    # Retain other features from test_df that are not in colorimetry_cols.
    df_test_other = test_df.drop(columns=colorimetry_cols)

    # Merge PCA components with the rest of the test features.
    test_df_final = pd.concat([df_test_other, df_test_pca], axis=1)

    return train_df_final, test_df_final, pca, scaler
