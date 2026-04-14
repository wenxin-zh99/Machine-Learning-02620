import numpy as np


class PCA:
    """
    Principal Component Analysis (PCA) implemented with NumPy SVD.

    This class follows the common sklearn-style interface:
    `fit`, `transform`, `fit_transform`, and `inverse_transform`.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of principal components to keep.
        If None, keep all available components.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space, sorted by explained variance
        in descending order.
    explained_variance_ : ndarray of shape (n_components,)
        Variance explained by each selected component.
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Fraction of total variance explained by each selected component.
    singular_values_ : ndarray of shape (n_components,)
        Singular values corresponding to each selected component.
    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean computed from the training data.
    n_components_ : int
        Actual number of components kept.
    n_features_in_ : int
        Number of features seen during fit.
    n_samples_in_ : int
        Number of samples seen during fit.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        """
        Fit PCA on X using the SVD of the centered data matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = self._validate_data(X)
        n_samples, n_features = X.shape

        if n_samples < 2:
            raise ValueError("PCA requires at least 2 samples.")

        self.n_samples_in_ = n_samples
        self.n_features_in_ = n_features
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        U, Vt = self._svd_flip(U, Vt)

        max_components = min(n_samples, n_features)
        self.n_components_ = self._resolve_n_components(self.n_components, max_components)

        explained_variance = (S ** 2) / (n_samples - 1)
        total_variance = explained_variance.sum()

        self.components_ = Vt[: self.n_components_]
        self.singular_values_ = S[: self.n_components_]
        self.explained_variance_ = explained_variance[: self.n_components_]
        if total_variance == 0:
            self.explained_variance_ratio_ = np.zeros(self.n_components_, dtype=float)
        else:
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        return self

    def transform(self, X):
        """
        Project X onto the principal component space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_components)
        """
        self._check_is_fitted()
        X = self._validate_data(X)
        self._check_n_features(X)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        """
        Fit the model on X and return the projected data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_components)
        """
        X = self._validate_data(X)
        n_samples, n_features = X.shape

        if n_samples < 2:
            raise ValueError("PCA requires at least 2 samples.")

        self.n_samples_in_ = n_samples
        self.n_features_in_ = n_features
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        U, Vt = self._svd_flip(U, Vt)

        max_components = min(n_samples, n_features)
        self.n_components_ = self._resolve_n_components(self.n_components, max_components)

        explained_variance = (S ** 2) / (n_samples - 1)
        total_variance = explained_variance.sum()

        self.components_ = Vt[: self.n_components_]
        self.singular_values_ = S[: self.n_components_]
        self.explained_variance_ = explained_variance[: self.n_components_]
        if total_variance == 0:
            self.explained_variance_ratio_ = np.zeros(self.n_components_, dtype=float)
        else:
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        return U[:, : self.n_components_] * self.singular_values_

    def inverse_transform(self, X_transformed):
        """
        Map data from principal component space back to feature space.

        Parameters
        ----------
        X_transformed : array-like of shape (n_samples, n_components)

        Returns
        -------
        ndarray of shape (n_samples, n_features)
        """
        self._check_is_fitted()
        X_transformed = np.asarray(X_transformed, dtype=float)
        if X_transformed.ndim != 2:
            raise ValueError(f"Expected 2-D array, got {X_transformed.ndim}-D array.")
        if X_transformed.shape[1] != self.n_components_:
            raise ValueError(
                "X_transformed has the wrong number of columns: "
                f"expected {self.n_components_}, got {X_transformed.shape[1]}."
            )
        return X_transformed @ self.components_ + self.mean_

    def _validate_data(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D array, got {X.ndim}-D array.")
        if X.shape[1] == 0:
            raise ValueError("PCA requires at least 1 feature.")
        return X

    def _check_is_fitted(self):
        required_attrs = (
            "components_",
            "explained_variance_",
            "explained_variance_ratio_",
            "mean_",
            "n_components_",
            "n_features_in_",
            "n_samples_in_",
            "singular_values_",
        )
        missing = [attr for attr in required_attrs if not hasattr(self, attr)]
        if missing:
            raise RuntimeError("This PCA instance is not fitted yet. Call fit() first.")

    def _check_n_features(self, X):
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has the wrong number of features: "
                f"expected {self.n_features_in_}, got {X.shape[1]}."
            )

    @staticmethod
    def _resolve_n_components(n_components, max_components):
        if n_components is None:
            return max_components

        if isinstance(n_components, bool) or not isinstance(n_components, (int, np.integer)):
            raise TypeError("n_components must be an integer or None.")

        if not 1 <= n_components <= max_components:
            raise ValueError(
                f"n_components={n_components} must be between 1 and {max_components}."
            )

        return int(n_components)

    @staticmethod
    def _svd_flip(U, Vt):
        """
        Make SVD output deterministic by enforcing a consistent sign convention.
        """
        max_abs_cols = np.argmax(np.abs(U), axis=0)
        signs = np.sign(U[max_abs_cols, np.arange(U.shape[1])])
        signs[signs == 0] = 1.0
        U *= signs
        Vt *= signs[:, np.newaxis]
        return U, Vt

    def __repr__(self):
        return f"PCA(n_components={self.n_components})"


if __name__ == "__main__":
    X = np.array(
        [
            [2.5, 2.4],
            [0.5, 0.7],
            [2.2, 2.9],
            [1.9, 2.2],
            [3.1, 3.0],
            [2.3, 2.7],
            [2.0, 1.6],
            [1.0, 1.1],
            [1.5, 1.6],
            [1.1, 0.9],
        ],
        dtype=float,
    )

    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_pca)

    print("components_:\n", pca.components_)
    print("\nexplained_variance_:\n", pca.explained_variance_)
    print("\nexplained_variance_ratio_:\n", pca.explained_variance_ratio_)
    print("\nsingular_values_:\n", pca.singular_values_)
    print("\nX_pca shape:", X_pca.shape)
    print("\nReconstruction MSE:", np.mean((X - X_reconstructed) ** 2))
