import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler

class SupervisedLearningVisualizer:
    def __init__(self):
        np.random.seed(42)
        
    def regression_comparison(self):
        """Compare different regression algorithms"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Regression Algorithms Comparison', fontsize=16)
        
        # Generate synthetic data
        n_samples = 100
        X = np.linspace(0, 1, n_samples).reshape(-1, 1)
        noise = np.random.normal(0, 0.1, n_samples)
        y_true = 1.5 * X.ravel() + np.sin(2 * np.pi * X.ravel()) + noise
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_true, test_size=0.3, random_state=42)
        
        # Models to compare
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42),
            'Polynomial (degree 3)': None,  # Will handle separately
            'Polynomial (degree 8)': None   # Will handle separately
        }
        
        X_plot = np.linspace(0, 1, 200).reshape(-1, 1)
        
        for i, (name, model) in enumerate(models.items()):
            row, col = divmod(i, 3)
            
            if 'Polynomial' in name:
                from sklearn.preprocessing import PolynomialFeatures
                from sklearn.pipeline import Pipeline
                
                degree = 3 if '3' in name else 8
                poly_model = Pipeline([
                    ('poly', PolynomialFeatures(degree=degree)),
                    ('linear', LinearRegression())
                ])
                poly_model.fit(X_train, y_train)
                y_pred = poly_model.predict(X_plot)
                mse = mean_squared_error(y_test, poly_model.predict(X_test))
                
            elif name == 'Random Forest':
                # Use RandomForestRegressor instead
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=10, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_plot)
                mse = mean_squared_error(y_test, model.predict(X_test))
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_plot)
                mse = mean_squared_error(y_test, model.predict(X_test))
            
            # Plot
            axes[row, col].scatter(X_train, y_train, alpha=0.6, s=20, label='Training data')
            axes[row, col].scatter(X_test, y_test, alpha=0.6, s=20, color='red', label='Test data')
            axes[row, col].plot(X_plot, y_pred, color='green', linewidth=2, label='Prediction')
            
            axes[row, col].set_title(f'{name}\nMSE: {mse:.4f}')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].set_xlabel('X')
            axes[row, col].set_ylabel('y')
        
        plt.tight_layout()
        plt.show()
        
    def classification_decision_boundaries(self):
        """Visualize decision boundaries for different classifiers"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Classification Decision Boundaries', fontsize=16)
        
        # Generate synthetic data
        X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, 
                                 n_informative=2, n_clusters_per_class=1, random_state=42)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42)
        
        # Define classifiers
        classifiers = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42),
            'SVM (RBF)': SVC(kernel='rbf', random_state=42),
            'SVM (Linear)': SVC(kernel='linear', random_state=42),
            'K-NN (k=5)': KNeighborsClassifier(n_neighbors=5)
        }
        
        # Create a mesh for plotting decision boundaries
        h = 0.02
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        for i, (name, clf) in enumerate(classifiers.items()):
            row, col = divmod(i, 3)
            
            # Train classifier
            clf.fit(X_train, y_train)
            
            # Make predictions
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Plot decision boundary
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            
            Z = Z.reshape(xx.shape)
            axes[row, col].contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
            
            # Plot data points
            scatter = axes[row, col].scatter(X_scaled[:, 0], X_scaled[:, 1], 
                                           c=y, cmap='RdYlBu', edgecolors='black')
            
            axes[row, col].set_title(f'{name}\nAccuracy: {accuracy:.3f}')
            axes[row, col].set_xlabel('Feature 1')
            axes[row, col].set_ylabel('Feature 2')
        
        plt.tight_layout()
        plt.show()
        
    def bias_variance_tradeoff(self):
        """Demonstrate bias-variance tradeoff"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Bias-Variance Tradeoff', fontsize=16)
        
        # Generate true function
        def true_function(x):
            return 1.5 * x + 0.3 * np.sin(2 * np.pi * x)
        
        X_true = np.linspace(0, 1, 200)
        y_true = true_function(X_true)
        
        # Generate multiple datasets
        n_datasets = 50
        n_samples = 50
        noise_std = 0.2
        
        predictions_low_complexity = []
        predictions_high_complexity = []
        predictions_optimal = []
        
        for _ in range(n_datasets):
            # Generate noisy dataset
            X_train = np.random.uniform(0, 1, n_samples)
            y_train = true_function(X_train) + np.random.normal(0, noise_std, n_samples)
            
            X_train = X_train.reshape(-1, 1)
            X_test = X_true.reshape(-1, 1)
            
            # Low complexity model (underfitting)
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.pipeline import Pipeline
            
            low_model = Pipeline([
                ('poly', PolynomialFeatures(degree=1)),
                ('linear', LinearRegression())
            ])
            low_model.fit(X_train, y_train)
            pred_low = low_model.predict(X_test)
            predictions_low_complexity.append(pred_low)
            
            # High complexity model (overfitting)
            high_model = Pipeline([
                ('poly', PolynomialFeatures(degree=15)),
                ('linear', LinearRegression())
            ])
            high_model.fit(X_train, y_train)
            pred_high = high_model.predict(X_test)
            predictions_high_complexity.append(pred_high)
            
            # Optimal complexity
            optimal_model = Pipeline([
                ('poly', PolynomialFeatures(degree=3)),
                ('linear', Ridge(alpha=0.1))
            ])
            optimal_model.fit(X_train, y_train)
            pred_optimal = optimal_model.predict(X_test)
            predictions_optimal.append(pred_optimal)
        
        # Calculate bias and variance
        predictions_low = np.array(predictions_low_complexity)
        predictions_high = np.array(predictions_high_complexity)  
        predictions_opt = np.array(predictions_optimal)
        
        mean_pred_low = np.mean(predictions_low, axis=0)
        mean_pred_high = np.mean(predictions_high, axis=0)
        mean_pred_opt = np.mean(predictions_opt, axis=0)
        
        # Plot results
        axes[0,0].plot(X_true, y_true, 'r-', linewidth=3, label='True function')
        for i in range(min(10, n_datasets)):
            alpha = 0.3 if i == 0 else 0.1
            label = 'Individual predictions' if i == 0 else None
            axes[0,0].plot(X_true, predictions_low[i], 'b-', alpha=alpha, label=label)
        axes[0,0].plot(X_true, mean_pred_low, 'b-', linewidth=3, label='Mean prediction')
        axes[0,0].set_title('High Bias, Low Variance\n(Underfitting)')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        axes[0,1].plot(X_true, y_true, 'r-', linewidth=3, label='True function')
        for i in range(min(10, n_datasets)):
            alpha = 0.3 if i == 0 else 0.1
            label = 'Individual predictions' if i == 0 else None
            axes[0,1].plot(X_true, predictions_high[i], 'g-', alpha=alpha, label=label)
        axes[0,1].plot(X_true, mean_pred_high, 'g-', linewidth=3, label='Mean prediction')
        axes[0,1].set_title('Low Bias, High Variance\n(Overfitting)')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        axes[1,0].plot(X_true, y_true, 'r-', linewidth=3, label='True function')
        for i in range(min(10, n_datasets)):
            alpha = 0.3 if i == 0 else 0.1
            label = 'Individual predictions' if i == 0 else None
            axes[1,0].plot(X_true, predictions_opt[i], 'purple', alpha=alpha, label=label)
        axes[1,0].plot(X_true, mean_pred_opt, 'purple', linewidth=3, label='Mean prediction')
        axes[1,0].set_title('Balanced Bias-Variance\n(Good fit)')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Bias-variance decomposition plot
        complexities = range(1, 16)
        bias_squared = []
        variance = []
        noise = noise_std**2
        
        for degree in complexities:
            predictions_deg = []
            
            for _ in range(20):  # Fewer samples for speed
                X_train = np.random.uniform(0, 1, n_samples)
                y_train = true_function(X_train) + np.random.normal(0, noise_std, n_samples)
                
                model = Pipeline([
                    ('poly', PolynomialFeatures(degree=degree)),
                    ('linear', Ridge(alpha=0.1))
                ])
                model.fit(X_train.reshape(-1, 1), y_train)
                pred = model.predict(X_true.reshape(-1, 1))
                predictions_deg.append(pred)
            
            predictions_deg = np.array(predictions_deg)
            mean_pred = np.mean(predictions_deg, axis=0)
            
            # Bias squared
            bias_sq = np.mean((mean_pred - y_true)**2)
            bias_squared.append(bias_sq)
            
            # Variance
            var = np.mean(np.var(predictions_deg, axis=0))
            variance.append(var)
        
        total_error = np.array(bias_squared) + np.array(variance) + noise
        
        axes[1,1].plot(complexities, bias_squared, 'b-o', linewidth=2, label='Bias²')
        axes[1,1].plot(complexities, variance, 'g-s', linewidth=2, label='Variance')
        axes[1,1].plot(complexities, total_error, 'r-^', linewidth=2, label='Total Error')
        axes[1,1].axhline(y=noise, color='orange', linestyle='--', label='Irreducible Error')
        
        axes[1,1].set_xlabel('Model Complexity (Polynomial Degree)')
        axes[1,1].set_ylabel('Error')
        axes[1,1].set_title('Bias-Variance Decomposition')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def run_all_visualizations(self):
        self.regression_comparison()
        self.classification_decision_boundaries()
        self.bias_variance_tradeoff()

# Usage
supervised_ml = SupervisedLearningVisualizer()
supervised_ml.run_all_visualizations()