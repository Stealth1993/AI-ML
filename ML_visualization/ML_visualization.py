import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Create datasets
X_class, y_class = make_classification(n_samples=300, n_features=2, n_redundant=0, 
                                      n_informative=2, n_clusters_per_class=1, random_state=42)
X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_cluster, _ = make_blobs(n_samples=300, centers=4, n_features=2, 
                         random_state=42, cluster_std=1.5)

# Create comprehensive ML visualization
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# 1. Classification with Decision Boundary
ax1 = fig.add_subplot(gs[0, 0])
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.3, random_state=42)

# Train logistic regression
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Plot decision boundary
h = 0.02
x_min, x_max = X_class[:, 0].min() - 1, X_class[:, 0].max() + 1
y_min, y_max = X_class[:, 1].min() - 1, X_class[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

ax1.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
scatter = ax1.scatter(X_class[:, 0], X_class[:, 1], c=y_class, cmap='RdYlBu', edgecolors='black')
ax1.set_title('Logistic Regression Decision Boundary')
plt.colorbar(scatter, ax=ax1)

# 2. Decision Tree Visualization
ax2 = fig.add_subplot(gs[0, 1])
tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf.fit(X_train, y_train)

# Plot decision boundary for tree
Z_tree = tree_clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z_tree = Z_tree.reshape(xx.shape)

ax2.contourf(xx, yy, Z_tree, levels=50, alpha=0.6, cmap='RdYlBu')
scatter2 = ax2.scatter(X_class[:, 0], X_class[:, 1], c=y_class, cmap='RdYlBu', edgecolors='black')
ax2.set_title('Decision Tree Classification')

# 3. Linear Regression
ax3 = fig.add_subplot(gs[0, 2])
reg = LinearRegression()
reg.fit(X_reg, y_reg)

ax3.scatter(X_reg, y_reg, alpha=0.6)
X_plot = np.linspace(X_reg.min(), X_reg.max(), 100).reshape(-1, 1)
y_plot = reg.predict(X_plot)
ax3.plot(X_plot, y_plot, 'r-', linewidth=2)
ax3.set_title(f'Linear Regression\nR² = {reg.score(X_reg, y_reg):.3f}')
ax3.set_xlabel('X')
ax3.set_ylabel('y')

# 4. Polynomial Regression
ax4 = fig.add_subplot(gs[0, 3])
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Create polynomial features
poly_reg = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('linear', LinearRegression())
])
poly_reg.fit(X_reg, y_reg)

y_poly = poly_reg.predict(X_plot)
ax4.scatter(X_reg, y_reg, alpha=0.6)
ax4.plot(X_plot, y_plot, 'r-', linewidth=2, label='Linear')
ax4.plot(X_plot, y_poly, 'g-', linewidth=2, label='Polynomial (degree 3)')
ax4.set_title('Linear vs Polynomial Regression')
ax4.legend()
ax4.set_xlabel('X')
ax4.set_ylabel('y')

# 5. K-Means Clustering
ax5 = fig.add_subplot(gs[1, 0])
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(X_cluster)

scatter3 = ax5.scatter(X_cluster[:, 0], X_cluster[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
centers = kmeans.cluster_centers_
ax5.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3)
ax5.set_title('K-Means Clustering (k=4)')

# 6. PCA Visualization
ax6 = fig.add_subplot(gs[1, 1])
# Create higher dimensional data
X_high_dim = np.random.randn(200, 4)
# Add some structure
X_high_dim[:100, 0] += 3
X_high_dim[100:, 1] += 3

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_high_dim)

colors = ['red' if i < 100 else 'blue' for i in range(200)]
ax6.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.7)
ax6.set_title(f'PCA Projection\nExplained variance: {pca.explained_variance_ratio_.sum():.2f}')
ax6.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
ax6.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')

# 7. Learning Curves
ax7 = fig.add_subplot(gs[1, 2])
from sklearn.model_selection import learning_curve

# Generate learning curve for logistic regression
train_sizes, train_scores, val_scores = learning_curve(
    LogisticRegression(), X_class, y_class, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10))

ax7.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
ax7.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Cross-validation score')
ax7.fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1)
ax7.fill_between(train_sizes, np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.1)
ax7.set_xlabel('Training Set Size')
ax7.set_ylabel('Accuracy Score')
ax7.set_title('Learning Curves')
ax7.legend()
ax7.grid(True)

# 8. Overfitting Demonstration
ax8 = fig.add_subplot(gs[1, 3])
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Generate simple dataset
X_simple = np.linspace(0, 1, 20).reshape(-1, 1)
y_simple = 1.5 * X_simple.ravel() + np.sin(1.5 * np.pi * X_simple.ravel()) + np.random.normal(0, 0.1, 20)

degrees = [1, 4, 15]
colors_poly = ['green', 'red', 'blue']
X_plot_simple = np.linspace(0, 1, 100).reshape(-1, 1)

ax8.scatter(X_simple, y_simple, alpha=0.8, s=50)

for degree, color in zip(degrees, colors_poly):
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X_simple)
    poly_reg_demo = LinearRegression()
    poly_reg_demo.fit(X_poly, y_simple)
    
    X_plot_poly = poly_features.transform(X_plot_simple)
    y_plot_poly = poly_reg_demo.predict(X_plot_poly)
    
    ax8.plot(X_plot_simple, y_plot_poly, color=color, linewidth=2, 
            label=f'Degree {degree}')

ax8.set_xlabel('X')
ax8.set_ylabel('y')
ax8.set_title('Overfitting Demonstration')
ax8.legend()
ax8.set_ylim(-2, 3)

# 9. Confusion Matrix Visualization
ax9 = fig.add_subplot(gs[2, 0])
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax9)
ax9.set_title('Confusion Matrix')
ax9.set_xlabel('Predicted')
ax9.set_ylabel('Actual')

# 10. ROC Curve
ax10 = fig.add_subplot(gs[2, 1])
from sklearn.metrics import roc_curve, auc

y_pred_proba = clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

ax10.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax10.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax10.set_xlim([0.0, 1.0])
ax10.set_ylim([0.0, 1.05])
ax10.set_xlabel('False Positive Rate')
ax10.set_ylabel('True Positive Rate')
ax10.set_title('ROC Curve')
ax10.legend(loc="lower right")
ax10.grid(True)

# 11. Feature Importance (for tree)
ax11 = fig.add_subplot(gs[2, 2])
feature_names = ['Feature 1', 'Feature 2']
importances = tree_clf.feature_importances_
ax11.bar(feature_names, importances)
ax11.set_title('Feature Importance (Decision Tree)')
ax11.set_ylabel('Importance')

# 12. Validation Curve
ax12 = fig.add_subplot(gs[2, 3])
from sklearn.model_selection import validation_curve

param_range = np.logspace(-4, 1, 10)
train_scores_val, test_scores_val = validation_curve(
    LogisticRegression(max_iter=1000), X_class, y_class, 
    param_name='C', param_range=param_range, cv=5)

ax12.semilogx(param_range, np.mean(train_scores_val, axis=1), 'o-', 
             label='Training score')
ax12.semilogx(param_range, np.mean(test_scores_val, axis=1), 'o-', 
             label='Cross-validation score')
ax12.fill_between(param_range, np.mean(train_scores_val, axis=1) - np.std(train_scores_val, axis=1),
                 np.mean(train_scores_val, axis=1) + np.std(train_scores_val, axis=1), alpha=0.1)
ax12.fill_between(param_range, np.mean(test_scores_val, axis=1) - np.std(test_scores_val, axis=1),
                 np.mean(test_scores_val, axis=1) + np.std(test_scores_val, axis=1), alpha=0.1)
ax12.set_xlabel('Regularization Parameter C')
ax12.set_ylabel('Accuracy Score')
ax12.set_title('Validation Curve')
ax12.legend()
ax12.grid(True)

plt.suptitle('Machine Learning Visualization Suite', fontsize=16)
plt.show()