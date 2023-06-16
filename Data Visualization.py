## For Principal Component Analysis

X_PCA = stats.zscore(X)
pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(X_PCA)

# Store the principal component values in a dataframe
principal_df = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])

# Print the total variance (percentage remain after dimension reduction)
print("The total variance is: " + str(pca.explained_variance_ratio_.sum() * 100))

# Plot the reduced dimension with colored labels
plt.figure(figsize=(8,8))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('PC1: ' + str(round(100*pca.explained_variance_ratio_[0], 2)) + '%',fontsize=20)
plt.ylabel('PC2: ' + str(round(100*pca.explained_variance_ratio_[1], 2)) + '%',fontsize=20)
plt.title("Principal Component Analysis of Water Potability",fontsize=20)
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = y == target
    plt.scatter(principal_df.loc[indicesToKeep, 'PC1']
               , principal_df.loc[indicesToKeep, 'PC2'], c = color, s = 50)
plt.legend(['Non Potable', 'Potable'],prop={'size': 15})

## Use t-Distributed Stochastic Neighbor Embedding with 2 components only

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X_PCA)
df_subset = pd.DataFrame()

# Store the reduced dimension in a dataframe
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]

# Plot the reduced dimension with colored labels
plt.figure(figsize=(8,8))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.title("t-SNE of Water Potability Dataset",fontsize=20)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = y == target
    plt.scatter(df_subset.loc[indicesToKeep, 'tsne-2d-one']
               , df_subset.loc[indicesToKeep, 'tsne-2d-two'], c = color, s = 50)

plt.legend(['Not Potable', 'Potable'],prop={'size': 15})
