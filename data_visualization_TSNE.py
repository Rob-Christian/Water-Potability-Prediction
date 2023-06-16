## Use t-Distributed Stochastic Neighbor Embedding with 2 components only

X_tsne = stats.zscore(X)
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(X_tsne)
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
