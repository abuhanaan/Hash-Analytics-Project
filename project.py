import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Importing the problem case study xslx file
excel_file = pd.ExcelFile('Problem-case-study.xlsx')
dataset1 = excel_file.parse('Existing employees')
dataset2 = excel_file.parse('Employees who have left')

# Encoding categorical data
salary = {"low": 0, "medium": 1, "high": 2}
dataset1.iloc[:, 9] = dataset1.iloc[:, 9].map(salary)
dataset2.iloc[:, 9] = dataset2.iloc[:, 9].map(salary)

X1 = dataset1.iloc[:, [1, 5]].values
X2 = dataset2.iloc[:, [1, 5]].values


# To determine the number of clusters(k) in the 'Employees who have left' dataset, we use the elbow method
wcss2 = []
for i in range(1, 11):  # We have selected the no of clusters to b 10 here
    kmeans2 = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
#     Then we need to predict object to data
    kmeans2.fit(X2)  # Then we need to append this to wcss variable
    wcss2.append(kmeans2.inertia_)  # This is to compute result
plt.plot(range(1, 11), wcss2)  # X axis is range(1, 11) and Y axis is wcss
plt.title('Elbow Method for Employees who have left')
plt.xlabel('Number of Clusters')
plt.ylabel('wcss2')
plt.savefig('Elbow_Graph_of_Emps_who_have_left.png')
plt.show()
# From the graph, the optimal value for k is 7

# To determine the number of clusters(k) in the 'Existing Employees' dataset, we use the elbow method as well
wcss1 = []
for i in range(1, 11):  # We have selected the no of clusters to b 10 here
    kmeans1 = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
#     Then we need to predict object to data
    kmeans1.fit(X2)  # Then we need to append this to wcss variable
    wcss1.append(kmeans1.inertia_)  # This is to compute result
plt.plot(range(1, 11), wcss1)  # X axis is range(1, 11) and Y axis is wcss
plt.title('Elbow Method for Existing Employees')
plt.xlabel('Number of Clusters')
plt.ylabel('wcss1')
plt.savefig('Elbow_graph_of_Existing_Emps')
plt.show()
# From the graph, the optimal value for k is 6 for the 2 elbow graphs

# Now we need to fit k-means to Eployees who have left dataset and set n_clusters to 6
kmeans2 = KMeans(n_clusters=6, init='k-means++', n_init=10, max_iter=300, random_state=0)
Y_kmeans2 = kmeans2.fit_predict(X2)


# Visualising the clusters
plt.scatter(X2[Y_kmeans2 == 0, 0], X2[Y_kmeans2 == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X2[Y_kmeans2 == 1, 0], X2[Y_kmeans2 == 1, 1], s=100, c='cyan', label='Cluster 2')
plt.scatter(X2[Y_kmeans2 == 2, 0], X2[Y_kmeans2 == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X2[Y_kmeans2 == 3, 0], X2[Y_kmeans2 == 3, 1], s=100, c='blue', label='Cluster 4')
plt.scatter(X2[Y_kmeans2 == 4, 0], X2[Y_kmeans2 == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.scatter(X2[Y_kmeans2 == 5, 0], X2[Y_kmeans2 == 5, 1], s=100, c='orange', label='Cluster 6')

plt.scatter(kmeans2.cluster_centers_[:, 0], kmeans2.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of Employees who have left')
plt.xlabel('Satisfactory Level')
plt.ylabel('Number of Years Spent in the Company')
plt.legend()
plt.savefig('Cluster_Graph_of_Emps_who_have_left.png')
plt.show()

# Now we need to fit k-means to Existing Employees dataset and set n_clusters to 6
kmeans1 = KMeans(n_clusters=6, init='k-means++', n_init=10, max_iter=300, random_state=0)
Y_kmeans1 = kmeans1.fit_predict(X1)


# Visualising the clusters
plt.scatter(X1[Y_kmeans1 == 0, 0], X1[Y_kmeans1 == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X1[Y_kmeans1 == 1, 0], X1[Y_kmeans1 == 1, 1], s=100, c='cyan', label='Cluster 2')
plt.scatter(X1[Y_kmeans1 == 2, 0], X1[Y_kmeans1 == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X1[Y_kmeans1 == 3, 0], X1[Y_kmeans1 == 3, 1], s=100, c='blue', label='Cluster 4')
plt.scatter(X1[Y_kmeans1 == 4, 0], X1[Y_kmeans1 == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.scatter(X1[Y_kmeans1 == 5, 0], X1[Y_kmeans1 == 5, 1], s=100, c='orange', label='Cluster 6')

plt.scatter(kmeans1.cluster_centers_[:, 0], kmeans1.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of Existing Employees')
plt.xlabel('Satisfactory Level')
plt.ylabel('Number of Years Spent in the Company')
plt.legend()
plt.savefig('Cluster_graph_of_Existing_Emps.png')
plt.show()

