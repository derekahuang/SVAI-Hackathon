import numpy as np
import random
import sys
import time
import math

#Calculates cosine similarity between two vectors
def calculateCosSim(vec1,vec2):
	return np.dot(vec1,vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))

#Initiates the first K centroids using the k-means++ algorithm
def initKPlusPlus(X,K):

	#Chooses random index for first centroid
	centroid_init = random.choice(list(range(len(X))))

	#Initiates centroid matrix with initially one chosen centroid
	centroids = [X[centroid_init]]

	#Keeps track of indices of documents chosen as centroids
	indices_chosen = [centroid_init]

	#Continuously adds centroids until we get K centroids
	while len(centroids) < K:
		centroids = np.append(centroids,[chooseNewCentroids(X,centroids,indices_chosen)],axis=0)

	return centroids

#Chooses new centroid to append to matrix of centroids for k-means++
def chooseNewCentroids(X,centroids,indices_chosen):

	#Calculates the shortest distances from centroids for each
	#document in X
	dist_closest_cents = shortestDistCentroids(X,centroids,indices_chosen)

	#Calculates corresponding probabilities for each document in X to be
	#next centroid
	probabilities = dist_closest_cents/sum(dist_closest_cents)

	#Converts each probability to a cumulative probability, which
	#makes it easier to correctly choose a document based on the
	#distribution
	probabilities = np.cumsum(probabilities)

	#Random number in [0.0,1.0)
	choice = random.random()

	#Choose a document index based on the cumulative probability distribution
	centroid = np.argmax(probabilities >= choice)

	return X[centroid]

#Multiplier at indices ?

#Calculates the shortest distances from centroids for each
#document in X
def shortestDistCentroids(X,centroids,indices_chosen):

	#Initializes the list that will be returned
	#in form of numpy array
	dist_closest_cents = []

	#Main logic to do the calculations
	for i in range(len(X)):
		if i in indices_chosen:
			#If document is already chosen as a centroid
			dist_closest_cents.append(0.0)
		else:
			#This will store the distances from centroids
			#for document i
			curr_closest_dists = []

			#Initializes document i
			curr_doc = X[i]

			for j in range(len(centroids)):
				#Calculates distance from document i to centroid j
				curr_closest_dists.append(np.linalg.norm(curr_doc - centroids[j])**2)

			#Adds the min distance to the return matrix
			dist_closest_cents.append(min(curr_closest_dists))

	return np.array(dist_closest_cents)

#Calculates the index of the closest centroid for document vector x
def closestCentroid(x,centroids):

	#Initialize the index and closest centroid to the 0th
	index_closest = 0
	centroid_closest = calculateCosSim(x,centroids[0])

	#Logic that calculates cosine similarity for index i in [1,K)
	#comparing and updating (if necessary) the closest index and
	#centroid
	for i in range(1,len(centroids)):
		temp_cos_sim = calculateCosSim(x,centroids[i])
		if temp_cos_sim > centroid_closest:
			centroid_closest = temp_cos_sim
			index_closest = i

	return index_closest


#Outputs a one-dimensional matrix of indices that represent row indices
#of the cluster assigned to each document, where the ith element
#in the list represents the row index of the cluster assigned
#to the ith document
def assignClusters(X,centroids):

	#Initializes the return matrix of assigned clusters
	cluster_indices = np.zeros(len(X),dtype=np.int32)
	for i in range(len(X)):
		cluster_indices[i] = closestCentroid(X[i],centroids)

	return cluster_indices

#Recomputes centroids based on clustering, with centroid_indices
#being a one-dimensional matrix representing the indices
#of centroid groupings for each document
def reComputeCentroids(X,centroid_indices):
	centroids = np.zeros((0,X.shape[1]))
	i = 0
	while i <= max(centroid_indices):
		#Checks if any documents were assigned to this cluster
		if i in centroid_indices:
			#Boolean vector of documents in cluster
			member_indices = centroid_indices == i

			#Total number of documents in cluster
			number_members = sum(member_indices)

			#Gets matrix of all the documents that are part of cluster
			members = X[member_indices,:]

			#Resets this specific centroid
			centroids = np.append(centroids,[sum(members)/float(number_members)],axis=0)

		i += 1

	return centroids

#Runs the K-means algorithm, asserting the K passed in is > 0
#Outputs tuple of centroids, and cluster indices of documents
def runKMeans(X,K,num_iterations,use_k_means_pp):

	#Checks whether or not to use k-means++ algo for
	#initiating centroids
	if use_k_means_pp:
		centroids = initKPlusPlus(X,K)
	else:
		centroids = initKRandomCentroids(X,K)

	#This will be the indices of assigned cluster per document
	centroid_indices = None
	for i in range(num_iterations):

		#Assigns each document to cluster
		centroid_indices = assignClusters(X,centroids)

		#Recomputes centroids
		centroids = reComputeCentroids(X,centroid_indices)

	return (centroids,centroid_indices)