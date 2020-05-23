import os, sys, copy, math
import numpy as np

## The algorithm for computing the representative vectors given a collection
## of vectors. The 
class CritMat(object):
	def __init__(self, training_vectors, distance_function):
		self.number_of_training_snapshots = len(training_vectors)
		self.training_traffic_vectors =  training_vectors
		return

	## returns the cost(i,j), essentially the cost function
	def _cost(self, vector_i, vector_j):
		distance = 0
		for index in range(len(vector_i)):
			distance += (vector_i[index] - vector_j[index]) ** 2
		return math.sqrt(distance)

	# Computes the euclidean distance between two vectors. 
	# A helper function for other routines.
	def __euclidean_distance(self, vector_dimensions, vector_i, vector_j):
		distance = 0.
		for i in range(vector_dimensions):
			distance += (vector_i[i] - vector_j[i]) ** 2
		return math.sqrt(distance)

	def _cost_euclidean(self, vector_dimensions, vector_i, vector_j):
		head_ij = self._find_head(vector_dimensions, [vector_i, vector_j])
		return max(self.__euclidean_distance(vector_dimensions, head_ij, vector_i), self.__euclidean_distance(vector_dimensions, head_ij, vector_j))

	## Given a collection of vectors, finds the head vector, which is a vector that dominates 
	## all other vectors in every dimension of the vector
	def _find_head(self, vector_dimension, collection_of_vectors):
		head = np.zeros((vector_dimension,))
		for i in range(vector_dimension):
			for vector in collection_of_vectors:
				head[i] = max(head[i], vector[i])
		return head

	def check_valid(self):
		return

	## Derives the critical representative traffic matrices
	def train(self, number_of_clusters=1):
		number_of_clusters = int(number_of_clusters)
		assert(number_of_clusters >= 1)
		assert(self.number_of_training_snapshots >= number_of_clusters)
		vector_dimensions = len(self.training_traffic_vectors[0])

		## Step 1 : Initialization phase
		cluster_heads = []
		clusters = []
		intercluster_scores = [0] * (self.number_of_training_snapshots * (self.number_of_training_snapshots - 1) / 2)
		for cluster_id in range(self.number_of_training_snapshots):
			cluster_heads.append(copy.deepcopy(self.training_traffic_vectors[cluster_id]))
			clusters.append(copy.deepcopy(self.training_traffic_vectors[cluster_id]))
		print("weee")
		## compute the scores
		print("num iterations: {}".format(self.number_of_training_snapshots * (self.number_of_training_snapshots - 1) / 2))
		index = 0
		for i in range(self.number_of_training_snapshots - 1):
			for j in range(i + 1, self.number_of_training_snapshots, 1):
				head_ij = self._find_head(vector_dimensions, [self.training_traffic_vectors[i], self.training_traffic_vectors[j]])
				cost_ij = self._cost_euclidean(vector_dimensions, clusters[i], clusters[j])
				intercluster_scores[index] = (i, j, cost_ij,)
				index += 1
			print("{}".format(i))
		intercluster_scores.sort(key=lambda x:x[2]) 
		print("Initialization done")
		## Step 2 : Agglomeration steps
		## To remove a cluster index at j, we just move it into the tail, so that after each
		## step, we reduce the leftover clusters by 1. This means that the clusters at the 
		## tail are not considered anymore
		percentage = 0.05
		number_of_steps = self.number_of_training_snapshots - number_of_clusters
		for step in range(self.number_of_training_snapshots - number_of_clusters):
			leftover_clusters = self.number_of_training_snapshots - step
			## agglomerate the two clusters that have the least cost
			(i, j, score) = intercluster_scores[0]
			new_head = self._find_head(vector_dimensions, [cluster_heads[i], cluster_heads[j], ])
			cluster_heads[i] = new_head

			## Swap cluster j and head j to the last entry, which is the entry we no longer 
			## look at in the subsequent interation here
			tmp = clusters[leftover_clusters - 1]
			clusters[leftover_clusters - 1] = clusters[j]
			clusters[j] = tmp
			tmp = cluster_heads[leftover_clusters - 1]
			cluster_heads[leftover_clusters - 1] = cluster_heads[j]
			cluster_heads[j] = tmp


			## remove the cluster that is not new_head
			intercluster_scores = []
			for ii in range(leftover_clusters - 2):
				for jj in range(ii + 1, leftover_clusters - 1, 1):
					head_ij = self._find_head(vector_dimensions, [self.training_traffic_vectors[ii], self.training_traffic_vectors[jj]])
					cost_ij = self._cost_euclidean(vector_dimensions, clusters[ii], clusters[jj])
					intercluster_scores.append((ii, jj, cost_ij,))
			# sorts the intercluster scores based on cost (sorts in place)
			intercluster_scores.sort(key=lambda x:x[2]) 
			print("step {}".format(step))
			#if percentage <= float(step ) / number_of_steps:
			#	print("training {}% COMPLETED...".format(float(step ) / number_of_steps))
			#	percentage += 0.05
		return [np.array(x) for x in cluster_heads[:number_of_clusters]]