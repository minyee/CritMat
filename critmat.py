import os, sys, copy, math
import numpy as np
from sklearn.cluster import KMeans

## The algorithm for computing the representative vectors given a collection
## of vectors. The 
class CritMat(object):
	def __init__(self, training_vectors, distance_function):
		self.number_of_training_snapshots = len(training_vectors)
		self.training_traffic_vectors =  training_vectors
		return
	# checks if v1 dominates v2
	def __dominated(self, v1, v2):
		dims = len(v1)
		for i in range(dims):
			if v2[i] > v1[i]:
				return False
		return True

	def __trim(self):
		dominated = [False] * self.number_of_training_snapshots
		for i in range(self.number_of_training_snapshots - 1):
			is_dominated = dominated[i]
			if is_dominated:
				continue
			else:
				for j in range(self.number_of_training_snapshots):
					if j != i and not dominated[j] and self.__dominated(self.training_traffic_vectors[j], self.training_traffic_vectors[i]):
						dominated[i] = True
						is_dominated = True
						break
		remaining_entries = 0
		for boolean in dominated:
			if not boolean:
				remaining_entries += 1
		print("trimmed, remaining entries : {}".format(remaining_entries))
		return


	# Computes the euclidean distance between two vectors. 
	# A helper function for other routines.
	def __euclidean_distance(self, vector_dimensions, vector_i, vector_j):
		distance = 0.
		for i in range(vector_dimensions):
			distance += (vector_i[i] - vector_j[i]) ** 2
		return math.sqrt(distance)

	def _cost_euclidean(self, vector_dimensions, head_ij, vector_i, vector_j):
		return max(self.__euclidean_distance(vector_dimensions, head_ij, vector_i), self.__euclidean_distance(vector_dimensions, head_ij, vector_j))

	## Critical-ness Aware Clustering
	def __CritAC__(self, training_vectors, number_of_clusters):
		vector_dimensions = len(training_vectors[0])
		num_training_snapshots = len(training_vectors)
		
		## Step 1 : Initialization phase
		cluster_heads = []
		clusters = []
		intercluster_scores = [0] * (num_training_snapshots * (num_training_snapshots - 1) / 2)
		for cluster_id in range(num_training_snapshots):
			cluster_heads.append(copy.deepcopy(training_vectors[cluster_id]))
			clusters.append(copy.deepcopy(training_vectors[cluster_id]))
		index = 0
		for i in range(num_training_snapshots - 1):
			for j in range(i + 1, num_training_snapshots, 1):
				head_ij = np.maximum(cluster_heads[i], cluster_heads[j])
				cost_ij = sum(head_ij) - max(sum(clusters[i]), sum(clusters[j]))
				#cost_ij = max(sum([(a - b)**2 for a, b in zip(head_ij, clusters[i])]), sum([(a - b)**2 for a, b in zip(head_ij, clusters[j])]))
				intercluster_scores[index] = (i, j, cost_ij)
				index += 1
			print("{}".format(i))
		intercluster_scores.sort(key=lambda x:x[2]) 
		print("Initialization done")

		## Step 2 : Agglomeration steps
		## To remove a cluster index at j, we just move it into the tail, so that after each
		## step, we reduce the leftover clusters by 1. This means that the clusters at the 
		## tail are not considered anymore
		for step in range(num_training_snapshots - number_of_clusters):
			leftover_clusters = num_training_snapshots - step
			## agglomerate the two clusters that have the least cost
			(i, j, score) = intercluster_scores[0]
			#dist = [(a - b)**2 for a, b in zip(vector1, vector2)]
			head_ij = np.maximum(cluster_heads[i], cluster_heads[j])
			cluster_heads[i] = head_ij
			index_to_merge = j
			if sum(clusters[j] < clusters[i]):
				index_to_merge = i
			## Swap cluster j and head j to the last entry, which is the entry we no longer 
			## look at in the subsequent interation here
			tmp = clusters[leftover_clusters - 1]
			clusters[leftover_clusters - 1] = clusters[index_to_merge]
			clusters[index_to_merge] = tmp
			tmp = cluster_heads[leftover_clusters - 1]
			cluster_heads[leftover_clusters - 1] = cluster_heads[index_to_merge]
			cluster_heads[index_to_merge] = tmp

			## remove the cluster that is not new_head
			leftover_clusters -= 1
			intercluster_scores = [(0,0,0)] * ((leftover_clusters) * (leftover_clusters - 1) / 2)
			index = 0
			for ii in range(leftover_clusters - 1):
				for jj in range(ii + 1, leftover_clusters, 1):
					head_ij = np.maximum(cluster_heads[ii], cluster_heads[jj])
					cost_ij = sum(head_ij) - max(sum(clusters[i]), sum(clusters[j]))
					#cost_ij = max(sum([(a - b)**2 for a, b in zip(head_ij, clusters[ii])]), sum([(a - b)**2 for a, b in zip(head_ij, clusters[jj])]))
					intercluster_scores[index] = (i, j, cost_ij)
					index += 1
			# sorts the intercluster scores based on cost (sorts in place)
			intercluster_scores.sort(key=lambda x:x[2]) 
			print("step {}".format(step))
		check_vector = np.zeros((vector_dimensions,))
		for i in range(vector_dimensions):
			check_vector[i] = cluster_heads[0][i] - training_vectors[0][i]
		print("check vector \n\n{}\n\n".format(check_vector))
		return [np.array(x) for x in cluster_heads[:number_of_clusters]]

	def __kmc(self, training_vectors, number_of_clusters):
		vector_dimensions = len(self.training_traffic_vectors[0])
		## need to copy the vectors one by one
		## use k means to first find the cluster centroids, find out the dominating cluster heads
		# substep 1 : run k means clustering
		kmeans = KMeans(n_clusters=number_of_clusters, random_state=0)
		kmeans.fit(training_vectors)
		# substep 2 : figure out all the vectors, and which cluster they are binned into
		point_labels = kmeans.predict(training_vectors)
		# substep 3 : for each cluster centroid, find the points that belong to this cluster, and find the head of each cluster
		training_vectors = []
		for _ in range(number_of_clusters):
			head = np.zeros( (vector_dimensions,) )
			training_vectors.append(head)
		for training_vector, cluster_label in zip(self.training_traffic_vectors, point_labels):
			training_vectors[cluster_label] = np.maximum(training_vector, training_vectors[cluster_label])
		return training_vectors

	## Derives the critical representative traffic matrices
	def train(self, number_of_clusters=1, critac_snapshot_limit=100):
		number_of_clusters = int(number_of_clusters)
		assert(number_of_clusters >= 1)
		assert(self.number_of_training_snapshots >= number_of_clusters)
		assert(critac_snapshot_limit > 1)
		vector_dimensions = len(self.training_traffic_vectors[0])
		training_vectors = []

		return self.__kmc(self.training_traffic_vectors, number_of_clusters)

		if len(self.training_traffic_vectors) <= critac_snapshot_limit:
			training_vectors = self.training_traffic_vectors
		else:
			## need to copy the vectors one by one
			## use k means to first find the cluster centroids, find out the dominating cluster heads
			# substep 1 : run k means clustering
			kmeans = KMeans(n_clusters=critac_snapshot_limit, random_state=0)
			kmeans.fit(self.training_traffic_vectors)
			# substep 2 : figure out all the vectors, and which cluster they are binned into
			point_labels = kmeans.predict(self.training_traffic_vectors)
			# substep 3 : for each cluster centroid, find the points that belong to this cluster, and find the head of each cluster
			training_vectors = []
			for _ in range(critac_snapshot_limit):
				head = np.zeros( (vector_dimensions,) )
				training_vectors.append(head)
			assert(len(point_labels) == len(self.training_traffic_vectors))

			for training_vector, cluster_label in zip(self.training_traffic_vectors, point_labels):
				training_vectors[cluster_label] = np.maximum(training_vector, training_vectors[cluster_label])
		return self.__CritAC__(training_vectors, number_of_clusters)
