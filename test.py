import critmat
import sys, os
sys.path.append("..")
import facebook_dcn_traffic.traffic_snapshot_pb2
import facebook_dcn_traffic.utility
from sklearn.cluster import KMeans

def read_in_facebook_traffic(trace_file_name, network_ids_filename):
	valid_network_ids = facebook_dcn_traffic.utility.read_in_network_ids(network_ids_filename)
	traffic_matrices = facebook_dcn_traffic.utility.read_traffic_matrix_protobuf(trace_file_name, considered_network_id=valid_network_ids)
	return traffic_matrices, valid_network_ids

def flatten_traffic_matrix(tm, num_nodes):
	vector = []
	for i in range(num_nodes):
		for j in range(num_nodes):
			if i != j:
				vector.append(tm[i][j])
	return vector

def main(trace_filename, network_ids_filename):
	traffic_matrices, valid_network_ids = read_in_facebook_traffic(trace_filename, network_ids_filename)
	number_of_pods = len(valid_network_ids)
	assert(number_of_pods == len(traffic_matrices[0]))
	print("Finished reading the traffic matrices")
	traffice_vectors = [flatten_traffic_matrix(x, number_of_pods) for x in traffic_matrices]

	# run kmc
	kmeans = KMeans(n_clusters=3, random_state=0).fit(traffice_vectors)
	cluster_centroids = kmeans.cluster_centers_

	print("flatened vectors have dimension = {}".format(len(traffice_vectors[0])))
	print("Finished flattening the traffic matrices into vectors")
	critm = critmat.CritMat(traffice_vectors, None)
	print("total number of traffic matrices : {}".format(len(traffice_vectors)))
	representative_vectors = critm.train(number_of_clusters=3)
	print("\n\n\n")
	for index, representative_vector in zip(range(len(representative_vectors)), representative_vectors):
		print("{}. {}".format(index, representative_vector))
	return


if __name__ == "__main__":
	print("Critmat test")

	test_vectors = [[3, 3, 1],
					[0, 2, 4], 
					[4, 2, 0],
					[3, 3, 1]]

	traffic_representative = critmat.CritMat(test_vectors, None)

	representative_traffic_matrices = traffic_representative.train(number_of_clusters=1)

	for traffic_matrix, index in zip(representative_traffic_matrices, range(1, len(representative_traffic_matrices) + 1, 1)):
		print("{}. {}".format(index, traffic_matrix) )


	representative_traffic_matrices = traffic_representative.train(number_of_clusters=2)

	for traffic_matrix, index in zip(representative_traffic_matrices, range(1, len(representative_traffic_matrices) + 1, 1)):
		print("{}. {}".format(index, traffic_matrix) )

	representative_traffic_matrices = traffic_representative.train(number_of_clusters=3)

	for traffic_matrix, index in zip(representative_traffic_matrices, range(1, len(representative_traffic_matrices) + 1, 1)):
		print("{}. {}".format(index, traffic_matrix) )

	tm_snapshots_protobuf_filename = "/Users/minyee/src/facebook_dcn_traffic/traffic_matrices/clusterC/hadoop_aggregationwindow_30.pb"
	valid_network_ids_filename = "/Users/minyee/src/facebook_dcn_traffic/traffic_matrices/clusterC/clusterC_pods.txt"
	#tm_snapshots_protobuf_filename = "/Users/minyee/src/facebook_dcn_traffic/traffic_matrices/clustercombined/combined_aggregationwindow_1.pb"
	#valid_network_ids_filename = "/Users/minyee/src/facebook_dcn_traffic/traffic_matrices/clustercombined/clustercombined_pods.txt"
	main(tm_snapshots_protobuf_filename, valid_network_ids_filename)

