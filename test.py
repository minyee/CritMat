import critmat
import sys, os, math
import numpy as np
sys.path.append("..")
import facebook_dcn_traffic.traffic_snapshot_pb2
import facebook_dcn_traffic.utility
from sklearn.cluster import KMeans

## import traffic engineering stuff
import robust_topology_engineering.aurora_network as aurora_module
import robust_topology_engineering.traffic_engineer as te_module
import robust_topology_engineering.topology_engineer as toe_module
import robust_topology_engineering.path_selector as path_selector_module

## import for plotting uses 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib import cm

def read_in_facebook_traffic(trace_file_name, network_ids_filename, aggregation_window):
	valid_network_ids = facebook_dcn_traffic.utility.read_in_network_ids(network_ids_filename)
	traffic_matrices = facebook_dcn_traffic.utility.read_traffic_matrix_protobuf(trace_file_name, considered_network_id=valid_network_ids)
	print("Number of traffic matrices : {}".format(len(traffic_matrices)))
	for tm in traffic_matrices:
		for i in range(len(tm)):
			for j in range(len(tm)):
				tm[i][j] = tm[i][j] / 1000000. / float(aggregation_window)
	return traffic_matrices, valid_network_ids

def flatten_traffic_matrix(tm, num_nodes):
	vector = []
	for i in range(num_nodes):
		for j in range(num_nodes):
			if i != j:
				vector.append(tm[i][j])
	return np.array(vector)

def _compute_link_utilization_statistics(nblocks, link_utilization_matrix, adj_matrix, total_links):
		total_links = 0.
		lu_distribution = []
		total_weighted_lu = 0.
		for i in range(nblocks):
			for j in range(nblocks):
				if i != j:
					lu_distribution.append( (link_utilization_matrix[i][j], adj_matrix[i][j], ) )
					total_links += adj_matrix[i][j]
					total_weighted_lu += (adj_matrix[i][j] * link_utilization_matrix[i][j])
		lu_distribution_sorted = sorted(lu_distribution, key=lambda x: x[0])	
		cumulative_link_counts = (total_links - total_links)
		p50_index = 0.5 * (total_links)
		p90_index = 0.9 * (total_links)
		lu50 = -1.
		lu90 = -1.
		mlu = lu_distribution_sorted[-1][0]
		for (lu, num_links) in lu_distribution_sorted:
			if lu50 < 0 and cumulative_link_counts >= p50_index:
				lu50 = lu
			elif lu90 < 0 and cumulative_link_counts >= p90_index:
				lu90 = lu
			cumulative_link_counts += num_links
		alu = total_weighted_lu / total_links
		return (mlu, lu90, lu50, alu, lu_distribution)

# computes the routing performance for a topology with a given set of routing weights
# part of static routing
def _evaluate_snapshot_performance(traffic_matrix, topology_adj_matrix, routing_weights, link_capacity, total_links, return_lu_distribution=False):
	nblocks = len(topology_adj_matrix)
	link_utilization = np.zeros((nblocks, nblocks,))
	hop_counter = 0.
	total_traffic = 0.
	#unable_to_route = False
	for path in routing_weights:
		src = path[0]
		dst = path[-1]
		weight = routing_weights[path]
		if traffic_matrix[src][dst] == 0.:
			continue
		if weight < 0.00001:
			continue
		traffic_load = weight * traffic_matrix[src][dst]
		
		total_traffic += traffic_load
		hop_counter += ((len(path) - 1) * traffic_load)
		curr_node = src
		for index in range(1, len(path), 1):
			next_node = path[index]
			path_cap = link_capacity * topology_adj_matrix[curr_node][next_node]
			#if path_cap == 0. and traffic_load >= 0.0001:
				#assert(false)
			if traffic_load / path_cap > 2.:
				print("link_load exploded! from src: {} to dst: {} via path {}".format(src, dst, path))
				print("traffic demand: {} weight : {}, adj_matrix has number of links : {}".format(traffic_matrix[src][dst], weight, topology_adj_matrix[curr_node][next_node]))

			link_utilization[curr_node][next_node] += (traffic_load / path_cap)
			curr_node = next_node
	ave_hop_count = hop_counter / total_traffic
	(mlu, lu90, lu50, alu, lu_distribution) = _compute_link_utilization_statistics(nblocks, link_utilization, topology_adj_matrix, total_links)
	if not return_lu_distribution:
		return (mlu, lu90, lu50, ave_hop_count)
	else:
		return (mlu, lu90, lu50, ave_hop_count, [x[0] for x in lu_distribution])

def main(trace_filename, network_ids_filename, aggregation_window):
	traffic_matrices, valid_network_ids = read_in_facebook_traffic(trace_filename, network_ids_filename, aggregation_window)
	traffic_matrices = traffic_matrices[1:-1]

	considered_snapshots = 3000
	offset = 3000
	traffic_matrices = traffic_matrices[offset:offset + considered_snapshots]

	number_of_pods = len(valid_network_ids)
	assert(number_of_pods == len(traffic_matrices[0]))
	print("Finished reading the traffic matrices")
	traffice_vectors = [flatten_traffic_matrix(x, number_of_pods) for x in traffic_matrices]

	number_of_snapshots = len(traffice_vectors)
	#for i in range(number_of_snapshots * (number_of_snapshots - 1) / 2):
	#	print("{}".format(i))
	#for i in range(number_of_snapshots - 1):
	#	for j in range(i + 1, number_of_snapshots, 1):
	#		print("{}, {}".format(i, j))

	#for i in range(len(traffice_vectors) - 1):
	#	for j in range(i+1, len(traffice_vectors), 1):
	#		if check_dominate(traffice_vectors[i], traffice_vectors[j]):
	#			print("{} dominates {}".format(i, j))

	print("Finished flattening the traffic matrices into vectors")

	num_clusters = 4

	critm = critmat.CritMat(traffice_vectors, None)
	print("total number of traffic matrices : {}".format(len(traffice_vectors)))
	representative_vectors = critm.train(number_of_clusters=num_clusters)

	maximum_vector = np.zeros((number_of_pods * (number_of_pods - 1),))
	for traffic_vector in traffice_vectors:
		maximum_vector = np.maximum(maximum_vector, traffic_vector)


	#if num_clusters == 1:
	#	assert(len(representative_vectors) == 1)
	#	lefover = np.zeros((len(maximum_vector),))
	#	for i in range(len(lefover)):
	#		lefover[i] = maximum_vector[i] - representative_vectors[0][i]
	#	print("max vector : {}".format(maximum_vector))
	#	print("representative : {}".format(representative_vectors[0]))
	#	print("leftover array : \n{}".format(lefover))
	#	exit()

	representative_matrices = []
	nrow = int(math.ceil(math.sqrt(num_clusters)))
	ncol = int(math.ceil(float(num_clusters) / nrow))
	fig, axes = plt.subplots(nrows=nrow, ncols=ncol, squeeze=True)
	axes_vector = [axes]
	if num_clusters > 1:
		axes_vector = axes.flatten()
	for ax, index in zip(axes_vector, range(num_clusters)):
		traffic_matrix = np.zeros((number_of_pods, number_of_pods,))
		offset = 0
		for i in range(number_of_pods):
			for j in range(number_of_pods):
				if i != j:
					traffic_matrix[i][j] = representative_vectors[index][offset]
					offset += 1
		ax.imshow(traffic_matrix)
		representative_matrices.append(traffic_matrix)
	#plt.title("{}".format(cluster_alias), fontsize=14)
	

	# Define the traffic_engineering class
	# Uniform Topology
	uniform_topology = [0] * number_of_pods

	per_node_pair_num_links = 10
	for i in range(number_of_pods):
		uniform_topology[i] = [0] * number_of_pods
		for j in range(number_of_pods):
			if i != j:
				uniform_topology[i][j] = per_node_pair_num_links

	link_capacity = 5
	block_params = {aurora_module.BlockType.SUPERBLOCK : {}, aurora_module.BlockType.BORDER_ROUTER : {}}
	block_params[aurora_module.BlockType.SUPERBLOCK]["link capacity"] = float(link_capacity)
	block_params[aurora_module.BlockType.SUPERBLOCK]["num links"] = float((number_of_pods - 1) * per_node_pair_num_links)
	block_params[aurora_module.BlockType.BORDER_ROUTER]["link capacity"] = float(link_capacity) # in gbps
	block_params[aurora_module.BlockType.BORDER_ROUTER]["num links"] = float((number_of_pods - 1) * per_node_pair_num_links)
	block_names_list = range(1, number_of_pods + 1, 1)
	block_names_list = ["ju{}".format(x) for x in block_names_list]
	aurora_network = aurora_module.AuroraNetwork("customized", block_params, block_names_list)
	allpaths_selector = path_selector_module.PathSelector(aurora_network, use_multihop=True)
	all_interblock_paths = allpaths_selector.get_all_paths()
	directpaths_selector = path_selector_module.PathSelector(aurora_network, use_multihop=False)
	direct_interblock_paths = directpaths_selector.get_all_paths()



	## Topology Engineer classes
	max_toe = toe_module.HistoricalMaxTrafficTopologyEngineer(aurora_network, 10, 10, all_interblock_paths, traffic_matrices)
	max_topology = max_toe.topology_engineer_given_TMs(traffic_matrices, all_interblock_paths)
	max_topology = aurora_network.round_fractional_topology_giant_switch(max_topology, [])
	multi_tm_toe = toe_module.RobustMultiTrafficTopologyEngineer(aurora_network, 10,
																10, all_interblock_paths, 
																traffic_matrices, num_clusters)
	multi_tm_topology = multi_tm_toe.topology_engineer_given_representative_TMs(representative_matrices, all_interblock_paths)
	multi_tm_topology = aurora_network.round_fractional_topology_giant_switch(multi_tm_topology, [])
	ave_toe = toe_module.HistoricalMaxTrafficTopologyEngineer(aurora_network, 10, 10, all_interblock_paths, traffic_matrices)
	ave_topology = ave_toe.topology_engineer_given_TMs(traffic_matrices, all_interblock_paths)
	ave_topology = aurora_network.round_fractional_topology_giant_switch(ave_topology, [])

	## max tm
	max_tm = np.zeros((number_of_pods, number_of_pods))
	for tm in traffic_matrices:
		for i in range(number_of_pods):
			for j in range(number_of_pods):
				max_tm[i][j] = max(max_tm[i][j], tm[i][j])

	## Compute the routing weights - Uniform topology
	multi_tm_traffic_engineer = te_module.RobustMultiClusterScaleUpWeightedTrafficEngineer(aurora_network, all_interblock_paths, 4, 1, num_clusters, reduce_multihop=True)
	unif_multi_tm_routing_weights = multi_tm_traffic_engineer.compute_path_weights(uniform_topology, representative_matrices)
	#multi_tm_routing_weights = multi_tm_traffic_engineer.compute_path_weights(uniform_topology, [max_tm,])
	
	max_traffic_engineering = te_module.HistoricalMaxTrafficEngineer(aurora_network, all_interblock_paths, 10, 10)
	unif_max_routing_weights = max_traffic_engineering.compute_path_weights(uniform_topology, traffic_matrices)

	ave_traffic_engineering = te_module.HistoricalAveTrafficEngineer(aurora_network, all_interblock_paths, 10, 10)
	unif_ave_routing_weights = ave_traffic_engineering.compute_path_weights(uniform_topology, traffic_matrices)

	## Compute the routing weights - Topology Engineer
	multi_toe_multi_te_routing_weights = multi_tm_traffic_engineer.compute_path_weights(multi_tm_topology, representative_matrices)
	#multi_tm_routing_weights = multi_tm_traffic_engineer.compute_path_weights(uniform_topology, [max_tm,])
	
	max_toe_max_te_routing_weights = max_traffic_engineering.compute_path_weights(max_topology, traffic_matrices)

	ave_toe_ave_te_routing_weights = ave_traffic_engineering.compute_path_weights(ave_topology, traffic_matrices)

	unif_multi_tm_mlu = []
	unif_max_tm_mlu = []
	unif_ave_tm_mlu = []
	unif_multi_tm_ahc = []
	unif_max_tm_ahc = []
	unif_ave_tm_ahc = []

	multi_toe_multi_te_mlu = []
	ave_toe_ave_te_mlu = []
	max_toe_max_te_mlu = []
	multi_toe_multi_te_ahc = []
	ave_toe_ave_te_ahc = []
	max_toe_max_te_ahc = []
	total_links = sum([sum(x) for x in uniform_topology])
	for tm in traffic_matrices:
		unif_multi_tm_perf = _evaluate_snapshot_performance(tm, uniform_topology, unif_multi_tm_routing_weights, link_capacity, total_links, return_lu_distribution=False)
		unif_max_tm_perf = _evaluate_snapshot_performance(tm, uniform_topology, unif_max_routing_weights, link_capacity, total_links, return_lu_distribution=False)
		unif_ave_tm_perf = _evaluate_snapshot_performance(tm, uniform_topology, unif_ave_routing_weights, link_capacity, total_links, return_lu_distribution=False)
		unif_multi_tm_mlu.append(unif_multi_tm_perf[0])
		unif_max_tm_mlu.append(unif_max_tm_perf[0])
		unif_ave_tm_mlu.append(unif_ave_tm_perf[0])
		unif_multi_tm_ahc.append(unif_multi_tm_perf[3])
		unif_max_tm_ahc.append(unif_max_tm_perf[3])
		unif_ave_tm_ahc.append(unif_ave_tm_perf[3])

		multi_toe_multi_te_perf = _evaluate_snapshot_performance(tm, multi_tm_topology, multi_toe_multi_te_routing_weights, link_capacity, total_links, return_lu_distribution=False)
		ave_toe_ave_te_perf = _evaluate_snapshot_performance(tm, ave_topology, ave_toe_ave_te_routing_weights, link_capacity, total_links, return_lu_distribution=False)
		max_toe_max_te_perf = _evaluate_snapshot_performance(tm, max_topology, max_toe_max_te_routing_weights, link_capacity, total_links, return_lu_distribution=False)
		multi_toe_multi_te_mlu.append(multi_toe_multi_te_perf[0])
		ave_toe_ave_te_mlu.append(ave_toe_ave_te_perf[0])
		max_toe_max_te_mlu.append(max_toe_max_te_perf[0])
		multi_toe_multi_te_ahc.append(multi_toe_multi_te_perf[3])
		ave_toe_ave_te_ahc.append(ave_toe_ave_te_perf[3])
		max_toe_max_te_ahc.append(max_toe_max_te_perf[3])
	fig = plt.figure()
	plt.plot(sorted(unif_multi_tm_mlu))
	plt.plot(sorted(unif_max_tm_mlu))
	plt.plot(sorted(unif_ave_tm_mlu))
	plt.plot(sorted(multi_toe_multi_te_mlu))
	plt.plot(sorted(max_toe_max_te_mlu))
	plt.plot(sorted(ave_toe_ave_te_mlu))
	plt.legend(["unif - multi tm", "unif - max", "unif - ave", "multi toe - multi tm", "max toe - max te", "ave toe - ave te"])
	plt.xlim(xmin=0)
	plt.ylim(ymin=0)

	fig = plt.figure()
	plt.plot(sorted(unif_multi_tm_ahc))
	plt.plot(sorted(unif_max_tm_ahc))
	plt.plot(sorted(unif_ave_tm_ahc))
	plt.plot(sorted(multi_toe_multi_te_ahc))
	plt.plot(sorted(max_toe_max_te_ahc))
	plt.plot(sorted(ave_toe_ave_te_ahc))
	plt.legend(["unif - multi tm", "unif - max", "unif - ave", "multi toe - multi tm", "max toe - max te", "ave toe - ave te"])
	plt.xlim(xmin=0)
	plt.ylim(ymin=1, ymax=2)
	plt.show()
	return


if __name__ == "__main__":
	print("Critmat test")
	'''
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
	'''
	aggregation_window = 1
	tm_snapshots_protobuf_filename = "/Users/minyee/src/facebook_dcn_traffic/traffic_matrices/clusterB/web_aggregationwindow_{}.pb".format(aggregation_window)
	valid_network_ids_filename = "/Users/minyee/src/facebook_dcn_traffic/traffic_matrices/clusterB/clusterB_pods.txt"
	tm_snapshots_protobuf_filename = "/Users/minyee/src/facebook_dcn_traffic/traffic_matrices/clustercombined/combined_aggregationwindow_1.pb"
	valid_network_ids_filename = "/Users/minyee/src/facebook_dcn_traffic/traffic_matrices/clustercombined/clustercombined_pods.txt"
	main(tm_snapshots_protobuf_filename, valid_network_ids_filename, aggregation_window)

