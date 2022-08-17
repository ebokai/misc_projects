import numpy as np 
from itertools import combinations, permutations

def gsbm(nc, npc, p, q):

	"""
	gsbm: graph stochastic block model

	--- parameters ---

	nc: number of clusters
	npc: nodes per cluster
	p: probability of connection within cluster
	q: probability of connection outside cluster 

	--- returns ---

	t: adjacency matrix

	"""

	n = nc * npc # number of nodes 

	clusters = [[i + npc * j for i in range(npc)] for j in range(nc)]

	t = np.zeros((n,n))

	for i in range(nc):
		for j in range(i,nc):

			cluster1 = clusters[i]
			cluster2 = clusters[j]

			# generate all edges and set probability P
			if i == j:
				edges = np.array(list(combinations(cluster1, 2)))
				P = p	
			else:
				# generate internal edges and all edges
				in_edges = list(combinations(cluster1, 2)) + list(combinations(cluster2, 2))
				all_edges = list(combinations(list(cluster1) + list(cluster2), 2))

				# remove internal edges
				edges = np.array([edge for edge in all_edges if edge not in in_edges])
				P = q

			# select edges with probability P 
			connections = np.random.binomial(1,P,len(edges)).astype(bool)
			selected_edges = edges[connections]

			# construct adjacency matrix
			for edge in selected_edges:
				t[edge[0],edge[1]] = 1
				t[edge[1],edge[0]] = 1

	return t

def hgsbm(nc, npc, p, q, d, edge_mode = False):

	"""
	hgsbm: hyper-graph stochastic block model

	d-uniform version

	--- parameters ---

	nc: number of clusters
	npc: nodes per cluster
	p: probability of connection within cluster
	q: probability of connection outside cluster 
	d: degree of hyperedge
	edge_mode: returns edges instead of adjacency matrix

	--- returns ---

	t: adjacency tensor (if edge_mode = False)
	edges: hyper-edges of the network (if edge_mode = True)

	"""

	n = nc * npc 

	clusters = np.array([[i + npc * j for i in range(npc)] for j in range(nc)])

	# set tensor dimensions
	dim = tuple([n for i in range(d)])
	t = np.zeros((dim))

	# loop over cluster combinations of length d
	for i in np.arange(1,2**nc):

		# convert to index mask
		mask = np.flip(np.array([int(x) for x in format(i,f'0{nc}b')], dtype = bool))

		# select nodes in cluster combination
		selected_nodes = clusters[mask]

		# generate all edges
		all_edges = list(combinations(selected_nodes.flatten(), d))

		# check if cluster combination involves external cluster
		check = bin(i).count('1')

		# if external, remove internal edges and set connection probability
		if check != 1:

			P = q

			# generate internal edges
			in_edges = []
			for cluster_nodes in selected_nodes:
				in_edges += list(combinations(cluster_nodes.flatten(), d))

			# remove internal edges 
			edges = np.array([edge for edge in all_edges if edge not in in_edges])
		
		else:
			P = p
			edges = np.array(all_edges)

		# select connected edges
		connections = np.random.binomial(1,P,len(edges)).astype(bool)
		selected_edges = edges[connections]

	if edge_mode:

		return selected_edges

	else:

		for edge in selected_edges:
			
			# generate all permutations of edge
			edge_perms = np.array(list(permutations(edge,d)))
			
			# construct adjacency tensor
			for edge_idx in edge_perms:
				t[edge_idx] = 1

		return t


def sbm_model(nc, npc, p, q, orders):

	"""
	generate a model (list of interactions)
	using the hgsbm algorithm

	--- parameters ---

	nc: number of clusters
	npc: number of nodes per cluster
	p: probability of connection within cluster
	q: probability of connection outside cluster 
	orders: list of interaction orders OR single interaction order

		example: 
		input [2,3,5] would result in a network with only
		2-, 3-, and 5-body interactions

	--- returns ---
	
	n: number of nodes corresponding to network structure
	model: sorted list of interactions in integer format


	"""

	# convert orders to list (allows input of single integer)
	orders = np.array([orders]).flatten() 

	n = nc * npc # number of nodes 

	# check if interaction orders are valid
	for order in orders:

		if order <= 1 or order > n:

			raise ValueError(f'List of interaction orders contains invalid order {order}')

	# check if probabilities are valid

	if p < 0 or p > 1:

		raise ValueError('Probability p is invalid. Select a value between 0 and 1')

	if q < 0 or q > 1:

		raise ValueError('Probability p is invalid. Select a value between 0 and 1')

	model = []

	print('generating model...')
	print(f'{n} nodes, {nc} clusters, {npc} nodes per cluster')

	for order in orders:

		edges = hgsbm(nc, npc, p, q, order, edge_mode = True)

		print(f'{len(edges)} interactions of order {order}')

		for edge in edges:

			op = np.sum([2**x for x in edge])

			model.append(op)

	return n, sorted(model)






		

		



