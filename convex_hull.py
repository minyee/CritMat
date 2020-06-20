import os, sys, copy
import numpy as np

class ConvexHull(object):
	def __init__(self, hull_points):
		assert(len(hull_points) > 0)
		ndim = len(hull_points[0])
		for hull_point in hull_points:
			assert(len(hull_point) == ndim)
		self.hull_points = hull_points
		self.dimension = ndim
		self.num_hull_points = len(self.hull_points)

		self.element_wise_max = [0] * ndim
		for index in range(self.num_hull_points):
			self.element_wise_max = np.maximum(self.element_wise_max, self.hull_points[index])
		return

	## Checks if the point lies strictly within the convex hull, not as simple as 
	## simply determining whether if the point is element-wise bounded
	def is_bounded(self, vi):
		# use LP?
		return True

	## Checks if the vector is bounded element-wise by at least one of 
	## the hull points
	def is_element_wise_dominated(self,vi):
		assert(len(vi) == self.ndim)
		for i in range(self.ndim):
			if self.element_wise_max[i] < vi[i]:
				return False
		return True