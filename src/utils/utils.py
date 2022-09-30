# Optimization Project: Biscuit Optimizer
# Roberto Basla
# Politecnico di Milano
# A.Y. 2021/2022
#
# Utilities for Python notebooks

from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import colorsys
import cv2
import os
import math


# ---------------
# Input utilities
# ---------------


def load_base_image(image_path: str, bitmask_path: str, display: bool = False) -> tuple[np.array, np.array]:
	"""
	Loads the base image and its bitmask from the given paths
	Args:
		image_path: string path to the base image
		bitmask_path: string path to the base image's binary mask
		display: if true the images are displayed
	
	Returns:
		base_image: the initial image
		bitmask: the image's binary mask
	"""
	# Images input
	base_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
	bitmask = np.where(cv2.imread(bitmask_path, cv2.IMREAD_GRAYSCALE), 1, 0)

	# Crop to the mask
	base_image = crop_image(base_image, bitmask)
	bitmask = crop_image(bitmask)

	if display:
		print("Bitmask shape:", bitmask.shape)
		display_initial_images(base_image, bitmask)
	
	return base_image, bitmask


def load_cutters(cutters_path: str, use_image: bool = False, display: bool = False, title: str = "Cookie cutters") -> list[dict[str, Any]]:
	"""
	Loads the cutters' bitmasks. Files are assumed to be named as <cutter_name>.<cutter_value>.png
	Args:
		cutter_path: path to the folder containing only cutter bitmasks
		display: if True shows the cutter masks
	Returns:
		cutters: list of cutter dictionaries having keys name, mask, value and area
	"""
	cutters = []
	for image_name in os.listdir(cutters_path):
		# Captures information from the file name
		cutter_name, cutter_value, _ = image_name.split(".")
		cutter_mask = cv2.imread(os.path.join(cutters_path, image_name), cv2.IMREAD_GRAYSCALE)
		cutters.append({
			"name": cutter_name,
			"mask": np.where(cutter_mask, 1, 0) if not use_image else cutter_mask,
			"value": int(cutter_value),
			"area": np.count_nonzero(cutter_mask)
		})

	if display:
		display_cutters(cutters, title)
	
	return cutters
		

# -----------------------------
# Image preprocessing functions
# -----------------------------

def crop_coordinates(mask):
	"""
	Computes the coordinates of a crop given a full mask
	Args:
		mask: mask of which to compute the 
	"""
	mask = np.where(mask, 255, 0)
	y0 = np.min(np.argwhere(mask)[:, 0])
	x0 = np.min(np.argwhere(mask)[:, 1])
	y1 = np.max(np.argwhere(mask)[:, 0])
	x1 = np.max(np.argwhere(mask)[:, 1])
	return y0, x0+1, y1, x1+1


def crop_image(image: np.array, mask: np.array = None) -> np.array:
	"""
	Crops the image according to the mask (or to the image itself if the mask is not provided)
	Args:
		image: the image to crop
		mask: the optional mask for cropping
	Returns:
		image: the cropped image
	"""
	if mask is None:
		mask = image
	y0, x0, y1, x1 = crop_coordinates(mask)
	return np.copy(image[y0:y1, x0:x1])


def subsample_base_image(base_image: np.array, bitmask: np.array, resize_factor: float, 
		display: bool = False) -> tuple[np.array, np.array]:
	"""
	Reduces the base image's dimensions according to the resize factor
	Args:
		base_image: the initial image
		bitmask: the initial image's binary mask
		resize_factor: the target resize factor
		display: if True shows the images
	Returns:
		subsampled_image: the resized base image
		subsampled_bitmask: the resized binary mask
	"""
	# Computes the resizing shape
	new_shape = (int(base_image.shape[1] * resize_factor), int(base_image.shape[0] * resize_factor))
	# Resizes the images
	subsampled_image = cv2.resize(base_image, new_shape, interpolation=cv2.INTER_LINEAR)
	subsampled_bitmask = cv2.resize(bitmask, new_shape, interpolation=cv2.INTER_NEAREST)
	
	if display:
		print("Subsampled shape:", new_shape)
		display_initial_images(subsampled_image, subsampled_bitmask)
	
	return subsampled_image, subsampled_bitmask


# ------------------------
# Images display utilities
# ------------------------


def display_initial_images(image: np.array, bitmask: np.array):
	"""
	Shows the initial image and its bitmask
	Args:
		image: the base image
		bitmask: the image's binary mask
	"""
	fig, axes = plt.subplots(1, 2, figsize=(20, 8))
	axes[0].imshow(image)
	axes[0].set_title("Base image")
	axes[1].imshow(bitmask, cmap="gray")
	axes[1].set_title("Dough bitmask")
	fig.suptitle("Input dough")
	plt.show()


def display_cutters(cutters: list[dict[int, Any]], title):
	"""
	Shows the loaded cutters
	Args:
		cutters: the list of cutter dictionaries
	"""
	n_cutters = len(cutters)
	# Build a plot with 3 columns and as many rows as needed
	n_rows = math.ceil(n_cutters / 3)
	figure, axes = plt.subplots(n_rows, 3, figsize=(10, 3 * n_rows + 1))
	for i, cutter in enumerate(cutters):
		x = math.floor(i / 3)
		y = i % 3
		axes[x, y].imshow(cutter["mask"], cmap="gray")
		axes[x, y].set_title(f"{cutter['name']}, c={cutter['value']}, a={cutter['area']}") 
	figure.suptitle(title)


def show_pan(biscuit_images: dict[int, np.array], biscuit_counts: dict[int, float], 
		margin: int = 5, pan_side: int = 250, biscuit_color: tuple[int, int, int] = (239, 204, 162), 
		fill_color: tuple[int, int, int] = (50, 50, 50)):
	"""
	Visualization of a solution as a biscuit pan
	Args:
		biscuit_images: dictionary associating cutter ids to their masks
		cutter_names: list of cutter names
		biscuit_counts: dictionary associating each cutter id to the number of biscuits in the solution
		margin: number of pixels between biscuits
		pan_side: side length (in pixels) of the square pan image
		fill_color: pan background color
	"""
	# Pan initialization
	pan = np.full(shape=(pan_side, pan_side, 3), fill_value=fill_color)
	# Row height based on the tallest biscuit
	current_row_height = 0
	# Current x position in the pan
	pan_x = margin
	# Current y position in the pan
	pan_y = margin

	for biscuit_id, n in biscuit_counts.items():
		biscuit = biscuit_images[biscuit_id]
		biscuit_height, biscuit_width = biscuit.shape[:2]
		for _ in range(int(n)):
			if pan_x + biscuit_width > pan_side:
				# Change row if overflowing
				pan_y += current_row_height + margin
				pan_x = margin
				current_row_height = 0

			# Overwrites the pan with the biscuit
			pan[pan_y:pan_y+biscuit_height, pan_x:pan_x+biscuit_width] = \
				np.where(np.stack((biscuit, ) * 3, axis=-1), biscuit_color, fill_color)
			pan_x += biscuit_width + margin
			current_row_height = max(current_row_height, biscuit_height)
	
	# Displays the pan
	plt.figure(figsize=(10, 10))
	plt.imshow(pan)
	plt.axis("off")
	plt.show()


def show_cut_dough(dough_image: np.array, biscuit_images: dict[int, np.array], 
		biscuit_coords: dict[int, list[tuple[int, int]]], title: str = None, figure: plt.figure = None):
	"""
	Displays the cutter masks overlayed to the dough image
	Args:
		dough_image: the initial image over which cutters are drawn
		biscuit_images: dictionary associating cutter names to masks
		biscuit_coords: dictionary associating to each cutter id the list of y coordinates
		title: optional title of the image
		figure: optional matplotlib figure to use for the image
	"""
	# Image initialization
	base_image = Image.fromarray(dough_image).convert("RGBA")
	color_h = 0
	for biscuit_id, coordinates in biscuit_coords.items():		
		for coordinate in coordinates:
			# Creates the 4-channel image
			biscuit_array = np.stack((biscuit_images[biscuit_id], ) * 4, -1).astype(np.uint8)
			# Sets the color and alpha from HSV to cycle them
			biscuit_array[:, :, 3] = np.where(biscuit_array[:, :, 3], 175, 0)
			color = tuple(int(channel * 255) for channel in colorsys.hsv_to_rgb(color_h, 1, 1))
			biscuit_array[:, :, :3] = np.where(biscuit_array[:, :, :3], color, 0)

			# Copies the semi-transparent biscuit image on the base image
			biscuit_image = Image.fromarray(biscuit_array)
			base_image.paste(biscuit_image, coordinate[::-1], biscuit_image)
			
			# Updates the next color
			color_h = color_h + 0.1

	if figure is None:
		# Creates the figure
		plt.figure(figsize=(10, 10))

	plt.imshow(base_image)

	if title is not None:
		plt.title(title)
	plt.axis("off")

	if figure is None:
		plt.show()


def show_solution_time(all_solutions: list[dict[str, Any]]):
	"""
	Plots the solution values against time

	Args:
		all_solutions: list of solutions generated by the solution callback
	"""
	# List of times
	xs = []
	# List of values
	ys = []
	for solution in all_solutions:
		xs.append(solution["time"])
		ys.append(solution["value"])

	plt.figure(figsize=(15, 7))
	plt.plot(xs, ys, marker="o")
	plt.xlabel("Time (s)")
	plt.ylabel("Objective value")
	plt.title("Time and value of each improving solution found")


def show_cost_map(cost_map, i=0, cmap="cividis"):
	"""
	Displays the given cost map with a layer detail
	Args:
		cost_map: The cost map to display
		i: index of the detail over axis 0
		cmap: color map to use
	"""
	flat_cost_map = np.sum(cost_map, axis=0)
	print("(Flattened) Shape:", flat_cost_map.shape, "Min:", np.min(flat_cost_map), "Max:", np.max(flat_cost_map))
	print(f"(Layer {i}) Shape:", cost_map[0].shape, "Min:", np.min(cost_map[0]), "Max:", np.max(cost_map[0]))

	fig, ax = plt.subplots(1, 2, figsize=(8, 4))
	ax[0].imshow(flat_cost_map, cmap=cmap)
	ax[0].set_title("Flattened cost map")
	ax[1].imshow(cost_map[0], cmap=cmap)
	ax[1].set_title(f"Cost map layer {i}")
	plt.show()


def show_test_results(values_dict: dict[int, int], n_tests, title: str = None):
	"""
	Displays the distribution of results of in values_dict
	Args:
		values_dict: Dictionary of result: frequency
		title: Title of the plot
	"""
	print("Average value:", sum(v * n for v, n in values_dict.items()) / n_tests)
	plt.figure(figsize=(10, 7))
	plt.bar(values_dict.keys(), values_dict.values(), width=50)
	if title is None:
		title = "Heuristics"
	plt.title(f"{title} - {n_tests} simulations")
	plt.xlabel("Final value")
	plt.ylabel("Frequency")
	plt.show()


def show_all_results(results_dict: dict[str, dict[int, int]]):
	"""
	Displays the distribution of results of all different strategies in results_dict
	Args:
		results_dict: Dictionary of label: result
	"""
	plt.figure(figsize=(15, 10))
	labels = []
	shift = -30
	for title, results in results_dict.items():
		plt.bar(np.array(list(results.keys())) + shift, results.values(), width=40)
		labels.append(title)
		shift += 15

	plt.legend(labels=labels)
	plt.xlabel("Final value")
	plt.ylabel("Frequency")
	plt.show()


# ---------------
# Model utilities
# ---------------


def can_host(base_mask: np.array, cutter_mask: np.array, n: int, m: int) -> bool:
	"""
	Checks if cutter_mask can be placed over base_mask in coordinates (n, m)
	Args:
		base_mask: The binary mask of the available dough
		cutter_mask: The binary mask of the cutter to place
		n: Row where to place cutter_mask
		m: Column where to place cutter_mask
	Returns:
		True if the cutter mask falls entirely in the base_mask
	"""
	in_range = n + cutter_mask.shape[0] <= base_mask.shape[0] and m + cutter_mask.shape[1] <= base_mask.shape[1]
	if not in_range:
		# (n, m) is in OUT_i for cutter_mask i
		return False
	# Overlaps the two masks; if base_mask can entirely host cutter_mask, the result equals cutter_mask
	masks_and = np.logical_and(base_mask[n:n+cutter_mask.shape[0], m:m+cutter_mask.shape[1]], cutter_mask)
	return np.array_equal(masks_and, cutter_mask)


def compute_feasible_y(dough_mask: np.array, cutters: dict[int, dict[str, Any]]) -> np.array:
	"""
	Computes which (n, m) can host a cutter for each cutter type.
	Args:
		dough_mask: The starting mask
		cutters: The problem's cutters
	Returns:
		An np array of bools with one set of feasible (n, m) for each cutter
	"""
	y = np.zeros((len(cutters), *dough_mask.shape))
	for i, cutter in enumerate(cutters):
		cutter_mask = cutter["mask"] > 0
		for n in range(dough_mask.shape[0]):
			for m in range(dough_mask.shape[1]):
				y[i, n, m] = can_host(dough_mask, cutter_mask, n, m)					
	return y


def update_feasible_y(current_y: np.array, occupancy_table: np.array, cutters: dict[int, dict[str, Any]], i: int,
		n: int, m: int) -> np.array:
	"""
	Changes the y according to the placement of the given cutter
	Args:
		current_y: The current y np array
		occupancy_table: Array of sets describing which ys are incompatible with the activation of a given (i, n, m)
		cutters: The problem's cutters
		i: The id of the placed cutter
		n: The row where the cutter is placed
		m: The column where the cutter is placed
	Returns:
		The updated y array
	"""
	cutter_mask = cutters[i]["mask"] > 0
	updated_y = np.copy(current_y)
	for h in range(cutter_mask.shape[0]):
		for k in range(cutter_mask.shape[1]):
			if cutter_mask[h, k]:
				for (j, l, o) in occupancy_table[n+h, m+k]:
					updated_y[j, l, o] = False
	return updated_y


def mask_from_y(feasible_y: np.array, cutters: list[dict[int, Any]]) -> np.array:
	"""
	Converts a 3-dimensional solution to the cut dough mask
	Args:
		feasible_y: NumPy array of the solution
		cutters: The dictionary of cutters
	Returns:
		The cut dough mask as a 2D NumPy array
	"""
	full_mask = np.zeros(feasible_y.shape[1:])
	for (i, n, m) in np.argwhere(feasible_y):
		cutter_mask = cutters[i]["mask"] > 0
		full_mask[n:n+cutter_mask.shape[0], m:m+cutter_mask.shape[1]] = np.where(cutter_mask, True, 
			full_mask[n:n+cutter_mask.shape[0], m:m+cutter_mask.shape[1]])
	return full_mask


def mask_usable(dough_mask: np.array, cutters: dict[int, dict[str, Any]]) -> np.array:
	"""
	Computes the parts of the dough_mask that can actually be used
	Args:
		dough_mask: The mask of the input dough
		cutters: Dictionary of cutters
	Returns:
		usable_mask: Mask True where z variables can be activated
	"""
	full_mask = np.zeros(dough_mask.shape)
	for n in range(dough_mask.shape[0]):
		for m in range(dough_mask.shape[1]):
			for i, cutter in enumerate(cutters):
				cutter_mask = cutter["mask"] > 0
				if can_host(dough_mask, cutter_mask, n, m):
					# Activates the parts of the mask covered by y[n, m, i]
					full_mask[n:n+cutter_mask.shape[0], m:m+cutter_mask.shape[1]] = \
						np.logical_or(full_mask[n:n+cutter_mask.shape[0], m:m+cutter_mask.shape[1]], cutter_mask)

	return np.logical_and(dough_mask, full_mask)


def compute_waste(mask_1: np.array, mask_2: np.array) -> int:
	"""
	Computes the number of wasted pixels in going from mask_1 to mask_2
	Args:
		mask_1: First mask to compare
		mask_2: Second mask to compare
	Returns:
		The number of wasted pixels
	"""
	return np.count_nonzero(np.logical_xor(mask_1, mask_2))


def compute_occupancy_table(dough_mask: np.array, cutters: dict[int, dict[str, Any]]) -> np.array:
	"""
	Computes the occupancy table where each entry at index (i, n, m) consists in the set of ys that would be made 
	infeasible by activating y[i, n, m]
	Args:
		dough_mask: The initial mask
		cutters: The set of cutters
	Returns:
		The cost map
	"""
	occupancy_table = [[set() for m in range(dough_mask.shape[1])] for n in range(dough_mask.shape[0])]
	for i, cutter in enumerate(cutters): 
		cutter_mask = cutter["mask"] > 0
		for n in range(dough_mask.shape[0]): 
			for m in range(dough_mask.shape[1]): 
				if can_host(dough_mask, cutter_mask, n, m):
					for h in range(cutter_mask.shape[0]):
						for k in range(cutter_mask.shape[1]):
							if cutter_mask[h, k]:
								occupancy_table[n+h][m+k].add((i, n, m))
	return np.array(occupancy_table)


def compute_value_cost(dough_mask: np.array, cutters: dict[int, dict[str, Any]]) -> np.array:
	"""
	Computes the negative value cost map
	Args:
		dough_mask: The initial mask
		cutters: The set of cutters
	Returns:
		The cost map
	"""
	return -np.stack([np.full_like(dough_mask, cutter["value"]) for cutter in cutters], axis=0)


def compute_area_cost(dough_mask: np.array, cutters: dict[int, dict[str, Any]]) -> np.array:
	"""
	Computes the area cost map
	Args:
		dough_mask: The initial mask
		cutters: The set of cutters
	Returns:
		The cost map
	"""
	return np.stack([np.full_like(dough_mask, np.count_nonzero(cutter["mask"] > 0)) for cutter in cutters], axis=0)


def compute_occupancy_cost(dough_mask: np.array, cutters: dict[int, dict[str, Any]], 
		occupancy_table: np.array) -> np.array:
	"""
	Computes the occupancy cost map
	Args:
		dough_mask: The initial mask
		cutters: The set of cutters
	Returns:
		The cost map
	"""
	occupancy_cost_map = np.zeros((len(cutters), *dough_mask.shape))
	for i in range(len(cutters)):
		cutter_mask = cutters[i]["mask"] > 0
		for n in range(dough_mask.shape[0]):
			for m in range(dough_mask.shape[1]):
				if can_host(dough_mask, cutter_mask, n, m):
					for h in range(cutter_mask.shape[0]):
						for k in range(cutter_mask.shape[1]):
							if cutter_mask[h, k]:
								occupancy_cost_map[i, n, m] += len(occupancy_table[n+h, m+k])
	# Dvided to be comparable to values
	occupancy_cost_map /= 10000
	return occupancy_cost_map
