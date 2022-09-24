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


def load_cutters(cutters_path: str, display: bool = False) -> list[dict[str, Any]]:
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
			"mask": cutter_mask,
			"value": int(cutter_value),
			"area": np.count_nonzero(cutter_mask)
		})

	if display:
		display_cutters(cutters)
	
	return cutters
		

# -----------------------------
# Image preprocessing functions
# -----------------------------


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
	mask = np.where(mask, 255, 0)
	y0 = np.min(np.argwhere(mask)[:, 0])
	x0 = np.min(np.argwhere(mask)[:, 1])
	y1 = np.max(np.argwhere(mask)[:, 0])
	x1 = np.max(np.argwhere(mask)[:, 1])
	return image[y0:y1+1, x0:x1+1]


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


def display_cutters(cutters: list[dict[int, Any]]):
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
		axes[x, y].imshow(cutter["mask"], cmap="gray", vmin=0, vmax=255)
		axes[x, y].set_title(f"{cutter['name']}, c={cutter['value']}, a={cutter['area']}") 
	figure.suptitle("Cookie cutters")


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
			biscuit_array = np.stack((biscuit_images[biscuit_id], ) * 4, -1)
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
