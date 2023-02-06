import math
import matplotlib.pyplot as plt
import time

import numpy as np

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer():
	def __init__(self):
		self._start_time = None
		self.elapsed_time = None

	def start(self):
		"""Start a new timer"""
		if self._start_time is not None:
			raise TimerError(f"Timer is running. Use .stop() to stop it")

		self._start_time = time.perf_counter()

	def stop(self):
		"""Stop the timer, and report the elapsed time"""
		if self._start_time is None:
			raise TimerError(f"Timer is not running. Use .start() to start it")

		self.elapsed_time = time.perf_counter() - self._start_time
		self._start_time = None

	def __repr__(self) -> str:
		if self.elapsed_time == None:
			raise TimerError(f"Timer is running. Use .stop() to stop it.")
		return f"{self.elapsed_time:0.4f}"

def view_image(title, img):
	plt.figure()
	plt.title(title)
	plt.imshow(img, cmap="gray")
	plt.waitforbuttonpress(0)