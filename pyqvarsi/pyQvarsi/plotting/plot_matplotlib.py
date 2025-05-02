#!/usr/bin/env python
#
# pyQvarsi, utils.
#
# Plotting utility routines.
#
# Last rev: 19/10/2021
from __future__ import print_function, division

import math

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import matplotlib.tri as tri

from mpl_toolkits.axes_grid1 import make_axes_locatable

class Plotter(object):
	def __init__(self,opt_dict=None):
		self.options = {"globalfontsize": 13,
		                "fontsize": 13,
		                "usetex": True,
		                "fontfamily": "cm",
		                "latex_preamble": r"\usepackage{amsmath}",
		                "max_open_warning": False,
		                "xlabelsize": 13,
		                "ylabelsize": 13,
		                "titlesize": 13,
		                "lineswidth": 0.5,
		                "unicode_minus:": False}
		self.set_options(opt_dict)

	def set_options(self,opt_dict=None):
		if opt_dict is not None:
			for k, v in opt_dict.items():
				self.options[k] = v

		if self.options["globalfontsize"]:
			self.options["fontsize"] = self.options["globalfontsize"]
			self.options["xlabelsize"] = self.options["globalfontsize"]
			self.options["ylabelsize"] = self.options["globalfontsize"]
			self.options["titlesize"] = self.options["globalfontsize"]

		plt.rcParams.update({
			"font.size": self.options["fontsize"],
			"text.usetex": self.options["usetex"],
			"font.family": self.options["fontfamily"],
		})

		mpl.rcParams.update({
			"text.latex.preamble": self.options["latex_preamble"],
			"figure.max_open_warning": self.options["max_open_warning"],
			"xtick.labelsize": self.options["xlabelsize"],
			"ytick.labelsize": self.options["ylabelsize"],
			"axes.titlesize": self.options["titlesize"],
			"axes.linewidth": self.options["lineswidth"],
			#"axes.unicode_minus": self.options["unicode_minus"],
		})

		return

	def plot2Dtri(self, field, connectList, x, y, **kwargs):
		"""

		"""
		import matplotlib.tri as tri
		from mpl_toolkits.axes_grid1 import make_axes_locatable

		fig, ax = plt.subplots(1, 1)

		# Get contour options
		contour = kwargs.get('contour', False)
		contour_color = kwargs.get('contour_color', 'black')
		contourf = kwargs.get('contourf', True)
		levels = kwargs.get('levels', 50)
		lim = kwargs.get('lim', [np.min(field), np.max(field)])
		lim_max = kwargs.get('lim_max', lim)
		cmap = kwargs.get('cmap', 'Blues')
		title = kwargs.get('title', '')
		n_ticks = kwargs.get('n_ticks', 10)
		n_decimals = kwargs.get('n_decimals', 2)
		ticks_flag = kwargs.get('ticks_flag', True)
		cbar_flag = kwargs.get('cbar_flag', False)
		eps = kwargs.get('eps', 0.0001)
		dpi = kwargs.get('dpi', 100)
		pad_inches = kwargs.get('pad_inches', 0)
		fname = kwargs.get('fname', 0),

		elems = tri.Triangulation(x, y, connectList)

		# Create contourf given a normalized (norm) colormap (cmap)
		clvls = levels
		if lim[0] < 0 < lim[1]:
			ll = np.linspace(lim[0], -eps, int(math.ceil(levels/2)))
			rr = np.linspace(eps, lim[1], int(math.ceil(levels/2)))
			lvls = np.append(ll, rr)
			if contour:
				extra_levels = 5
				dl = lvls[1] - lvls[0]
				clvls = np.append(lvls, np.linspace(lim[1] + dl, lim[1] + extra_levels * dl, extra_levels))
		else:
			lvls = np.linspace(lim[0], lim[1], levels + 1)

		if contour:
			ax.tricontour(elems, field, clvls, linewidths=0.1, colors=contour_color)
		if contourf:
			pt = ax.tricontourf(elems, field, lvls, vmin=lim_max[0], vmax=lim_max[1], cmap=cmap, extend='both')

		# Format figure
		ax.set_aspect(1)
		ax.set_xlim(np.min(x), np.max(x))
		ax.set_ylim(np.min(y), np.max(y))

		# Add colorbar
		if cbar_flag and contourf:
			divider = make_axes_locatable(ax)
			cax = divider.append_axes("right", size="5%", pad=0.15)
			if lim[0] < 0:
				tick1 = np.linspace(lim[0], 0, int(math.ceil(n_ticks/2)))
				dl = tick1[1] - tick1[0]
				tick2 = np.linspace(dl, lim[1], int(math.ceil(n_ticks/2)))
				ticks = np.append(tick1, tick2)
			else:
				ticks = np.linspace(lim[0], lim[1], n_ticks + 1)
			cbar = fig.colorbar(pt, cax=cax, ticks=ticks)
			fmt_str = r'${:.' + str(n_decimals) + 'f}$'
			cbar.ax.set_yticklabels([fmt_str.format(t) for t in ticks])
			cbar.ax.yaxis.set_tick_params(pad=5, direction='out', size=1)  # your number may vary
			cbar.ax.set_title(title, x=1, y=1.02, loc='left', size=12)

		if ticks_flag:
			ax.tick_params(bottom="on", top="on", right="on", which='both', direction='in', length=2)
		else:
			ax.tick_params(bottom="off", top="off", right="off", left="off")
			ax.xaxis.set_ticks([])
			ax.yaxis.set_ticks([])

		# if fname:
		# 	fig.savefig(fname)#transparent=True, bbox_inches='tight', pad_inches=pad_inches, dpi=dpi)

		plt.show()
		plt.clf()
		return