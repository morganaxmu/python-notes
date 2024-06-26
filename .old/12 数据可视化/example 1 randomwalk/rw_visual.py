#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  rw_visual.py
#  
#  Copyright 2019 billy huang <billy huang@DESKTOP-77CQ0AV>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  


import matplotlib.pyplot as plt
from random_walk import Randomwalk
while True:
	rw = Randomwalk()
	rw.fill_walk()
	point_numbers = list(range(rw.num_points))
	plt.scatter(rw.x_values, rw.y_values, c=point_numbers, cmap=plt.cm.Blues, s=15)
	plt.scatter(0, 0, c='green', s=100)
	plt.scatter(rw.x_values[-1], rw.y_values[-1], c='red', s=100)
	plt.axes().get_xaxis().set_visible(False)
	plt.axes().get_yaxis().set_visible(False)
	plt.show()
	keep_running = input("Make another walk?(y/n):")
	if keep_running == 'n':
		break
	else:
		continue
	
