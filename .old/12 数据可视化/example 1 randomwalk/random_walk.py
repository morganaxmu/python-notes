#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  randomwalk.py
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


from random import choice

class Randomwalk():
	def __init__(self, num_points=5000):
		self.num_points = num_points
		self.x_values = [0]
		self.y_values = [0]
	def fill_walk(self):
		while len(self.x_values) < self.num_points:
			x_direction = choice([1, -1])
			x_distance = choice([0, 1, 2 ,3 ,4])
			x_step = x_direction * x_distance
			y_direction = choice([1, -1])
			y_distance = choice([0, 1, 2 ,3 ,4])
			y_step = y_direction * y_distance
			if x_step ==0 and y_step ==0:
				continue
			next_x = self.x_values[-1] +x_step
			next_y = self.y_values[-1] +y_step
			self.x_values.append(next_x)
			self.y_values.append(next_y)
