#!/usr/bin/env python
'''

ers_viewer_prototype.py

 This script interactively displays radar altimetry waveforms, and has been
 written specifically for ers1/ers2 ra waveforms. For envisat ra waveforms,
 use env_bokeh_viewer.py. Command line arguments needed: (1) ers1/2 netcdf file,
 (2) start point for track display, (3) end point for track display, (4) name of output
 html file.

 example usage:
 python3 ers_bokeh_viewer.py ers_granuls.nc 0 100 Waveforms.html

 '''


import numpy as np
import rautils as ra
import glob
import os
import sys
import re
import h5py
from bokeh.plotting import figure, output_file, show
from bokeh.palettes import all_palettes
from bokeh.layouts import column, grid, row
import matplotlib.pyplot as plt
from bokeh.models import ColumnDataSource, HoverTool, TapTool, Button, CustomJS, Div, Slider
from bokeh.transform import factor_cmap
from bokeh.io.doc import curdoc
import pandas as pd
from bokeh import events
from bokeh.io import output_file, show
from bokeh.models.tickers import FixedTicker
import xarray as xr
import geoviews as gv
import geoviews.feature as gf
from geoviews import dim, opts
import geoviews.tile_sources as gvts

# choose bokeh compatibility for geoviews
gv.extension('bokeh')

# helper functions
def xlistToDict(lst):
    op = { i : lst[int(i/20)] for i in range(0, (len(lst))*20, 20 ) }
    return op

def ylistToDict(lst):
    op = { i : lst[i] for i in range(0, len(lst) ) }
    return op

# javascript callbacks
hover_code = """
// initialize js variables
const data = {'x0': [], 'y0': []}
const data2 = {'x1': [], 'y1': []}
const avdata = {'avx0': [], 'avy0': []}
const run_av = {'rav': []}
var fin_av = 0
// read mouse position
const indices = cb_data.geometry.x
// write waveform to data source
for (var i = 0; i < 64; i++) {
    data['x0'].push(i)
    data['y0'].push(radar[63-i][Math.floor(indices)])
    avdata['avx0'].push(i)
    // average 20 waveforms centered on current point
    fin_av = 0
    for (var j = 0; j < 21; j++) {
        fin_av = fin_av + radar[63-i][Math.floor(indices)-10+j]
    }
    fin_av = fin_av / 21
    avdata['avy0'].push(fin_av)
}
// write lat/lon info to data source
const clat = latt[Math.floor(indices)]
const clon = lonn[Math.floor(indices)]
data2['x1'].push(clon)
data2['y1'].push(clat)
// push changes and update figures
line.data = data
circle.data = data2
average.data = avdata
"""

slider_code = """
const data = {'x2': [], 'y2': []}
const s_val0 = cb_obj.value - 1
const s_val1 = s_val0 + inc
const lon2_data = {'lon2': []}
const lat2_data = {'lat2': []}
const lon3_data = {'lon3': []}
const lat3_data = {'lat3': []}
for (var i = s_val0; i < s_val1; i++) {
    data['x2'].push(lonn[i])
    data['y2'].push(latt[i])
}
for (var i = 0; i < inc; i++) {
    lat2_data['lat2'].push(latt[i+s_val0])
    lon2_data['lon2'].push(lonn[i+s_val0])
}
track.data = data

"""

# get data directory and track start and end points from command line arguments
data_file = sys.argv[1]
ch1 = int(sys.argv[2]) # ch1,ch2 are start stop points for track
ch2 = int(sys.argv[3])
html_file = sys.argv[4]

# open netcdf file
f = ra.netcdfopen(data_file)

# read radar waveforms, latitude, longitude, and quality flags
lat = ra.ers_lat(f)
lon = ra.ers_lon(f)
lat20 = ra.ers_lat20hz(f)
lon20 = ra.ers_lon20hz(f)
wf = ra.ers_ku_wf(f)
qual = ra.ers_track_qual(f)
if ch2 == 0:
    ch2 = len(lat)
slid_inc = 50

#close netcdf file
ra.netcdfclose(f)

# select part of orbit to plot
lat2 = lat[ch1:ch2+1]
lon2 = lon[ch1:ch2+1]
lat3 = np.zeros(len(lat2)*20)
lon3 = np.zeros(len(lon2)*20)

# create lat/lon arrays same length as number of waveforms
count = 0
for i1 in range(0,len(lat2)):
    for j1 in range(0,20):
        lat3[count] = lat20[i1+ch1,j1]
        lon3[count] = lon20[i1+ch1,j1]
        count = count + 1

# calculate along-track distance
dist = ra.gps_distance(lat2,lon2)
# create waveform stack image
rt0 = ra.ers_rangetime(wf,ch1,ch2,64)
global rt
rt = np.flip(rt0,axis=0)
dist = ra.gps_distance(lat2,lon2)
locs = np.arange(0,(ch2-ch1+1)*20,20)

# create bokeh linked plots
select_tools = ['box_select', 'lasso_select', 'poly_select', 'tap', 'reset', 'hover','box_zoom']
TOOLTIPS = [
    ("x", "$x"),
    ("y", "$y"),
]

# set data sources
# where the mouse is hovering on the waveform image
wf_pos_source = ColumnDataSource({'x0': [], 'y0': []})
# the position of circle on map
map_circ_source = ColumnDataSource({'x1': [], 'y1': []})
# track segment displayed on map
map_track_source = ColumnDataSource({'x2': [], 'y2': []})
# average waveform source
wf_av_source = ColumnDataSource({'avx0': [], 'avy0': []})
# waveform stack image source




#set html output file
#output_file("Waveforms.html", title="Waveforms")
output_file(html_file, title=data_file)

# create slider for data display
slider = Slider(start=0, end=(len(lat)-slid_inc), value=1, step=1, title="Position",height=50,width=750)

waveforms = figure(plot_width=1800, plot_height=400, tools=select_tools,tooltips=TOOLTIPS,title=data_file)
waveforms.x_range.range_padding = waveforms.y_range.range_padding = 0
waveforms.xaxis.axis_label = 'Along track distance (km)'
waveforms.yaxis.axis_label = 'Relative range'
numrows = rt.shape[0]
numcols = rt.shape[1]
columns = [f'col_{num}' for num in range(numcols)]
index = [f'index_{num}' for num in range(numrows)]
waveforms.image(image=[rt], x=0, y=0, dw=lat2.size*20, dh=64, palette="Inferno256")
waveforms.xaxis.ticker = FixedTicker(ticks=locs)
xstrList = ['{:.1f}'.format(x) for x in dist.tolist()]
xlabs = xlistToDict(xstrList)
waveforms.xaxis.major_label_overrides = xlabs
r = np.arange(0,64,1)
r = np.flip(r,axis=0)
ystrList = ['{:.0f}'.format(x) for x in r.tolist()]
ylabs = ylistToDict(ystrList)
waveforms.yaxis.ticker = FixedTicker(ticks=np.arange(0,64,10))
waveforms.yaxis.major_label_overrides = ylabs

coastline_data = np.loadtxt('Coastline.txt',skiprows=1)
map = figure(plot_width=850,plot_height=500,x_range=(-180, 180),y_range=(-90,90))
map.line(coastline_data[:,0],coastline_data[:,1],color='black')
tr = map.circle(x='x2', y='y2', source=map_track_source, color='blue', size=10)
cr = map.circle(x='x1', y='y1', source=map_circ_source,color='red',size=10)
map.xaxis.axis_label = 'Longitude (degrees)'
map.yaxis.axis_label = 'Latitude (degrees)'
map.yaxis.ticker = FixedTicker(ticks=np.arange(-90,90,10))
map.xaxis.ticker = FixedTicker(ticks=np.arange(-180,180,20))

ind_wf = figure(plot_width=850, plot_height=500)
ln = ind_wf.line(x='x0', y='y0',source=wf_pos_source,color='black', legend_label='waveform')
av = ind_wf.line(x='avx0', y='avy0', source=wf_av_source,color='red', legend_label='average')
ind_wf.xaxis.axis_label = 'Relative range'
ind_wf.yaxis.axis_label = 'Normalized power'
ind_wf.legend.location = 'top_left'
ind_wf.legend.title_text_font = 'Arial'
ind_wf.legend.title_text_font_size = '20pt'

# add hover tool to waveforms figure
hover_callback = CustomJS(args={'line': ln.data_source, 'average': av.data_source, 'radar': rt, 'circle': cr.data_source, 'latt': lat3, 'lonn': lon3}, code=hover_code)
waveforms.add_tools(HoverTool(tooltips=None, callback=hover_callback, renderers=[ln,cr,av,tr]))

# add slider functionality
slider_callback = CustomJS(args = {'track': tr.data_source, 'latt': lat, 'lonn':lon, 'inc':slid_inc}, code=slider_code)
slider.js_on_change('value', slider_callback)

# create layout to display all figures
p = grid([
    [slider],
    [waveforms],
    [map, ind_wf],
], sizing_mode='fixed')

# display figures
show(p)
