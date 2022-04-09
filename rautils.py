#!/usr/bin/env python
'''

rautils.py

A collection of functions for reading netcdf files for envisat and ers1/ers2
radar altimetry waveforms.

'''

import netCDF4 as nc
import numpy as np
import os
import fnmatch
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors
import matplotlib.patches as patches
import matplotlib as mpl
import glob
from geopy.distance import distance

##
# generic netcdf funcions
def netcdfopen(filename):
    # open a netcdf file
    f = nc.Dataset(filename)
    return f

def netcdfclose(f):
    # close a netcdf file
    f.close()

##
# envisat read functions
def env_file_find(path,trk):
    # find an envisat netcdf file for a given track and cycle number (cycle should be included in file path)
    ra_file=glob.glob(f"{path}*{trk}____PAC_R_NT_003.nc")[0]
    return ra_file

def env_waveform_power_2k(f):
    # read power waveforms from envisat netcdf file
    waveform_power_2k = f.variables['waveform_power_2k']
    pwr = np.array(waveform_power_2k[:])
    return pwr

def env_waveform_phase_2k(f):
    # read phase waveforms from envisat netcdf file
    waveform_phase_2k = f.variables['waveform_phase_2k']
    phs = np.array(waveform_phase_2k[:])
    return phs

def env_lat2k(f):
    # read waveform lat data from envisat netcdf file
    lat = f.variables['lat_2k']
    l = np.array(lat[:])
    return l

def env_lon2k(f):
    # read waveform lon data from envisat netcdf file
    lon = f.variables['lon_2k']
    l = np.array(lon[:])
    return l

def env_fft_sample_ind_ku(f):
	# read wavenumber sample number from netcdf file
	fft_sam = f.variables['fft_sample_ind_ku']
	fs = np.array(fft_sam[:])
	return fs

def env_time_20(f):
	# read time_20 from netcdf file
	t20 = f.variables['time_20']
	t = np.array(t20[:])
	return t

def env_alt_20(f):
	# read alt_20 from netcdf file
	alt20 = f.variables['alt_20']
	a = np.array(alt20[:])
	return a

def env_ref_power_20(f):
	# read ref_power_20 from netcdf file
	rp = f.variables['ref_power_20']
	r = np.array(rp[:])
	return r

def env_lat_20(f):
	# read lat_20 from netcdf file
	l20 = f.variables['lat_20']
	l = np.array(l20[:])
	return l

def env_lon_20(f):
	# read lon_20 from netcdf file
	l20 = f.variables['lon_20']
	l = np.array(l20[:])
	return l

def env_ind_meas_1hz_20(f):
	# read ind_meas_1hz_20 from netcdf file
	im = f.variables['ind_meas_1hz_20']
	i = np.array(im[:])
	return i

def env_lat_cor_20(f):
    # read lat_cor_20 data from envisat netcdf files
    lat = f.variables['lat_cor_20']
    l = np.array(lat[:])
    return l

def env_lon_cor_20(f):
    # read lon_cor_20 data from envisat netcdf files
    lon = f.variables['lon_cor_20']
    l = np.array(lon[:])
    return l

def env_nominal_tracking_20(f):
    # read nominal_tracking_20 from envisat netcdf files
    nt20 = f.variables['nominal_tracking_20']
    n = np.array(nt20[:])
    return n

def env_surf_type_20(f):
    # read surf_type_20 from envisat netcdf file
    st20 = f.variables['surf_type_20']
    s = np.array(st20[:])
    return s

def env_surf_class_20(f):
    # read surf_class_20 from envisat netcdf file
    sc20 = f.variables['surf_class_20']
    s = np.array(sc20[:])
    return s

def env_orb_alt_rate_20(f):
    # read orb_alt_rate_20 from envisat netcdf file
    oa20 = f.variables['orb_alt_rate_20']
    o = np.array(oa20[:])
    return o

def env_slope_ice1_qual_20(f):
    # read slope_ice1_qual_20 from envisat netcdf file
    sq20 = f.variables['slope_ice1_qual_20']
    s = np.array(sq20[:])
    return s

def env_offset_tracking_20(f):
    # read offset_tracking from envisat netcdf file
    ot20 = f.variables['offset_tracking_20']
    o = np.array(ot20[:])
    return o

def env_lat_ref_track_20(f):
    # read lat_ref_track_20 from envisat netcdf file
    lr20 = f.variables['lat_ref_track_20']
    l = np.array(lr20[:])
    return l

def env_lon_ref_track_20(f):
    # read lon_ref_track_20 from envisat netcdf file
    lr20 = f.variables['lon_ref_track_20']
    l = np.array(lr20[:])
    return l

def env_fault_id_20(f):
    # read fault_id_20 from envisat netcdf file
    fi20 = f.variables['fault_id_20']
    f = np.array(fi20[:])
    return f

def env_waveform_fault_id_20(f):
    # read waveform_fault_id_20 from envisat netcdf file
    wf20 = f.variables['waveform_fault_id_20']
    w = np.array(wf20[:])
    return w

def env_waveform_fft_20_ku(f):
    # read waveform_fft_20_ku from envisat netcdf file
    wf20 = f.variables['waveform_fft_20_ku']
    w = np.array(wf20[:])
    return w

def env_tracker_range_20_ku(f):
    # read tracker_range_20_ku from envisat netcdf file
    tr20 = f.variables['tracker_range_20_ku']
    t = np.array(tr20[:])
    return t

def env_tracker_range_qual_20_ku(f):
    # read tracker_range_qual_20_ku from envisat netcdf file
    tr20 = f.variables['tracker_range_20_ku']
    t = np.array(tr20[:])
    return t

def env_chirp_band_20_ku(f):
    # read chirp_band_20_ku from envisat netcdf file
    cb20 = f.variables['chirp_band_20_ku']
    c = np.array(cb20[:])
    return c

def env_chirp_band_qual_20_ku(f):
    # read chirp_band_qual_20_ku from envisat netcdf file
    cb20 = f.variables['chirp_band_qual_20_ku']
    c = np.array(cb20[:])
    return c

def env_range_ocean_20_ku(f):
    # read range_ocean_20_ku from envisat netcdf file
    ro20 = f.variables['range_ocean_20_ku']
    r = np.array(ro20[:])
    return r

def env_range_ice1_20_ku(f):
    # read range_ice1_20_ku from envisat netcdf file
    ro20 = f.variables['range_ice1_20_ku']
    r = np.array(ro20[:])
    return r

def env_elevation_ice1_20_ku(f):
    # read elevation_ice1_20_ku from envisat netcdf file
    ro20 = f.variables['elevation_ice1_20_ku']
    r = np.array(ro20[:])
    return r

def env_elevation_ice1_01_ku(f):
    # read elevation_ice1_1_ku from envisat netcdf file
    ro20 = f.variables['elevation_ice1_01_ku']
    r = np.array(ro20[:])
    return r

def env_time_01(f):
    # read time_01 from envisat netcdf file
    ro20 = f.variables['time_01']
    r = np.array(ro20[:])
    return r

def env_lon_01(f):
    # read lon_01 from envisat netcdf file
    ro20 = f.variables['lon_01']
    r = np.array(ro20[:])
    return r

def env_lat_01(f):
    # read lat_01 from envisat netcdf file
    ro20 = f.variables['lat_01']
    r = np.array(ro20[:])
    return r

##
# ers1/ers2 read functions
def ers_rafile_find(path):
    # find all ers files for a given path
    ra_files = glob.glob(f"{path}*.NC")
    return ra_files

def ers_zipfile_find(path,cyc):
    # find a ers1/ers2 zip file for a given cycle number
    zipfile = glob.glob(f"{path}*{cyc}.ZIP")[0]
    return zipfile

def ers_time(f):
    # read time from ers1/ers2 netcdf file
    time = f.variables['time']
    t = np.array(time[:])
    return t

def ers_meas_ind(f):
    # read meas_ind from ers1/ers2 netcdf file
    meas_ind = f.variables['meas_ind']
    m = np.array(meas_ind[:])
    return m

def ers_time20hz(f):
    # read time 20 hz from ers1/ers2 netcdf file
    time20 = f.variables['time_20hz']
    t = np.array(time20[:])
    return t

def ers_lat(f):
    # read latitudes from ers1/ers2 netcdf files
    lat = f.variables['lat']
    l = np.array(lat[:])
    return l

def ers_lon(f):
    # read longitudes from ers1/ers2 netcdf files
    lon = f.variables['lon']
    l = np.array(lon[:])
    return l

def ers_lat20hz(f):
    # read latitude 20 hz from ers1/ers2 netcdf files
    lat20 = f.variables['lat_20hz']
    l = np.array(lat20[:])
    return l

def ers_lon20hz(f):
    # read longitude 20 hz from ers1/ers2 netcdf files
    lon20 = f.variables['lon_20hz']
    l = np.array(lon20[:])
    return l

def ers_track_qual(f):
    # read tracking quality flag from ers1/ers2 netcdf file
    qual_wf_not_tracking_20hz = f.variables['qual_wf_not_tracking_20hz']
    q = np.array(qual_wf_not_tracking_20hz[:])
    return q

def ers_alt(f):
    # read alt from ers1/ers2 netcdf file
    alt = f.variables['alt']
    a = np.array(alt[:])
    return a

def ers_alt20hz(f):
    # read alt 20 hz from ers1/ers2 netcdf file
    alt20 = f.variables['alt_20hz']
    a = np.array(alt20[:])
    return a

def ers_ku_wf(f):
    # read waveforms from ers1/ers2 netcdf files
    ku_wf = f.variables['ku_wf']
    wf = np.array(ku_wf[:])
    return wf

def ers_surface_type(f):
    # read surface type from ers1/ers2 netcdf file
    surf = f.variables['surface_type']
    s = np.array(surf[:])
    return s

def ers_wvf_ind(f):
    # read wvf_ind from ers1/ers2 netcdf file
    wvf_ind = f.variables['wvf_ind']
    w = np.array(wvf_ind[:])
    return w

##
# data processing/display functions
def track_plot(lat,lon,trk):
    # plot satellite ground track on map using lat/lon coordinates for track number, trk
    coastline_data = np.loadtxt('Coastline.txt',skiprows=1)
    w, h = plt.figaspect(0.5)
    fig = plt.figure(figsize=(w,h))
    ax = fig.gca()
    ##
    #fgcolor='white'
    #bgcolor='black'
    fgcolor='black'
    bgcolor='white'
    fig.suptitle("Track %d" %trk,fontsize=16,color=fgcolor)
    plt.plot(coastline_data[:,0],coastline_data[:,1],'k')
    ax.set_xlabel('Longitude (deg)',fontsize=14,color=fgcolor)
    ax.set_ylabel('Latitude (deg)',fontsize=14,color=fgcolor)
    ax.axes.tick_params(color=fgcolor,labelcolor=fgcolor)
    ax.patch.set_facecolor(bgcolor)
    plt.xlim(-180,180)
    plt.ylim(-90,90)
    plt.yticks([-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90])
    plt.xticks([-180,-150,-120,-90,-60,-30,0,30,60,90,120,150,180])
    ##
    for i1 in range(0,len(lat)):
        if lon[i1] > 180:
            lon[i1] = lon[i1] - 360
        if i1 < 50:
            plt.plot(lon[i1],lat[i1],'b.',markersize=5)
        else:
            plt.plot(lon[i1],lat[i1],'r.',markersize=5)
    ax.grid(True)
    plt.show()
    #plt.savefig('env_track.pdf')

def ers_rangetime(wf,ch1,ch2,bins):
    # create numpy array of waveforms (wf) that is 2D rather than 3D
    # bins is number of range bins for each waveform
    rt = np.zeros((bins,(ch2-ch1+1)*20))
    #m = np.amax(wf,axis=None)
    ii1 = 0
    for i1 in range(0,ch2-ch1):
        for j1 in range(0,20):
            ii1 = ii1 + 1
            m = np.amax(wf[i1+ch1,j1,:],axis=None)
            for k1 in range(0,bins):
                rt[k1,ii1] = wf[i1+ch1,j1,k1] / m
    return rt

def wave_stack(ch1,ch2,pwr,num_gates,dist):
    # create RA cross section by shifting range-wise slices to align range of
    # max return and display aligned wave stack image.
    # ch1, ch2 are beginning and end indices of the 'chunk' of power (pwr) data
    # to be processed and displayed
    stack = np.empty((num_gates*2,(ch2-ch1)*20))
    stack[:] = np.nan
    for i1 in range(0,(ch2-ch1)*20):
        s = max(pwr[:,i1+ch1])
        ind = np.where(pwr[:,i1+ch1] == s)[0]
        for i2 in range(0,num_gates):
            new_ind = (num_gates-ind)+i2
            stack[new_ind,i1] = pwr[i2,i1+ch1]
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_axes([0.075,0.1,0.9,0.85])
    plt.imshow(stack,cmap='inferno',aspect='auto')
    ax.set_xticks(np.arange(0,(ch2-ch1+1)*20,len(dist)))
    ax.set_xticklabels(dist)
    ax.set_xlabel("Along track distance (km)")
    ax.set_ylabel("Relative range")
    plt.show()
    return np.transpose(stack)

def gps_distance(lat,lon):
    # calculate distance between two gps coordinates
    dist = np.empty((len(lat)))
    dist[0] = 0
    for i1 in range(1,len(lat)):
        lt0 = lat[i1-1]
        ln0 = lon[i1-1]
        lt1 = lat[i1]
        ln1 = lon[i1]
        if ln0 > 180:
            ln0 = ln0 - 360
        if ln1 > 180:
            ln1 = ln1 - 360
        c0 = (lt0,ln0)
        c1 = (lt1,ln1)
        dist[i1] = np.round(distance(c0, c1).km + dist[i1-1],1)
    return dist

def env_latlon_search(lat1,lat2,lon1,lon2):
    # search env_track_lookup.txt for tracks with data within a rectangle defined
    # by lat1, lat2, lon1, lon2
    lookup = open("env_track_lookup.txt", "r")
    # read line-by-line looking for tracks in ROI box
    k = 2
    a = 0
    tracks = []
    while 1:
        line = []
        lat0 = 0
        lon0 = 0
        try:
            ll = lookup.readline()
            line = ll.split()
            n = 0
            for i1 in line:
                if n == 0:
                    n = n + 1
                    #print(i1)
                    continue
                else:
                    n = n + 1
                    if (k % 2) == 0:
                        # latitude line
                        if (float(i1) > lat1) and (float(i1) < lat2):
                            lat0 = 1
                            continue
                    else:
                        # longitude line
                        if (float(i1) > lon1) and (float(i1) < lon2):
                            lon0 = 1
                            continue
            if lat0 == 1 and lon0 == 1:
                tracks[a] = line[0]
                print(tracks[a])
                a = a + 1
            k = k + 1
        except:
            print("end of file")
            break
    lookup.close()
    return tracks
