# -*- coding: utf-8 -*-
"""
USGS HTMC HISTORICAL MAP PROCESSOR
Created 2021-2022
@author: Johannes H. Uhl, University of Colorado Boulder, USA
"""

import pandas as pd
import numpy as np
import geopandas as gp                
import requests
from xml.etree import ElementTree as ET   
import time  
import os,sys
from shapely.geometry import Polygon
from numpy.lib.stride_tricks import as_strided
from scipy import stats
import subprocess
from osgeo import gdal
from gdalconst import GA_ReadOnly
from skimage.feature import local_binary_pattern
import cv2
from skimage.filters.rank import windowed_histogram
from zipfile import ZipFile

metadata_zip_url = r'https://thor-f5.er.usgs.gov/ngtoc/metadata/misc/topomaps_all.zip'
base_folder = r'H:\USGS_HTMC_MAPPROCESSOR'

#########################################################################
### paths to gdal binaries:
gdalwarp = r'C:\OSGeo4W\bin\gdalwarp.exe'
gdal_translate = r'C:\OSGeo4W\bin\gdal_translate.exe'
gdal_edit = r'python C:\OSGeo4W\bin\gdal_edit.py'
gdalsrsinfo = r'C:\OSGeo4W\bin\gdalsrsinfo.exe'
ogr2ogr = r'C:\OSGeo4W\bin\ogr2ogr.exe' 
gdal_merge = r'python C:\OSGeo4W\bin\gdal_merge.py'

#########################################################################
download_metadata=False ## fetches USGS HTMC metadata csv file
get_relevant_maps=False ## performs user-specified query
download_and_process_maps=False # downloads and processes maps, stores aggregated data in numpy dumps.
export_indiv_maps=False ## generated geotiffs of the aggregated map data (RGB)
create_vrt=True ## misaics geotiffs to VRT

#########################################################################
### map data is aggregated into a target grid of cell size windowsize x windowsize.
### the windowsize parameter (measured in pixels) can be specified individually for each map scale.
windowsize_dict={24000:50, #### 2m pixelsize at ground, aggregating in 50x50 pixel blocks roughly corresponds to a 100m target grid.
                 62500:20  #### 5m pixelsize at ground, aggregating in 20x20 pixel blocks roughly corresponds to a 100m target grid.
                 }
            
######################################################################### 
### keep all downloaded geotiff files:
keep_orig_geotiff=False ## set to false for large-scale processing
           
### per default, we extract color moments (average, std, kurtosis, skewness per R,G,B band, per tile)
### optionally, Local Binary Pattern (LBP) textural descriptor extraction is implemented.
### Users can easily insert other descriptors (GLCM, (dense) SIFT, HOG,...)
do_lbp=False    
if do_lbp:
    # some LBP params:
    n_points = 24
    METHOD = 'uniform'
    radius=2
    num_histbins=40
### optionally, a color reduction step can be performed using k-means.
### however, this process slows down the data processing considerably.
do_kmeans=False
kmeans_numcolors=5

#########################################################################

metadatacsv=base_folder+os.sep+'historicaltopo.csv'
metadata_xml_folder = base_folder+os.sep+'xml_metadata'
tiff_folder = base_folder+os.sep+'tiff_download' 
tiff_clip_folder = base_folder+os.sep+'tiff_clipped'
tempdata = base_folder+os.sep+'temp' 
shp_path = base_folder+os.sep+'quad_shps'
histfolder = base_folder+os.sep+'descriptors' 
blockstat_tif_folder = base_folder+os.sep+'blockstat_tif' 
mosaic_dir = base_folder+os.sep+'mosaics'
selection_shp = shp_path+os.sep+'temp.shp'

#########################################################################
# get and / or read HTMC metadata 

if not os.path.exists(metadatacsv):
    remotefile = requests.get(metadata_zip_url) 
    localfile = base_folder+os.sep+metadata_zip_url.split('/')[-1]
    open(localfile, 'wb').write(remotefile.content)   
    with ZipFile(localfile, 'r') as zipObj:
       zipObj.extractall(base_folder)
    with ZipFile(base_folder+os.sep+'historicaltopo.zip', 'r') as zipObj:
       zipObj.extractall(base_folder)        
metadatadf = pd.read_csv(metadatacsv)

#########################################################################

# query map sheets of interest by year:
selectiondf=metadatadf
selectiondf = selectiondf[selectiondf.date_on_map<1950]

###spatial selection, set to conus, or use subset
lon_min=-125  
lon_max=-60
lat_min=23
lat_max=51

###select by state(s). add state names in a list, e.g. ['Colorado','Utah']. set to [] if not used.
relevant_states=['Rhode Island']
rel_scales=[24000,62500]

#if this query returns multiple map sheets per quadrangle, only the oldest map sheet is used.
keep_earliest_only=True

#########################################################################
### perform query:
if len(relevant_states)>0:
    selectiondf=selectiondf[selectiondf['primary_state'].isin(relevant_states)]
selectiondf=selectiondf[selectiondf['northbc']<lat_max]
selectiondf=selectiondf[selectiondf['southbc']>lat_min]
selectiondf=selectiondf[selectiondf['westbc']>lon_min]
selectiondf=selectiondf[selectiondf['eastbc']<lon_max]
selectiondf=selectiondf[selectiondf['map_scale'].isin(rel_scales)]
if len(selectiondf)==0:
    print('query empty. exiting...')
    sys.exit(0)

### create data subfolders:
if not os.path.exists(metadata_xml_folder):
    os.mkdir(metadata_xml_folder)
if not os.path.exists(tiff_folder):
    os.mkdir(tiff_folder)
if not os.path.exists(tiff_clip_folder):
    os.mkdir(tiff_clip_folder)
if not os.path.exists(tempdata):
    os.mkdir(tempdata)
if not os.path.exists(shp_path):
    os.mkdir(shp_path)
if not os.path.exists(histfolder):
    os.mkdir(histfolder)
if not os.path.exists(blockstat_tif_folder):
    os.mkdir(blockstat_tif_folder)
if not os.path.exists(mosaic_dir):
    os.mkdir(mosaic_dir)

################ some functions #########################

### source: https://community.esri.com/t5/python-blog/sliding-moving-window-operations-in-rasters-and/ba-p/893965
def _check(a, r_c):
    if isinstance(r_c, (int, float)):
        r_c = (1, int(r_c))
    r, c = r_c
    a = np.atleast_2d(a)
    shp = a.shape
    r, c = r_c = ( min(r, a.shape[0]), min(c, shp[1]) ) 
    a = np.ascontiguousarray(a)
    return a, shp, r, c, tuple(r_c)    
def stride(a, stepsize, r_c):
    a, shp, r, c, r_c = _check(a, r_c)
    shape = (a.shape[0] - r + 1, a.shape[1] - c + 1) + r_c
    #shape = (int(a.shape[0]/float(stepsize)), int(a.shape[1]/float(stepsize))) + r_c
    strides = a.strides * 2
    a_s = (as_strided(a, shape=shape, strides=strides)).squeeze()
    return a_s
def block(a, r_c=(3, 3)):
    a, shp, r, c, r_c = _check(a, r_c)
    shape = (a.shape[0]/r, a.shape[1]/c) + r_c
    strides = (r*a.strides[0], c*a.strides[1]) + a.strides
    a_b = as_strided(a, shape=shape, strides=strides).squeeze()
    return a_b
### source: J.H. Uhl
def gdalNumpy2floatRaster(array,outname,template_georef_raster,x_pixels,y_pixels,px_type):            
    #use px_type = gdal.GDT_Int16 or gdal.GDT_Float32)            
    dst_filename = outname        
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(dst_filename,x_pixels, y_pixels, 1, px_type)   
    dataset.GetRasterBand(1).WriteArray(array)                
    mapraster = gdal.Open(template_georef_raster, GA_ReadOnly)
    proj=mapraster.GetProjection() #you can get from a existing tif or import 
    dataset.SetProjection(proj)
    dataset.FlushCache()
    dataset=None                        
    #set bounding coords
    ulx, xres, xskew, uly, yskew, yres  = mapraster.GetGeoTransform()
    lrx = ulx + (mapraster.RasterXSize * xres)
    lry = uly + (mapraster.RasterYSize * yres)            
    mapraster = None                            
    gdal_cmd = gdal_edit+' -a_ullr %s %s %s %s "%s"' % (ulx,uly,lrx,lry,outname)
    print(gdal_cmd)
    response=subprocess.check_output(gdal_cmd, shell=True)
    print(response)
### source: J.H. Uhl    
def gdalNumpy2floatRaster_reproj(array,outname,template_georef_raster,x_pixels,y_pixels,px_type,ulx,uly,lrx,lry,proj):            
    dst_filename = outname        
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(dst_filename,x_pixels, y_pixels, 1, px_type)   
    dataset.GetRasterBand(1).WriteArray(array)                
    dataset.SetProjection(proj)
    dataset.FlushCache()
    dataset=None                                                   
    gdal_cmd = gdal_edit+' -a_ullr %s %s %s %s "%s"' % (ulx,uly,lrx,lry,outname)
    print(gdal_cmd)
    response=subprocess.check_output(gdal_cmd, shell=True)
    print(response)
### source: J.H. Uhl
def gdalNumpy2floatRaster_reproj_RGB(array,outname,x_pixels,y_pixels,px_type,ulx,uly,lrx,lry,proj):            
    #use px_type = gdal.GDT_Int16 or gdal.GDT_Float32)            
    dst_filename = outname        
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(dst_filename,x_pixels, y_pixels, 3, px_type)   
    dataset.GetRasterBand(1).WriteArray(array[:,:,0])
    dataset.GetRasterBand(2).WriteArray(array[:,:,1])  
    dataset.GetRasterBand(3).WriteArray(array[:,:,2])                  
    dataset.SetProjection(proj)
    dataset.FlushCache()
    dataset=None                                                   
    gdal_cmd = gdal_edit+' -a_ullr %s %s %s %s "%s"' % (ulx,uly,lrx,lry,outname)
    print(gdal_cmd)
    response=subprocess.check_output(gdal_cmd, shell=True)
    print(response)   
#################################################################################

if get_relevant_maps:
    scales=selectiondf.map_scale.unique()
    for scale in scales:
        selectiondf_scale=selectiondf[selectiondf.map_scale==scale]
        
        if keep_earliest_only:
            earliest_ids_cell=[]
            for cell,celldf in selectiondf_scale.groupby('gnis_cell_id'):
                currcscanid=int(celldf.sort_values(by='date_on_map').head(1)['scan_id'].values[0])
                earliest_ids_cell.append(currcscanid)
            selectiondf_scale=selectiondf_scale[selectiondf_scale['scan_id'].isin(earliest_ids_cell)]
        selectiondf_scale.to_csv(base_folder+os.sep+'map_selection_%s.csv' %scale, index=False) 
        print('found %s maps at scale 1:%s' %(len(selectiondf_scale),scale))
        
	
if download_and_process_maps:
            
    for scale in rel_scales:
        selectiondf_scale=pd.read_csv(base_folder+os.sep+'map_selection_%s.csv' %scale)        
        total=len(selectiondf_scale)
        
        #######################################################
        try:
            windowsize=windowsize_dict[scale]
        except:
            print('error. user needs to define a suitable tile size for scale 1:%s.' %scale) 
        stepsize=int(0.5*windowsize)
                      
        #######################################################
        
        
        structel_size = windowsize
        structel_arr = np.ones((structel_size,structel_size))
        structel_arr = structel_arr.astype(np.uint8)
        
        counter=0
        for i,row in selectiondf_scale.iterrows():
            counter+=1            
            mapstart = time.time()            
            md_xml_url = row['metadata_url'] 
            scanid = str(int(row['scan_id']))
            year = int(row['date_on_map'])
            
            ##############################
            #if not scanid=='450915':
            #    continue
            ##############################

            try:
                remotexmlfile = requests.get(md_xml_url) 
                localxmlfile = metadata_xml_folder+os.sep+md_xml_url.split('/')[-1]
                open(localxmlfile, 'wb').write(remotexmlfile.content)                                     
                doc = ET.parse(localxmlfile).getroot()        
                geotiff_url=''
                for distinfo in doc.findall('distinfo'):
                    for stdorder in distinfo.findall('stdorder'):
                        for digform in stdorder.findall('digform'):
                            for digtinfo in digform.findall('digtinfo'):
                                filetype = digtinfo.find('formname').text                        
                                if filetype == 'GeoTIFF':
                                    print (filetype)                                            
                                    for digtopt in digform.findall('digtopt'):
                                        geotiff_url = digtopt.find('./onlinopt/computer/networka/networkr').text
                                        break                                
                print(geotiff_url)  
                ###download geotiff
                remotetifffile = requests.get(geotiff_url) 
                localtifffile = tiff_folder+os.sep+geotiff_url.split('/')[-1]
                open(localtifffile, 'wb').write(remotetifffile.content)
            except: 
                print(scanid,file = open(base_folder+os.sep+'errorfile.txt','a'))                    
                print('error',scanid )
                continue
                
            if geotiff_url=='':
                print(scanid,file = open(base_folder+os.sep+'errorfile.txt','a'))                    
                print('error',scanid )
                continue
                        
            ### now read and process tiff            
            ### first create quadrangle shp
            rowgdf=gp.GeoDataFrame(pd.DataFrame(row)).transpose()
            rowgdf['geometry'] = rowgdf.apply(lambda row:Polygon([[row['westbc'],row['northbc']],
                                                                    [row['eastbc'],row['northbc']],
                                                                     [row['eastbc'],row['southbc']],
                                                                      [row['westbc'],row['southbc']],
                                                                       [row['westbc'],row['northbc']]]),axis=1)
            rowgdf.crs='+proj=longlat +datum=NAD27 +no_defs'
            rowgdf.to_file(filename=selection_shp)
            ### clip to boundaries:
                
            # 1) get the quadrangle for the map: 
            #infile = os.path.split(selection_shp)[1].replace('.shp','')
            #currquadfile = tempdata + os.sep + 'quad_%s.shp' %scanid
            #select_cmd = """ogr2ogr -sql "SELECT * FROM %s WHERE (mapid='%s')" "%s" "%s" """ %(infile,scanid,currquadfile,selection_shp)
            ##print select_cmd
            #response=subprocess.check_output(select_cmd, shell=True)
            #print response 
            ####################################################################################
    
            # 2) reproject quadr into map CRS:        
            call = gdalsrsinfo+' -o proj4 "'+localtifffile+'"'
            crs_raster=subprocess.check_output(call, shell=True).decode().strip().replace("'","")
            
            call = gdalsrsinfo+' -o proj4 "'+selection_shp+'"'
            crs_quads=subprocess.check_output(call, shell=True).decode().strip().replace("'","")
            
            try:
                quads_proj = selection_shp.replace('.shp','_proj.shp')
                call = ogr2ogr+' -t_srs "'+crs_raster+'" -s_srs "'+crs_quads+'" "'+quads_proj+'" "'+selection_shp+'"'
                #print(call)
                response=subprocess.check_output(call, shell=True)
                #print(response)

            
                ####################################################################################
    
                # 3) clip
                #clip raster to quadrangle        
                map_clipped =  tiff_clip_folder+os.sep+scanid+'%s_clip.tif'%year
                
                if os.path.exists(map_clipped):
                    os.remove(map_clipped)
                    
                quad_shp_noPath = os.path.split(quads_proj)[1].replace('.shp','')
                call = gdalwarp+' -of GTiff -cutline "'+quads_proj+'" -cl "'+quad_shp_noPath+'" -crop_to_cutline "'+localtifffile+'" "'+map_clipped+'"'
                #print(call)
                response=subprocess.check_output(call, shell=True) 
                #print(response)
    
                localtifffile_comp = map_clipped.replace('.tif','_comp.tif')
                
                call = gdal_translate+' -co compress=JPEG "%s" "%s"' %(map_clipped,localtifffile_comp)
                #print(call)
                response=subprocess.check_output(call, shell=True)  
                
                ####################################################################################
                #print 'spatial preproc done.', time.time() - start
                #start = time.time()
                            
                mapfile = map_clipped        
                combcount=0
                #######################
                
                template_georef_raster = mapfile
                outfile_key = os.path.split(mapfile)[1].replace('.tif','')
                        
                lbp_list=[]
                #mapdata = ndimage.imread(mapfile)
                ds = gdal.Open(mapfile)
                band1 = np.array(ds.GetRasterBand(1).ReadAsArray())
                band2 = np.array(ds.GetRasterBand(2).ReadAsArray())
                band3 = np.array(ds.GetRasterBand(3).ReadAsArray())
                mapdata_col = np.dstack((band1,band2,band3))
                del ds 
                
                print(time.time()-mapstart)
            
                if do_kmeans:
                    Z = mapdata_col.reshape((-1,3))    
                    Z = np.float32(Z)        
                    K = kmeans_numcolors           
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    # Set flags (Just to avoid line break in the code)
                    flags = cv2.KMEANS_RANDOM_CENTERS
                    # Apply KMeans
                    ret,label,center = cv2.kmeans(Z,K,None,criteria,10,flags)                                        
                    center = np.uint8(center)
                    res = center[label.flatten()]
                    mapdata_colred = res.reshape((mapdata_col.shape)) 
                    mapdata = cv2.cvtColor(mapdata_colred,cv2.COLOR_BGR2GRAY)
                    #print 'color reduction done.', time.time() - start
                    print(time.time()-mapstart)
                else:
                    mapdata = cv2.cvtColor(mapdata_col,cv2.COLOR_BGR2GRAY)

            except:
                print(scanid,file = open(base_folder+os.sep+'errorfile.txt','a'))                    
                print('error',scanid )
                continue


            try:
                if do_lbp:
                    lbp_mapdata = local_binary_pattern(mapdata, n_points, radius, METHOD)                        
                    lbp_hist_img = windowed_histogram(lbp_mapdata.astype(np.int8), structel_arr,n_bins=num_histbins)
                    
                    ### applying the stride:
                    half_window = np.int(windowsize/float(2))
                    lbp_hist_img = lbp_hist_img[half_window::stepsize,half_window::stepsize]
                    
                a = stride(mapdata_col[:,:,0],stepsize, r_c=(windowsize, windowsize))       
                a = a[::stepsize,::stepsize]
                                     
                a_mean = np.nanmean(a,axis=(2, 3))
                a_stnd = np.nanstd(a,axis=(2, 3))
                a_resh = np.reshape(a,(a.shape[0],a.shape[1],a.shape[2]*a.shape[2]))
                a_kurt = stats.kurtosis(a_resh,axis=2)
                a_skew = stats.skew(a_resh,axis=2)
                col_mom_0 = np.dstack((a_mean,a_stnd,a_kurt,a_skew))

                a = stride(mapdata_col[:,:,1],stepsize, r_c=(windowsize, windowsize))                                                
                a = a[::stepsize,::stepsize]

                a_mean = np.nanmean(a,axis=(2, 3))
                a_stnd = np.nanstd(a,axis=(2, 3))
                a_resh = np.reshape(a,(a.shape[0],a.shape[1],a.shape[2]*a.shape[2]))
                a_kurt = stats.kurtosis(a_resh,axis=2)
                a_skew = stats.skew(a_resh,axis=2)
                col_mom_1 = np.dstack((a_mean,a_stnd,a_kurt,a_skew))              
                                
                a = stride(mapdata_col[:,:,2],stepsize, r_c=(windowsize, windowsize))                                                
                a = a[::stepsize,::stepsize]

                a_mean = np.nanmean(a,axis=(2, 3))
                a_stnd = np.nanstd(a,axis=(2, 3))
                a_resh = np.reshape(a,(a.shape[0],a.shape[1],a.shape[2]*a.shape[2]))
                a_kurt = stats.kurtosis(a_resh,axis=2)
                a_skew = stats.skew(a_resh,axis=2)
                col_mom_2 = np.dstack((a_mean,a_stnd,a_kurt,a_skew))
                
                if do_lbp:
                    if col_mom_0.shape[0]<lbp_hist_img.shape[0]:
                        lbp_hist_img = lbp_hist_img[1:,:]
                    if col_mom_0.shape[1]<lbp_hist_img.shape[1]:
                        lbp_hist_img = lbp_hist_img[:,1:,:]
                    descr = np.dstack((lbp_hist_img,col_mom_0,col_mom_1,col_mom_2))                               
                else:
                    descr = np.dstack((col_mom_0,col_mom_1,col_mom_2)) 
                
                #np.savez_compressed(histfolder+os.sep+'_colmom_%s_%s_%s_%s_%s_%s_%s.npz' %(scanid,year,n_points,METHOD,num_histbins,stepsize,windowsize), descr=descr)
                np.savez_compressed(histfolder+os.sep+'_colmom_%s_%s_%s_%s.npz' %(scanid,year,stepsize,windowsize), descr=descr)

                #print 'LBP computation done.', time.time() - start
                print( scale, counter,'/', total, scanid, 'done.', time.time() - mapstart)
            except:    
                print(scanid,file = open(base_folder+os.sep+'errorfile.txt','a'))                    
                print('error',scanid )
                continue                    
                    
            try:
                os.remove(map_clipped)
                if not keep_orig_geotiff:
                    os.remove(localtifffile)
            except:
                continue
            #sys.exit(0)
            
if export_indiv_maps:
    
    for scale in rel_scales:
        selectiondf_scale=pd.read_csv(base_folder+os.sep+'map_selection_%s.csv' %scale)

        total=len(selectiondf_scale)
        
        #######################################################
        try:
            windowsize=windowsize_dict[scale]
        except:
            print('error. user needs to define a suitable tile size for scale 1:%s.' %scale) 
        stepsize=int(0.5*windowsize)
                      
        #######################################################
        
        counter=0
        for i,row in selectiondf_scale.iterrows():

            counter+=1
            scanid = str(int(row['scan_id']))
                        
            year = int(row['date_on_map'])
            scale =str(int(row['map_scale']))
            blockstat_file = histfolder+os.sep+'_colmom_%s_%s_%s_%s.npz' %(scanid,year,stepsize,windowsize)
            
            if not os.path.exists(blockstat_file):
                continue
            
            blockstat_arr = np.load(blockstat_file)
            blockstat_arr = blockstat_arr['descr']
 
            channelnames=['R','G','B']
            blockstat_arr_means=blockstat_arr[:,:,[0,4,8]]  
            cols = blockstat_arr_means.shape[1]
            rows = blockstat_arr_means.shape[0]
            #### in NAD 1927
            ulx=row['westbc'] 
            uly=row['northbc']
            lrx=row['eastbc']
            lry=row['southbc']
            outproj='+proj=longlat +datum=NAD27 +no_defs'
            gdalNumpy2floatRaster_reproj_RGB(blockstat_arr_means,blockstat_tif_folder+os.sep+'blockstats_nad1927_RGB_%s_%s_%s.tif' %(scale,scanid,year),cols,rows,gdal.GDT_Float32,ulx,uly,lrx,lry,outproj)
            print(counter,'exported RGB %s' %scanid)
                
   

if create_vrt: 
    
    for scale in rel_scales:

        selectiondf_scale=pd.read_csv(base_folder+os.sep+'map_selection_%s.csv' %scale)
                            
        total=len(selectiondf_scale)
        
        selectiondf_scale=selectiondf_scale[selectiondf_scale['westbc']>=lon_min]
        selectiondf_scale=selectiondf_scale[selectiondf_scale['westbc']<=lon_max]
        selectiondf_scale=selectiondf_scale[selectiondf_scale['southbc']>=lat_min]
        selectiondf_scale=selectiondf_scale[selectiondf_scale['southbc']<=lat_max]
        
        counter=0
        vrt_input=[]
        cellids=[]
        for i,row in selectiondf_scale.iterrows():
            counter+=1
            scanid = str(int(row['scan_id']))
            year = int(row['date_on_map'])
            scale =str(int(row['map_scale']))    
            cellids.append(row['gnis_cell_id'])
            
            curr_rast=blockstat_tif_folder+os.sep+'blockstats_nad1927_RGB_%s_%s_%s.tif' %(scale,scanid,year)
    
            if os.path.exists(curr_rast):
                vrt_input.append(curr_rast)
                print(curr_rast)
                    

        vrt_inputdf=pd.DataFrame(vrt_input)
        vrt_inputdf.to_csv(mosaic_dir+os.sep+'create_vrt_input.csv',header=None,index=False)

        outvrt=mosaic_dir+os.sep+'input_%s.vrt' %scale   

        infiles=mosaic_dir+os.sep+'create_vrt_input.csv'
        gdal_cmd = """gdalbuildvrt "%s" -input_file_list "%s" """ %(outvrt,infiles) #-a_srs "EPSG:4267" 
        print(gdal_cmd)
        response=subprocess.check_output(gdal_cmd, shell=True)
        print(response) 
        
        ## VRT to compressed geotiff:
        outname_lzw=outvrt.replace('.vrt','_lzw.tif')
        gdal_translate = r'gdal_translate %s %s -co COMPRESS=LZW' %(outvrt,outname_lzw)
        print(gdal_translate)
        response=subprocess.check_output(gdal_translate, shell=True)
        print(response) 
        
        ## pyramids:
        cmd = r'gdaladdo -ro --config BIGTIFF_OVERVIEW YES %s 2 4 8 16' %outname_lzw
        print(cmd)
        response=subprocess.check_output(cmd, shell=True)
        print(response)     

