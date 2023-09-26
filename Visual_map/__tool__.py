from sklearn import neighbors
from pykrige.ok import OrdinaryKriging
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *
from tensorflow.python.keras import backend as K
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Path, PathPatch
from datetime import datetime 

import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib
import gc
import sqlite3

import numpy as np
import pandas as pd
import tensorflow_addons as tfa
import json
import requests

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger().disabled = True
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def CWB_Table(cwb_total_list):

#     cwb_total_list = np.load('cwb_stats_info_align.npy',allow_pickle=True)

    title_api = ''
    wea_api_data = 'O-A0001-001' #https://opendata.cwb.gov.tw/dataset/observation/O-A0001-001
    rain_api_data = 'O-A0002-001' #https://opendata.cwb.gov.tw/dataset/observation/O-A0002-001
    key_api = ''

    cwb_wea_api_website = title_api + wea_api_data + key_api
    cwb_rain_api_website = title_api + rain_api_data + key_api

    cwb_wea_r = requests.get(cwb_wea_api_website,verify=True)
    cwb_rain_r = requests.get(cwb_rain_api_website,verify=True)

    if (str(cwb_wea_r) == '<Response [200]>') and (str(cwb_rain_r) == '<Response [200]>'):
        cwb_wea = cwb_wea_r.json()
        cwb_rain = cwb_rain_r.json()

    # get weather api whole station 
    cwb_wea_station_list, cwb_rain_align_wea_station_list, cwb_wea_total, cwb_rain_total = ([] for i in range(4))
    cwb_rain_value_list, cwb_wea_time_list = ([] for i in range(2))
    cwb_wea_WDIR_list, cwb_wea_WDSD_list, cwb_wea_TEMP_list, cwb_wea_HUMD_list = ([] for i in range(4))

    for _ in range (len(cwb_wea.get('records').get('location'))):
        cwb_wea_station_list.append(cwb_wea.get('records').get('location')[_]['stationId'])

    # alignment rain api station to weather station
    for _ in range (len(cwb_rain.get('records').get('location'))):
        if cwb_rain.get('records').get('location')[_]['stationId'] in cwb_wea_station_list:
            cwb_rain_align_wea_station_list.append(cwb_rain.get('records').get('location')[_]['stationId'])
            cwb_rain_total.append(cwb_rain.get('records').get('location')[_])

    # aligment wea to rain again
    for _ in range (len(cwb_wea.get('records').get('location'))):
        if cwb_wea.get('records').get('location')[_]['stationId'] in cwb_rain_align_wea_station_list:
            cwb_wea_total.append(cwb_wea.get('records').get('location')[_])

    # both of tables are aligment stationID
    cwb_wea_table = pd.DataFrame(cwb_wea_total).sort_index(by = 'stationId').reset_index()
    cwb_rain_table = pd.DataFrame(cwb_rain_total).sort_index(by = 'stationId').reset_index()

    del cwb_wea_table['parameter'], cwb_wea_table['index']

    # get rain value, -998 mean last 6 hour rain  = 0
    for _ in range(len(cwb_rain_table['weatherElement'])):
        for __ in range(len(cwb_rain_table['weatherElement'][_])):
            if cwb_rain_table['weatherElement'][_][__]['elementName'] == 'RAIN':
                if cwb_rain_table['weatherElement'][_][__]['elementValue'] == '-998.00':
                    cwb_rain_value_list.append('0.00')
                else:
                    cwb_rain_value_list.append(cwb_rain_table['weatherElement'][_][__]['elementValue'])

    # get weather api time stamp
    for _ in range(len(cwb_wea_table['time'])): cwb_wea_time_list.append(cwb_wea_table['time'][_]['obsTime'])                

    # get weather api attribute -99 mean no value, HUMD : 0 ~ 1, need *100
    for _ in range(len(cwb_wea_table['weatherElement'])):
        for __ in range(len(cwb_wea_table['weatherElement'][_])):
            if cwb_wea_table['weatherElement'][_][__]['elementName'] == 'WDIR':
                if np.float(cwb_wea_table['weatherElement'][_][__]['elementValue']) == -99:cwb_wea_WDIR_list.append(0)
                else:cwb_wea_WDIR_list.append(round(np.float(cwb_wea_table['weatherElement'][_][__]['elementValue']),0))
            if cwb_wea_table['weatherElement'][_][__]['elementName'] == 'WDSD':
                if np.float(cwb_wea_table['weatherElement'][_][__]['elementValue']) == -99:cwb_wea_WDSD_list.append(0)
                else:cwb_wea_WDSD_list.append(round(np.float(cwb_wea_table['weatherElement'][_][__]['elementValue']),0))
            if cwb_wea_table['weatherElement'][_][__]['elementName'] == 'TEMP':
                if np.float(cwb_wea_table['weatherElement'][_][__]['elementValue']) == -99:cwb_wea_TEMP_list.append(0)
                else:cwb_wea_TEMP_list.append(round(np.float(cwb_wea_table['weatherElement'][_][__]['elementValue']),0))
            if cwb_wea_table['weatherElement'][_][__]['elementName'] == 'HUMD':
                if np.float(cwb_wea_table['weatherElement'][_][__]['elementValue']) == -99:cwb_wea_HUMD_list.append(0)
                else:cwb_wea_HUMD_list.append(round(np.float(cwb_wea_table['weatherElement'][_][__]['elementValue']) * 100,0))

                    
    if len(cwb_rain_value_list) == len(cwb_wea_time_list) == len(cwb_wea_WDIR_list) == len(cwb_wea_WDSD_list) \
       == len(cwb_wea_TEMP_list) == len(cwb_wea_HUMD_list):
    
        cwb_wea_table.insert(0,'Rain',np.array(cwb_rain_value_list,dtype=float))
        cwb_wea_table.insert(1,'Time',cwb_wea_time_list)
        cwb_wea_table.insert(2,'WDIR',cwb_wea_WDIR_list)
        cwb_wea_table.insert(3,'WDSD',cwb_wea_WDSD_list)
        cwb_wea_table.insert(4,'TEMP',cwb_wea_TEMP_list)
        cwb_wea_table.insert(5,'HUMD',cwb_wea_HUMD_list)

        del cwb_wea_table['time'], cwb_wea_table['weatherElement']
        cwb_wea_table = cwb_wea_table[['locationName','stationId','Rain','WDIR','WDSD','TEMP','HUMD','lat','lon','Time']]

        # change lat lon to grid
        cwb_lat2grid_list, cwb_lon2grid_list = ([] for _ in range(2))
        for _ in range(len(cwb_wea_table['lat'].values)):
            cwb_lat2grid_list.append(int(round((np.float(cwb_wea_table['lat'].values[_])-21.87)/((25.35-21.87)/348),0)))
            cwb_lon2grid_list.append(int(round((np.float(cwb_wea_table['lon'].values[_])-120)/((122.04-120)/204),0)))

        del cwb_wea_table['lat'], cwb_wea_table['lon']

        cwb_wea_table.insert(7,'lat',cwb_lat2grid_list)
        cwb_wea_table.insert(8,'lon',cwb_lon2grid_list)

        return (cwb_wea_table, datetime.strptime(list(set(cwb_wea_time_list))[0], "%Y-%m-%d %H:%M:%S"))
    else:
        return (None, None)

def CWB_Attribute_KNN(cwb_wea_table):
    lats = cwb_wea_table['lat']
    lons = cwb_wea_table['lon']
    rain = cwb_wea_table['Rain']
    wdir = np.round((cwb_wea_table['WDIR'] - 180)/180,2) # normalization to -1 ~ 1
    wdsd = cwb_wea_table['WDSD']
    temp = cwb_wea_table['TEMP']
    humd = cwb_wea_table['HUMD']
    
    test_x = ([[i,j] for i in range(348) for j in range(204)])

    total_attribute = []
    train_x = np.vstack((lats,lons)).T
    for train_y in ([rain,wdir,wdsd,temp,humd]):
        result = []
        knn = neighbors.KNeighborsRegressor(3, weights = 'distance')
        knn_fit = knn.fit(train_x, train_y)
        test_y = knn_fit.predict(test_x)
        for k in range(348): result.append(test_y[k*204:(k+1)*204])
        # --------------KNN--------------
        total_attribute.append(np.reshape(result,(348,204,1)))    
    return (np.array(total_attribute))

def EPA_Table(info, custom_cmap, custom_norm, GeographyData_path):
    api_website = "https://data.epa.gov.tw/api/v1/aqx_p_432?format=json&limit=100&api_key="
    r = requests.get(api_website,verify=True)
    
    total_list = ['安南', '板橋', '菜寮', '彰化', '潮州', '嘉義', '大里', '大寮', '大同', '大園', '冬山',
                '斗六', '二林', '鳳山', '豐原', '復興', '關山', '觀音', '古亭', '恆春', '新竹', '花蓮',
                '湖口', '基隆', '林口', '林園', '龍潭', '崙背', '麥寮', '美濃', '苗栗', '南投', '楠梓',
                '屏東', '平鎮', '埔里', '朴子', '前金', '前鎮', '橋頭', '仁武', '三重', '三義', '沙鹿',
                '善化', '士林', '松山', '臺南', '臺東', '臺西', '淡水', '桃園', '頭份', '土城', '萬華',
                '萬里', '線西', '小港', '新店', '新港', '新營', '新莊', '西屯', '汐止', '陽明', '宜蘭',
                '永和', '中壢', '忠明', '中山', '竹東', '竹山', '左營']    
    
    if str(r) == '<Response [200]>':
        list_of_dicts = r.json()
        pm25_lst, lat_lst, lon_lst, sitename_lst, tempory, result, m_lats, m_lons = ([] for _ in range(8))
        for iter_nb in range(len(list_of_dicts.get('records'))):
            pm25_lst.append(list_of_dicts.get('records')[iter_nb]['PM2.5_AVG'])
            lat_lst.append(list_of_dicts.get('records')[iter_nb]['Latitude'])
            lon_lst.append(list_of_dicts.get('records')[iter_nb]['Longitude'])
            sitename_lst.append(list_of_dicts.get('records')[iter_nb]['SiteName'])

        date_newest = ([list_of_dicts.get('records')[i]['PublishTime'] for i in range(len(list_of_dicts.get('records')))])

        del api_website, r, list_of_dicts

        if len(np.unique(date_newest)) == 1:
            
            year = date_newest[0][:4]; month = date_newest[0][5:7]; day = date_newest[0][8:10]; hour = date_newest[0][12:13]

            data = pd.DataFrame()
            data.insert(0,'SiteName',sitename_lst); data.insert(1,'PM2.5_AVG',pm25_lst); data.insert(2,'lat',lat_lst); 
            data.insert(3,'lon',lon_lst); data.insert(4,'Publish_Time',date_newest)

            del sitename_lst, pm25_lst, lat_lst, lon_lst, date_newest
            
            ex_list = list(data['SiteName'])
            outlier = ['高雄(阿蓮)','南投(草屯)','新北(樹林)','彰化(員林)','臺南(北門)','屏東(琉球)','臺南(麻豆)','彰化(大城)',
                       '富貴角','馬公','金門','馬祖','永和(環河)','大園(竹圍)','枋寮','桃園(竹圍)','屏東(枋寮)']

            ex_list = [ele for ele in ex_list if ele not in outlier] 

            data_filter = data[data['SiteName'].isin(ex_list)]
            data_filter_reindex = data_filter.reset_index()
            del data_filter_reindex['index']

            data_filter_reindex['SiteName'] = pd.Categorical(data_filter_reindex['SiteName'],total_list)

            data_filter_reindex = data_filter_reindex.sort_values("SiteName")
            data = data_filter_reindex.values
            data_filter_reindex = pd.DataFrame(data)
            data_filter_reindex.columns = ['SiteName','PM2.5_AVG','lat','lon','Publish_Time']
            data_filter_reindex['lat'] = info['coordinate_lat']
            data_filter_reindex['lon'] = info['coordinate_lon']        
            data_filter_reindex = data_filter_reindex.reset_index()
            del ex_list, data_filter_reindex['index']  

            for i in range(len(total_list)):
                if data_filter_reindex['SiteName'][i] != total_list[i]:
                    missing_line = pd.DataFrame({"SiteName": total_list[i], 
                                              "PM2.5_AVG": '', 
                                              'lat': int(info[info['SiteName'] == total_list[i]]['coordinate_lat'].values),
                                              'lon': int(info[info['SiteName'] == total_list[i]]['coordinate_lon'].values),
                                              'Publish_Time': data_filter_reindex['Publish_Time'][0],
                                             }, index=[i])            
            
                    data_filter_reindex = pd.concat([data_filter_reindex.iloc[:i], 
                                                     missing_line, 
                                                     data_filter_reindex.iloc[i:]]).reset_index(drop=True)                 
            if len(data_filter_reindex) > 73:
                Redundant_site_number = len(data_filter_reindex)-73
                for redundant_loop in range(Redundant_site_number):
                    data_filter_reindex = data_filter_reindex.drop(data_filter_reindex.index[len(data_filter_reindex)-1])

            data_filter_reindex.insert(1,'SiteEngName',info['SiteEngName'].values)                    
                    
            # interpolation, if station value not exists. 20210113 need modify to use old value, not whole station mean!!
            for i in range(73):
                if ("".join(filter(str.isdigit, np.array(data_filter_reindex['PM2.5_AVG'].values)[i]))) !='':
                    tempory.append(("".join(filter(str.isdigit, np.array(data_filter_reindex['PM2.5_AVG'].values)[i]))))
            mean_values = np.array(tempory,dtype='float32').mean()

            for i in range(73):
                if data_filter_reindex['PM2.5_AVG'].values[i] == '':
                    data_filter_reindex.set_value(i, 'PM2.5_AVG', mean_values) 
                    print(str(data_filter_reindex['Publish_Time'][i])+' Find Space value in '+
                          str(data_filter_reindex['SiteName'][i])+' PM2.5_AVG, Replace to mean value')
                    
    return(data_filter_reindex, datetime.strptime(data_filter_reindex['Publish_Time'].unique()[0], "%Y/%m/%d %H:%M:%S"))


def EPA_PM25_KNN(epa_pm25_table):
    mou = pd.read_csv('mou_process_done.csv')
    mou_lat = [mou['matrix_lat'][0], mou['matrix_lat'][71], mou['matrix_lat'][92]] #玉山, 馬比杉山, 北大武山
    mou_lon = [mou['matrix_lon'][0], mou['matrix_lon'][71], mou['matrix_lon'][92]]

    result = []
    lats = epa_pm25_table['lat']
    lons = epa_pm25_table['lon']
    pm25 = epa_pm25_table['PM2.5_AVG'].astype(float)
    
    matrix_lat_array = np.concatenate([np.array(lats),mou_lat])
    matrix_lon_array = np.concatenate([np.array(lons),mou_lon])
    
    test_x = ([[i,j] for i in range(348) for j in range(204)])
    train_x = np.vstack((matrix_lat_array,matrix_lon_array)).T
    
    mon_value = np.zeros_like(mou_lat)
    train_y = np.concatenate([pm25,mon_value])  

    knn = neighbors.KNeighborsRegressor(3, weights = 'distance')
    #knn = neighbors.KNeighborsRegressor(5)

    knn_fit = knn.fit(train_x, train_y)
    test_y = knn_fit.predict(test_x)
    for k in range(348): result.append(test_y[k*204:(k+1)*204])
    # --------------KNN--------------
    KNN_matrix = np.reshape(result,(1,348,204,1))
    
    return (KNN_matrix)

def EPA_PM25_KNN_NoFake(epa_pm25_table):
    result = []
    lats = epa_pm25_table['lat']
    lons = epa_pm25_table['lon']
    pm25 = epa_pm25_table['PM2.5_AVG'].astype(float)
    
    
    test_x = ([[i,j] for i in range(348) for j in range(204)])
    train_x = np.vstack((np.array(lats),np.array(lons))).T

    train_y = (pm25)  

    knn = neighbors.KNeighborsRegressor(3, weights = 'distance')
    #knn = neighbors.KNeighborsRegressor(5)

    knn_fit = knn.fit(train_x, train_y)
    test_y = knn_fit.predict(test_x)
    for k in range(348): result.append(test_y[k*204:(k+1)*204])
    # --------------KNN--------------
    KNN_matrix = np.reshape(result,(1,348,204,1))
    
    return (KNN_matrix)

def EPA_PM25_Kriging(epa_pm25_table):
    mou = pd.read_csv('mou_process_done.csv')
    mou_lat = [mou['matrix_lat'][0], mou['matrix_lat'][71], mou['matrix_lat'][92]] #玉山, 馬比杉山, 北大武山
    mou_lon = [mou['matrix_lon'][0], mou['matrix_lon'][71], mou['matrix_lon'][92]]

    result = []
    lats = epa_pm25_table['lat']
    lons = epa_pm25_table['lon']
    pm25 = epa_pm25_table['PM2.5_AVG'].astype(float)
    
    matrix_lat_array = np.concatenate([np.array(lats),mou_lat])
    matrix_lon_array = np.concatenate([np.array(lons),mou_lon])
    
    grid_lon = np.array(np.arange(204),dtype='float32')
    grid_lat = np.array(np.arange(348),dtype='float32') 
    train_x = np.vstack((matrix_lon_array, matrix_lat_array)).T
    
    mon_value = np.zeros_like(mou_lat)
    train_y = np.concatenate([pm25,mon_value])  

#     knn = neighbors.KNeighborsRegressor(3, weights = 'distance')
#     knn = neighbors.KNeighborsRegressor(5, weights = 'distance')
    OK = OrdinaryKriging(train_x[:,0], train_x[:,1], train_y, variogram_model='linear', weight = True,
                         exact_values = False, verbose=False)

    values, ss1 = OK.execute('grid', grid_lon, grid_lat)

    Kriging_matrix = np.reshape(values,(1,348,204,1))
    
    return (Kriging_matrix)

def Encoder(channel_number):
    input_img = Input(shape=(348,204,1))
    e1 = Conv2D(16, (3, 3), activation='tanh', padding='same')(input_img)
    e2 = MaxPooling2D((2, 2), padding='same')(e1)
    e3 = Conv2D(8, (3, 3), activation='tanh', padding='same')(e2)
    e4 = MaxPooling2D((2, 2), padding='same')(e3)
    e5 = Conv2D(channel_number, (3, 3), activation='tanh', padding='same')(e4)
    e6 = MaxPooling2D((2, 2), padding='same')(e5)
    e7 = Activation('tanh')(e6)
    return Model(input_img, e7)

def Decoder(channel_number):
    input_img = Input(shape=(44,26,channel_number))
    d1 = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
    d2 = UpSampling2D((2, 2))(d1)
    d3 = Conv2D(16, (3, 3), activation='relu', padding='same')(d2)
    d4 = UpSampling2D((2, 2))(d3)
    d5 = Conv2D(32, (3, 3), activation='relu')(d4)
    d6 = UpSampling2D((2, 2))(d5)
    d7 = Conv2D(1, (3, 3), activation='relu', padding='same')(d6)
    return Model(input_img, d7)

def TanhDecoder(channel_number):
    input_img = Input(shape=(44,26,channel_number))
    d1 = Conv2D(8, (3, 3), activation='tanh', padding='same')(input_img)
    d2 = UpSampling2D((2, 2))(d1)
    d3 = Conv2D(16, (3, 3), activation='tanh', padding='same')(d2)
    d4 = UpSampling2D((2, 2))(d3)
    d5 = Conv2D(32, (3, 3), activation='tanh')(d4)
    d6 = UpSampling2D((2, 2))(d5)
    d7 = Conv2D(1, (3, 3), activation='tanh', padding='same')(d6)
    return Model(input_img, d7)

def calculate_73_Loss(y_true,y_pred):
    
    def MSE(y, y_pred):
        mse = np.mean((y - y_pred)**2) 
        return np.round(mse,4)    
    
    loss = 0
    
    sta_lat=[9,61,66,70,70,74,74,76,77,81,81,82,87,89,89,
             102,112,118,118,125,144,160,160,169,185,185,
             189,189,189,205,206,210,211,220,223,227,229,
             230,236,239,252,270,277,283,288,288,294,300,
             304,309,309,311,312,313,315,315,316,317,317,
             318,319,320,320,320,320,320,321,321,324,326,
             330,331,332]    
    
    sta_lon=[79,42,57,34,43,31,32,36,29,30,49,34,33,31,116,
             54,21,22,117,30,32,25,45,35,21,55,26,35,68,69,
             41,97,160,55,68,47,65,62,57,75,76,83,180,90,109,
             175,98,122,104,121,123,154,146,131,146,152,153,
             109,144,151,158,121,149,152,153,165,137,150,152,
             177,145,169,153]    
    for lat,lon in zip(sta_lat,sta_lon):
        loss += MSE(y_true[0,lat,lon,0],y_pred[0,lat,lon,0])
    return np.round((loss/73),4)

def plot_map(predict_metrix, save_path, time, mse, KNN_matrix, custom_cmap, custom_norm, GeographyData_path):
    def gridtolatlon(grid_x,grid_y):
        lon = (np.array(grid_x)/100)+21.87
        lat = (np.array(grid_y)/100)+120
        return np.round(lon,3),np.round(lat,3)
    
    lon_min = 120;lon_max = 122.04;lat_min = 21.87;lat_max = 25.35
    color_min = 0; color_max = 370
    
    sta_lat=[9,61,66,70,70,74,74,76,77,81,81,82,87,89,89,102,112,118,118,125,144,160,160,169,185,185,
             189,189,189,205,206,210,211,220,223,227,229,230,236,239,252,270,277,283,288,288,294,300,
             304,309,309,311,312,313,315,315,316,317,317,318,319,320,320,320,320,320,321,321,324,326,330,331,332]

    sta_lon=[79,42,57,34,43,31,32,36,29,30,49,34,33,31,116,54,21,22,117,30,32,25,45,35,21,55,26,35,68,69,
             41,97,160,55,68,47,65,62,57,75,76,83,180,90,109,175,98,122,104,121,123,154,146,131,146,152,153,
             109,144,151,158,121,149,152,153,165,137,150,152,177,145,169,153]    
    

    fig = plt.figure(figsize=(10,8))
    ax = plt.subplot(121)
    plt.axis('off')

    m = Basemap(llcrnrlon=lon_min,llcrnrlat=lat_min,urcrnrlon=lon_max,urcrnrlat=lat_max,
                projection='merc', resolution='i',area_thresh=1000.)
    m.drawcoastlines()
    m.readshapefile(GeographyData_path,'metro', linewidth=.15)

    x, y = m(gridtolatlon(sta_lat,sta_lon)[1], gridtolatlon(sta_lat,sta_lon)[0])
    for i in range(len(sta_lat)):
        sc = ax.scatter(x, y, c = KNN_matrix[0,sta_lat,sta_lon,0], 
                        vmin=color_min, vmax=color_max, cmap=custom_cmap, norm = custom_norm,
                        s=50, edgecolors='None', linewidth=0.5)

    x0,x1 = ax.get_xlim();y0,y1 = ax.get_ylim()
    map_edges = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    polys = [p.boundary for p in m.landpolygons]
    polys = [map_edges]+polys[:]
    codes = [ [Path.MOVETO]+[Path.LINETO for p in p[1:]] for p in polys]
    polys_lin = [v for p in polys for v in p]
    codes_lin = [xx for cs in codes for xx in cs]
    path = Path(polys_lin, codes_lin)
    patch = PathPatch(path,facecolor='white', lw=0)
    ax.add_patch(patch)
    plt.title(time+'EPA Original 73 stations MA-PM2.5')
    
    
    ax1 = plt.subplot(122)
    plt.axis('off')
    m = Basemap(llcrnrlon=lon_min,llcrnrlat=lat_min,urcrnrlon=lon_max,urcrnrlat=lat_max,
                projection='merc', resolution='i',area_thresh=1000.)
    m.drawcoastlines()
    m.readshapefile(GeographyData_path,'metro', linewidth=.15)    
    
    grid_lon,grid_lat = np.linspace(lon_min, lon_max, 204),np.linspace(lat_min, lat_max, 348)
    xintrp, yintrp = np.meshgrid(grid_lon, grid_lat);x,y=m(xintrp, yintrp)
    sc1 = ax1.scatter(x, y,  c= predict_metrix[0,:,:,0],
                      vmin = color_min, vmax = color_max, cmap = custom_cmap, norm = custom_norm,
                      s=50, edgecolors='none')  
    
    x0,x1 = ax1.get_xlim();y0,y1 = ax1.get_ylim()
    map_edges = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    polys = [p.boundary for p in m.landpolygons]
    polys = [map_edges]+polys[:]
    codes = [ [Path.MOVETO]+[Path.LINETO for p in p[1:]] for p in polys]
    polys_lin = [v for p in polys for v in p]
    codes_lin = [xx for cs in codes for xx in cs]
    path = Path(polys_lin, codes_lin)
    patch = PathPatch(path,facecolor='white', lw=0)
    ax1.add_patch(patch)
    plt.title('CycleGan Prediction 73 staiotns MSE Ave:'+str(mse))

    position=fig.add_axes([0.15, 0.05, 0.7, 0.03])#位置[左,下,右,上]
    fig.colorbar(sc,cax=position,orientation='horizontal')
    plt.savefig(save_path, dpi=300, pad_inches = 0, bbox_inches = 'tight')
    plt.cla()
    return (None)


def bar_chart(data_filter_reindex, decoder_matrix, mse, save_path):
    data_filter_reindex['PM2.5_AVG'] = data_filter_reindex['PM2.5_AVG'].astype('float16')
    plot_data_pd = data_filter_reindex.sort_values(by=['PM2.5_AVG'])
    
    epa_sort_lat = plot_data_pd['lat'].values; epa_sort_lon = plot_data_pd['lon'].values

    CycleGan_sort_pm25 = []
    for epa_lat, epa_lon in zip(epa_sort_lat, epa_sort_lon): 
        CycleGan_sort_pm25.append(decoder_matrix[0, int(epa_lat), int(epa_lon), 0])


    bar_width = 0.4
    index = np.arange(73)

    plt.figure(figsize=(16,8))
    plt.bar(index, plot_data_pd['PM2.5_AVG'], bar_width, label='EPA')
    plt.bar(index+0.4, CycleGan_sort_pm25, bar_width, label='CycleGan')
    plt.xticks(list(index),plot_data_pd['SiteEngName'].values, rotation=90)
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 14)

    plt.title(plot_data_pd['Publish_Time'].values[0]+' Avg MSE:'+ str(mse), fontsize=16)
    plt.xlabel('Stations', fontsize=16)
    plt.ylabel('PM2.5_AVG', fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig(save_path, dpi=300, pad_inches = 0, bbox_inches = 'tight')
    plt.cla()
    return(None)

def plot_Animation(predict_metrix, time, mse, KNN_matrix_NoFake, KNN_matrix, Kriging_Graph, 
             custom_cmap, custom_norm, GeographyData_path):
    
    def gridtolatlon(grid_x,grid_y):
        lon = (np.array(grid_x)/100)+21.87
        lat = (np.array(grid_y)/100)+120
        return np.round(lon,3),np.round(lat,3)
    
    lon_min = 120;lon_max = 122.04;lat_min = 21.87;lat_max = 25.35
    color_min = 0; color_max = 370
    
    sta_lat=[9,61,66,70,70,74,74,76,77,81,81,82,87,89,89,102,112,118,118,125,144,160,160,169,185,185,
             189,189,189,205,206,210,211,220,223,227,229,230,236,239,252,270,277,283,288,288,294,300,
             304,309,309,311,312,313,315,315,316,317,317,318,319,320,320,320,320,320,321,321,324,326,330,331,332]

    sta_lon=[79,42,57,34,43,31,32,36,29,30,49,34,33,31,116,54,21,22,117,30,32,25,45,35,21,55,26,35,68,69,
             41,97,160,55,68,47,65,62,57,75,76,83,180,90,109,175,98,122,104,121,123,154,146,131,146,152,153,
             109,144,151,158,121,149,152,153,165,137,150,152,177,145,169,153]    
    

    fig = plt.figure(figsize=(14,8))
    ax = plt.subplot(141)
    plt.axis('off')

    m = Basemap(llcrnrlon=lon_min,llcrnrlat=lat_min,urcrnrlon=lon_max,urcrnrlat=lat_max,
                projection='merc', resolution='i',area_thresh=1000.)
    m.drawcoastlines()
    m.readshapefile(GeographyData_path,'metro', linewidth=.15)

    x, y = m(gridtolatlon(sta_lat,sta_lon)[1], gridtolatlon(sta_lat,sta_lon)[0])
    for i in range(len(sta_lat)):
        sc = ax.scatter(x, y, c = KNN_matrix[0,sta_lat,sta_lon,0], 
                        vmin=color_min, vmax=color_max, cmap=custom_cmap, norm = custom_norm,
                        s=50, edgecolors='None', linewidth=0.5)

    x0,x1 = ax.get_xlim();y0,y1 = ax.get_ylim()
    map_edges = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    polys = [p.boundary for p in m.landpolygons]
    polys = [map_edges]+polys[:]
    codes = [ [Path.MOVETO]+[Path.LINETO for p in p[1:]] for p in polys]
    polys_lin = [v for p in polys for v in p]
    codes_lin = [xx for cs in codes for xx in cs]
    path = Path(polys_lin, codes_lin)
    patch = PathPatch(path,facecolor='white', lw=0)
    ax.add_patch(patch)
    plt.title('EPA 73 stations')

    
    
    ax1 = plt.subplot(142)
    plt.axis('off')
    m = Basemap(llcrnrlon=lon_min,llcrnrlat=lat_min,urcrnrlon=lon_max,urcrnrlat=lat_max,
                projection='merc', resolution='i',area_thresh=1000.)
    m.drawcoastlines()
    m.readshapefile(GeographyData_path,'metro', linewidth=.15)    
    
    grid_lon,grid_lat = np.linspace(lon_min, lon_max, 204),np.linspace(lat_min, lat_max, 348)
    xintrp, yintrp = np.meshgrid(grid_lon, grid_lat);x,y=m(xintrp, yintrp)
    sc1 = ax1.scatter(x, y,  c = KNN_matrix_NoFake[0,:,:,0],
                      vmin = color_min, vmax = color_max, cmap = custom_cmap, norm = custom_norm,
                      s=50, edgecolors='none')  
    
    x0,x1 = ax1.get_xlim();y0,y1 = ax1.get_ylim()
    map_edges = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    polys = [p.boundary for p in m.landpolygons]
    polys = [map_edges]+polys[:]
    codes = [ [Path.MOVETO]+[Path.LINETO for p in p[1:]] for p in polys]
    polys_lin = [v for p in polys for v in p]
    codes_lin = [xx for cs in codes for xx in cs]
    path = Path(polys_lin, codes_lin)
    patch = PathPatch(path,facecolor='white', lw=0)
    ax1.add_patch(patch)
    plt.title('KNN3')    
    
    ax1 = plt.subplot(143)
    plt.axis('off')
    m = Basemap(llcrnrlon=lon_min,llcrnrlat=lat_min,urcrnrlon=lon_max,urcrnrlat=lat_max,
                projection='merc', resolution='i',area_thresh=1000.)
    m.drawcoastlines()
    m.readshapefile(GeographyData_path,'metro', linewidth=.15)    
    
    grid_lon,grid_lat = np.linspace(lon_min, lon_max, 204),np.linspace(lat_min, lat_max, 348)
    xintrp, yintrp = np.meshgrid(grid_lon, grid_lat);x,y=m(xintrp, yintrp)
    sc1 = ax1.scatter(x, y,  c = Kriging_Graph[0,:,:,0],
                      vmin = color_min, vmax = color_max, cmap = custom_cmap, norm = custom_norm,
                      s=50, edgecolors='none')  
    
    x0,x1 = ax1.get_xlim();y0,y1 = ax1.get_ylim()
    map_edges = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    polys = [p.boundary for p in m.landpolygons]
    polys = [map_edges]+polys[:]
    codes = [ [Path.MOVETO]+[Path.LINETO for p in p[1:]] for p in polys]
    polys_lin = [v for p in polys for v in p]
    codes_lin = [xx for cs in codes for xx in cs]
    path = Path(polys_lin, codes_lin)
    patch = PathPatch(path,facecolor='white', lw=0)
    ax1.add_patch(patch)
    plt.title('Kriging')        
    
    
    ax1 = plt.subplot(144)
    plt.axis('off')
    m = Basemap(llcrnrlon=lon_min,llcrnrlat=lat_min,urcrnrlon=lon_max,urcrnrlat=lat_max,
                projection='merc', resolution='i',area_thresh=1000.)
    m.drawcoastlines()
    m.readshapefile(GeographyData_path,'metro', linewidth=.15)    
    
    grid_lon,grid_lat = np.linspace(lon_min, lon_max, 204),np.linspace(lat_min, lat_max, 348)
    xintrp, yintrp = np.meshgrid(grid_lon, grid_lat);x,y=m(xintrp, yintrp)
    sc1 = ax1.scatter(x, y,  c= predict_metrix[0,:,:,0],
                      vmin = color_min, vmax = color_max, cmap = custom_cmap, norm = custom_norm,
                      s=50, edgecolors='none')  
    
    x0,x1 = ax1.get_xlim();y0,y1 = ax1.get_ylim()
    map_edges = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    polys = [p.boundary for p in m.landpolygons]
    polys = [map_edges]+polys[:]
    codes = [ [Path.MOVETO]+[Path.LINETO for p in p[1:]] for p in polys]
    polys_lin = [v for p in polys for v in p]
    codes_lin = [xx for cs in codes for xx in cs]
    path = Path(polys_lin, codes_lin)
    patch = PathPatch(path,facecolor='white', lw=0)
    ax1.add_patch(patch)
    plt.title('Target Embedding CycleGan')
    

    position=fig.add_axes([0.15, 0.05, 0.7, 0.03])#位置[左,下,右,上]
    fig.colorbar(sc,cax=position,orientation='horizontal')
    plt.savefig('Animation/'+time+'.jpg')
    return (None)



def main_func_ex6(epa_pm25_table, cwb_wea_table, epa_time, cwb_time):
    scale_lst = np.load('color_setting/scale_lst.npy')
    hex_lst = np.load('color_setting/hex_lst.npy')
    custom_cmap = matplotlib.colors.ListedColormap(hex_lst)
    custom_norm = matplotlib.colors.BoundaryNorm(scale_lst, len(hex_lst))    
    GeographyData_path = 'GeographyData/COUNTY_MOI_1081121'
    TWMTP = np.reshape(np.load('TW_Mountainous_Terrain_percent.npy'),(1,348,204,1))

    EPA_PM25_Graph = EPA_PM25_KNN(epa_pm25_table)
    CWB_attribute_Graph = CWB_Attribute_KNN(cwb_wea_table)

    x = Input(shape=(348, 204, 1))
    channel_number = 6
    PM25_Encoder_model = Encoder(channel_number)
    PM25_Decoder_model = Decoder(channel_number)
    PM25_Autoencoder = Model(x,PM25_Decoder_model(PM25_Encoder_model(x)))
    PM25_Autoencoder.load_weights('Weight/Target_Embedding/0323-train_58.341145-test_28.762764.h5')
    PM25_pred = PM25_Decoder_model.predict(PM25_Encoder_model.predict(EPA_PM25_Graph))
    PM25_AutoEncoder_MSE = np.round(np.mean((EPA_PM25_Graph - PM25_pred)**2) ,4)
    PM25_code = PM25_Encoder_model.predict(EPA_PM25_Graph)

    # CWB_attribute_Graph : rain,wdir,wdsd,temp,humd
    Rain_Encoder_model = Encoder(channel_number)
    Rain_Decoder_model = Decoder(channel_number)
    Rain_Autoencoder = Model(x,Rain_Decoder_model(Rain_Encoder_model(x)))
    Rain_Autoencoder.load_weights('Weight/Attribute/CWBEPA_Content_RAIN_channel6_K3/0122-train_0.030204-test_0.006936.h5')
    Rain_pred = Rain_Decoder_model.predict(Rain_Encoder_model.predict(CWB_attribute_Graph[0:1]))
    Rain_AutoEncoder_MSE = np.round(np.mean((CWB_attribute_Graph[0:1] - Rain_pred)**2) ,4)
    Rain_code = Rain_Encoder_model.predict(CWB_attribute_Graph[0:1])

    WDIR_Encoder_model = Encoder(channel_number)
    WDIR_Decoder_model = TanhDecoder(channel_number)
    WDIR_Autoencoder = Model(x,WDIR_Decoder_model(WDIR_Encoder_model(x)))
    WDIR_Autoencoder.load_weights('Weight/Attribute/CWBEPA_Content_WIND_DIREC_channel6_K3/0137-train_0.004478-test_0.003393.h5')
    WDIR_pred = WDIR_Decoder_model.predict(WDIR_Encoder_model.predict(CWB_attribute_Graph[1:2]))
    WDIR_AutoEncoder_MSE = np.round(np.mean((CWB_attribute_Graph[1:2] - WDIR_pred)**2) ,4)
    WDIR_code = WDIR_Encoder_model.predict(CWB_attribute_Graph[1:2])

    WDSD_Encoder_model = Encoder(channel_number)
    WDSD_Decoder_model = Decoder(channel_number)
    WDSD_Autoencoder = Model(x,WDSD_Decoder_model(WDSD_Encoder_model(x)))
    WDSD_Autoencoder.load_weights('Weight/Attribute/CWBEPA_Content_WIND_SPEED_channel6_K3/0063-train_0.024618-test_0.045998.h5')
    WDSD_pred = WDSD_Decoder_model.predict(WDSD_Encoder_model.predict(CWB_attribute_Graph[2:3]))
    WDSD_AutoEncoder_MSE = np.round(np.mean((CWB_attribute_Graph[2:3] - WDSD_pred)**2) ,4)
    WDSD_code = WDSD_Encoder_model.predict(CWB_attribute_Graph[2:3])

    TEMP_Encoder_model = Encoder(channel_number)
    TEMP_Decoder_model = Decoder(channel_number)
    TEMP_Autoencoder = Model(x,TEMP_Decoder_model(TEMP_Encoder_model(x)))
    TEMP_Autoencoder.load_weights('Weight/Attribute/CWBEPA_Content_TEMP_channel6_K3/0075-train_0.282232-test_0.111152.h5')
    TEMP_pred = TEMP_Decoder_model.predict(TEMP_Encoder_model.predict(CWB_attribute_Graph[3:4]))
    TEMP_AutoEncoder_MSE = np.round(np.mean((CWB_attribute_Graph[3:4] - TEMP_pred)**2) ,4)
    TEMP_code = TEMP_Encoder_model.predict(CWB_attribute_Graph[3:4])

    RH_Encoder_model = Encoder(channel_number)
    RH_Decoder_model = Decoder(channel_number)
    RH_Autoencoder = Model(x,RH_Decoder_model(RH_Encoder_model(x)))
    RH_Autoencoder.load_weights('Weight/Attribute/CWBEPA_Content_RH_channel6_K3/0461-train_4.235878-test_3.270598.h5')
    RH_pred = RH_Decoder_model.predict(RH_Encoder_model.predict(CWB_attribute_Graph[4:5]))
    RH_AutoEncoder_MSE = np.round(np.mean((CWB_attribute_Graph[4:5] - RH_pred)**2) ,4)
    RH_code = RH_Encoder_model.predict(CWB_attribute_Graph[4:5])

    month_format = np.zeros((44,26,1))  # month
    weekday_format = np.zeros((44,26,1))  # month

    month_format[:,:,:] = datetime.now().month
    weekday_format[:,:,:] = datetime.isoweekday(datetime.strptime(str(datetime.now().year)+'-'+
                                           str(datetime.now().month)+'-'+
                                           str(datetime.now().day),
                                          "%Y-%m-%d"))

    TIME_code = np.reshape(np.concatenate([month_format,weekday_format],axis=2),(1,44,26,2))

    #/CycleGan/Experiment6/Data/TFA_CWBEPA_BIGD_Unpair_TE_CycleGan_AIRSAT_RMSprop_Res9_Channel6_Reload/39.5526
    g_model_AtoB = load_model('Weight/CycleGan/g_model_AtoB_0002450.h5')
    pred_full_code = g_model_AtoB.predict([PM25_code, Rain_code, RH_code, TEMP_code, TIME_code, WDIR_code, WDSD_code])
    pred_full_graph = (PM25_Decoder_model.predict(pred_full_code))* np.flip(TWMTP,1)

    mse = np.round(calculate_73_Loss(EPA_PM25_Graph, pred_full_graph),2)

    time = ((str(epa_pm25_table['Publish_Time'][0].replace('/','-')).replace(' ','-')).replace(':','-'))
    time_format = datetime.strptime(time, "%Y-%m-%d-%H-%M-%S")
    time_name = str(time_format.date())+'-'+str(time_format.hour)

    if not os.path.exists("History_data/testing_ex6/"+str(time_format.date())+'/'):
        os.makedirs("History_data/testing_ex6/"+str(time_format.date())+'/')
        os.makedirs("History_data/fig_ex6/"+str(time_format.date())+'/')
    if not os.path.exists('static/'+str(time_format.date())+'/'):
        os.makedirs('static/'+str(time_format.date())+'/')


    f4 = open("History_data/testing_ex6/"+str(time_format.date())+'/'+time_name+'_A2B_mse_AEReuslt.txt', 'a')
    print('---------------------------',file=f4)
    print('AE_RealTime_PM25_MSE:',PM25_AutoEncoder_MSE, 'AE_Test_PM25_MSE :28.7627',file=f4)        
    print('AE_RealTime_Rain_MSE:',Rain_AutoEncoder_MSE, 'AE_Test_Rain_MSE :0.0069',file=f4)
    print('AE_RealTime_WDIR_MSE:',WDIR_AutoEncoder_MSE, 'AE_Test_WDIR_MSE :0.0033',file=f4)
    print('AE_RealTime_WDSD_MSE:',WDSD_AutoEncoder_MSE, 'AE_Test_WDSD_MSE :0.045',file=f4)
    print('AE_RealTime_TEMP_MSE:',TEMP_AutoEncoder_MSE, 'AE_Test_TEMP_MSE :0.1111',file=f4)
    print('AE_RealTime_RH_MSE:',RH_AutoEncoder_MSE, 'AE_Test_RH_MSE :3.2705',file=f4)
    print('---------------------------',file=f4)
    f4.close()              

    epa_pm25_table.to_csv("History_data/testing_ex6/"+str(time_format.date())+'/'+time_name+'_A2B_mse_'+str(mse)+'.csv')

    plot_map(pred_full_graph, 'History_data/fig_ex6/'+str(time_format.date())+'/'+time_name+'_A2B_mse_'+str(mse)+'.jpg', 
             time_name, mse, EPA_PM25_Graph, custom_cmap, custom_norm, GeographyData_path)
    #for API Web
    plot_map(pred_full_graph, 'static/'+str(time_format.date())+'/'+time_name+'_ex6.jpg', 
             time_name, mse, EPA_PM25_Graph, custom_cmap, custom_norm, GeographyData_path)

    bar_chart(epa_pm25_table, pred_full_graph, mse, save_path = 'static/'+str(time_format.date())+'/'+time_name+'_bar_ex6.jpg')

    print((str(time_format.date())+'-'+str(time_format.hour)+'_A2B_mse_'+str(mse)))
    print('figure save done')

    del pred_full_graph, epa_pm25_table, pred_full_code, g_model_AtoB, EPA_PM25_Graph
    
    gc.collect()
    return(None)
