import pandas as pd 
import csv
import random as rd 
from random import random
from random import sample
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import heatmap
import geopandas as gpd
import datetime 
import descartes

#import geoplot
from shapely.geometry import Point, Polygon
from statsmodels.stats.outliers_influence import variance_inflation_factor
from numpy.linalg import svd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, BaggingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence 
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

#----Tree visualization--------------------------------
from sklearn.tree import export_graphviz
from IPython.display import Image
from sklearn.metrics import r2_score

pd.options.display.max_colwidth = 100

#fullcrimes = pd.read_csv("../Capstone_2/data/Crimes_-_2001_to_present.csv")
#chicagocrimes2018 = fullcrimes.where(fullcrimes['Year'] >= 2018)
#chicagocrimes2018 = chicagocrimes2018.to_csv('data/crimes2018.csv')

# chicagocrimes = pd.read_csv('data/crimes2018.csv')
# crimes = chicagocrimes.where(chicagocrimes['Year'] == 2018) 
crimes = pd.read_csv('data/2018crimes.csv')
# crimes = readfile.where(readfile['Year'] == 2018, inplace=True)
#wardnums = pd.read_csv('data/WardpopsPerc.csv', header=0)
commareanums = pd.read_csv('data/CensusCA.csv', header = 0)
aerosol=pd.read_excel('../../../Downloads/Aerosol_Optical_Depth_v2.xlsx')
coarsepart=pd.read_excel('../../../Downloads/Coarse_Particle_Pollution__Inverse_Distance_Weighting.xlsx')
finepart=pd.read_excel('../../../Downloads/Fine_Particle_Pollution__Inverse_Distance_Weighting_v2.xlsx')
nitdioxide=pd.read_excel('../../../Downloads/Nitrogen_Dioxide_Annual_Air_Concentration_v2.xlsx')
ozone=pd.read_excel('../../../Downloads/Ozone_Annual_Concentration__Inverse_Distance_Weighting_v2.xlsx')
# annualepaconcs = pd.read_csv('data/annual_conc_by_monitor_2018.csv')
# illconc = annualepaconcs[annualepaconcs['State Name'] == 'Illinois'] 
# chiccoc = illconc[illconc['City Name']=='Chicago']
# illconcozone = illconc[illconc['Parameter Name']=='Ozone']
# chicozone = chiccoc[chiccoc['Parameter Name']=='Ozone']
crimesdom=crimes[crimes['Domestic'] == True]

arrestsbyca = crimes.groupby('Community Area')['Arrest'].sum()
arrestsdom=crimesdom.groupby('Community Area')['Arrest'].sum()

dombyca = crimes.groupby('Community Area')['Domestic'].sum()
commareanums['arrestsbyca'] = arrestsbyca
commareanums['dombyca'] = dombyca 
commareanums['arrestsdom'] = arrestsdom

commareanums['Aerosol'] = aerosol['Ave_Annual_Number']
commareanums['CoarsePart'] = coarsepart['Ave_Annual_Number'] 
commareanums['Ozone'] = ozone['Ave_Annual_Number']
commareanums['FinePart'] = finepart['Ave_Annual_Number']
commareanums['NitDioxide'] = nitdioxide['Ave_Annual_Number']


commareanums['arrestrate'] = (commareanums['arrestsbyca']/commareanums['POPULATION'])*1000
commareanums['domrate'] = (commareanums['dombyca']/commareanums['POPULATION'])*1000
commareanums['domarrestrate'] = commareanums['arrestsdom']/commareanums['dombyca']
commareanums['arrestrate'] = commareanums['arrestrate'].astype('float')
commareanums['domrate'] = commareanums['domrate'].astype('float')
    
commareaanal = commareanums.drop(['COMMUNITY AREA NAME', 'POPULATION','arrestsbyca','dombyca','arrestsdom','domarrestrate'], axis = 1) 
commareaanal=commareaanal.dropna()

X = commareaanal.drop(['Community Area Number', 'arrestrate','domrate'], axis =1)
y = commareaanal['domrate']

num_estimator_list = [1,2,5,10,20,40,100,500,1000]




#########plotting error values on map
street_map = gpd.read_file('data/geo_export_e603e826-68c1-4029-95cf-2feaa0cc9da2.shp')
street_map["area_numbe"] = street_map["area_numbe"].astype(int)

merged = street_map.set_index('area_numbe').join(commareanums.set_index('Community Area Number'))

def plot_map(measure):
    plotvar = merged[measure]
    vmin=plotvar.min()
    vmax=plotvar.max()
    fig, ax = plt.subplots(1, figsize=(10, 6))
    merged.plot(column=plotvar, linewidth=0.8, ax=ax)
    # Create colorbar as a legend
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # empty array for the data range
    sm._A = []
    # add the colorbar to the figure
    cbar = fig.colorbar(sm)


if __name__ == '__main__':
    ######EDA
    # fig,ax = plt.subplots()
    # pd.plotting.scatter_matrix(commareaanal) 
    # plt.xticks(rotation=45, fontsize=5)
    # plt.yticks(rotation=90, fontsize=5)

    # cacorr=commareaanal.corr()
    # sns.heatmap(cacorr)
    #############plots
    # commareanums=commareanums.sort_values('domrate')
    # plt.barh(commareanums['COMMUNITY AREA NAME'], commareanums['domrate'])
    # plt.yticks(fontsize = 5)
    # plt.title('Domestic Call Rate per 1000 Population')

    # commareanums=commareanums.sort_values('arrestrate')
    # plt.barh(commareanums['COMMUNITY AREA NAME'], commareanums['arrestrate'])
    # plt.yticks(fontsize = 5)
    # plt.title('Arrest Rate per 1000 Population')

    # commareanums=commareanums.sort_values('domarrestrate')
    # plt.barh(commareanums['COMMUNITY AREA NAME'],commareanums['domarrestrate'])
    # plt.yticks(fontsize = 5)
    # plt.title('Arrest Rate for Domestic Calls by Community Area')

    # plt.scatter(commareanums['arrestrate'],commareanums['domrate'])
    # plt.title('Arrest Population Rate v. Domestic Call Poopulation Rate')

    np.corrcoef(commareanums['arrestrate'], commareanums['domrate'])

    # f = plt.figure(figsize=(19, 15))
    # plt.matshow(commareaanal.corr(), fignum=f.number)
    # plt.xticks(range(commareaanal.shape[1]), commareaanal.columns, fontsize=8, rotation=45)
    # plt.yticks(range(commareaanal.shape[1]), commareaanal.columns, fontsize=8)
    # cb = plt.colorbar()
    # cb.ax.tick_params(labelsize=10)
    ########mapping epa monitors in chicago

    street_map = gpd.read_file('data/geo_export_e603e826-68c1-4029-95cf-2feaa0cc9da2.shp')
# site_points = chicozone[['Longitude', 'Latitude']].apply(lambda row:
#               Point(row["Longitude"], row["Latitude"]), axis=1)
# geo_sites = gpd.GeoDataFrame({"geometry": site_points,
# 			"site_names": chicozone["Local Site Name"]})
# geo_sites.crs = {"init": 'epsg:4326'}
# geometrydom = [Point(xy) for xy in zip(chicozone["Longitude"], chicozone["Latitude"])]
# geo_dfdom = gpd.GeoDataFrame(chicozone, crs=geo_sites.crs, geometry = geometrydom)
# fig,ax = plt.subplots(figsize = (15, 15))
# # street_map.plot(ax = ax, alpha = .4, color='grey')

# street_map.plot(ax = ax, alpha = .4)


# fig, ax = plt.subplots(1, figsize=(3.5,7))
# base = street_map.plot(ax=ax, color='gray')
# geo_sites.plot(ax=base, marker="o", markersize=10, alpha=0.5, color='red')
# _ = ax.axis('off')
# ax.set_title("Plot of monitors in Chicago")

    ##############Random Forest Regression
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=123)

    # train_errors_rf = []
    # test_errors_rf = []

    # for num_est in num_estimator_list:
    #     rf = RandomForestRegressor(n_estimators = num_est, n_jobs=-1)
    #     rf.fit(X_train, y_train)
    #     y_pred_test =  rf.predict(X_test)
    #     y_pred_train =  rf.predict(X_train)
    
    #     train_errors_rf.append(mean_squared_error(y_pred_train, y_train)) 
    #     test_errors_rf.append(mean_squared_error(y_pred_test, y_test))  

        

    # plt.figure(figsize=(15,10))
    # plt.plot(num_estimator_list, train_errors_rf, label='Training MSE')
    # plt.plot(num_estimator_list, test_errors_rf, label='Test MSE')
    # plt.xlabel('Number of Estimators')
    # plt.ylabel('MSE')
    # plt.xscale('log')
    # plt.title('Random Forest MSE vs. Num Estimators')
    # plt.legend()   

    ##########Getting each CA MSE
   
    # train_errors_rf = []
    # test_errors_rf = []
    
    rf = RandomForestRegressor(n_estimators = 20, n_jobs=-1)
    rf.fit(X, y)
    y_pred =  rf.predict(X)
    commareanums['diff2'] = (y - y_pred)**2
    commareanums['diffbyca'] = commareanums.groupby('Community Area Number')['diff2'].mean()

    # plt.barh(commareanums['COMMUNITY AREA NAME'],diffbyca)
    # plt.yticks(fontsize=5)
     

#########plotting error values on map
    street_map = gpd.read_file('data/geo_export_e603e826-68c1-4029-95cf-2feaa0cc9da2.shp')
    street_map["area_numbe"] = street_map["area_numbe"].astype(int)
    #cpd.merge(counties, us_data, on="FIPS")
    merged = street_map.set_index('area_numbe').join(commareanums.set_index('Community Area Number'))
    meandiff=merged['diffbyca'].mean()
    stddiff=merged['diffbyca'].std()
    merged['diffstdzd']=(merged['diffbyca'] - meandiff)/stddiff
    # q = merged['diffbyca'].quantile(0.75)
    # merged[merged["diffbyca"] < q]
    
    plotvar = merged['diffbyca']
    #vmin, vmax = 0, 100
    vmin=plotvar.min()
    vmax=plotvar.max()
    fig, ax = plt.subplots(1, figsize=(10, 6))
    merged.plot(column=plotvar, linewidth=0.8, ax=ax, cmap='PRGn')
    #plt.clim(plotvar.min(), plotvar.max())
    # Create colorbar as a legend
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap='PRGn')
    # empty array for the data range
    sm._A = []
    # add the colorbar to the figure
    cbar = fig.colorbar(sm)
    
    


    ######OOB Score R2

    # train_r2_rf = []
    # oob_r2_rf = []

    # for num_est in num_estimator_list:
    #     rf = RandomForestRegressor(n_estimators = num_est, oob_score=True, n_jobs=-1)
    #     rf.fit(X, y)
    #     #y_pred_test =  rf.predict(X_test)
    #     y_pred =  rf.predict(X)
    
    #     train_r2_rf.append(r2_score(y, y_pred)) 
    #     oob_r2_rf.append(rf.oob_score_)

    
    # plt.figure(figsize=(15,10))
    # plt.plot(num_estimator_list, train_r2_rf, label='Training R2')
    # plt.plot(num_estimator_list, oob_r2_rf, label='oob R2')
    # plt.xlabel('Number of Estimators')
    # plt.ylabel('R2')
    # plt.xscale('log')
    # plt.title('Random Forest R2 vs. Num Estimators')
    # plt.legend()     

    #######feature importances
    # rf = RandomForestRegressor(n_estimators = 10, n_jobs=-1)
    # rf.fit(X_train, y_train)

    # feature_names = commareaanal.columns.values[1:14]
    # plt.figure(figsize = (15,10))
    # plt.barh(feature_names, rf.feature_importances_)
    # #plt.xticks(rotation=20, fontsize=8)
    # plt.title('Feature Importances')
    # plt.xlabel("Normalized decrease in node impurity")
    # plt.ylabel("Features")

    #########Partial dependence plots

    # gb = GradientBoostingRegressor(n_estimators = 20)
    # gb.fit(X_train, y_train)

    
    
    # fig, axs = plot_partial_dependence(gb, X_train, [2],
    #                                    feature_names = commareaanal.columns.values[1:14],
    #                                    n_jobs=-1, grid_resolution=50)
    # fig.subplots_adjust(hspace = .2)
    # fig.suptitle('Partial Dependence Plot')
    # fig.set_figwidth(15)

    

    


        

