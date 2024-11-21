#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 18:22:12 2023

@author: Wangdong
"""

import os
import pandas as pd
from wsimod.orchestration.model import Model, to_datetime
from wsimod.core import constants
from datetime import datetime
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
import numpy as np
import statistics

suffix = '_gwh_reservoir_timevaryingdemand'
# %%

#Directories
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v0")
model_dir = os.path.join(data_dir, "model")
wr_dir = os.path.dirname(os.path.abspath(__file__))

#Q value
Abs_subs=['1683','1823','1701','1684',"1685",
          '1893','1818','1896','1821','1894','1704','1702','1703',
          '1686','1820','1695','1219','1897','1676']



   
Sub_Arcs={'1823':'node_10393-to-node_10393-breaknode_3174',
'1683':'node_3174-to-1683-river',
'1685':'1685-river-to-node_3221-breaknode_2912',
'1701':'node_2912-to-1701-river',
'1684':'1684-river-to-node_3219-breaknode_3220',
'1818':'node_10375-to-node_10375-breaknode_10376',
'1893':'1893-river-to-node_3224-breaknode_3220',
'1896':'1896-river-to-node_3183-breaknode_3180',
'1821':'node_10382-to-node_10382-breaknode_10383',
'1695':'1695-river-to-node_10227-breaknode_10228',
'1897':'node_10641-to-node_10641-breaknode_9698',
'815':'815-river-to-node_2506',
'1676':'node_9720-to-node_9720-breaknode_9698',
'1704':'node_2974-to-1704-river',
'1892':'node_10714-to-node_10714-breaknode_2974',
'1821':'node_10382-to-node_10382-breaknode_10383',
'1894':'1894-river-to-node_3179-breaknode_3180',
'1895':'1895-river-to-node_3181-breaknode_3182',
'1686':'1686-river-to-node_3222-breaknode_3223',
'1822':'node_10385-to-node_10385-breaknode_10386',
'1819':'1819-river-to-node_2976-breaknode_2977',
'1697':'1697-river-to-node_2902-breaknode_2903',
'1702':'node_10241-to-node_10386',
'1820':'node_10378-to-node_10378-breaknode_10255',
'1703':'node_10365-to-node_10365-breaknode_10646',
'1694':'1694-river-to-node_2581-breaknode_2582',
'1221':'1221-river-to-node_2577-breaknode_2578',
'1681':'1681-river-to-node_2579-breaknode_2580',
'1220':'node_9719-to-node_9719-breaknode_9720',
'1219':'1219-river-to-node_1940-breaknode_1941',
'1577':'node_2870-to-node_2870-breaknode_2871',
'1898':'node_10750-to-node_10750-breaknode_10751',
'1576':'1576-river-to-node_2868-breaknode_2869',
'1578':'1578-river-to-node_2872-breaknode_2873',
'1700':'1700-river-to-node_2910-breaknode_2911',
'1217':'1217-river-to-node_2869-breaknode_2506'}



Zero_flow = pd.read_pickle(os.path.join(model_dir,'results1','_gwh_reservoir_timevaryingdemand', 'flows_FullList_check.pkl'))
def Qn(list, n):
    a = len(list)
    percentile = 100-n
    position = (percentile / 100) * a
    
    if position.is_integer(): #whether integer
        Qn = list[int(position) - 1]
    else: # linear interpolation
        lower_index = int(np.floor(position)) - 1 #floor 1.3-1
        upper_index = int(np.ceil(position)) - 1 #ceil 1.3-2
        Qn = list[lower_index] + (position - (lower_index + 1)) * (list[upper_index] - list[lower_index])
    return Qn

dic_Q30={}
dic_Q50={}
dic_Q70={}
dic_Q95={}
for sub in Abs_subs:
    data = Zero_flow.groupby('arc').get_group(Sub_Arcs[sub])['flow']
    Q30 = Qn(sorted(data),30)
    Q50 = Qn(sorted(data),50)
    Q70 = Qn(sorted(data),70)
    Q95 = Qn(sorted(data),95)
    dic_Q30[sub]=Q30
    dic_Q50[sub]=Q50
    dic_Q70[sub]=Q70
    dic_Q95[sub]=Q95


results_dir = os.path.join(model_dir, "results1", suffix)
if not os.path.exists(results_dir):
    os.mkdir(results_dir)
    

# %%
#Load model
my_model = Model()
my_model.load(model_dir, config_name = 'config_ModifyRecharge_NW_noneABS'+suffix+'.yml')

dates = [to_datetime(str(x)) for x in pd.date_range('2000-01-01','2015-12-31')]

# %%
# Custom groundwater abstractions already added in the yml
# add sewer leakage to 1823-gw
nodelist = []
nodelist.append({'name': '1823-storm-add',
                 'capacity': 33447258.251555994,
                 'pipe_time': 0,
                 'chamber_area': 1,
                 'pipe_timearea': '*id001',
                 'chamber_floor': 10,
                 'type_': 'Sewer',
                 'node_type_override': 'Sewer'})
my_model.add_nodes(nodelist)
arclist = []
arclist.append({'capacity': 1000000000000000.0,
                         'name': '1823-storm-to-1823-storm-add',
                         'preference': 1,
                         'type_': 'Arc',
                         'in_port': '1823-storm',
                         'out_port': '1823-storm-add'})
arclist.append({'capacity': 1000000000000000.0,
                         'name': '1823-storm-add-to-1823-gw',
                         'preference': 1e10,
                         'type_': 'Arc',
                         'in_port': '1823-storm-add',
                         'out_port': '1823-gw'})
my_model.add_arcs(arclist)

# %%
# Boundary conditions for gwh
# Customise and create the Catchment object - for inflows from wfdid-external-gw
def custom_route(self):
    def inner_function():
        """Send any water that has not already been abstracted downstream
        """
        #Get amount of water
        avail = self.get_avail()
        #Route excess flow onwards
        reply = self.push_distributed(avail, of_type = ['Groundwater',
                                                        'Groundwater_h',
                                                        'Waste'])
        self.unrouted_water = self.sum_vqip(self.unrouted_water, reply)
        if reply['volume'] > constants.FLOAT_ACCURACY:
            pass
    return inner_function


# typical parameter values (from Barney's COVID19 paper)
groundwater_pollutants = {'do' : 7.6,
                         'org-phosphorus' : 0.04,
                         'phosphate' : 0.25,
                         'ammonia' : 0.03,
                         'solids' : 8,
                         'bod' : 3,
                         'cod' : 36,
                         'ph' : 0.007,
                         'temperature' : 11,
                         'nitrate' : 4.6,
                         'nitrite' : 0.004,
                         'org-nitrogen' : 4.3}


cat_tf = [node_name for node_name in my_model.nodes.keys() if '-external-gw' in node_name]

for cat_tf_ in cat_tf:
    my_model.nodes[cat_tf_].route = custom_route(my_model.nodes[cat_tf_])

for cat_tf_ in cat_tf:
    wfdid = cat_tf_.split('-')[0]
    BC = pd.read_csv(os.path.join(data_dir, os.pardir, 'additional_data_LL', 'Budget_EXTERNALBOUND', wfdid+'.csv')).set_index('time')
    data_input_dict_flow = {}
    for date in dates:
        for key in constants.POLLUTANTS:
            my_model.nodes[cat_tf_].data_input_dict[(key, date)] = groundwater_pollutants[key]
            if key in constants.ADDITIVE_POLLUTANTS:
                my_model.nodes[cat_tf_].data_input_dict[(key, date)] *= constants.MG_L_TO_KG_M3
        data_input_dict_flow[('flow', date)] = BC.loc[pd.to_datetime(str(date)).month, 'GW_IN'] # [m3/day]
    my_model.nodes[cat_tf_].data_input_dict = my_model.nodes[cat_tf_].data_input_dict | data_input_dict_flow

# Customise and create the arcs - for outflows to external-gw-waste
def bc_outflow_distribute_gw(arc, custom_flows, func):
    def customise_arc():
        month = arc.in_port.t.month
        amount_to_extract = custom_flows.loc[month, 'GW_OUT']
        # make a pull request from the GW
        arc.send_pull_request({'volume': amount_to_extract})
        func()
    return customise_arc
def bc_outflow_distribute_gw_1685(arc, custom_flows, func): # designed for 1685, which uses the exchange flows derived from the generalhead boundaries
    def customise_arc():
        t = str(arc.in_port.t)
        amount_to_extract = custom_flows.loc[t, 'flow']
        # make a pull request from the GW
        arc.send_pull_request({'volume': amount_to_extract})
        func()
    return customise_arc
arcs_to_external_gw_waste = [arc_name for arc_name in my_model.arcs.keys() if 'external-gw-waste' in arc_name]
for arc_name in arcs_to_external_gw_waste:
    arc = my_model.arcs[arc_name]
    gw_name = arc_name.split('-to-')[0]
    gw = my_model.nodes[gw_name]
    wfdid = gw_name.split('-')[0]
    if wfdid == '1685':
        BC = pd.read_csv(os.path.join(data_dir, os.pardir, 'additional_data_LL', 'Budget_EXTERNALBOUND', wfdid+'_fromgeneralhead.csv')).set_index('time')
        gw.distribute_gw_gw = bc_outflow_distribute_gw_1685(arc, BC, gw.distribute_gw_gw)
    else:
        BC = pd.read_csv(os.path.join(data_dir, os.pardir, 'additional_data_LL', 'Budget_EXTERNALBOUND', wfdid+'.csv')).set_index('time')
        gw.distribute_gw_gw = bc_outflow_distribute_gw(arc, BC, gw.distribute_gw_gw)
# %%
#Update parameters for Lea navigation

#Add custom water resources elements (because these are not default WSIMOD they
#must be added separately)
ltoa = pd.read_csv(os.path.join(wr_dir, "ltoa.csv"))
ltoa = ltoa.set_index('month')['level-1'].to_dict()

nlars = pd.read_csv(os.path.join(wr_dir, "nlars.csv"))
nlars = nlars.set_index('month')['capacity'].to_dict()

def month_difference(d1, d2):
    date1_dt = datetime.strptime(d1._date, "%Y-%m")
    date2_dt = datetime.strptime(d2._date, "%Y-%m")
    
    return relativedelta(date1_dt, date2_dt).months

def eval_ltoa(self, direction = 'pull', vqip = None, tag = 'default'):
    pct_full = self.out_port.tank.storage['volume'] / self.out_port.capacity
    month = self.out_port.t.month
    
    if self.nlars_month_tracker != 0:
        nlars_duration = month_difference(self.out_port.monthyear, self.nlars_month_tracker)
        if nlars_duration > self.nlars_reset:
            self.nlars_month_tracker = 0
            
    if pct_full < self.ltoa[month]:
        if self.nlars_month_tracker == 0:
            nlars_duration = 0
            self.nlars_month_tracker = self.out_port.monthyear
        
        if nlars_duration in self.nlars.keys():
            capacity = self.nlars[nlars_duration]
        else:
            capacity = 0
    else:
        capacity = 0
    capacity -= self.flow_in
    # avail = self.out_port.pull_check({'volume' : capacity}, tag = 'default')
    return self.v_change_vqip(self.empty_vqip(), capacity)

my_model.arcs['nlars_arc'].nlars = nlars
my_model.arcs['nlars_arc'].ltoa = ltoa
my_model.arcs['nlars_arc'].nlars_month_tracker = 0
my_model.arcs['nlars_arc'].nlars_reset = 21
my_model.arcs['nlars_arc'].get_excess = lambda **x : eval_ltoa(my_model.arcs['nlars_arc'], **x)

# customise the push_set_handler for the kinggeorgev reservoir
my_model.nodes['815-kinggeorgev-lake'].tank.capacity = my_model.nodes['815-kinggeorgev-lake'].depth * my_model.nodes['815-kinggeorgev-lake'].area
my_model.nodes['815-kinggeorgev-lake'].push_set_handler['default'] = my_model.nodes['815-kinggeorgev-lake'].push_set_storage
my_model.nodes['815-kinggeorgev-lake'].push_check_handler['default'] = my_model.nodes['815-kinggeorgev-lake'].tank.get_excess
# river damp
my_model.nodes['815-river'].damp = 1#2

#for 1217
my_model.arcs['london_deepham_stw-1217-foul-to-1217-river'].preference = 100 / my_model.nodes['london_deepham_stw-wwtw'].stormwater_tank.capacity
my_model.arcs['london_deepham_stw-1217-foul-to-london_deepham_stw-wwtw'].preference =1 / my_model.nodes['1217-river'].tank.capacity # foul to river/wwtw = 50/50
my_model.arcs['london_deepham_stw-1217-foul-to-1217-storm'].preference = 1e-10

#for 1219
my_model.arcs['london_deepham_stw-1219-foul-to-1219-river'].preference = 10 / my_model.nodes['london_deepham_stw-wwtw'].stormwater_tank.capacity
my_model.arcs['london_deepham_stw-1219-foul-to-london_deepham_stw-wwtw'].preference =1 / my_model.nodes['1219-river'].tank.capacity # foul to river/wwtw = 50/50
my_model.arcs['london_deepham_stw-1219-foul-to-1219-storm'].preference = 1e-10                                                               
#my_model.nodes['hertford_stw-1700-demand'].per_capita = 0.2

# for 1221
my_model.arcs['london_deepham_stw-1221-foul-to-1221-river'].preference = 1 / my_model.nodes['london_deepham_stw-wwtw'].stormwater_tank.capacity
my_model.arcs['london_deepham_stw-1221-foul-to-london_deepham_stw-wwtw'].preference = 4 / my_model.nodes['1221-river'].tank.capacity # foul to river/wwtw = 50/50
my_model.arcs['london_deepham_stw-1221-foul-to-1221-storm'].preference = 1e-10
# for 1220
my_model.arcs['london_deepham_stw-1220-foul-to-1220-river'].preference =  4 / my_model.nodes['1220-river'].tank.capacity
my_model.arcs['london_deepham_stw-1220-foul-to-london_deepham_stw-wwtw'].preference = 1 / my_model.nodes['london_deepham_stw-wwtw'].stormwater_tank.capacity
my_model.arcs['london_deepham_stw-1220-foul-to-1220-storm'].preference = 1e-10

# for 1577
my_model.arcs['london_deepham_stw-1577-foul-to-london_deepham_stw-wwtw'].preference = 1 / my_model.nodes['london_deepham_stw-wwtw'].stormwater_tank.capacity
my_model.arcs['london_deepham_stw-1577-foul-to-1577-river'].preference = 4 / my_model.nodes['1577-river'].tank.capacity # foul to river/wwtw = 50/50
my_model.arcs['london_deepham_stw-1577-foul-to-1577-storm'].preference = 1e-10
# for 1578
my_model.arcs['london_deepham_stw-1578-foul-to-london_deepham_stw-wwtw'].preference = 1e-10
my_model.arcs['london_deepham_stw-1578-foul-to-1578-river'].preference = 1e10
my_model.arcs['london_deepham_stw-1578-foul-to-1578-storm'].preference = 1e-10
# for 1676
my_model.arcs['hertford_stw-1676-foul-to-hertford_stw-wwtw'].preference = 1 / my_model.nodes['hertford_stw-wwtw'].stormwater_tank.capacity
my_model.arcs['hertford_stw-1676-foul-to-1676-river'].preference = 50 / my_model.nodes['1676-river'].tank.capacity # foul to river/wwtw = 50/50
my_model.arcs['hertford_stw-1676-foul-to-1676-storm'].preference = 1e-10
my_model.arcs['london_deepham_stw-1676-foul-to-london_deepham_stw-wwtw'].preference = 1 / my_model.nodes['london_deepham_stw-wwtw'].stormwater_tank.capacity
my_model.arcs['london_deepham_stw-1676-foul-to-1676-river'].preference = 40 / my_model.nodes['1676-river'].tank.capacity # foul to river/wwtw = 50/50
my_model.arcs['london_deepham_stw-1676-foul-to-1676-storm'].preference = 1e-10
#for 1697
my_model.arcs['little_hallingbury_stw-1697-foul-to-little_hallingbury_stw-wwtw'].preference=1/ my_model.nodes['little_hallingbury_stw-wwtw'].stormwater_tank.capacity
my_model.arcs['little_hallingbury_stw-1697-foul-to-1697-river'].preference=40 / my_model.nodes['1697-river'].tank.capacity
my_model.arcs['little_hallingbury_stw-1697-foul-to-1697-storm'].preference=1e-10
# for 1694
my_model.arcs['hertford_stw-1694-foul-to-hertford_stw-wwtw'].preference = 1 / my_model.nodes['london_deepham_stw-wwtw'].stormwater_tank.capacity
my_model.arcs['hertford_stw-1694-foul-to-1694-river'].preference = 4 / my_model.nodes['1694-river'].tank.capacity # foul to river/wwtw = 50/50
my_model.arcs['hertford_stw-1694-foul-to-1694-storm'].preference = 1e-10
#for 1695
my_model.arcs['hertford_stw-1695-foul-to-hertford_stw-wwtw'].preference = 1 / my_model.nodes['hertford_stw-wwtw'].stormwater_tank.capacity
my_model.arcs['hertford_stw-1695-foul-to-1695-river'].preference = 50 / my_model.nodes['1695-river'].tank.capacity # foul to river/wwtw = 50/50
my_model.arcs['hertford_stw-1695-foul-to-1695-storm'].preference = 1e-10



#for 1892
my_model.arcs['widford_widford_herts_stw-1892-foul-to-widford_widford_herts_stw-wwtw'].preference = 1 / my_model.nodes['widford_widford_herts_stw-wwtw'].stormwater_tank.capacity
my_model.arcs['widford_widford_herts_stw-1892-foul-to-1892-river'].preference= 10 / my_model.nodes['1892-river'].tank.capacity
my_model.arcs['widford_widford_herts_stw-1892-foul-to-1892-storm'].preferenc =1e-10
my_model.arcs['bishops_stortford_bishops_sto_stw-1892-foul-to-bishops_stortford_bishops_sto_stw-wwtw'].preference = 1 / my_model.nodes['bishops_stortford_bishops_sto_stw-wwtw'].stormwater_tank.capacity
my_model.arcs['bishops_stortford_bishops_sto_stw-1892-foul-to-1892-river'].preference=10 / my_model.nodes['1892-river'].tank.capacity
my_model.arcs['bishops_stortford_bishops_sto_stw-1892-foul-to-1892-storm'].preference=1e-10
my_model.arcs['furneux_pelham_stw-1892-foul-to-furneux_pelham_stw-wwtw'].preference = 1 / my_model.nodes['furneux_pelham_stw-wwtw'].stormwater_tank.capacity
my_model.arcs['furneux_pelham_stw-1892-foul-to-1892-river'].preference= 10 / my_model.nodes['1892-river'].tank.capacity
my_model.arcs['furneux_pelham_stw-1892-foul-to-1892-storm'].preference=1e-10

#for 1893
my_model.arcs['hertford_stw-1893-foul-to-hertford_stw-wwtw'].preference = 1 / my_model.nodes['hertford_stw-wwtw'].stormwater_tank.capacity
my_model.arcs['hertford_stw-1893-foul-to-1893-river'].preference = 0.35 / my_model.nodes['1893-river'].tank.capacity # foul to river/wwtw = 50/50
my_model.arcs['hertford_stw-1893-foul-to-1893-storm'].preference = 1e-10

#for 1894
my_model.arcs['braughing_stw-1894-foul-to-1894-river'].preference = 1e10
my_model.arcs['barkway_stw-1894-foul-to-1894-river'].preference = 1e10
#for 1898
my_model.arcs['london_deepham_stw-1898-foul-to-london_deepham_stw-wwtw'].preference = 1 / my_model.nodes['london_deepham_stw-wwtw'].stormwater_tank.capacity
my_model.arcs['london_deepham_stw-1898-foul-to-1898-river'].preference = 1 / my_model.nodes['1898-river'].tank.capacity # foul to river/wwtw = 50/50
my_model.arcs['london_deepham_stw-1898-foul-to-1898-storm'].preference = 1e-10

# for 1684
my_model.arcs['hertford_stw-1684-foul-to-1684-river'].preference = 1e10
my_model.arcs['hertford_stw-1684-foul-to-hertford_stw-wwtw'].preference = 1e-10
my_model.arcs['hertford_stw-1684-foul-to-1684-storm'].preference = 1e-10
# for 1823
my_model.arcs['luton_stw-1823-foul-to-1823-river'].preference = 1e-10
my_model.arcs['chalton_stw-1823-foul-to-1823-river'].preference = 1e-10
my_model.arcs['1823-storm-to-1823-river'].preference *= 50


def adjust_atmospheric_deposition(surface, ratio = 0.15):
    def atmospheric_deposition():
        """Inflow function to cause dry atmospheric deposition to occur, updating the 
        surface tank

        Returns:
            (tuple): A tuple containing a VQIP amount for model inputs and outputs 
                for mass balance checking. 
        """
        #TODO double check units in preprocessing - is weight of N or weight of NHX/noy?

        #Read data and scale
        nhx = surface.get_data_input_surface('nhx-dry') * surface.area * ratio
        noy = surface.get_data_input_surface('noy-dry') * surface.area * ratio
        srp = surface.get_data_input_surface('srp-dry') * surface.area * ratio

        #Assign pollutants
        vqip = surface.empty_vqip()
        vqip['ammonia'] = nhx
        vqip['nitrate'] = noy
        vqip['phosphate'] = srp

        #Update tank
        in_ = surface.dry_deposition_to_tank(vqip)

        #Return mass balance
        return (in_, surface.empty_vqip())
    return atmospheric_deposition

 

def adjust_precipitation_deposition(surface, ratio = 0.15):
    def precipitation_deposition():
        """Inflow function to cause wet precipitation deposition to occur, updating 
        the surface tank

 

        Returns:
            (tuple): A tuple containing a VQIP amount for model inputs and outputs 
                for mass balance checking. 
        """
        #TODO double check units - is weight of N or weight of NHX/noy?

 

        #Read data and scale
        nhx = surface.get_data_input_surface('nhx-wet') * surface.area * ratio
        noy = surface.get_data_input_surface('noy-wet') * surface.area * ratio
        srp = surface.get_data_input_surface('srp-wet') * surface.area * ratio

        #Assign pollutants
        vqip = surface.empty_vqip()
        vqip['ammonia'] = nhx
        vqip['nitrate'] = noy
        vqip['phosphate'] = srp

        #Update tank
        in_ = surface.wet_deposition_to_tank(vqip)

        #Return mass balance
        return (in_, surface.empty_vqip())
    return precipitation_deposition


# %%
# adjust wq parameters
# read calibration parameters
wq_parameters = pd.read_csv('wq_parameters.csv').set_index('wfdid')
wq_parameters.index = wq_parameters.index.astype(str)

for wfdid in wq_parameters.index:
    # surface/nutrient_pool parameters
    surfaces = my_model.nodes[wfdid+'-land'].surfaces
    for surface in surfaces:
        if surface.surface not in ['Impervious', 'Garden']:
            surface.denpar = wq_parameters.loc[wfdid, 'denpar']
            surface.nutrient_pool.minfpar = {'N': wq_parameters.loc[wfdid, 'minfpar_N'],
                                              'P': wq_parameters.loc[wfdid, 'minfpar_P']}
            surface.nutrient_pool.immobdpar = {'N': wq_parameters.loc[wfdid, 'immobdpar_N'],
                                              'P': wq_parameters.loc[wfdid, 'immobdpar_P']}    
            if wfdid in ['1676', '1694', '1700']:
                surface.kfr = 0.1537
            elif wfdid in ['1697']:
                surface.kfr = 1.537
        elif surface.surface == "Impervious":
            surface.atmospheric_deposition = adjust_atmospheric_deposition(surface)
            surface.precipitation_deposition = adjust_precipitation_deposition(surface)
            surface.inflows[0] = surface.atmospheric_deposition
            surface.inflows[1] = surface.precipitation_deposition
    # river parameters
    river = my_model.nodes[wfdid+'-river']
    river.muptNpar = wq_parameters.loc[wfdid, 'muptNpar']
    river.denpar_w = wq_parameters.loc[wfdid, 'denpar_w']
    river.muptPpar = wq_parameters.loc[wfdid, 'muptPpar']
# #gw 
# np1 = {'nitrate': 8.6,
#        'phsophate': 0.1}
# np2 = {'nitrate': 0.6,
#         'phsophate': 0.25}
# gws_conc = {'1704': np1,
#             '1818': np2,
#             '1577': np2,
#             '1221': np2,
#             '1819': np1,
#             '1576': np2,
#             '1681': np2,
#             '1894': np1,
#             '1896': np2,
#             '1821': np1,
#             '1219': np2,
#             '1220': np2,
#             '1676': np2,
#             '1686': np1,
#             '1684': np2,
#             '1893': np2,
#             '1694': np2,
#             '1700': np2,
#             '1683': np2,
#             '1578': np2,
#             '1697': np2,
#             '1823': np2,
#             '1892': np1
#             }
# for wfdid, concs in gws_conc.items():
#     gw = wfdid+'-gw'
#     for pol, conc in concs.items():
#         my_model.nodes[gw].tank.storage[pol] = conc * my_model.nodes[gw].tank.storage['volume'] / constants.KG_M3_TO_MG_L # [kg]

# urban
# discharge preference
# my_model.arcs['bishops_stortford_bishops_sto_stw-1892-foul-to-1892-river'].preference = 1e-11
# my_model.arcs['furneux_pelham_stw-1892-foul-to-1892-river'].preference = 1e-11
# my_model.arcs['widford_widford_herts_stw-1892-foul-to-1892-river'].preference = 1e-11
# my_model.arcs['bramfield_stw-1818-foul-to-1818-river'].preference = 1e-11
# my_model.arcs['dane_end_stw-1818-foul-to-1818-river'].preference = 1e-11
# my_model.arcs['bishops_stortford_bishops_sto_stw-1892-foul-to-1892-river'].preference = 1e-11
# my_model.arcs['furneux_pelham_stw-1892-foul-to-1892-river'].preference = 1e-11
# my_model.arcs['widford_widford_herts_stw-1892-foul-to-1892-river'].preference = 1e-11
my_model.arcs['hertford_stw-1221-foul-to-1221-river'].preference = 1e10
my_model.arcs['hertford_stw-1681-foul-to-1681-river'].preference = 1e10
my_model.arcs['hertford_stw-1694-foul-to-1694-river'].preference = 1e10
# demand
demands_conc = {'1217': {'nitrate':8},
                '1219': {'phosphate':0.6},
                '1220': {'nitrate':0.01,
                         'phosphate':0.7},
                
                '1221': {'phosphate':1.13},
                '1577': {'nitrate':3,
                         'phosphate':0.9},
                '1578': {'nitrate':3,
                         'phosphate':1.066},
                '1676': {
                         'phosphate':1.2},

                '1684': {'nitrate': 4,
                        'phosphate':0.124},#0.15
                '1694': {'nitrate': 10,
                    'phosphate':0.706},
                '1697': {'phosphate':5.4},
                '1700': {'phosphate':10},#10
                '1704':{'phosphate':4},#4
                '1818':{'phosphate':0.01},
                '1819':{'phosphate':6.6},
                '1821':{'phosphate':0.1},
                '1893':{#'nitrate': 1,
                      'phosphate':3.6},
                '1894':{'phosphate':10.6},
                '1896':{'phosphate':4.2},
                '1898':{'nitrate':0.6, # 2
                        'phosphate':12}
                }
for wfdid, concs in demands_conc.items():
    demands = [node for node in my_model.nodes.keys() if node[-11:] == wfdid + '-demand'] #TODO '815' is not considered
    for pol, conc in concs.items():
        for demand in demands:
            my_model.nodes[demand].pollutant_load[pol] = my_model.nodes[demand].per_capita * \
                                                                conc / constants.KG_M3_TO_MG_L
# wwtw

# '1217'
wwtw = 'london_deepham_stw-wwtw'
my_model.nodes[wwtw].process_parameters['nitrate'] = {'constant': 28, 'exponent': 1.05} #constant 28
my_model.nodes[wwtw].process_parameters['phosphate'] = {'constant': 0.6, 'exponent': 1.001} 

#wwtw = 'hertford_stw-wwtw'
#my_model.nodes[wwtw].process_parameters['nitrate'] = {'constant': 5, 'exponent': 1.001} 
#'1695':
wwtw = 'standon_standon_herts_stw-wwtw'
my_model.nodes[wwtw].process_parameters['phosphate'] = {'constant': 1.2, 'exponent': 1.001} 
#'1821'
wwtw = 'standon_standon_herts_stw-wwtw'
my_model.nodes[wwtw].process_parameters['phosphate'] = {'constant': 0.8, 'exponent': 1.001} 
#'1686'
wwtw ='stansted_mountfitchet_stansted_stw-wwtw'
my_model.nodes[wwtw].process_parameters['phosphate'] = {'constant': 0.1, 'exponent': 1.001}
# for '1695'
wwtw = 'luton_stw-wwtw'
my_model.nodes[wwtw].process_parameters['phosphate'] = {'constant': 1.5, 'exponent': 1.01} # 1.4 to 1.5 1.001 to1.01

wwtw = 'hertford_stw-wwtw'
my_model.nodes[wwtw].process_parameters['phosphate'] = {'constant': 2, 'exponent': 1.001} #升到1.6
#for 1896
wwtw = 'buntingford_aspenden_lane_bun_stw-wwtw'
my_model.nodes[wwtw].process_parameters['phosphate'] = {'constant': 1.25, 'exponent': 1.001} 
my_model.nodes[wwtw].process_parameters['nitrate'] = {'constant':18 , 'exponent': 1.001}   

#for1704
wwtw = 'widford_widford_herts_stw-wwtw'
my_model.nodes[wwtw].process_parameters['phosphate'] = {'constant': 3, 'exponent': 1.001} 

def wrapper(f,node,variable, value, date):
    def new_end_timestep():
        f()
        if str(node.t) == date:
            node.process_parameters[variable]['constant'] = value
    return new_end_timestep
for wwtw, new_constants, variable, date in zip(['london_deepham_stw-wwtw','luton_stw-wwtw','hertford_stw-wwtw'],
                                         [0.01,0.01,0.2],
                                         ['phosphate','phosphate','phosphate'],
                                         ['2012-03-01','2005-01-01','2012-03-01']):
    node = my_model.nodes[wwtw]
    node.end_timestep = wrapper(node.end_timestep, node, variable, new_constants, date)
    
abstraction_arcs = ['1683-gw-to-1683-gw-demand',
                   '1823-gw-to-1823-gw-demand',
                   '1701-gw-to-1701-gw-demand',
                   '1685-gw-to-1685-gw-demand',
                   '1684-gw-to-1684-gw-demand',
                   "1818-gw-to-1818-gw-demand",
                   '1893-gw-to-1893-gw-demand',
                   '1896-gw-to-1896-gw-demand',
                   '1821-gw-to-1821-gw-demand',
                   '1894-gw-to-1894-gw-demand',
                   '1704-gw-to-1704-gw-demand',
                   '1702-gw-to-1702-gw-demand',
                   '1703-gw-to-1703-gw-demand',
                   '1686-gw-to-1686-gw-demand',
                   '1820-gw-to-1820-gw-demand',
                   '1695-gw-to-1695-gw-demand',
                   '1219-gw-to-1219-gw-demand',
                   '1897-gw-to-1897-gw-demand',
                   '1676-gw-to-1676-gw-demand',
                   ]

Sub_abs={'1683':'1683-gw-to-1683-gw-demand',
         "1823":'1823-gw-to-1823-gw-demand',
         "1701":'1701-gw-to-1701-gw-demand',
         "1685":'1685-gw-to-1685-gw-demand',
         "1684":'1684-gw-to-1684-gw-demand',
         "1818":"1818-gw-to-1818-gw-demand",
         "1893":'1893-gw-to-1893-gw-demand',
         "1896":'1896-gw-to-1896-gw-demand',
         "1821":'1821-gw-to-1821-gw-demand',
         "1894":'1894-gw-to-1894-gw-demand',
         "1704":'1704-gw-to-1704-gw-demand',
         "1702":'1702-gw-to-1702-gw-demand',
         "1703":'1703-gw-to-1703-gw-demand',
         "1686":'1686-gw-to-1686-gw-demand',
         "1820":'1820-gw-to-1820-gw-demand',
         "1695":'1695-gw-to-1695-gw-demand',
         "1219":'1219-gw-to-1219-gw-demand',
         "1897":'1897-gw-to-1897-gw-demand',
         "1676":'1676-gw-to-1676-gw-demand'}
def decorate_abstraction_hof(Abs_subs,Sub_Arcs, method):
    def wrapper():
        for sub in Abs_subs:
            if my_model.arcs[Sub_Arcs[sub]].flow_in < dic_Q95[sub]:
                my_model.arcs[Sub_abs[sub]].capacity=my_model.arcs[Sub_abs[sub]].default_capacity*0
            elif dic_Q95[sub]<=my_model.arcs[Sub_Arcs[sub]].flow_in<dic_Q70[sub]:
                my_model.arcs[Sub_abs[sub]].capacity = my_model.arcs[Sub_abs[sub]].default_capacity*0.7 #0.4
            elif dic_Q70[sub]<=my_model.arcs[Sub_Arcs[sub]].flow_in<=dic_Q50[sub]:
                my_model.arcs[Sub_abs[sub]].capacity = my_model.arcs[Sub_abs[sub]].default_capacity*0.9 #0.6
            elif dic_Q50[sub]<my_model.arcs[Sub_Arcs[sub]].flow_in<=dic_Q30[sub]:
                my_model.arcs[Sub_abs[sub]].capacity = my_model.arcs[Sub_abs[sub]].default_capacity*1.1 #1.4
            else:
                my_model.arcs[Sub_abs[sub]].capacity = my_model.arcs[Sub_abs[sub]].default_capacity*1.3 #1.6
        return method()
    return wrapper      
    


for arc in abstraction_arcs:
    my_model.arcs[arc].default_capacity = my_model.arcs[arc].capacity

river_node = my_model.nodes['815-river']
river_node.distribute = decorate_abstraction_hof(Abs_subs,Sub_Arcs,river_node.distribute)


# %%
#Set dates
my_model.dates = dates
constants.FLOAT_ACCURACY = 1E-9


#Run
flows, tanks, _, _, _ = my_model.run(dates = dates)
flows = pd.DataFrame(flows)
tanks = pd.DataFrame(tanks)
flows.time = [str(x) for x in flows.time]
sim_dates = pd.to_datetime(flows.time).dt.date.astype(str).unique()

import pickle
filepath = os.path.join(results_dir, "tanks_DS2.1.pkl")
with open(filepath, 'wb') as fp:
    pickle.dump(tanks, fp)
filepath = os.path.join(results_dir, "flows_DS2.1.pkl")
with open(filepath, 'wb') as fp:
    pickle.dump(flows, fp)
a=[]
for arc in abstraction_arcs:
    flow = flows.groupby('arc').get_group(arc).set_index('time').flow
    abstraction = sum(flow)
    a.append(abstraction)
b = 0
for i in a:
    b=b+i
b
    
