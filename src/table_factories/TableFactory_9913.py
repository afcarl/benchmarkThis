#!/usr/bin/env python
##################################################
#   Last edit 9/13/2013                          #
#   checked by sophie and will                   #
##################################################

# general imports
from numpy import (array, arange, sin, cos, tan, pi, hstack, vstack, corrcoef, 
    e, insert, linspace)
from scipy.stats.distributions import lognorm, uniform, gamma, expon
from numpy.random import seed, shuffle
from scipy.signal import square, sawtooth, gausspulse
from correlations.generators.timeseries import (cube_d5_indices, 
    generate_otu_from_pt_in_R5, subsample_otu_evenly, subsample_otu_zero, 
    subsample_otu_choose, make_pop_growth_func, subsample_otu_random)
from correlations.generators.lokta_volterra import (dX_dt_template, 
    lokta_volterra)
from correlations.generators.null import (model1_otu, model1_table, 
    model2_table, model3_table, alter_table)
from correlations.generators.ga import evolve
from correlations.generators.ecological import *
from biom.table import table_factory
from numpy.random import seed




################ Eco Table #######################
# seed(5000)

# p_zero = .3
# p_ln_dist = 1 - p_zero
# tmp = []
# zero_dist = uniform.rvs(0,1,size=(500,400))
# os = lognorm.rvs(3.0,1.8,size=(500,400))
# for i in range(50):
#     p_zero = .5 + i/100.
#     tmp.append(where(zero_dist[i] < p_zero, 0, os[i]))
# vals = vstack(tmp)
# tmp = []
# for i in range(1,11):
#     base = lognorm.rvs(3.0,1.8,size=(i**3, 400))
#     zi = uniform.rvs(0,1,size=(i**3,400))
#     tmp.append(where(zi < i*e, 0, base))
# vals = vstack(tmp)


# # Amensalism 1D, st5
# strength = .5
# #os = lognorm.rvs(3,1.5,size=(50,400)).astype(int)
# #os = expon.rvs(0, 5,1000,size=(50,400)).astype(int)
# amensally_related_1d_st_5 = []
# for i in range(25):
#     ind_i, ind_j = 2*i, 2*i+1
#     am_otu = amensal_1d(os[ind_i], os[ind_j], strength)
#     amensally_related_1d_st_5.extend([os[ind_i], am_otu])

# # Commenasalism 1D, st5
# strength = .5
# os = lognorm.rvs(3,0,size=(50,400)).astype(int)
# commensually_related_1d_st_5 = []
# for i in range(25):
#     ind_i, ind_j = 2*i, 2*i+1
#     boosted_otu = commensal_1d(os[ind_i], os[ind_j], strength)
#     commensually_related_1d_st_5.extend([os[ind_i], boosted_otu])

# # Mutualism 1D, st5
# strength = .5
# mutually_related_1d_st_5 = []
# os = lognorm.rvs(3,0,size=(50,400)).astype(int)
# for i in range(25): # no base otu to skip
#     ind_i = 2*i
#     ind_j = 2*i + 1
#     # even numbered otus will be related to the next odd otu, i.e. 0 and 1 are 
#     # mutualistic, 2,3 and so forth
#     moi, moj = mutual_1d(os[ind_i], os[ind_j], strength)
#     mutually_related_1d_st_5.extend([moi, moj])

# # Parasatism 1D, st5
# strength = .5
# os = lognorm.rvs(3,0,size=(50,400)).astype(int)
# parasitically_related_1d_st_5 = []
# for i in range(25):
#     ind_i, ind_j = 2*i, 2*i + 1
#     # even numbered otus will be the parasite so 0 eats 1, 2 eats 3 etc.
#     moi, moj = parasite_1d(os[ind_i], os[ind_j], strength)
#     parasitically_related_1d_st_5.extend([moi, moj])

# # Competition 1D, st5
# strength = .5
# os = lognorm.rvs(3,0,size=(50,400)).astype(int)
# competitively_related_1d_st_5 = []
# for i in range(25):
#     ind_i, ind_j = 2*i, 2*i + 1
#     # even numbered otus will be the related to the next odd otu as in o2 
#     # related to o3, o4 with o5. 
#     moi, moj = competition_1d(os[ind_i], os[ind_j], strength)
#     competitively_related_1d_st_5.extend([moi, moj])

# # Obligate syntrophy 1D, st5
# strength = .5
# obligate_related_1d_st_5 = []
# os = lognorm.rvs(3,0,size=(25,400)).astype(int)
# for otu in os:
#     obs_otu = obligate_syntroph_1d(otu, strength)
#     obligate_related_1d_st_5.extend([otu, obs_otu])

# # Partial obligate syntrophy 1D, st5
# os = lognorm.rvs(3,0,size=(50,400)).astype(int)
# partial_obligate_syntrophic_related_1d = []
# for i in range(25):
#     ind_i, ind_j = 2*i, 2*i + 1
#     # even numbered otus will be the independent so 0 allows 1, 2 allows 3 etc.
#     moj = partial_obligate_syntroph_1d(os[ind_i], os[ind_j])
#     partial_obligate_syntrophic_related_1d.extend([os[ind_i], moj])

#1400 unrelated OTUs as noise
# os = lognorm.rvs(3,0,size=(1400,400))
# unrelated_otus = [i for i in os]

# typing as int then float guarantees we get about 50k more 0's than if we 
# just did .round(0)
# otu_data = vstack([amensally_related_1d_st_5,
#     commensually_related_1d_st_5,
#     mutually_related_1d_st_5,
#     parasitically_related_1d_st_5,
#     competitively_related_1d_st_5,
#     obligate_related_1d_st_5,
#     partial_obligate_syntrophic_related_1d,
#     unrelated_otus]).astype(int).astype(float)


# otu_data = vstack([amensally_related_1d_st_5,
#     commensually_related_1d_st_5,
#     mutually_related_1d_st_5,
#     parasitically_related_1d_st_5,
#     competitively_related_1d_st_5,
#     obligate_related_1d_st_5,
#     partial_obligate_syntrophic_related_1d]).astype(int).astype(float)

# def make_ids(data):
#     sids = ['s%i' % i for i in range(data.shape[1])]
#     oids = ['o%i' % i for i in range(data.shape[0])]
#     return sids, oids

# sids, oids = make_ids(otu_data)

# bt = table_factory(otu_data, sids, oids)


################ GA Table #######################

seed(300)
ref_a = uniform.rvs(100,1000,size=50)
ref_b = ref_a + uniform.rvs(0,20,size=50)
ref_gene = array([ref_a,ref_b]).T

igp = [ref_gene[:]+uniform.rvs(100,1000,size=ref_gene.shape) for i in range(500)]
gc, fmeans, fmaxes = evolve(igp, ref_gene, 1000)
tmp_ga_table = vstack([i.T for i in gc])

ga_table = []
for i in range(500):
    if (tmp_ga_table[2*i:2*(i+1)]>0).all():
        ga_table.append(tmp_ga_table[2*i:2*(i+1)])
ga_table = vstack(ga_table)


# this removes all otus (rows) from the table for which the correlation between
# the otu and its generating partner (its gene partner in the GS 
# orthonormalization procedure) is not the highest of all the correlations 
# that otu has. this is an overly aggressive procedure since its likely that 
# removing a few rows would suffice, but this is simple and effective enough. 
inds = []
cvals = corrcoef(ga_table)
for i in range(0, cvals.shape[0]-2, 2):
    if all(cvals[i][i+1]>cvals[i,0:i]) and all(cvals[i][i+1]>cvals[i,i+2:]):
        inds.append(i)

i_inds = array(inds)
j_inds = i_inds+1
all_inds = array([i_inds,j_inds]).flatten()
all_inds.sort() # ensures that each OTU and its generating partner are next to
# one another
ga_table = ga_table.take(all_inds, 0)

###############
# group1 otus #
###############
'''
These otus will be variations of a sin signal with changing frequency, 
phase, and subsampling routine.

These otus will not have values that dip below 0 due to noise effects. 
'''
seed(652323)
freq = [0.25, 0.5, 1, 1.25, 1.5, 2, 3, 25, 50, 100, 200]
amp = [100]
amp_logistic = [10]
phase = [0, .5*pi]
noise = [0.5]
adj = [[subsample_otu_evenly, .26], [subsample_otu_random, .26], [subsample_otu_zero, .26, .26]]
q = cube_d5_indices(freq, amp, phase, noise, adj)
ts_g1_otus_sin = vstack([generate_otu_from_pt_in_R5(q[i], sin, 150) 
    for i in range(65)])
ts_g1_otus_square = vstack([generate_otu_from_pt_in_R5(q[i], square, 150) 
    for i in range(65)])
otus1_1 = vstack([generate_otu_from_pt_in_R5(q[i], square, 150) 
    for i in range(65)])
# second half of signal
otus2_1 = vstack([generate_otu_from_pt_in_R5(q[i], cos, 150) 
    for i in range(65)])
q_logistic = cube_d5_indices(freq, amp_logistic, phase, noise, adj)
ts_g1_otus_logistic = vstack([generate_otu_from_pt_in_R5(q_logistic[i], make_pop_growth_func(10,2, 1), 10) 
    for i in range(65)])
# make one signal of 26 samples
ts_g1_otus_square_cos = hstack([otus1_1, otus2_1])
ts_g1_otus_square_cos = ts_g1_otus_square_cos.take(arange(0,52,2),1)
ts_g1_otus = vstack([ts_g1_otus_sin, ts_g1_otus_square, ts_g1_otus_square_cos, ts_g1_otus_logistic]).astype(int)


###############
# group2 otus #
###############
'''
These otus will be variations of a sin signal with changing frequency, 
phase, and subsampling routine.

These otus will not have values that dip below 0 due to noise effects. 
'''
freq = [0.25, 0.5, 1, 1.25, 1.5, 2, 3, 25, 50, 100, 200]
amp = [100]
amp_logistic = [10]
phase = [0, .5*pi]
noise = [0.5]
adj = [[subsample_otu_evenly, .50], [subsample_otu_random, .50], [subsample_otu_zero, .50, .50]]
q = cube_d5_indices(freq, amp, phase, noise, adj)
ts_g2_otus_sin = vstack([generate_otu_from_pt_in_R5(q[i], sin, 150) 
    for i in range(65)])
ts_g2_otus_square = vstack([generate_otu_from_pt_in_R5(q[i], square, 150) 
    for i in range(65)])
otus1_2 = vstack([generate_otu_from_pt_in_R5(q[i], square, 150) 
    for i in range(65)])
# second half of signal
otus2_2 = vstack([generate_otu_from_pt_in_R5(q[i], cos, 150) 
    for i in range(65)])
q_logistic = cube_d5_indices(freq, amp_logistic, phase, noise, adj)
ts_g2_otus_logistic = vstack([generate_otu_from_pt_in_R5(q_logistic[i], make_pop_growth_func(10,2, 1), 10) 
    for i in range(65)])
# make one signal of 50 samples
ts_g2_otus_square_cos = hstack([otus1_2, otus2_2])
ts_g2_otus_square_cos = ts_g2_otus_square_cos.take(arange(0,100,2),1)
ts_g2_otus = vstack([ts_g2_otus_sin, ts_g2_otus_square, ts_g2_otus_square_cos, ts_g2_otus_logistic]).astype(int)


###############
# group3 otus #
###############
'''
These otus will be variations of a sin signal with changing frequency, 
phase, and subsampling routine.

These otus will not have values that dip below 0 due to noise effects. 
'''
freq = [0.25, 0.5, 1, 1.25, 1.5, 2, 3, 25, 50, 100, 200]
amp = [100]
amp_logistic = [10]
phase = [0, .5*pi]
noise = [0.5]
adj = [[subsample_otu_evenly, .74], [subsample_otu_random, .74], [subsample_otu_zero, .74, .74]]
q = cube_d5_indices(freq, amp, phase, noise, adj)
ts_g3_otus_sin = vstack([generate_otu_from_pt_in_R5(q[i], sin, 150) 
    for i in range(65)])
ts_g3_otus_square = vstack([generate_otu_from_pt_in_R5(q[i], square, 150) 
    for i in range(65)])
otus1_3 = vstack([generate_otu_from_pt_in_R5(q[i], square, 150) 
    for i in range(65)])
# second half of signal
otus2_3 = vstack([generate_otu_from_pt_in_R5(q[i], cos, 150) 
    for i in range(65)])
q_logistic = cube_d5_indices(freq, amp_logistic, phase, noise, adj)
ts_g3_otus_logistic = vstack([generate_otu_from_pt_in_R5(q_logistic[i], make_pop_growth_func(10,2, 1), 10) 
    for i in range(65)])
# make one signal of 74 samples
ts_g3_otus_square_cos = hstack([otus1_3, otus2_3])
ts_g3_otus_square_cos = ts_g3_otus_square_cos.take(arange(0,148,2),1)
ts_g3_otus = vstack([ts_g3_otus_sin, ts_g3_otus_square, ts_g3_otus_square_cos, ts_g3_otus_logistic]).astype(int)


###############
# group4 otus #
###############
'''
Just half sampling of group1 otus.  
'''
ts_g4_otus = ts_g1_otus.take(arange(0,26,2),1).astype(int)


###############
# group5 otus #
###############
'''
Just half sampling of group2 otus.  
'''
ts_g5_otus = ts_g2_otus.take(arange(0,50,2),1).astype(int)


###############
# group6 otus #
###############
'''
Just half sampling of group3 otus.  
'''
ts_g6_otus = ts_g3_otus.take(arange(0,74,2),1).astype(int)


###############
# group7 otus #
###############
'''
These otus will include a pulse for some of the samples.  The pulses are 
all shifted by one sample, and also includes their envelope.  Here, the
center frequency is 1Hz
'''
#make pulse of 50 points, with amplitude 200
t = linspace(-3, 3, 1*50, endpoint=False)   
real, envelope = gausspulse(t, fc=1, retenv=True)
real = 200*(real+1)
envelope = 200*(envelope+1)
# plt.plot(t, i, t, e, '--')
# plt.show()

samples = linspace(0, 199, 200)
otu_real =[[0 for a in range(200)] for k in range(200)]
otu_envelope =[[0 for a in range(200)] for k in range(200)]

#insert pulse into 150 point vector of average amplitude
#at different indices for total length of 200
for j in samples:
    z = [200]*150
    j = int(j)
    for x in real:
        z.insert(j,x)
    otu_real[j]=z
    z = [200]*150
    for y in envelope:
        z.insert(j,y)
    otu_envelope[j]=z
otus_real = array(otu_real)
otus_envelope = array(otu_envelope)
otus_gausspulse_fc1 = vstack([otus_real, otus_envelope]).astype(int)

# plt.plot(samples,otus_gausspulse_fc1[0], samples, otus_gausspulse_fc1[200]) 
# plt.show()


###############
# group8 otus #
###############
'''
same as group 7 with center frequency of 10Hz
'''
#make pulse of 50 points, with amplitude 200
t = linspace(-3, 3, 1*50, endpoint=False)   
real, envelope = gausspulse(t, fc=10, retenv=True)
real = 200*(real+1)
envelope = 200*(envelope+1)

samples = linspace(0, 199, 200)
otu_real =[[0 for a in range(200)] for k in range(200)]
otu_envelope =[[0 for a in range(200)] for k in range(200)]

#insert pulse into 150 point vector of average amplitude
#at different indices for total length of 200
for j in samples:
    z = [200]*150
    j = int(j)
    for x in real:
        z.insert(j,x)
    otu_real[j]=z
    z = [200]*150
    for y in envelope:
        z.insert(j,y)
    otu_envelope[j]=z
otus_real = array(otu_real)
otus_envelope = array(otu_envelope)
otus_gausspulse_fc10 = vstack([otus_real, otus_envelope]).astype(int)


###############
# group9 otus #
###############
'''
same as group 7 with center frequency of 0.1Hz
'''
#make pulse of 50 points, with amplitude 200
t = linspace(-3, 3, 1*50, endpoint=False)   
real, envelope = gausspulse(t, fc=.1, retenv=True)
real = 200*(real+1)
envelope = 200*(envelope+1)

samples = linspace(0, 199, 200)
otu_real =[[0 for a in range(200)] for k in range(200)]
otu_envelope =[[0 for a in range(200)] for k in range(200)]

#insert pulse into 150 point vector of average amplitude
#at different indices for total length of 200
for j in samples:
    z = [200]*150
    j = int(j)
    for x in real:
        z.insert(j,x)
    otu_real[j]=z
    z = [200]*150
    for y in envelope:
        z.insert(j,y)
    otu_envelope[j]=z
otus_real = array(otu_real)
otus_envelope = array(otu_envelope)
otus_gausspulse_fcpt1 = vstack([otus_real, otus_envelope]).astype(int)


all_tables = [ts_g1_otus, ts_g2_otus, ts_g3_otus, ts_g4_otus, ts_g5_otus, 
    ts_g6_otus, otus_gausspulse_fc1, otus_gausspulse_fc10, 
    otus_gausspulse_fcpt1, ga_table]


from biom.table import table_factory

names = ['table_34.biom',
 'table_35.biom',
 'table_36.biom',
 'table_37.biom',
 'table_38.biom',
 'table_39.biom',
 'table_40.biom',
 'table_41.biom',
 'table_42.biom',
 'table_43.biom']

def make_ids(data):
    sids = ['s%i' % i for i in range(data.shape[1])]
    oids = ['o%i' % i for i in range(data.shape[0])]
    return sids, oids

for table, name in zip(all_tables,names):
    sids, oids = make_ids(table)
    bt = table_factory(table, sids, oids)
    json_str = bt.getBiomFormatJsonString(generated_by='Sophie_Will')
    o = open('/Users/wdwvt1/Desktop/'+name, 'w')
    o.write(json_str)
    o.close()

# description of tables 0-33
# the base table for tables 0-33 was derived from sequences provided by 
# Ridaura et al. in "Cultered gut bacterial consortia from twins discordant 
# for obesity modulate adiposity and metabolic phenotypes in gnotobiotic mice". 
# for tables 0-19, we clustered the sequences at 97 percent and picked closed 
# ref against green genes 13_5. rarifying the table at 1000 seqs/sample, (and 
# doing so 10 times independenlty) produced tables 0-9. rarifying at 2000 seqs/
# sample produced tables 12-19. 
# tables 20-23 were created by taking the first rarified table (table_0.biom) 
# and filtering OTUs that did not occur in some percentage of the samples. 
# table_20.biom filtered out otus that were found in 5 percent of the samples or 
# less, 21 filtered out otus found in 10 percent of the samples or less, 22 in 
# 20 percent or less, and 23 in 50 percent or less. 
# tables 24-26 were created by filtering the raw 97 percent closed picked table 
# to remove otus whose overall sequence count in that table was below a certain 
# percentage (and then rarifaction at 1000 seqs/sample). table_24.biom removed all 
# otus for which the fraction of all sequences in the table that belonged to that 
# OTU was less than .00005, 25 less than .00010, 26 less than .000025. 
# table_27.biom was created by taking table 24 and adding the additional step of 
# removing otus found in less than 20 percent of the samples. 
# table_28.biom was created by summarizing the whole table at L6, then rarifying 
# at 1000, then removing otus not found in at least 20 percent of samples. 
# table_29.biom was created by picking the original sequences at 94 percent 
# against gg_13_5, rarifying to 1000, and removing otus found in less than 20 
# samples. 
# table 30 was the same as table 28, but collapsed at L5. table 31 was the same as
# 29, but at 91 percent. 
# table 32 was the same as table 28, but at L4. table 33 was the same as table 29 
# but at 88 percent. 


out samples found in 10 