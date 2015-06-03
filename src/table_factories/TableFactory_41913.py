##################################################
#   Last edit 4/19/2013                          #
#   checked by sophie and will                   #
##################################################

# general imports
from numpy import array, arange, sin, cos, tan, pi, hstack, vstack
from scipy.stats.distributions import lognorm, uniform, gamma
from numpy.random import seed, shuffle
from scipy.signal import square, sawtooth
from generators.timeseries import (cube_d5_indices, generate_otu_from_pt_in_R5,
    subsample_otu_evenly, subsample_otu_zero, subsample_otu_choose)
from generators.lokta_volterra import (dX_dt_template, lokta_volterra)
from generators.null import (model1_otu, model1_table, model2_table, 
    model3_table, alter_table)
from generators.ecological import *


################################################################################
#                                 Timeseries                                   #
################################################################################
# this is failing with
# seed(908292934381041909001101)
# /Users/wdwvt1/src/numpy/build/lib.macosx-10.6-intel-2.7/numpy/random/mtrand.so in mtrand.RandomState.seed (numpy/random/mtrand/mtrand.c:5270)()

# ValueError: object of too small depth for desired array
# why would this not work on here but would work on macosx 10.6.5?


seed(100230249)

###############
# group1 otus #
###############
'''
These otus will be variations of a sin signal with changing frequency, 
amplitude, phase, noise, and subsampling routine.

These otus will not have values that dip below 0 due to noise effects. 
'''
freq = [1, 2, 3]
amp = [100, 50, 25]
phase = [0, .25*pi, .5*pi]
noise = [0, .25, .5]
adj = [[subsample_otu_evenly, .5], [subsample_otu_zero, .5, .3], 
    [subsample_otu_zero, .5, .75]]
q = cube_d5_indices(freq, amp, phase, noise, adj)
ts_g1_otus = vstack([generate_otu_from_pt_in_R5(q[i], sin, 150) 
    for i in range(243)])


###############
# group2 otus #
###############
'''
These otus will be variations a square wave for half of the samples, then a
cos wave for half the samples. 
'''
freq = [1, 2, 3]
amp = [100, 50, 25]
phase = [0, .25*pi, .5*pi]
noise = [0, .25, .5]
adj = [[subsample_otu_evenly, .5], [subsample_otu_zero, .5, .3], 
    [subsample_otu_zero, .5, .75]]
q = cube_d5_indices(freq, amp, phase, noise, adj)
# first half of signal
otus1 = vstack([generate_otu_from_pt_in_R5(q[i], square, 150) 
    for i in range(243)])
# second half of signal
otus2 = vstack([generate_otu_from_pt_in_R5(q[i], cos, 150) 
    for i in range(243)])
# make one signal of 100 samples
ts_g2_otus = hstack([otus1, otus2])


###############
# group3 otus #
###############
'''
Just half sampling of group2 otus.  
'''
ts_g3_otus = ts_g2_otus.take(arange(0,100,2),1)


###############
# group4 otus #
###############
'''
These otus will be summations of a sawtooth and cos wave. 
'''
freq = [1, 2, 3]
amp = [100, 50, 25]
phase = [0, .25*pi, .5*pi]
noise = [0, .25, .5]
adj = [[subsample_otu_evenly, .5], [subsample_otu_zero, .5, .3], 
    [subsample_otu_zero, .5, .75]]
q = cube_d5_indices(freq, amp, phase, noise, adj)
# first half of signal
otus1 = vstack([generate_otu_from_pt_in_R5(q[i], sawtooth, 150) 
    for i in range(243)])
# second half of signal
otus2 = vstack([generate_otu_from_pt_in_R5(q[i], cos, 150) 
    for i in range(243)])
# make one signal of 100 samples
ts_g4_otus = otus1+otus2


###############
# group5 otus #
###############
'''
Significantly undersampled wave + low frequency wave.
'''
# carrier wave
freq = [1, 2, 3]
amp = [100, 50, 25]
phase = [0, .25*pi, .5*pi]
noise = [0, .25, .5]
adj = [[subsample_otu_evenly, .5], [subsample_otu_zero, .5, .3], 
    [subsample_otu_zero, .5, .75]]
q = cube_d5_indices(freq, amp, phase, noise, adj)
otus1 = vstack([generate_otu_from_pt_in_R5(q[i], sawtooth, 300) 
    for i in range(243)])

# undersampled wave
freq = [20, 20, 20]
amp = [100, 50, 25]
phase = [0, .25*pi, .5*pi]
noise = [0, .25, .5]
adj = [[subsample_otu_evenly, .5], [subsample_otu_zero, .5, .3], 
    [subsample_otu_zero, .5, .75]]
q = cube_d5_indices(freq, amp, phase, noise, adj)
otus2 = vstack([generate_otu_from_pt_in_R5(q[i], cos, 0) 
    for i in range(243)])
# make one signal of 100 samples
ts_g5_otus = otus1+otus2


################################################################################
#                                Lotka-Volterra                                #
################################################################################

seed(30968340693)

two_species_interaction_matrices = [\
array([[1., 0, -.1],
[-1.5, 0.075, 0]]),
array([[1., 0, -.1],
[-0.5, 0.075, 0]]),
array([[1, 0, -.1],
[-10, 0.075, 0]]),
array([[10, 0, -.1],
[-10, 0.075, 0]]),
array([[-1, 0, -.1],
[-1.5, 0.075, 0]]),
array([[-0.1, 0, -.1],
[-1.5, 0.075, 0]]),
array([[0, 0, -.1],
[-1.5, 0.075, 0]]),
array([[0, 0, -.1],
[0, 0.075, 0]]),
array([[1., 0, -1],
[-1.5, 1, 0]]),
array([[4, -2, -2],
[1, 1, -2]])]

six_species_interaction_matrices = [\
array([[2, 0, -2, 0, 0, 0, 0],
[1, 2, -1, -2, 0, 0, 0],
[4, 0, 2, 0, -2, -2, -2],
[-1, 0, 0, 2, 0, -1, 0],
[-3, 0, 0, 2, 1, 0, 0],
[-1, 0, 0, 2, 0, 0, -1]]),
array([[2,-10,-2,0,0,0,0],
[1,10,-1,-2,0,0,0],
[4,10,2,0,-2,-2,-2],
[-1,-10,0,2,0,-1,0],
[-3,-10,0,2,1,0,0],
[-10,-10,0,2,0,0,-1]]),
array([[10,-10,-2,0,0,0,0],
[1,10,-1,-2,0,0,0],
[4,10,2,0,-2,-2,-2],
[-1,-10,0,2,0,-1,0],
[-3,-10,0,2,1,0,0],
[-1,-10,0,2,0,0,-1]])]


# two_species_otu_table
two_species_otu_table = []
for C in two_species_interaction_matrices:
    f = dX_dt_template(C)
    Y = lokta_volterra(f, array([10]*C.shape[0]), 0, 20, 10000)
    two_species_otu_table.extend([Y[0],Y[1]])

# shape is 20 x 10000
two_species_otu_table = vstack(two_species_otu_table)

# make 5 tables:
# 1. relative abundance table with pts taken at equal intervals
# 2. counts table with pts taken at equal intervals (same pts as 1)
# 3. relative abundance table with pts taken at random indices
# 4. counts table with pts taken at random intervals (same pts as 3)
# 5. table with 60 percent sparsity 

# must add 80 completely random OTUs to confound + pad the table

# generate 40 otus from lognorm distribution 2,0,1 
dfs_and_params = [[lognorm, 2, 0]]*40
otus_lognorm_2_0 = model1_table(dfs_and_params, 50)
# generate 40 otus from gamma distribution 1,0,10
dfs_and_params = [[gamma, 1, 0, 10]]*40
otus_gamma_1_0_10 = model1_table(dfs_and_params, 50)

evenly_sampled_indices = arange(50)*200
random_indices = arange(10000)
shuffle(random_indices)
random_indices = random_indices[:50]

lv_table_2sp_12_base = vstack([two_species_otu_table[:,evenly_sampled_indices], 
    otus_gamma_1_0_10, otus_lognorm_2_0])
lv_table_2sp_1 = lv_table_2sp_12_base/lv_table_2sp_12_base.sum(0)
lv_table_2sp_2 = 100*lv_table_2sp_12_base.round(0)
lv_table_2sp_34_base = vstack([two_species_otu_table[:,random_indices], 
    otus_gamma_1_0_10, otus_lognorm_2_0])
lv_table_2sp_3 = lv_table_2sp_34_base/lv_table_2sp_34_base.sum(0)
lv_table_2sp_4 = 100*lv_table_2sp_34_base.round(0)
lv_table_2sp_5 = alter_table(lv_table_2sp_12_base, sparsity=.6).round(0)


# six_species_otu_tables
six_species_otu_tables = []
for C in six_species_interaction_matrices:
    f = dX_dt_template(C)
    Y = lokta_volterra(f, array([10]*C.shape[0]), 0, 20, 10000)
    six_species_otu_tables.append(Y)

six_species_otu_table = vstack(six_species_otu_tables)

# make 5 tables:
# 1. relative abundance table with pts taken at equal intervals
# 2. counts table with pts taken at equal intervals (same pts as 1)
# 3. relative abundance table with pts taken at random indices
# 4. counts table with pts taken at random intervals (same pts as 3)
# 5. table with 60 percent sparsity 

# must add 82 completely random OTUs to confound + pad the table

# generate 42 otus from lognorm distribution 1,0,1 
dfs_and_params = [[lognorm, 1, 0]]*42
otus_lognorm_1_0 = model1_table(dfs_and_params, 50)
# generate 42 otus from gamma distribution 1,0,10
dfs_and_params = [[gamma, 1, 0, 10]]*42
otus_gamma_1_0_10 = model1_table(dfs_and_params, 50)

evenly_sampled_indices = arange(50)*200
random_indices = arange(10000)
shuffle(random_indices)
random_indices = random_indices[:50]

lv_table_6sp_12_base = vstack([six_species_otu_table[:,evenly_sampled_indices], 
    otus_gamma_1_0_10, otus_lognorm_1_0])
lv_table_6sp_1 = lv_table_6sp_12_base/lv_table_6sp_12_base.sum(0)
lv_table_6sp_2 = 100*lv_table_6sp_12_base.round(0)
lv_table_6sp_34_base = vstack([six_species_otu_table[:,random_indices], 
    otus_gamma_1_0_10, otus_lognorm_1_0])
lv_table_6sp_3 = lv_table_6sp_34_base/lv_table_6sp_34_base.sum(0)
lv_table_6sp_4 = 100*lv_table_6sp_34_base.round(0)
lv_table_6sp_5 = alter_table(lv_table_6sp_12_base, sparsity=.6)


################################################################################
#                                 Ecological                                   #
################################################################################


# going to test impact of relative abundance on ability to identify some of the
# ecological relationships. only going to test 1d relationships, but need to 
# have the other relationships here to make the calculations exactly the same
# as in the first table set; i.e. have to eat up the right number of calls to 
# the prng. 


# seed at 0 for reproducibility
seed(0)

######################
# Amensalism 1d
#####################

# choose 60 otus and relate them via o1^o2-> decrease to o2.
# note, odd otus will be affected by the last even otu. o0 will decrease 01, 
# o2 will decrease 03 etc.  
strength = .5
os = lognorm.rvs(3,0,size=(60,50))
amensally_related_1d_st_5 = []
for i in range(30):
    ind_i, ind_j = 2*i, 2*i+1
    am_otu = amensal_1d(os[ind_i], os[ind_j], strength)
    amensally_related_1d_st_5.extend([os[ind_i], am_otu])

strength = .3
os = lognorm.rvs(3,0,size=(60,50))
amensally_related_1d_st_3 = []
for i in range(30):
    ind_i, ind_j = 2*i, 2*i+1
    am_otu = amensal_1d(os[ind_i], os[ind_j], strength)
    amensally_related_1d_st_3.extend([os[ind_i], am_otu])

strength = .2
os = lognorm.rvs(3,0,size=(60,50))
amensally_related_1d_st_2 = []
for i in range(30):
    ind_i, ind_j = 2*i, 2*i+1
    am_otu = amensal_1d(os[ind_i], os[ind_j], strength)
    amensally_related_1d_st_2.extend([os[ind_i], am_otu])


######################
# Amensalism 2d
#####################

# require a network of 2 otus to be present to cause the amensal relationship
# eg O1^O2 -> decrease in O3. pick 90 otus where this happens. 
# note, o3 will be decreased if o1^o2
strength = .5
amensally_related_2d_st_5 = []
os = lognorm.rvs(3,0,size=(90,50))
for i in range(30):
    ind_i, ind_j, ind_k = 3*i, 3*i+1, 3*i+2
    depressed_otu = amensal_nd(vstack([os[ind_i], os[ind_j], os[ind_k]]), strength)
    amensally_related_2d_st_5.extend([os[ind_i], os[ind_j], depressed_otu])


strength = .3
amensally_related_2d_st_3 = []
os = lognorm.rvs(3,0,size=(90,50))
for i in range(30):
    ind_i, ind_j, ind_k = 3*i, 3*i+1, 3*i+2
    depressed_otu = amensal_nd(vstack([os[ind_i], os[ind_j], os[ind_k]]), strength)
    amensally_related_2d_st_3.extend([os[ind_i], os[ind_j], depressed_otu])


strength = .2
amensally_related_2d_st_2 = []
os = lognorm.rvs(3,0,size=(90,50))
for i in range(30):
    ind_i, ind_j, ind_k = 3*i, 3*i+1, 3*i+2
    depressed_otu = amensal_nd(vstack([os[ind_i], os[ind_j], os[ind_k]]), strength)
    amensally_related_2d_st_2.extend([os[ind_i], os[ind_j], depressed_otu])


######################
# Commensalism 1d
#####################

# choose 60 otus and relate them via o1^o2-> increase to o2.
# note, odd otus will be affected by the last even otu. o0 will increase 01, 
# o2 will increase 03 etc.  
strength = .5
os = lognorm.rvs(3,0,size=(60,50))
commensually_related_1d_st_5 = []
for i in range(30):
    ind_i, ind_j = 2*i, 2*i+1
    boosted_otu = commensal_1d(os[ind_i], os[ind_j], strength)
    commensually_related_1d_st_5.extend([os[ind_i], boosted_otu])

strength = .3
os = lognorm.rvs(3,0,size=(60,50))
commensually_related_1d_st_3 = []
for i in range(30):
    ind_i, ind_j = 2*i, 2*i+1
    boosted_otu = commensal_1d(os[ind_i], os[ind_j], strength)
    commensually_related_1d_st_3.extend([os[ind_i], boosted_otu])

strength = .2
os = lognorm.rvs(3,0,size=(60,50))
commensually_related_1d_st_2 = []
for i in range(30):
    ind_i, ind_j = 2*i, 2*i+1
    boosted_otu = commensal_1d(os[ind_i], os[ind_j], strength)
    commensually_related_1d_st_2.extend([os[ind_i], boosted_otu])


######################
# commensalism 2d
#####################

# require a network of 2 otus to be present to cause the commensal relationship
# eg O1^O2 -> increase in O3. pick 90 otus where this happens. 
# note, o3 will be increased if o1^o2
strength = .5
commensually_related_2d_st_5 = []
os = lognorm.rvs(3,0,size=(90,50))
for i in range(30):
    ind_i, ind_j, ind_k = 3*i, 3*i+1, 3*i+2
    boosted_otu = commensal_nd(vstack([os[ind_i], os[ind_j], os[ind_k]]),strength)
    commensually_related_2d_st_5.extend([os[ind_i], os[ind_j], boosted_otu])


strength = .3
commensually_related_2d_st_3 = []
os = lognorm.rvs(3,0,size=(90,50))
for i in range(30):
    ind_i, ind_j, ind_k = 3*i, 3*i+1, 3*i+2
    boosted_otu = commensal_nd(vstack([os[ind_i], os[ind_j], os[ind_k]]),strength)
    commensually_related_2d_st_3.extend([os[ind_i], os[ind_j], boosted_otu])


strength = .2
commensually_related_2d_st_2 = []
os = lognorm.rvs(3,0,size=(90,50))
for i in range(30):
    ind_i, ind_j, ind_k = 3*i, 3*i+1, 3*i+2
    boosted_otu = commensal_nd(vstack([os[ind_i], os[ind_j], os[ind_k]]),strength)
    commensually_related_2d_st_2.extend([os[ind_i], os[ind_j], boosted_otu])



####################
# mutualism 1d
####################

# create 30 pairs of related otus where o1^o2 -> boosting both
strength = .5
mutually_related_1d_st_5 = []
os = lognorm.rvs(3,0,size=(60,50))
for i in range(30): # no base otu to skip
    ind_i = 2*i
    ind_j = 2*i + 1
    # even numbered otus will be related to the next odd otu, i.e. 0 and 1 are 
    # mutualistic, 2,3 and so forth
    moi, moj = mutual_1d(os[ind_i], os[ind_j], strength)
    mutually_related_1d_st_5.extend([moi, moj])


strength = .3
mutually_related_1d_st_3 = []
os = lognorm.rvs(3,0,size=(60,50))
for i in range(30): # no base otu to skip
    ind_i = 2*i
    ind_j = 2*i + 1
    # even numbered otus will be related to the next odd otu, i.e. 0 and 1 are 
    # mutualistic, 2,3 and so forth
    moi, moj = mutual_1d(os[ind_i], os[ind_j], strength)
    mutually_related_1d_st_3.extend([moi, moj])


strength = .2
mutually_related_1d_st_2 = []
os = lognorm.rvs(3,0,size=(60,50))
for i in range(30): # no base otu to skip
    ind_i = 2*i
    ind_j = 2*i + 1
    # even numbered otus will be related to the next odd otu, i.e. 0 and 1 are 
    # mutualistic, 2,3 and so forth
    moi, moj = mutual_1d(os[ind_i], os[ind_j], strength)
    mutually_related_1d_st_2.extend([moi, moj])



####################
# mutualism 2d
####################

# create 30 triplets of related otus where o1^o2^o3 -> boost for all otus
# note that its o1 and o2 that are the network inducing the o3 boost so that
# if xor(o1, o2)==True then no boost to o3. 
strength = .5
mutually_related_2d_st_5 = []
os = lognorm.rvs(3,0,size=(90,50))
for i in range(30):
    ind_i, ind_j, ind_k = 3*i, 3*i + 1, 3*i + 2
    # otus will be related in groups of 3 0,1,2 are in mutualistic relationshi
    # as are 3,4,5 etc
    moi, moj, mok = mutual_nd(os.take([ind_i, ind_j, ind_k], 0), strength)
    mutually_related_2d_st_5.extend([moi, moj, mok])

strength = .3
mutually_related_2d_st_3 = []
os = lognorm.rvs(3,0,size=(90,50))
for i in range(30):
    ind_i, ind_j, ind_k = 3*i, 3*i + 1, 3*i + 2
    # otus will be related in groups of 3 0,1,2 are in mutualistic relationshi
    # as are 3,4,5 etc
    moi, moj, mok = mutual_nd(os.take([ind_i, ind_j, ind_k], 0), strength)
    mutually_related_2d_st_3.extend([moi, moj, mok])

strength = .2
mutually_related_2d_st_2 = []
os = lognorm.rvs(3,0,size=(90,50))
for i in range(30):
    ind_i, ind_j, ind_k = 3*i, 3*i + 1, 3*i + 2
    # otus will be related in groups of 3 0,1,2 are in mutualistic relationshi
    # as are 3,4,5 etc
    moi, moj, mok = mutual_nd(os.take([ind_i, ind_j, ind_k], 0), strength)
    mutually_related_2d_st_2.extend([moi, moj, mok])



####################
# parasatism 1d
####################

# create 30 pairs of otus where o1 ^ o2 -> increase o1, decrease o2
strength = .5
os = lognorm.rvs(3,0,size=(60,50))
parasitically_related_1d_st_5 = []
for i in range(30):
    ind_i, ind_j = 2*i, 2*i + 1
    # even numbered otus will be the parasite so 0 eats 1, 2 eats 3 etc.
    moi, moj = parasite_1d(os[ind_i], os[ind_j], strength)
    parasitically_related_1d_st_5.extend([moi, moj])


strength = .3
os = lognorm.rvs(3,0,size=(60,50))
parasitically_related_1d_st_3 = []
for i in range(30):
    ind_i, ind_j = 2*i, 2*i + 1
    # even numbered otus will be the parasite so 0 eats 1, 2 eats 3 etc.
    moi, moj = parasite_1d(os[ind_i], os[ind_j], strength)
    parasitically_related_1d_st_3.extend([moi, moj])

strength = .2
os = lognorm.rvs(3,0,size=(60,50))
parasitically_related_1d_st_2 = []
for i in range(30):
    ind_i, ind_j = 2*i, 2*i + 1
    # even numbered otus will be the parasite so 0 eats 1, 2 eats 3 etc.
    moi, moj = parasite_1d(os[ind_i], os[ind_j], strength)
    parasitically_related_1d_st_2.extend([moi, moj])


####################
# parasatism 2d
####################

# create 30 triplets of otus where o3 feeds on o1 and o2. this is basically
# a convenience wrapper for sending the same parasitizing otu through multiple
# round of parasite_1d with different otus
# note, o3 eats o1 and o2 so the 2,5,8 etc will be the parasitizing otus

strength = .5
parasitically_related_2d_st_5 = []
os = lognorm.rvs(3,0,size=(90,50))
for i in range(30):
    ind_i, ind_j, ind_k = 3*i, 3*i + 1, 3*i + 2
    # otus will be related in groups of 3 o2 eats o1 and o0. 
    moi, moj, mok = parasite_nd(os.take([ind_i, ind_j, ind_k], 0), strength)
    parasitically_related_2d_st_5.extend([moi, moj, mok])

strength = .3
parasitically_related_2d_st_3 = []
os = lognorm.rvs(3,0,size=(90,50))
for i in range(30):
    ind_i, ind_j, ind_k = 3*i, 3*i + 1, 3*i + 2
    # otus will be related in groups of 3 o2 eats o1 and o0. 
    moi, moj, mok = parasite_nd(os.take([ind_i, ind_j, ind_k], 0), strength)
    parasitically_related_2d_st_3.extend([moi, moj, mok])

strength = .2
parasitically_related_2d_st_2 = []
os = lognorm.rvs(3,0,size=(90,50))
for i in range(30):
    ind_i, ind_j, ind_k = 3*i, 3*i + 1, 3*i + 2
    # otus will be related in groups of 3 o2 eats o1 and o0. 
    moi, moj, mok = parasite_nd(os.take([ind_i, ind_j, ind_k], 0), strength)
    parasitically_related_2d_st_2.extend([moi, moj, mok])


####################
# competition 1d
####################

# create 30 pairs of otus where o1^o2 -> decrease for both o1 and o2
strength = .5
os = lognorm.rvs(3,0,size=(60,50))
competitively_related_1d_st_5 = []
for i in range(30):
    ind_i, ind_j = 2*i, 2*i + 1
    # even numbered otus will be the related to the next odd otu as in o2 
    # related to o3, o4 with o5. 
    moi, moj = competition_1d(os[ind_i], os[ind_j], strength)
    competitively_related_1d_st_5.extend([moi, moj])


strength = .3
os = lognorm.rvs(3,0,size=(60,50))
competitively_related_1d_st_3 = []
for i in range(30):
    ind_i, ind_j = 2*i, 2*i + 1
    # even numbered otus will be the related to the next odd otu as in o2 
    # related to o3, o4 with o5. 
    moi, moj = competition_1d(os[ind_i], os[ind_j], strength)
    competitively_related_1d_st_3.extend([moi, moj])

strength = .2
os = lognorm.rvs(3,0,size=(60,50))
competitively_related_1d_st_2 = []
for i in range(30):
    ind_i, ind_j = 2*i, 2*i + 1
    # even numbered otus will be the related to the next odd otu as in o2 
    # related to o3, o4 with o5. 
    moi, moj = competition_1d(os[ind_i], os[ind_j], strength)
    competitively_related_1d_st_2.extend([moi, moj])

# round to nearest float to avoid partial counts
eco_table_base = vstack([\
 competitively_related_1d_st_3,
 commensually_related_1d_st_2,
 commensually_related_1d_st_3,
 competitively_related_1d_st_5,
 competitively_related_1d_st_2,
 mutually_related_1d_st_5,
 commensually_related_1d_st_5,
 mutually_related_1d_st_3,
 amensally_related_1d_st_5,
 amensally_related_1d_st_3,
 amensally_related_1d_st_2,
 mutually_related_1d_st_2]).round(0)


eco_table_1 = eco_table_base/eco_table_base.sum(0)
eco_table_2 = alter_table(eco_table_base, sparsity=.5)
eco_table_3 = eco_table_2/eco_table_2.sum(0)

################################################################################
#                                 copula                                       #
################################################################################

# add copula elements where the input rho matrix actually has large correlations
# in it


from numpy.random import seed 
from numpy import array
from biom.parse import parse_biom_table
from scipy import corrcoef
from scipy.stats import lognorm, gamma, expon
bt = parse_biom_table(open('/Users/wdwvt1/src/correlations/tables/bioms/table_7.biom'))
data = array([bt.observationData(i) for i in bt.ObservationIds])
inp_rho_mat = corrcoef(data[0:45])


from generators.copula import copula

mu_mat = array([0]*45)
num_samples = 50
methods1 = [[lognorm, 3, 0]]*45
methods2 = [[gamma, 1, 0, 100]]*45
methods3 = [[expon, 0, 1000]]*45

seed(0)
copula_table1_lognorm_3_0 = copula(num_samples, inp_rho_mat, mu_mat, methods1)
seed(0)
copula_table2_gamma_1_0_100 = copula(num_samples, inp_rho_mat, mu_mat, methods2)
seed(0)
copula_table3_expon_0_1000 = copula(num_samples, inp_rho_mat, mu_mat, methods3)


all_tables = [ts_g1_otus.round(0), ts_g2_otus.round(0), 
    ts_g3_otus.round(0), ts_g4_otus.round(0), ts_g5_otus.round(0), 
    lv_table_2sp_1, lv_table_2sp_2, lv_table_2sp_3, lv_table_2sp_4,
    lv_table_2sp_5, lv_table_6sp_1, lv_table_6sp_2, lv_table_6sp_3,
    lv_table_6sp_4, lv_table_6sp_5, eco_table_1, eco_table_2, eco_table_3,
    copula_table1_lognorm_3_0, copula_table2_gamma_1_0_100, 
    copula_table3_expon_0_1000]



from biom.table import table_factory


names = ['table_1.biom',
 'table_2.biom',
 'table_3.biom',
 'table_4.biom',
 'table_5.biom',
 'table_6.biom',
 'table_7.biom',
 'table_8.biom',
 'table_9.biom',
 'table_10.biom',
 'table_11.biom',
 'table_12.biom',
 'table_13.biom',
 'table_14.biom',
 'table_15.biom',
 'table_16.biom',
 'table_17.biom',
 'table_18.biom',
 'table_19.biom',
 'table_20.biom',
 'table_21.biom']

def make_ids(data):
    sids = ['s%i' % i for i in range(data.shape[1])]
    oids = ['o%i' % i for i in range(data.shape[0])]
    return sids, oids

for table, name in zip(all_tables,names):
    sids, oids = make_ids(table)
    bt = table_factory(table, sids, oids)
    json_str = bt.getBiomFormatJsonString(generated_by='Sophie_Will')
    o = open('/Users/wdwvt1/Desktop/table2/'+name, 'w')
    o.write(json_str)
    o.close()











