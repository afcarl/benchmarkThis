from numpy import array, arange
from scipy.stats.distributions import lognorm
from numpy.random import seed
import matplotlib.pyplot as plt
from generators.ecological import *

##################################################
#   Last edit 3/6/2013                           #
#   checked by sophie and will                   #
##################################################


##################################################
#                 ecological table               #
##################################################

# seed at 0 for reproduccibility
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


####################
# competition 2d
####################

# create 30 triplets of otus where o1^o2^o3 -> decrease for all
# note that o1 and o2 form the network and thus must be present for the 
# competition to appear. 

strength = .5
competitively_related_2d_st_5 = []
os = lognorm.rvs(3,0,size=(90,50))
for i in range(30):
    ind_i, ind_j, ind_k = 3*i, 3*i + 1, 3*i + 2
    # otus will be related in groups of 3 o2 eats o1 and o0. 
    moi, moj, mok = competition_nd(os.take([ind_j, ind_j, ind_k], 0),strength)
    competitively_related_2d_st_5.extend([moi, moj, mok])

strength = .3
competitively_related_2d_st_3 = []
os = lognorm.rvs(3,0,size=(90,50))
for i in range(30):
    ind_i, ind_j, ind_k = 3*i, 3*i + 1, 3*i + 2
    # otus will be related in groups of 3 o2 eats o1 and o0. 
    moi, moj, mok = competition_nd(os.take([ind_j, ind_j, ind_k], 0),strength)
    competitively_related_2d_st_3.extend([moi, moj, mok])

strength = .2
competitively_related_2d_st_2 = []
os = lognorm.rvs(3,0,size=(90,50))
for i in range(30):
    ind_i, ind_j, ind_k = 3*i, 3*i + 1, 3*i + 2
    # otus will be related in groups of 3 o2 eats o1 and o0. 
    moi, moj, mok = competition_nd(os.take([ind_j, ind_j, ind_k], 0),strength)
    competitively_related_2d_st_2.extend([moi, moj, mok])

####################
# obligate syntroph 1d
####################

# choose 10 otus randomly and introduce 10 dependants in an obligate 
# syntrophic manner. 
# note that the even otus will be the independent ones and the odd otus will be
# dependent. 
strength = .5
obligate_related_1d_st_5 = []
os = lognorm.rvs(3,0,size=(10,50))
for otu in os:
    obs_otu = obligate_syntroph_1d(otu, strength)
    obligate_related_1d_st_5.extend([otu, obs_otu])

####################
# obligate syntrophy 2d
####################

# choose 60 otus where o1^o2 -> o3. o1 and o2 are the inducers
# otus 2,5,8 will be the induced ones
strength = .5
obligate_related_2d_st_5 = []
os = lognorm.rvs(3,0,size=(60,50))
for i in range(30):
    ind_i, ind_j = 2*i, 2*i + 1
    obs_otu = obligate_syntroph_nd(os.take([ind_i, ind_j],0), strength)
    obligate_related_2d_st_5.extend([os[ind_i], os[ind_j], obs_otu])


####################
# partial obligate syntrophy 1d
####################

# create 30 pairs of otus where o1 allows o2 but doesnt affect its abundance
os = lognorm.rvs(3,0,size=(60,50))
partial_obligate_syntrophic_related_1d = []
for i in range(30):
    ind_i, ind_j = 2*i, 2*i + 1
    # even numbered otus will be the independent so 0 allows 1, 2 allows 3 etc.
    moj = partial_obligate_syntroph_1d(os[ind_i], os[ind_j])
    partial_obligate_syntrophic_related_1d.extend([os[ind_i], moj])


####################
# partial obligate syntrophy 2d
####################

# create 30 pairs of otus where o1^o2 allows o3 but doesnt affect its abundance
# note that o3 is the allowed otu and o1 and o2 must be present to allow it
os = lognorm.rvs(3,0,size=(90,50))
partial_obligate_syntrophic_related_2d = []
for i in range(30):
    ind_i, ind_j, ind_k = 3*i, 3*i+1, 3*i+2
    oi, oj, depressed_otu = partial_obligate_syntroph_nd(vstack([os[ind_i], os[ind_j], os[ind_k]]))
    partial_obligate_syntrophic_related_2d.extend([oi, oj, depressed_otu])




print 'eco done'


######################
# stitching the table together
# 
# all in order
# partial_obligate_syntrophic_related_2d,
# partial_obligate_syntrophic_related_1d,
# obligate_related_2d_st_5,
# obligate_related_1d_st_5,
# amensally_related_1d_st_5,
# amensally_related_1d_st_3,
# amensally_related_1d_st_2,
# amensally_related_2d_st_5,
# amensally_related_2d_st_3,
# amensally_related_2d_st_2,
# commensually_related_1d_st_5,
# commensually_related_1d_st_3,
# commensually_related_1d_st_2,
# commensually_related_2d_st_5,
# commensually_related_2d_st_3,
# commensually_related_2d_st_2,
# mutually_related_1d_st_5,
# mutually_related_1d_st_3,
# mutually_related_1d_st_2,
# mutually_related_2d_st_5,
# mutually_related_2d_st_3,
# mutually_related_2d_st_2,
# parasitically_related_1d_st_5,
# parasitically_related_1d_st_3,
# parasitically_related_1d_st_2,
# parasitically_related_2d_st_5,
# parasitically_related_2d_st_3,
# parasitically_related_2d_st_2,
# competitively_related_1d_st_5,
# competitively_related_1d_st_3,
# competitively_related_1d_st_2,
# competitively_related_2d_st_5,
# competitively_related_2d_st_3,
# competitively_related_2d_st_2,



eco_table1 = vstack(
[competitively_related_2d_st_3,
 mutually_related_2d_st_5,
 commensually_related_2d_st_5,
 competitively_related_1d_st_3,
 parasitically_related_2d_st_2,
 commensually_related_1d_st_2,
 obligate_related_2d_st_5,
 competitively_related_2d_st_5,
 commensually_related_1d_st_3,
 partial_obligate_syntrophic_related_2d,
 commensually_related_2d_st_3,
 parasitically_related_2d_st_5,
 competitively_related_1d_st_5,
 competitively_related_1d_st_2,
 parasitically_related_1d_st_3,
 mutually_related_1d_st_5]).astype(int)

eco_table2 = vstack(
[mutually_related_2d_st_2,
 commensually_related_1d_st_5,
 competitively_related_2d_st_2,
 parasitically_related_1d_st_2,
 mutually_related_1d_st_3,
 amensally_related_1d_st_5,
 amensally_related_2d_st_2,
 mutually_related_2d_st_3,
 amensally_related_1d_st_3,
 partial_obligate_syntrophic_related_1d,
 amensally_related_1d_st_2,
 obligate_related_1d_st_5,
 mutually_related_1d_st_2,
 parasitically_related_2d_st_3,
 parasitically_related_1d_st_5,
 commensually_related_2d_st_2,
 amensally_related_2d_st_5,
 amensally_related_2d_st_3]).astype(int)

# #spot check
# from numpy import linspace
# import matplotlib.pyplot as plt
# from matplotlib.pylab import matshow
# from qiime.beta_diversity import get_nonphylogenetic_metric
# m = get_nonphylogenetic_metric('pearson')
# q = m(array(obligate_related_1d_st_5))
# matshow(q)
# bars = linspace(-.5,19.5,21)
# plt.vlines(bars,-.5,19.5)
# plt.hlines(bars,-.5,19.5)
# plt.colorbar()
# plt.show()



##################################################
#                 null table                     #
##################################################

from scipy.stats.distributions import (gamma, beta, lognorm, nakagami, chi2, 
    uniform)
from generators.null import (model1_otu, model1_table, model2_table, 
    model3_table)

##############
#############
# null table 1 no compositionality
#############
##############

seed(10000000)

# all tables will have 50 samples 
# generate 100 otus from lognorm distribution 2,0,1 
dfs_and_params = [[lognorm, 2, 0]]*100
otus_lognorm_2_0 = model1_table(dfs_and_params, 50)

# generate 100 otus from lognorm distribution 3,0,1
dfs_and_params = [[lognorm, 3, 0]]*100
otus_lognorm_3_0 = model1_table(dfs_and_params, 50)

# generate 100 otus from gamma distribution 1,0,100
dfs_and_params = [[gamma, 1, 0, 100]]*100
otus_gamma_1_0_100 = model1_table(dfs_and_params, 50)

# generate 100 otus from nakagami distribution .1,0,100
dfs_and_params = [[nakagami, .1, 0, 100]]*100
otus_nakagami_pt1_0_100 = model1_table(dfs_and_params, 50)

# generate 100 otus from chisquared distribution .1,0,100
dfs_and_params = [[chi2, .1, 0, 100]]*100
otus_chi2_pt1_0_100 = model1_table(dfs_and_params, 50)

# generate 100 otus from uniform distributions 0,1000,1
dfs_and_params = [[uniform, 0, 1000]]*100
otus_uniform_0_1000 = model1_table(dfs_and_params, 50)

# in order
# otus_uniform_0_1000
# otus_lognorm_2_0
# otus_lognorm_3_0
# otus_gamma_1_0_100
# otus_nakagami_pt1_0_100
# otus_chi2_pt1_0_100

# shuffled for table order
null_table1 = vstack(
[otus_lognorm_3_0,
 otus_gamma_1_0_100,
 otus_lognorm_2_0,
 otus_nakagami_pt1_0_100,
 otus_uniform_0_1000,
 otus_chi2_pt1_0_100])



##############
#############
# null table 2 compositionality
#############
##############

# open the otu_sums.txt doc to get otu sums
o = open('/Users/will/Desktop/otu_sums.txt')
l = o.readlines()[:500] # use only the first 500
o.close()
otu_sums = array(map(float, l))
samples = 50
seq_depth = 25000
tpk=1000
null_table2 = model2_table(otu_sums, samples, seq_depth, tpk)


print 'null done'


##################################################
#                 ga table                       #
##################################################

# run the genetic algorithms method on a known 
# otu sequence and maximize graphic dissimilarity
# pearson distance is about .007
from numpy.random import seed
seed(2039203920392039)
from generators.ga import *
from scipy.stats.distributions import uniform
ref_gene = uniform.rvs(100,1000,size=(50,2))
igp = [ref_gene[:]+uniform.rvs(100,1000,size=ref_gene.shape) for i in range(400)]

gc, fmeans, fmaxes = evolve(igp, ref_gene, 1000)
tmp_ga_table = vstack([i.T for i in gc])

# remove negative entries
ga_table = []
for i in range(400):
    if (tmp_ga_table[2*i:2*(i+1)]<0).sum() < 1:
        ga_table.append(tmp_ga_table[2*i:2*(i+1)])
ga_table = vstack(ga_table)

print 'ga done'

##################################################
#                 copula table                   #
##################################################


#NOTE: the input rho_mat must be positive definite correlation matrix. cov
# matrices have failed for. 
# load up a table of pvals we created
from numpy import load, array, arange
from numpy.random import seed
from scipy.stats.distributions import gamma, lognorm, uniform
import matplotlib.pyplot as plt
from matplotlib.pylab import matshow
from qiime.beta_diversity import get_nonphylogenetic_metric
from generators.copula import copula, generate_rho_matrix

mu_mat = array([0]*500)
num_samples = 50
# methods1 = [[lognorm, 3, 0]]*500
# methods2 = [[gamma, 1, 0, 100]]*500
methods1 = [[uniform, 10, 100]]*500
methods2 = [[uniform, 10, 10000]]*500
seed(0)
j = generate_rho_matrix(uniform, [-.01, .02], 500, 100)
seed(0)
copula_table1_lognorm_3_0 = copula(num_samples, j, mu_mat, methods1)
seed(0)
copula_table2_gamma_1_0_100 = copula(num_samples, j, mu_mat, methods2)

# proof that they have exactly the same pearson scores
# m = get_nonphylogenetic_metric('pearson')
# ps1 = m(copula_table1)
# ps2 = m(copula_table1)
# matshow(ps1-ps2)
# plt.colorbar()
# plt.show()

print 'copula done'

# all tables

from biom.table import table_factory
tables = [copula_table2_gamma_1_0_100,
copula_table1_lognorm_3_0,
ga_table,
null_table1,
null_table2,
eco_table1,
eco_table2]


names = ['table_1.biom','table_2.biom','table_3.biom','table_4.biom','table_5.biom','table_6.biom','table_7.biom']

def make_ids(data):
    sids = ['s%i' % i for i in range(data.shape[1])]
    oids = ['o%i' % i for i in range(data.shape[0])]
    return sids, oids

for table, name in zip(tables,names):
    sids, oids = make_ids(table)
    bt = table_factory(table, sids, oids)
    json_str = bt.getBiomFormatJsonString(generated_by='Sophie_Will')
    o = open('/Users/will/Desktop/'+name, 'w')
    o.write(json_str)
    o.close()



