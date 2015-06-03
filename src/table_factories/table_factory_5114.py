from numpy import array, arange, corrcoef
from scipy.stats.distributions import lognorm
from numpy.random import seed
from qiime.pycogent_backports.alpha_diversity import * 
from qiime.rarefaction import *
import cogent.maths.unifrac.fast_unifrac as fast_unifrac
import matplotlib.pyplot as plt
from generators.ecological import *

#table_1_rarify_depth_1000.biom = table_6.biom   (2=7,3=8,4=9,5=10)
#table_1_CSS.biom = table_11.biom       (2=12,3=13,4=14,5=15)
#table_1_DESeq.biom = table_16.biom       (2=17,3=18,4=19,5=20)

##inverse shannon alpha diversities: 36.30 24.47 18.61 9.998 3.99
##################################################
#                 copula table (112x80)          #
##################################################


#NOTE: the input rho_mat must be positive definite correlation matrix. cov
# matrices have failed for. 
# load up a table of pvals we created
from numpy import load, array, arange, shape, transpose
from numpy.random import seed
from scipy.stats import spearmanr
from scipy.stats.distributions import gamma, lognorm, uniform
import matplotlib.pyplot as plt
from matplotlib.pylab import matshow
from qiime.beta_diversity import get_nonphylogenetic_metric
from generators.copula import copula, generate_rho_matrix

######create random matrix with low correlations#########
num_otus = 500
num_samples = 80
mu_mat = array([0]*num_otus)
methods = [[lognorm, 2, 0]]*num_otus

seed(0)
j = generate_rho_matrix(uniform, [-.01, .02], num_otus, 100)
seed(0)
copula_table1_lognorm_3_0 = copula(num_samples, j, mu_mat, methods)


# #####create 2 highly correlated otus########
num_otus = 2
num_samples = 80
mu_mat = array([0]*num_otus) 
methods = [[lognorm, 2, 0]]*num_otus

seed(0)
j=array([[ 1.        , 1],
       [1,  1.        ]])
copula_table2_lognorm_3_0 = copula(num_samples, j, mu_mat, methods)

#####create 2 highly correlated otus########
num_otus = 2
num_samples = 80
mu_mat = array([0]*num_otus)
methods = [[lognorm, 2, 0]]*num_otus

seed(0)
j=array([[ 1.        , 0.9],    #in practice is 0.978
       [0.9,  1.        ]])
copula_table3_lognorm_3_0 = copula(num_samples, j, mu_mat, methods)

# #####create 2 highly correlated otus########
num_otus = 2
num_samples = 80
mu_mat = array([0]*num_otus)
methods = [[lognorm, 2, 0]]*num_otus

seed(0)
j=array([[ 1.        , 0.7],   #in practice is 0.85
       [0.7,  1.        ]])
copula_table4_lognorm_3_0 = copula(num_samples, j, mu_mat, methods)

# #####create 2 highly correlated otus########
num_otus = 2
num_samples = 80
mu_mat = array([0]*num_otus)
methods = [[lognorm, 2, 0]]*num_otus

seed(0)
j=array([[ 1.        , 0.5],  #in practice is 0.53
       [0.5,  1.        ]])
copula_table5_lognorm_3_0 = copula(num_samples, j, mu_mat, methods)

# #####create 2 highly correlated otus########
num_otus = 2
num_samples = 80
mu_mat = array([0]*num_otus)
methods = [[lognorm, 2, 0]]*num_otus

seed(0)
j=array([[ 1.        , 0.4],  #in practice is 0.33
       [0.4,  1.        ]])
copula_table6_lognorm_3_0 = copula(num_samples, j, mu_mat, methods)

# #####create 2 highly correlated otus########
num_otus = 2
num_samples = 80
mu_mat = array([0]*num_otus)
methods = [[lognorm, 2, 0]]*num_otus

seed(0)
j=array([[ 1.        , 0.25],  #in practice is 0.115
       [0.25,  1.        ]])
copula_table7_lognorm_3_0 = copula(num_samples, j, mu_mat, methods)

# #####create 2 highly correlated otus########
num_otus = 2
num_samples = 80
mu_mat = array([0]*num_otus) 
methods = [[lognorm, 2, 0]]*num_otus

seed(0)
j=array([[ 1.        , -1],
       [-1,  1.        ]])
copula_table8_lognorm_3_0 = copula(num_samples, j, mu_mat, methods)

#####create 2 highly correlated otus########
num_otus = 2
num_samples = 80
mu_mat = array([0]*num_otus)
methods = [[lognorm, 2, 0]]*num_otus

seed(0)
j=array([[ 1.        , -0.9],    #in practice is -0.91
       [-0.9,  1.        ]])
copula_table9_lognorm_3_0 = copula(num_samples, j, mu_mat, methods)

# #####create 2 highly correlated otus########
num_otus = 2
num_samples = 80
mu_mat = array([0]*num_otus)
methods = [[lognorm, 2, 0]]*num_otus

seed(0)
j=array([[ 1.        , -0.7],   #in practice is -0.72
       [-0.7,  1.        ]])
copula_table10_lognorm_3_0 = copula(num_samples, j, mu_mat, methods)

# #####create 2 highly correlated otus########
num_otus = 2
num_samples = 80
mu_mat = array([0]*num_otus)
methods = [[lognorm, 2, 0]]*num_otus

seed(0)
j=array([[ 1.        , -0.5],  #in practice is -0.54
       [-0.5,  1.        ]])
copula_table11_lognorm_3_0 = copula(num_samples, j, mu_mat, methods)

# #####create 2 highly correlated otus########
num_otus = 2
num_samples = 80
mu_mat = array([0]*num_otus)
methods = [[lognorm, 2, 0]]*num_otus

seed(0)
j=array([[ 1.        , -0.4],  #in practice spearmanr is -0.45
       [-0.4,  1.        ]])
copula_table12_lognorm_3_0 = copula(num_samples, j, mu_mat, methods)

# #####create 2 highly correlated otus########
num_otus = 2
num_samples = 80
mu_mat = array([0]*num_otus)
methods = [[lognorm, 2, 0]]*num_otus

seed(0)
j=array([[ 1.        , -0.25],  #in practice is -0.28
       [-0.25,  1.        ]])
copula_table13_lognorm_3_0 = copula(num_samples, j, mu_mat, methods)



copula_table_base = vstack([\
 copula_table1_lognorm_3_0,
 copula_table2_lognorm_3_0,
 copula_table3_lognorm_3_0,
 copula_table4_lognorm_3_0,
 copula_table5_lognorm_3_0,
 copula_table6_lognorm_3_0,
 copula_table7_lognorm_3_0,
 copula_table8_lognorm_3_0,
 copula_table9_lognorm_3_0,
 copula_table10_lognorm_3_0,
 copula_table11_lognorm_3_0,
 copula_table12_lognorm_3_0,
 copula_table13_lognorm_3_0]).round(0)

######### multiple correlation 1 OTU with values ranging from 1, 10, 100, 1000 ##########

copula_table_base2 = vstack([\
 copula_table1_lognorm_3_0,
 copula_table2_lognorm_3_0,
 200*copula_table3_lognorm_3_0,
 copula_table4_lognorm_3_0,
 copula_table5_lognorm_3_0,
 copula_table6_lognorm_3_0,
 copula_table7_lognorm_3_0,
 copula_table8_lognorm_3_0,
 copula_table9_lognorm_3_0,
 copula_table10_lognorm_3_0,
 copula_table11_lognorm_3_0,
 copula_table12_lognorm_3_0,
 copula_table13_lognorm_3_0]).round(0)

copula_table_base3 = vstack([\
 copula_table1_lognorm_3_0,
 copula_table2_lognorm_3_0,
 500*copula_table3_lognorm_3_0,
 copula_table4_lognorm_3_0,
 copula_table5_lognorm_3_0,
 copula_table6_lognorm_3_0,
 copula_table7_lognorm_3_0,
 copula_table8_lognorm_3_0,
 copula_table9_lognorm_3_0,
 copula_table10_lognorm_3_0,
 copula_table11_lognorm_3_0,
 copula_table12_lognorm_3_0,
 copula_table13_lognorm_3_0]).round(0)

copula_table_base4 = vstack([\
 copula_table1_lognorm_3_0,
 copula_table2_lognorm_3_0,
 3000*copula_table3_lognorm_3_0,
 copula_table4_lognorm_3_0,
 copula_table5_lognorm_3_0,
 copula_table6_lognorm_3_0,
 copula_table7_lognorm_3_0,
 copula_table8_lognorm_3_0,
 copula_table9_lognorm_3_0,
 copula_table10_lognorm_3_0,
 copula_table11_lognorm_3_0,
 copula_table12_lognorm_3_0,
 copula_table13_lognorm_3_0]).round(0)

copula_table_base5 = vstack([\
 copula_table1_lognorm_3_0,
 copula_table2_lognorm_3_0,
 20000*copula_table3_lognorm_3_0,
 copula_table4_lognorm_3_0,
 copula_table5_lognorm_3_0,
 copula_table6_lognorm_3_0,
 copula_table7_lognorm_3_0,
 copula_table8_lognorm_3_0,
 copula_table9_lognorm_3_0,
 copula_table10_lognorm_3_0,
 copula_table11_lognorm_3_0,
 copula_table12_lognorm_3_0,
 copula_table13_lognorm_3_0]).round(0)

div = sum([simpson_reciprocal(a) for a in transpose(copula_table_base)])/float(copula_table_base.shape[1])
div2 = sum([simpson_reciprocal(a) for a in transpose(copula_table_base2)])/float(copula_table_base2.shape[1])
div3 = sum([simpson_reciprocal(a) for a in transpose(copula_table_base3)])/float(copula_table_base3.shape[1])
div4 = sum([simpson_reciprocal(a) for a in transpose(copula_table_base4)])/float(copula_table_base4.shape[1])
div5 = sum([simpson_reciprocal(a) for a in transpose(copula_table_base5)])/float(copula_table_base5.shape[1])

print div, div2, div3, div4, div5
print 'copula done'

# def make_ids(data):
#     sids = ['s%i' % i for i in range(data.shape[1])]
#     oids = ['o%i' % i for i in range(data.shape[0])]
#     return sids, oids

# from biom.table import table_factory
# tables = [copula_table_base,
# copula_table_base2,
# copula_table_base3,
# copula_table_base4,
# copula_table_base5]

# names = ['table_1.biom','table_2.biom','table_3.biom','table_4.biom','table_5.biom']


# for table, name in zip(tables,names):
#     sids, oids = make_ids(table)
#     bt = table_factory(table, sids, oids, observation_metadata=[{'taxonomy':{'kingdom':'nonsense', 'phylum':'meganonsense'}} for i in range(len(oids))])
#     json_str = bt.getBiomFormatJsonString(generated_by='Sophie_Will')
#     o = open('/Users/sophie/Desktop/'+name, 'w')
#     o.write(json_str)
#     o.close()

#table_1_rarify_depth_1000.biom = table_6.biom   (2=7,3=8,4=9,5=10)
#table_1_CSS.biom = table_11.biom       (2=12,3=13,4=14,5=15)
#table_1_DESeq.biom = table_16.biom       (2=17,3=18,4=19,5=20)



############################################
#               Eco table  (112x100)       #
############################################

# seed(5000)

# # Amensalism 1D, st5
# strength = .5
# os = lognorm.rvs(3,0,size=(70,100)).astype(int)
# amensally_related_1d_st_5 = []
# for i in range(35):
#     ind_i, ind_j = 2*i, 2*i+1
#     am_otu = amensal_1d(os[ind_i], os[ind_j], strength)
#     amensally_related_1d_st_5.extend([os[ind_i], am_otu])

# # Commenasalism 1D, st5
# strength = .5
# os = lognorm.rvs(3,0,size=(70,100)).astype(int)
# commensually_related_1d_st_5 = []
# for i in range(35):
#     ind_i, ind_j = 2*i, 2*i+1
#     boosted_otu = commensal_1d(os[ind_i], os[ind_j], strength)
#     commensually_related_1d_st_5.extend([os[ind_i], boosted_otu])

# # Mutualism 1D, st5
# strength = .5
# mutually_related_1d_st_5 = []
# os = lognorm.rvs(3,0,size=(70,100)).astype(int)
# for i in range(35): # no base otu to skip
#     ind_i = 2*i
#     ind_j = 2*i + 1
#     # even numbered otus will be related to the next odd otu, i.e. 0 and 1 are 
#     # mutualistic, 2,3 and so forth
#     moi, moj = mutual_1d(os[ind_i], os[ind_j], strength)
#     mutually_related_1d_st_5.extend([moi, moj])

# # Parasatism 1D, st5
# strength = .5
# os = lognorm.rvs(3,0,size=(70,100)).astype(int)
# parasitically_related_1d_st_5 = []
# for i in range(35):
#     ind_i, ind_j = 2*i, 2*i + 1
#     # even numbered otus will be the parasite so 0 eats 1, 2 eats 3 etc.
#     moi, moj = parasite_1d(os[ind_i], os[ind_j], strength)
#     parasitically_related_1d_st_5.extend([moi, moj])

# # Competition 1D, st5
# strength = .5
# os = lognorm.rvs(3,0,size=(70,100)).astype(int)
# competitively_related_1d_st_5 = []
# for i in range(35):
#     ind_i, ind_j = 2*i, 2*i + 1
#     # even numbered otus will be the related to the next odd otu as in o2 
#     # related to o3, o4 with o5. 
#     moi, moj = competition_1d(os[ind_i], os[ind_j], strength)
#     competitively_related_1d_st_5.extend([moi, moj])

# # Obligate syntrophy 1D, st5
# strength = .5
# obligate_related_1d_st_5 = []
# os = lognorm.rvs(3,0,size=(35,100)).astype(int)
# for otu in os:
#     obs_otu = obligate_syntroph_1d(otu, strength)
#     obligate_related_1d_st_5.extend([otu, obs_otu])

# # Partial obligate syntrophy 1D, st5
# os = lognorm.rvs(3,0,size=(70,100)).astype(int)
# partial_obligate_syntrophic_related_1d = []
# for i in range(35):
#     ind_i, ind_j = 2*i, 2*i + 1
#     # even numbered otus will be the independent so 0 allows 1, 2 allows 3 etc.
#     moj = partial_obligate_syntroph_1d(os[ind_i], os[ind_j])
#     partial_obligate_syntrophic_related_1d.extend([os[ind_i], moj])

# ##14 unrelated OTUs as noise
# os = lognorm.rvs(3,0,size=(14,100))
# unrelated_otus = [i for i in os]

# ## typing as int then float guarantees we get about 50k more 0's than if we 
# ## just did .round(0)
# otu_data = vstack([amensally_related_1d_st_5,
#     commensually_related_1d_st_5,
#     mutually_related_1d_st_5,
#     parasitically_related_1d_st_5,
#     competitively_related_1d_st_5,
#     obligate_related_1d_st_5,
#     partial_obligate_syntrophic_related_1d,
#     unrelated_otus]).astype(int).astype(float)






# def make_ids(data):
#     sids = ['s%i' % i for i in range(data.shape[1])]
#     oids = ['o%i' % i for i in range(data.shape[0])]
#     return sids, oids

# sids, oids = make_ids(otu_data)

# bt = table_factory(otu_data, sids, oids)
# all tables

# from biom.table import table_factory
# tables = [copula_table2_gamma_1_0_100,
# copula_table1_lognorm_3_0,
# ga_table,
# null_table1,
# null_table2,
# eco_table1,
# eco_table2]


# names = ['table_1.biom','table_2.biom','table_3.biom','table_4.biom','table_5.biom','table_6.biom','table_7.biom']

# def make_ids(data):
#     sids = ['s%i' % i for i in range(data.shape[1])]
#     oids = ['o%i' % i for i in range(data.shape[0])]
#     return sids, oids

# for table, name in zip(tables,names):
#     sids, oids = make_ids(table)
#     bt = table_factory(table, sids, oids)
#     json_str = bt.getBiomFormatJsonString(generated_by='Sophie_Will')
#     o = open('/Users/will/Desktop/'+name, 'w')
#     o.write(json_str)
#     o.close()