import unittest

from _coverage import brive
from biom import load_table
from skbio.diversity.alpha import robbins
from skbio.stats.composition import closure
import numpy as np
import numpy.testing as npt

class TestCoverage(unittest.TestCase):
    def setUp(self):
        data_dir = "../../data/tick/meshnick_tech_reps"
        biom_file = "%s/373_otu_table.biom" % data_dir
        meta_file = "%s/meta.txt" % data_dir

        table = load_table(biom_file)
        Z = 1
        mat = np.array(table._get_sparse_data().todense()).T
        x = np.ravel(mat[Z, :])
        self.tick_pvals = closure(np.array(x[x > 0]))
        self.uniform_pvals = closure(np.array([10000] * len(self.tick_pvals)))
        self.exponential_pvals = closure(np.exp(
            np.linspace(0, 4,len(self.tick_pvals))))

    def test_brive_tick(self):
        samp_table = np.random.multinomial(n=500,
                                           pvals=self.tick_pvals)
        bvals = brive(samp_table, replace_zeros=False)
        rel = closure(samp_table)
        m = bvals.sum()
        npt.assert_array_less(rel-bvals, 1.1/500)
        self.assertLess(m, 1 - robbins(samp_table))

    def test_brive_uniform(self):
        samp_table = np.random.multinomial(n=500,
                                           pvals=self.uniform_pvals)
        bvals = brive(samp_table, replace_zeros=False)
        rel = closure(samp_table)
        m = bvals.sum()
        npt.assert_array_less(rel-bvals, 1.1/500)
        self.assertLess(m, 1 - robbins(samp_table))

    def test_exponential_uniform(self):
        samp_table = np.random.multinomial(n=500,
                                           pvals=self.exponential_pvals)
        bvals = brive(samp_table, replace_zeros=False)
        rel = closure(samp_table)
        m = bvals.sum()
        npt.assert_array_less(rel-bvals, 1.1/500)
        self.assertLess(m, 1 - robbins(samp_table))

if __name__=='__main__':
    unittest.main()
