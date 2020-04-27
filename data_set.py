import numpy as np 
import glob
import pandas as pd 
import matplotlib.pyplot as plt 

class data_set(object):
    def __init__(self, load_n, shuffle, return_joined, n_samples):
        self.return_joined = return_joined
        self.n_samples = n_samples

    def get_data(self, path):

        f = glob.glob(path+'/**_leads.npy')
        dd_ld = []
        dd_lb = []
        dd_ft = []

        miss_li = []
        for ix, leads in enumerate(f):
            label = leads.replace('leads', 'label')
            feats = leads.replace('leads', 'feats')
            
            lddt = np.load(leads, allow_pickle = True)
            for i in range(lddt.shape[1]):
                lddt[:, i] = pd.Series(lddt[:, i]).interpolate().values
            lddt[np.isnan(lddt)] = 0
            lbdt = np.load(label, allow_pickle = True)
            ftdt = np.load(feats, allow_pickle = True)
            dd_ld.append(lddt[np.newaxis])
            dd_lb.append(lbdt[np.newaxis])
            dd_ft.append(ftdt[np.newaxis])

        dd_ld = np.vstack(dd_ld)
        dd_lb = np.vstack(dd_lb)
        dd_ft = np.vstack(dd_ft)

        return dd_ld, dd_ft, dd_lb


    def get_cat_features(self):
        return self.cat_features

    def get_n_classes(self):
        return self.n_classes

    def generate(self):

        train_leads, train_feats, train_label = self.get_data('./dataset_train')
        x_train = [train_leads, train_feats]
        y_train = train_label
        print(len(x_train))
        test_leads, test_feats, test_label = self.get_data('./dataset_test')
        x_test = [test_leads, test_feats]
        y_test = test_label
        print(len(x_test))
        self.cat_features = int(test_feats.shape[1])
        self.n_classes = int(y_test.shape[1])

        return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    dataset = data_set(load_n = -1, shuffle = True, return_joined = True, n_samples = 8192)
    dataset.generate()