import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_X_y

class ClusterUnderSampler(ClassifierMixin, BaseEstimator):
    def __init__(self, clustering_model=None,  random_sample_seed=None, 
                 n_clusters = None, reduction_percentage_class=None, **clustering_params  ):
        self.clustering_model = clustering_model
        self.random_sample_seed = random_sample_seed
        self._rng = np.random.RandomState(random_sample_seed)
        self.n_clusters=n_clusters
        self.reduction_percentage_class = reduction_percentage_class
        self.clustering_params = clustering_params

    def fit_resample(self, X, y):    
        X = check_array(X)

        if y is not None:
            self.minority_class_ , self.non_minority_classes_ = self._identify_classes(y)
            self.minority_class_index_ = np.where(y==self.minority_class_)[0]
            self.non_minority_classes_index_ = [np.where(y==non_min_class_)[0] for non_min_class_ in self.non_minority_classes_  ]
        
        reduction_percentages = dict(zip(self.non_minority_classes_, [self.reduction_percentage_class]))
        self.under_sample_index_ = [self.minority_class_index_]
        
        for non_minority_class_index, cls in zip(self.non_minority_classes_index_, self.non_minority_classes_):
            # Separate the non_minority class
            X_non_minority = X[non_minority_class_index]

            # Calculate number of samples for each non-mionority class
            reduction_percentage = reduction_percentages[cls]
            if reduction_percentage < 1:
                sample_size = len(X_non_minority) * reduction_percentage
        
                # Cluster and label the non minority class
                cluster_labels = self._fit_cluster(X_non_minority)
                samples_index = self._stratified_sample_combination(cluster_labels, sample_size)

                # Sampled index are referenced to class, and not to the original data, we map that again
                self.under_sample_index_.append(non_minority_class_index[samples_index])

            else:
                # Sampled index are referenced to class, and not to the original data, we map that again
                self.under_sample_index_.append(non_minority_class_index)

        self.under_sample_index_ = np.concat(self.under_sample_index_)

        return X[self.under_sample_index_], y[self.under_sample_index_]

    def _get_unique_values_and_ids(self, X):
        unique_rows, ids = np.unique(X, axis=0, return_inverse=True)
        return unique_rows, ids
    
    def _identify_classes(self, y):
        unique_classes, counts = np.unique_counts(y)
        minority_class = unique_classes[np.argmin(counts)]
        non_minority_classes = unique_classes[np.argsort(counts)[::-1][:-1]]   #return sorted majority class labels, first element correspon to the most representative class
        return minority_class, non_minority_classes

    def _split_X(self, X):
        Xdiv1, Xdiv2 = np.split(X,2,axis=1)
        return Xdiv1, Xdiv2 
    
    def _fit_cluster(self, X):
        _, X_2 = self._split_X(X) 

        unique_X_2,unique_X_2_ids = self._get_unique_values_and_ids(X_2)

        if isinstance(self.clustering_model, type):
        # clustering_model is a class, instantiate it
            clustering = self.clustering_model(n_clusters = self.n_clusters, **self.clustering_params)
        else:
        # clustering_model is already an instance
            clustering = self.clustering_model

        # Predict cluster on compounds combination X_2        
        self.labels = clustering.fit_predict(unique_X_2)

        id_to_label = dict(enumerate(self.labels))
        X_2_label = [id_to_label[id] for id in unique_X_2_ids]

        return X_2_label

    def _stratified_sample_combination(self, compound_label, m_r):
        
        #get cluster_ids and counts in combination
        cluster_id, cluster_count  = np.unique_counts(compound_label)
        #Calculate compound cluster frequency on combination
        cluster_comn_freq = cluster_count / len(compound_label)
        
        samples_index = []
        # uniform sample
        for  id, freq in zip(cluster_id, cluster_comn_freq):
            # Calculate number of samples per cluster in combination
            number_sample = int(m_r * freq)

            cluster_comb_group = np.where(compound_label == id)[0]
            cluster_sample = self._rng.choice(cluster_comb_group, size=number_sample, replace=False,)
            samples_index.append(cluster_sample)

        return np.concat(samples_index)