import os
import torch
import metric
import numpy as np
import scipy.sparse as sp
from sklearn.mixture import GaussianMixture
from sparsesvd import sparsesvd


class HiGSP(object):
    def __init__(self, init_adj_mat,
                 alpha1=0.5, alpha2=0.5, order1=2, order2=2, n_clusters=20, pri_factor=256,
                 user_interacted_items_dict=None, topks=[5, 10, 20]):
        self.adj_mat = init_adj_mat

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.order1 = order1
        self.order2 = order2
        self.n_clusters = n_clusters
        self.pri_factor = pri_factor

        self.user_interacted_items_dict = user_interacted_items_dict
        self.topks = topks

    def normalize_adj_mat(self, adj_mat):
        adj_mat = sp.csr_matrix(adj_mat)
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        d_inv = 1.0 / d_inv
        d_inv[np.isinf(d_inv)] = 0.
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.A
        return norm_adj

    def normalize_adj_mat_sp(self, adj_mat):
        adj_mat = sp.csr_matrix(adj_mat)
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        d_mat_i = d_mat
        d_inv = 1.0 / d_inv
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_i_inv = sp.diags(d_inv)
        norm_adj = norm_adj.dot(d_mat)
        return norm_adj, d_mat_i, d_mat_i_inv

    def bmatpow(self, mat, order):
        with torch.no_grad():
            R = torch.FloatTensor(mat)
            mat = torch.FloatTensor(mat)
            I = torch.FloatTensor(np.expand_dims(np.identity(mat.shape[1]), axis=0).repeat(self.n_clusters, axis=0))
            if order == 1:
                return R.detach().numpy()
            for ord in range(2, order+1):
                R = torch.bmm(R.transpose(1, 2), mat)
            R = R.detach().numpy()
        return R

    def matpow(self, mat, order):
        with torch.no_grad():
            R = torch.FloatTensor(mat)
            mat = torch.FloatTensor(mat)
            if order == 1:
                return R.detach().numpy()
            for ord in range(2, order+1):
                R = torch.matmul(R.transpose(0, 1), mat)
            R = R.detach().numpy()
        return R

    def construct_cluster_wise_filter(self, adj_mat):
        # Cluster users based on their interactions
        clustering = GaussianMixture(n_components=self.n_clusters)
        cluster_labels = clustering.fit_predict(adj_mat)
        n_clusters = len(set(cluster_labels))
        C = np.zeros((n_clusters, adj_mat.shape[0], adj_mat.shape[1]))
        C[cluster_labels, [i for i in range(adj_mat.shape[0])], :] = adj_mat

        # Construct filters for each cluster
        A_tilde_list = []
        for i in range(n_clusters):
            adj_mat = C[i, :, :]
            C_tilde = adj_mat
            A_tilde = C_tilde.T @ C_tilde
            A_tilde = self.normalize_adj_mat(A_tilde)
            A_tilde = np.expand_dims(A_tilde, axis=0)
            A_tilde_list.append(A_tilde)
        A_tilde = np.concatenate(A_tilde_list, axis=0)
        L_tilde = np.expand_dims(np.identity(A_tilde.shape[1]), axis=0).repeat(self.n_clusters, axis=0) - A_tilde
        L_tilde_k = self.bmatpow(L_tilde, self.order1)
        local_filter = np.expand_dims(np.identity(A_tilde.shape[1]), axis=0).repeat(self.n_clusters, axis=0) - L_tilde_k
        return local_filter, cluster_labels

    def construct_global_aware_filter(self, adj_mat):
        # Construct ideal low-pass filter
        norm_adj, d_mat_i, d_mat_i_inv = self.normalize_adj_mat_sp(adj_mat)
        norm_adj = norm_adj.tocsc()
        ut, s, vt = sparsesvd(norm_adj, self.pri_factor)
        global_filter1 = d_mat_i @ vt.T @ vt @ d_mat_i_inv

        # Construct high-order low-pass filter
        R_tilde = self.normalize_adj_mat(adj_mat)
        P_tilde = R_tilde.T @ R_tilde
        L_tilde = np.identity(P_tilde.shape[1]) - P_tilde
        L_tilde_k = self.matpow(L_tilde, self.order2)
        global_filter2 = np.identity(P_tilde.shape[1]) - L_tilde_k
        return global_filter1, global_filter2

    def get_users_rating(self):
        ratings = 0.0

        # Predictions from cluster-wise filter
        n_clusters = len(set(self.item_cluster_labels))
        C = np.zeros((n_clusters, self.adj_mat.shape[0], self.adj_mat.shape[1]))
        C[self.item_cluster_labels, [i for i in range(self.adj_mat.shape[0])], :] = self.adj_mat
        with torch.no_grad():
            C = torch.FloatTensor(C)
            filter = torch.FloatTensor(self.item_cluster_filter)
            ratings += torch.bmm(C, filter)
            ratings = torch.sum(ratings, dim=0, keepdim=False)
            ratings = ratings.detach().numpy()
        # Predictions from ideal low-pass filter in globally-aware filter
        ratings += self.alpha1 * self.adj_mat @ self.global_filter1
        # Predictions from high-order low-pass filter in globally-aware filter
        ratings += self.alpha2 * self.adj_mat @ self.global_filter2
        return ratings

    def train(self):
        # Construct item-wise filter
        self.item_cluster_filter, self.item_cluster_labels = self.construct_cluster_wise_filter(self.adj_mat)
        # Construct globally-aware filter
        self.global_filter1, self.global_filter2 = self.construct_global_aware_filter(self.adj_mat)
        # Predict user future interactions
        ratings = self.get_users_rating()
        return ratings

    def eval_test(self, user_items_dict, ratings):
        users = list(user_items_dict.keys())
        user_prediction_items_list = []
        user_truth_items_list = []
        for user in users:
            rating = ratings[user, :]
            rating = torch.from_numpy(rating).view(1, -1)
            user_interacted_items = list(self.user_interacted_items_dict[user])
            rating[0, user_interacted_items] = -999999.0
            ranked_items = torch.topk(rating, k=max(self.topks))[1].numpy()[0]
            user_prediction_items_list.append(ranked_items)
            user_truth_items_list.append(user_items_dict[user])
        precisions, recalls, f1_scores, mrrs, ndcgs = metric.calculate_all(user_truth_items_list,
                                                                                      user_prediction_items_list,
                                                                                      self.topks)
        return f1_scores, mrrs, ndcgs
