# coding: utf-8

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
import os
from copy import deepcopy
import math


class MeAIM(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MeAIM, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']   #64
        self.feat_embed_dim = config['feat_embed_dim']   #64
        self.cf_model = config['cf_model']     #lightgcn
        self.n_vc_layer = config['n_vc_layers']   #
        self.n_tc_layer = config['n_tc_layers']   #
        self.n_knn_layers = config['n_knn_layers']
        self.n_ui_layers = config['n_ui_layers']  #[2] the layers of cge
        self.tau = config['tau']
        self.beta = config['beta']  #[1e-04]
        self.lamda = config['lamda']
        self.knn_k = config['knn_k']
        self.alpha = config['alpha']
        self.drop_rate = config['drop_rate']
        self.gamma = config['gamma']
        self.use_knn = config['use_knn']

        self.n_nodes = self.n_users + self.n_items
        

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.adj = self.scipy_matrix_to_sparse_tensor(self.interaction_matrix, torch.Size((self.n_users, self.n_items)))
        self.num_inters, self.norm_adj = self.get_norm_adj_mat()   #self.num_inters: degree; self.norm_adj: normalization adjacency matrix
        #self.num_inters = torch.FloatTensor(1.0 / (self.num_inters + 1e-7)).to(self.device)   
        
        self.norm_inter_adj = self.get_norm_inter_adj_mat()
        
        # init user and item ID embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)   #init userID embeddings
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)   #init itemID embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_freedomdsp_{}_{}.pt'.format(self.knn_k, int(10*self.alpha)))
        v_adj_file = os.path.join(dataset_path, 'v_adj_freedomdsp_{}.pt'.format(self.knn_k))
        t_adj_file = os.path.join(dataset_path, 't_adj_freedomdsp_{}.pt'.format(self.knn_k))

        # load item modal features and define hyperedges embeddings
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)   #original visual embeddings  
            self.item_image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim, bias=False)
            self.item_image_trs2 = nn.Linear(self.feat_embed_dim, self.feat_embed_dim, bias=True)
            self.v_meta_weights = VMetaWeightNet(self.feat_embed_dim, self.drop_rate)
            
            indices, self.image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
            torch.save(self.image_adj, v_adj_file)
           
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)    #original text embeddings
            self.item_text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim, bias=False)
            self.item_text_trs2 = nn.Linear(self.feat_embed_dim, self.feat_embed_dim, bias=True)
            self.t_meta_weights = TMetaWeightNet(self.feat_embed_dim, self.drop_rate)
            
            indices, self.text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
            torch.save(self.text_adj, t_adj_file)
            
        self.c_meta_weights = CMetaWeightNet(self.feat_embed_dim, self.drop_rate)
            
        if self.v_feat is not None and self.t_feat is not None:
            self.mm_adj = self.alpha * self.image_adj + (1.0 - self.alpha) * self.text_adj
            torch.save(self.mm_adj, mm_adj_file)
            
        self.meta_weights = MetaWeightNet(self.feat_embed_dim, self.drop_rate)
        
        #################
        
        self.gate_image_prefer = nn.Sequential(
            nn.Linear(self.feat_embed_dim, self.feat_embed_dim),
            nn.Sigmoid()
        )

        self.gate_text_prefer = nn.Sequential(
            nn.Linear(self.feat_embed_dim, self.feat_embed_dim),
            nn.Sigmoid()
        )
        
        self.softmax = nn.Softmax(dim=-1)
        ##################################
        
  
                
    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
 
        adj_size = sim.size()
        del sim
     
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
      
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)
        
        
    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        
        return torch.sparse.FloatTensor(indices, values, adj_size)
                
 
        
            
    def scipy_matrix_to_sparse_tensor(self, matrix, shape):
        row = matrix.row
        col = matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(matrix.data)
        return torch.sparse.FloatTensor(i, data, shape).to(self.device)
    
    
    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        return sumArr, self.scipy_matrix_to_sparse_tensor(L, torch.Size((self.n_nodes, self.n_nodes)))
        
        
    def get_norm_inter_adj_mat(self):
        A = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        data_dict = dict(zip(zip(inter_M.row, inter_M.col), [1] * inter_M.nnz))
        A._update(data_dict)
        
        sumArr1 = (A > 0).sum(axis=1)
        diag1 = np.array(sumArr1.flatten())[0] + 1e-7
        diag1 = np.power(diag1, -0.5)
        D1 = sp.diags(diag1)
        
        sumArr2 = (A > 0).sum(axis=0)
        diag2 = np.array(sumArr2.flatten())[0] + 1e-7
        diag2 = np.power(diag2, -0.5)
        D2 = sp.diags(diag2)
        
        L = D1 * A * D2
        L = sp.coo_matrix(L)
        return self.scipy_matrix_to_sparse_tensor(L, torch.Size((self.n_users, self.n_items)))
        
        
    

        
        
        
    # collaborative graph embedding
    def cge(self):
        if self.cf_model == 'mf':
            cge_embs = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        if self.cf_model == 'lightgcn':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            for _ in range(self.n_ui_layers):
                ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings) 

            cge_embs = ego_embeddings
            
        user_cge_embs, item_cge_embs = torch.split(cge_embs, [self.n_users, self.n_items], dim=0)
        
        return self.user_embedding.weight, self.item_id_embedding.weight, user_cge_embs, item_cge_embs
        
  
        
        
    
    # modality graph embedding
    def mge(self, str='v'):
        if str == 'v':
            item_feats = self.item_image_trs(self.image_embedding.weight) 
            item_feats = F.normalize(item_feats)
            item_feats = torch.mul(self.item_id_embedding.weight, torch.sigmoid(self.item_image_trs2(item_feats)))
        elif str == 't':
            item_feats = self.item_text_trs(self.text_embedding.weight)
            item_feats = F.normalize(item_feats)
            item_feats = torch.mul(self.item_id_embedding.weight, torch.sigmoid(self.item_text_trs2(item_feats)))

        
        user_feats = torch.sparse.mm(self.norm_inter_adj, item_feats)

        mge_feats = torch.cat([user_feats, item_feats], dim=0)
        
        if str == 'v':
            for _ in range(self.n_vc_layer):
                mge_feats = torch.sparse.mm(self.norm_adj, mge_feats) 
        elif str == 't': 
            for _ in range(self.n_tc_layer):
                mge_feats = torch.sparse.mm(self.norm_adj, mge_feats) 
          
        user_mge_embs, item_mge_embs = torch.split(mge_feats, [self.n_users, self.n_items], dim=0)
        
        return user_feats, item_feats, user_mge_embs, item_mge_embs

        
        
    def compute_score(self, uv_embs, ut_embs, uc_embs, iv_embs, it_embs, ic_embs):
    
        i_embs = torch.cat([iv_embs.T.unsqueeze(2), it_embs.T.unsqueeze(2), ic_embs.T.unsqueeze(2)], dim=2) 
        
        ##################################################
    
        v_weights = self.v_meta_weights(uv_embs, iv_embs, it_embs, ic_embs) #[7, 3]   
        iiv_embs = torch.sum(i_embs * v_weights, dim=2).T  #[7, 64]
        
        vv_scores = torch.sum(torch.mul(uv_embs, iv_embs), dim=1)
        vt_scores = torch.sum(torch.mul(uv_embs, it_embs), dim=1)
        vc_scores = torch.sum(torch.mul(uv_embs, ic_embs), dim=1)
        
        
        v_cat_scores = torch.cat([vv_scores.unsqueeze(1), vt_scores.unsqueeze(1), vc_scores.unsqueeze(1)], dim=1)  
        v_scores = torch.sum(torch.mul(v_cat_scores, v_weights), dim=1).unsqueeze(1)  #[7, 1]
        
        ############################################
        
        t_weights = self.t_meta_weights(ut_embs, iv_embs, it_embs, ic_embs) #[7, 3]   
        iit_embs = torch.sum(i_embs * t_weights, dim=2).T  #[7, 64]
        
        tv_scores = torch.sum(torch.mul(ut_embs, iv_embs), dim=1)
        tt_scores = torch.sum(torch.mul(ut_embs, it_embs), dim=1)
        tc_scores = torch.sum(torch.mul(ut_embs, ic_embs), dim=1)
        
        t_cat_scores = torch.cat([tv_scores.unsqueeze(1), tt_scores.unsqueeze(1), tc_scores.unsqueeze(1)], dim=1)  
        t_scores = torch.sum(torch.mul(t_cat_scores, t_weights), dim=1).unsqueeze(1)  #[7, 1]
        
        #############################################
        
        c_weights = self.c_meta_weights(uc_embs, iv_embs, it_embs, ic_embs) #[7, 3]   
        iic_embs = torch.sum(i_embs * c_weights, dim=2).T  #[7, 64]
        
        cv_scores = torch.sum(torch.mul(uc_embs, iv_embs), dim=1)
        ct_scores = torch.sum(torch.mul(uc_embs, it_embs), dim=1)
        cc_scores = torch.sum(torch.mul(uc_embs, ic_embs), dim=1)
        
        c_cat_scores = torch.cat([cv_scores.unsqueeze(1), ct_scores.unsqueeze(1), cc_scores.unsqueeze(1)], dim=1)  
        c_scores = torch.sum(torch.mul(c_cat_scores, c_weights), dim=1).unsqueeze(1)  #[7, 1]
        
        #####################################################
        weights = self.meta_weights(uv_embs, ut_embs, uc_embs, iiv_embs, iit_embs, iic_embs) 
        scores = torch.cat([v_scores, t_scores, c_scores], dim=1)  #[1,4]
        scores = torch.sum(torch.mul(scores, weights), dim=1).unsqueeze(0)
        
        return scores, [v_scores, t_scores, c_scores, weights]
        
        
        
        
        
    def forward(self):
        
        uv_initial_embs, iv_initial_embs, uv_mge_embs, iv_mge_embs = self.mge('v')
        ut_initial_embs, it_initial_embs, ut_mge_embs, it_mge_embs = self.mge('t')
        u_initial_embs, i_initial_embs, u_cge_embs, i_cge_embs = self.cge()
        
        if self.use_knn:
            for i in range(self.n_knn_layers):
                i_knn_embs = torch.sparse.mm(self.mm_adj, i_initial_embs)
                iv_knn_embs = torch.sparse.mm(self.mm_adj, iv_initial_embs)
                it_knn_embs = torch.sparse.mm(self.mm_adj, it_initial_embs)
                
                i_cge_embs = i_cge_embs + i_knn_embs
                iv_mge_embs = iv_mge_embs + iv_knn_embs
                it_mge_embs = it_mge_embs + it_knn_embs
                
        #####
        i_image_prefer = self.gate_image_prefer(i_cge_embs)
        i_text_prefer = self.gate_text_prefer(i_cge_embs)
        iv_mge_embs = torch.multiply(i_image_prefer, iv_mge_embs)
        it_mge_embs = torch.multiply(i_text_prefer, it_mge_embs)
        ##
        u_image_prefer = self.gate_image_prefer(u_cge_embs)
        u_text_prefer = self.gate_text_prefer(u_cge_embs)
        uv_mge_embs = torch.multiply(u_image_prefer, uv_mge_embs)
        ut_mge_embs = torch.multiply(u_text_prefer, ut_mge_embs)
        #####
        
        #torch.save(uv_mge_embs, 'uv_mge_embs0.pt')
        #torch.save(ut_mge_embs, 'ut_mge_embs0.pt')
        #torch.save(u_cge_embs, 'u_cge_embs0.pt')
        
        #torch.save(iv_mge_embs, 'iv_mge_embs0.pt')
        #torch.save(it_mge_embs, 'it_mge_embs0.pt')
        #torch.save(i_cge_embs, 'i_cge_embs0.pt')
                
        return u_cge_embs, i_cge_embs, uv_mge_embs, iv_mge_embs, ut_mge_embs, it_mge_embs
        
        
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return bpr_loss
    
    
    def ssl_triple_loss(self, emb1, emb2, all_emb):
        norm_emb1 = F.normalize(emb1)
        norm_emb2 = F.normalize(emb2)
        norm_all_emb = F.normalize(all_emb)
        pos_score = torch.exp(torch.mul(norm_emb1, norm_emb2).sum(dim=1) / self.tau)
        ttl_score = torch.exp(torch.matmul(norm_emb1, norm_all_emb.T) / self.tau).sum(dim=1)
        ssl_loss = -torch.log(pos_score / ttl_score).mean()
        return ssl_loss
    
    
    def reg_loss(self, *embs):
        reg_loss = 0
        for emb in embs:
            reg_loss += torch.norm(emb, p=2)
        reg_loss /= embs[-1].shape[0]
        return reg_loss
      
        
    def KL(self, p1, p2):
        return p1 * torch.log(p1) - p1 * torch.log(p2) + (1 - p1) * torch.log(1 - p1) - (1 - p1) * torch.log(1 - p2)


    def calculate_loss(self, interaction):
        
        u_cge_embs, i_cge_embs, uv_mge_embs, iv_mge_embs, ut_mge_embs, it_mge_embs = self.forward()
        
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        uv_embs = uv_mge_embs[users]
        pos_iv_embs = iv_mge_embs[pos_items]
        neg_iv_embs = iv_mge_embs[neg_items] 
        
        ut_embs = ut_mge_embs[users]
        pos_it_embs = it_mge_embs[pos_items]
        neg_it_embs = it_mge_embs[neg_items]
        
        uc_embs = u_cge_embs[users] 
        pos_ic_embs = i_cge_embs[pos_items] 
        neg_ic_embs = i_cge_embs[neg_items] 
        ###################################################
        
        pos_scores, [pos_v_scores, pos_t_scores, pos_c_scores, pos_weights] = self.compute_score(uv_embs, ut_embs, uc_embs, pos_iv_embs, pos_it_embs, pos_ic_embs)
        neg_scores, [neg_v_scores, neg_t_scores, neg_c_scores, neg_weights] = self.compute_score(uv_embs, ut_embs, uc_embs, neg_iv_embs, neg_it_embs, neg_ic_embs)
        
        ########################
        
        batch_v_bpr_loss = -F.logsigmoid(pos_v_scores - neg_v_scores)
        batch_t_bpr_loss = -F.logsigmoid(pos_t_scores - neg_t_scores)
        batch_c_bpr_loss = -F.logsigmoid(pos_c_scores - neg_c_scores)
        batch_vtc_bpr_loss = torch.cat((batch_v_bpr_loss, batch_t_bpr_loss, batch_c_bpr_loss), dim=1)
        weights = (pos_weights + neg_weights)/2
        batch_vtc_bpr_loss = torch.sum(torch.mul(batch_vtc_bpr_loss, weights), dim=1)
      
        batch_bpr_loss = self.gamma * batch_vtc_bpr_loss + (1 - self.gamma) * batch_c_bpr_loss.squeeze(1)
  
        batch_bpr_loss = torch.mean(batch_bpr_loss)
       
        ##########################
        

        batch_cl_loss = self.ssl_triple_loss(uc_embs, uv_embs, uv_mge_embs) + self.ssl_triple_loss(uc_embs, ut_embs, ut_mge_embs) + \
                        self.ssl_triple_loss(pos_ic_embs, pos_iv_embs, iv_mge_embs) + self.ssl_triple_loss(pos_ic_embs, pos_it_embs, it_mge_embs) + \
                        self.ssl_triple_loss(uv_embs, ut_embs, ut_mge_embs) + self.ssl_triple_loss(pos_iv_embs, pos_it_embs, it_mge_embs)
                                  
        ################################### KL Loss ######################################
        mm_item_embs = self.alpha*iv_mge_embs + (1-self.alpha)*it_mge_embs
        uid_imm_scores = torch.sigmoid(torch.mm(uc_embs, F.normalize(mm_item_embs).T))
        uid_iid_scores = torch.sigmoid(torch.mm(uc_embs, F.normalize(i_cge_embs).T))
        
        kl_loss = torch.mean(self.KL(uid_imm_scores, uid_iid_scores) + self.KL(uid_iid_scores, uid_imm_scores))
        
        #########################################################################
       
        loss = batch_bpr_loss + self.beta * batch_cl_loss + self.lamda * kl_loss
        

        return loss
        

    def full_sort_predict(self, interaction):
        
        u_cge_embs, i_cge_embs, uv_mge_embs, iv_mge_embs, ut_mge_embs, it_mge_embs = self.forward()
        
        user = interaction[0]
        user_tensor = user.unsqueeze(1).repeat(1, self.n_items).flatten()  #[7219200]
        item_tensor = torch.Tensor(list(range(self.n_items))).unsqueeze(1).repeat(1, len(user)).T.flatten().long()
        
        test_uv_embs = uv_mge_embs[user_tensor]
        test_ut_embs = ut_mge_embs[user_tensor]
        test_uc_embs = u_cge_embs[user_tensor]
        
        test_iv_embs = iv_mge_embs[item_tensor]
        test_it_embs = it_mge_embs[item_tensor]
        test_ic_embs = i_cge_embs[item_tensor]
        
        
        scores, _ = self.compute_score(test_uv_embs, test_ut_embs, test_uc_embs, test_iv_embs, test_it_embs, test_ic_embs)
        
        scores = scores.reshape(len(user), self.n_items)
        
        return scores
        
        
        
        
class VMetaWeightNet(nn.Module):
    def __init__(self, feat_embed_dim, drop_rate):
        super(VMetaWeightNet, self).__init__()
        
        self.feat_embed_dim = feat_embed_dim
        self.dropout = nn.Dropout(drop_rate)
        
        self.vv_weight_layer1 = nn.Linear(self.feat_embed_dim*3, int((self.feat_embed_dim*3)/2))
        self.vv_weight_layer2 = nn.Linear(int((self.feat_embed_dim*3)/2), 1)
        
        self.vt_weight_layer1 = nn.Linear(self.feat_embed_dim*3, int((self.feat_embed_dim*3)/2))
        self.vt_weight_layer2 = nn.Linear(int((self.feat_embed_dim*3)/2), 1)
        
        self.vc_weight_layer1 = nn.Linear(self.feat_embed_dim*3, int((self.feat_embed_dim*3)/2))
        self.vc_weight_layer2 = nn.Linear(int((self.feat_embed_dim*3)/2), 1)
    
    def forward(self, u_embs, iv_embs, it_embs, ic_embs, vars_dict=None):
    
        uv_loss = F.mse_loss(u_embs, iv_embs, reduction='none').sum(1)  
        ut_loss = F.mse_loss(u_embs, it_embs, reduction='none').sum(1)
        uc_loss = F.mse_loss(u_embs, ic_embs, reduction='none').sum(1)
        
        uv_loss = uv_loss.unsqueeze(1).repeat(1, self.feat_embed_dim)   
        ut_loss = ut_loss.unsqueeze(1).repeat(1, self.feat_embed_dim)
        uc_loss = uc_loss.unsqueeze(1).repeat(1, self.feat_embed_dim)
              
        v_meta_knowledge = torch.cat([uv_loss, u_embs, iv_embs], dim=1)  
        t_meta_knowledge = torch.cat([ut_loss, u_embs, it_embs], dim=1) 
        c_meta_knowledge = torch.cat([uc_loss, u_embs, ic_embs], dim=1)  
        
        v_scale = v_meta_knowledge.shape[1]
        t_scale = t_meta_knowledge.shape[1]
        c_scale = c_meta_knowledge.shape[1]
        
        
        v_meta_knowledge1 = self.dropout(self.vv_weight_layer1(v_meta_knowledge))
        v_meta_weights = torch.sigmoid(v_scale * self.dropout(self.vv_weight_layer2(v_meta_knowledge1)))
        
        t_meta_knowledge1 = self.dropout(self.vt_weight_layer1(t_meta_knowledge))
        t_meta_weights = torch.sigmoid(t_scale * self.dropout(self.vt_weight_layer2(t_meta_knowledge1)))
        
        c_meta_knowledge1 = self.dropout(self.vc_weight_layer1(c_meta_knowledge))
        c_meta_weights = torch.sigmoid(c_scale * self.dropout(self.vc_weight_layer2(c_meta_knowledge1)))
        
        meta_weights = torch.cat([v_meta_weights, t_meta_weights, c_meta_weights], dim=1)  
        meta_weights = F.softmax(meta_weights, dim=1)   
        
        return meta_weights

    
        
             
  
class TMetaWeightNet(nn.Module):
    def __init__(self, feat_embed_dim, drop_rate):
        super(TMetaWeightNet, self).__init__()
        
        self.feat_embed_dim = feat_embed_dim
        self.dropout = nn.Dropout(drop_rate)
        
        self.tv_weight_layer1 = nn.Linear(self.feat_embed_dim*3, int((self.feat_embed_dim*3)/2))
        self.tv_weight_layer2 = nn.Linear(int((self.feat_embed_dim*3)/2), 1)
        
        self.tt_weight_layer1 = nn.Linear(self.feat_embed_dim*3, int((self.feat_embed_dim*3)/2))
        self.tt_weight_layer2 = nn.Linear(int((self.feat_embed_dim*3)/2), 1)
        
        self.tc_weight_layer1 = nn.Linear(self.feat_embed_dim*3, int((self.feat_embed_dim*3)/2))
        self.tc_weight_layer2 = nn.Linear(int((self.feat_embed_dim*3)/2), 1)
    
    def forward(self, u_embs, iv_embs, it_embs, ic_embs, vars_dict=None):
    
        uv_loss = F.mse_loss(u_embs, iv_embs, reduction='none').sum(1)  
        ut_loss = F.mse_loss(u_embs, it_embs, reduction='none').sum(1)
        uc_loss = F.mse_loss(u_embs, ic_embs, reduction='none').sum(1)
        
        uv_loss = uv_loss.unsqueeze(1).repeat(1, self.feat_embed_dim)   
        ut_loss = ut_loss.unsqueeze(1).repeat(1, self.feat_embed_dim)
        uc_loss = uc_loss.unsqueeze(1).repeat(1, self.feat_embed_dim)
              
        v_meta_knowledge = torch.cat([uv_loss, u_embs, iv_embs], dim=1)  
        t_meta_knowledge = torch.cat([ut_loss, u_embs, it_embs], dim=1)  
        c_meta_knowledge = torch.cat([uc_loss, u_embs, ic_embs], dim=1)  
        
        v_scale = v_meta_knowledge.shape[1]
        t_scale = t_meta_knowledge.shape[1]
        c_scale = c_meta_knowledge.shape[1]
        

        v_meta_knowledge1 = self.dropout(self.tv_weight_layer1(v_meta_knowledge))
        v_meta_weights = torch.sigmoid(v_scale * self.dropout(self.tv_weight_layer2(v_meta_knowledge1)))
        
        t_meta_knowledge1 = self.dropout(self.tt_weight_layer1(t_meta_knowledge))
        t_meta_weights = torch.sigmoid(t_scale * self.dropout(self.tt_weight_layer2(t_meta_knowledge1)))
        
        c_meta_knowledge1 = self.dropout(self.tc_weight_layer1(c_meta_knowledge))
        c_meta_weights = torch.sigmoid(c_scale * self.dropout(self.tc_weight_layer2(c_meta_knowledge1)))
          
        meta_weights = torch.cat([v_meta_weights, t_meta_weights, c_meta_weights], dim=1)  
        meta_weights = F.softmax(meta_weights, dim=1)  
        
        return meta_weights





class CMetaWeightNet(nn.Module):
    def __init__(self, feat_embed_dim, drop_rate):
        super(CMetaWeightNet, self).__init__()
        
        self.feat_embed_dim = feat_embed_dim
        self.dropout = nn.Dropout(drop_rate)
        
        self.cv_weight_layer1 = nn.Linear(self.feat_embed_dim*3, int((self.feat_embed_dim*3)/2))
        self.cv_weight_layer2 = nn.Linear(int((self.feat_embed_dim*3)/2), 1)
        
        self.ct_weight_layer1 = nn.Linear(self.feat_embed_dim*3, int((self.feat_embed_dim*3)/2))
        self.ct_weight_layer2 = nn.Linear(int((self.feat_embed_dim*3)/2), 1)
        
        self.cc_weight_layer1 = nn.Linear(self.feat_embed_dim*3, int((self.feat_embed_dim*3)/2))
        self.cc_weight_layer2 = nn.Linear(int((self.feat_embed_dim*3)/2), 1)
    
    def forward(self, u_embs, iv_embs, it_embs, ic_embs, vars_dict=None):
    
        uv_loss = F.mse_loss(u_embs, iv_embs, reduction='none').sum(1)  
        ut_loss = F.mse_loss(u_embs, it_embs, reduction='none').sum(1)
        uc_loss = F.mse_loss(u_embs, ic_embs, reduction='none').sum(1)
        
        uv_loss = uv_loss.unsqueeze(1).repeat(1, self.feat_embed_dim)   
        ut_loss = ut_loss.unsqueeze(1).repeat(1, self.feat_embed_dim)
        uc_loss = uc_loss.unsqueeze(1).repeat(1, self.feat_embed_dim)
              
        v_meta_knowledge = torch.cat([uv_loss, u_embs, iv_embs], dim=1) 
        t_meta_knowledge = torch.cat([ut_loss, u_embs, it_embs], dim=1)  
        c_meta_knowledge = torch.cat([uc_loss, u_embs, ic_embs], dim=1)  
        
        v_scale = v_meta_knowledge.shape[1]
        t_scale = t_meta_knowledge.shape[1]
        c_scale = c_meta_knowledge.shape[1]
        
  
        v_meta_knowledge1 = self.dropout(self.cv_weight_layer1(v_meta_knowledge))
        v_meta_weights = torch.sigmoid(v_scale * self.dropout(self.cv_weight_layer2(v_meta_knowledge1)))
        
        t_meta_knowledge1 = self.dropout(self.ct_weight_layer1(t_meta_knowledge))
        t_meta_weights = torch.sigmoid(t_scale * self.dropout(self.ct_weight_layer2(t_meta_knowledge1)))
        
        c_meta_knowledge1 = self.dropout(self.cc_weight_layer1(c_meta_knowledge))
        c_meta_weights = torch.sigmoid(c_scale * self.dropout(self.cc_weight_layer2(c_meta_knowledge1)))
             
        meta_weights = torch.cat([v_meta_weights, t_meta_weights, c_meta_weights], dim=1) 
        meta_weights = F.softmax(meta_weights, dim=1)   
        
        return meta_weights
        
        
        
   
class MetaWeightNet(nn.Module):
    def __init__(self, feat_embed_dim, drop_rate):
        super(MetaWeightNet, self).__init__()
        
        self.feat_embed_dim = feat_embed_dim
        self.dropout = nn.Dropout(drop_rate)

        self.v_weight_layer1 = nn.Linear(self.feat_embed_dim*3, int((self.feat_embed_dim*3)/2))
        self.v_weight_layer2 = nn.Linear(int((self.feat_embed_dim*3)/2), 1)
        
        self.t_weight_layer1 = nn.Linear(self.feat_embed_dim*3, int((self.feat_embed_dim*3)/2))
        self.t_weight_layer2 = nn.Linear(int((self.feat_embed_dim*3)/2), 1)
        
        self.c_weight_layer1 = nn.Linear(self.feat_embed_dim*3, int((self.feat_embed_dim*3)/2))
        self.c_weight_layer2 = nn.Linear(int((self.feat_embed_dim*3)/2), 1)
    
    def forward(self, uv_embs, ut_embs, uc_embs, iv_embs, it_embs, ic_embs, vars_dict=None):
    
        uv_loss = F.mse_loss(uv_embs, iv_embs, reduction='none').sum(1)  
        ut_loss = F.mse_loss(ut_embs, it_embs, reduction='none').sum(1)
        uc_loss = F.mse_loss(uc_embs, ic_embs, reduction='none').sum(1)
        
        uv_loss = uv_loss.unsqueeze(1).repeat(1, self.feat_embed_dim)   
        ut_loss = ut_loss.unsqueeze(1).repeat(1, self.feat_embed_dim)
        uc_loss = uc_loss.unsqueeze(1).repeat(1, self.feat_embed_dim)
              
        v_meta_knowledge = torch.cat([uv_loss, uv_embs, iv_embs], dim=1)  
        t_meta_knowledge = torch.cat([ut_loss, ut_embs, it_embs], dim=1)  
        c_meta_knowledge = torch.cat([uc_loss, uc_embs, ic_embs], dim=1)  
        
        v_scale = v_meta_knowledge.shape[1]
        t_scale = t_meta_knowledge.shape[1]
        c_scale = c_meta_knowledge.shape[1]
        
       
        v_meta_knowledge1 = self.dropout(self.v_weight_layer1(v_meta_knowledge))
        v_meta_weights = torch.sigmoid(v_scale * self.dropout(self.v_weight_layer2(v_meta_knowledge1)))
        
        t_meta_knowledge1 = self.dropout(self.t_weight_layer1(t_meta_knowledge))
        t_meta_weights = torch.sigmoid(t_scale * self.dropout(self.t_weight_layer2(t_meta_knowledge1)))
        
        c_meta_knowledge1 = self.dropout(self.c_weight_layer1(c_meta_knowledge))
        c_meta_weights = torch.sigmoid(c_scale * self.dropout(self.c_weight_layer2(c_meta_knowledge1)))
        
        meta_weights = torch.cat([v_meta_weights, t_meta_weights, c_meta_weights], dim=1)  
        meta_weights = F.softmax(meta_weights, dim=1)   
        
        return meta_weights


        
        
       
