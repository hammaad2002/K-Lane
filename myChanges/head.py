import torch
import torch.nn as nn

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x):
        return self.fn(self.norm(x))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
    
        self.net = nn.Sequential(
                                nn.Linear(dim, hidden_dim),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(hidden_dim, dim),
                                nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()

        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
                                    nn.Linear(inner_dim, dim),
                                    nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        qkv_list = []
        for t in qkv:

            print('t shape: ',t.shape)
            b, n, hd = t.shape
            h = self.heads
            d = hd // h

            t = t.view(b, n, h, d)
            t = t.permute(0, 2, 1, 3)
            qkv_list.append(t)
        
        q, k, v = qkv_list
        k_transposed = k.permute(0, 1, 3, 2)  # This is equivalent to k.transpose(-1, -2)

        # Perform the matrix multiplication and scaling
        dots = torch.matmul(q, k_transposed) * self.scale
        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)

        # Preparing each dimension's value
        b, h, n, d = out.shape
        hd = h * d

        # First, reshape the tensor
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(b, n, hd)

        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.layerAttention = nn.ModuleList([PreNorm(1024, Attention(1024, heads = 16, dim_head = 64, dropout = 0.0))])
        self.layerFeedForward = nn.ModuleList([PreNorm(1024, FeedForward(1024, 2048, dropout = 0.0))])

    def forward(self, x):

        x = self.layerAttention[0](x) + x
        x = self.layerFeedForward[0](x) + x
        return x

class RowSharNotReducRef(nn.Module):
    def __init__(self,
                dim_feat=8,
                row_size=144,
                dim_shared=512,
                lambda_cls=1.,
                thr_ext = 0.3,
                off_grid = 2,
                dim_token = 1024,
                tr_depth = 1,
                tr_heads = 16,
                tr_dim_head = 64,
                tr_mlp_dim = 2048,
                tr_dropout = 0.,
                tr_emb_dropout = 0.,
                is_reuse_same_network = False,
                conf_thr = 0.5,
                cls_lane_color = [(0, 0, 255),
                                  (0, 50, 255),
                                  (0, 255, 255),
                                  (0, 255, 0),
                                  (255, 0, 0),
                                  (255, 0, 100)]):
        super(RowSharNotReducRef, self).__init__()

        self.b_size = 0
        self.conf_thr = conf_thr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Making Labels
        self.num_cls = 6
        self.lambda_cls=lambda_cls

        self.ext_0 =  nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,2,1,1,0),
            )
        
        self.cls_0 = nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,row_size,1,1,0),
            )
        
        self.ext_1 =  nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,2,1,1,0),
            )
        
        self.cls_1 = nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,row_size,1,1,0),
            )
        
        self.ext_2 =  nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,2,1,1,0),
            )
        
        self.cls_2 = nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,row_size,1,1,0),
            )
        
        self.ext_3 =  nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,2,1,1,0),
            )
        
        self.cls_3 = nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,row_size,1,1,0),
            )
        
        self.ext_4 =  nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,2,1,1,0),
            )
        
        self.cls_4 = nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,row_size,1,1,0),
            )

        self.ext_5 =  nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,2,1,1,0),
            )
        
        self.cls_5 = nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,row_size,1,1,0),
            )

        # Refinement (2nd Stage)
        self.thr_ext = thr_ext
        self.off_grid = off_grid
        in_token_channel = (2*self.off_grid+1)*row_size*dim_feat
        
        self.to_token = nn.Linear(in_token_channel, dim_token)

        self.emb_0 = nn.Parameter(torch.randn(dim_token)).to(self.device)
        self.emb_1 = nn.Parameter(torch.randn(dim_token)).to(self.device)
        self.emb_2 = nn.Parameter(torch.randn(dim_token)).to(self.device)
        self.emb_3 = nn.Parameter(torch.randn(dim_token)).to(self.device)
        self.emb_4 = nn.Parameter(torch.randn(dim_token)).to(self.device)
        self.emb_5 = nn.Parameter(torch.randn(dim_token)).to(self.device)

        self.tr_lane_correlator = nn.Sequential(
            Transformer(),
            nn.LayerNorm(dim_token),
            nn.Linear(dim_token, in_token_channel),
        )

        self.ext2_0 =  nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,2,1,1,0),
            )
        
        self.cls2_0 = nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,row_size,1,1,0)
            )
        
        self.ext2_1 =  nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,2,1,1,0),
            )
        
        self.cls2_1 = nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,row_size,1,1,0)
            )
        
        self.ext2_2 =  nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,2,1,1,0),
            )
        
        self.cls2_2 = nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,row_size,1,1,0)
            )
        
        self.ext2_3 =  nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,2,1,1,0),
            )
        
        self.cls2_3 = nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,row_size,1,1,0)
            )
        
        self.ext2_4 =  nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,2,1,1,0),
            )
        
        self.cls2_4 = nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,row_size,1,1,0)
            )

        self.ext2_5 =  nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,2,1,1,0),
            )
        
        self.cls2_5 = nn.Sequential(
                nn.Conv1d(dim_feat*row_size, dim_shared,1,1,0),
                nn.BatchNorm1d(dim_shared),
                nn.Conv1d(dim_shared,row_size,1,1,0)
            )
        
        self.store_val_idx_h = 0

    def forward(self, x):

        b, _, _, _ = x.shape
        self.b_size = b

        out_ext = torch.zeros(self.num_cls, b, 2, 144)
        out_cls = torch.zeros(self.num_cls, b, 144, 144)

        out_ext_2 = torch.zeros(self.num_cls, b, 2, 144)
        out_cls_2 = torch.zeros(self.num_cls, b, 144, 144)

        row_feat = x
        row_feat = row_feat.permute(0, 1, 3, 2)
        row_tensor = row_feat.reshape(b, 8*144, 144)

        out_ext[0] = torch.nn.functional.softmax(self.ext_0(row_tensor), dim=2)
        out_cls[0] = torch.nn.functional.softmax(self.cls_0(row_tensor), dim=2)
        out_ext[1] = torch.nn.functional.softmax(self.ext_1(row_tensor), dim=2)
        out_cls[1] = torch.nn.functional.softmax(self.cls_1(row_tensor), dim=2)
        out_ext[2] = torch.nn.functional.softmax(self.ext_2(row_tensor), dim=2)
        out_cls[2] = torch.nn.functional.softmax(self.cls_2(row_tensor), dim=2)
        out_ext[3] = torch.nn.functional.softmax(self.ext_3(row_tensor), dim=2)
        out_cls[3] = torch.nn.functional.softmax(self.cls_3(row_tensor), dim=2)
        out_ext[4] = torch.nn.functional.softmax(self.ext_4(row_tensor), dim=2)
        out_cls[4] = torch.nn.functional.softmax(self.cls_4(row_tensor), dim=2)
        out_ext[5] = torch.nn.functional.softmax(self.ext_5(row_tensor), dim=2)
        out_cls[5] = torch.nn.functional.softmax(self.cls_5(row_tensor), dim=2)

        #1st Stage Processing
        b_size, _, _, _ = row_feat.shape
        dim_feat = 8
        image_height = 144

        # Zero padding for offset
        off_grid = self.off_grid
        zero_left = torch.zeros((b_size, dim_feat, image_height, off_grid)).to(self.device)
        zero_right = torch.zeros((b_size, dim_feat, image_height, off_grid)).to(self.device)
        row_feat_pad = torch.cat([zero_left, row_feat, zero_right], dim=3)

        # Initialize lists to store tokens and indexes for all batches
        ext_lane_tokens_list = []
        ext_corr_idxs_list = []

        # Iterate over batches
        for sample in range(b_size):

            #-------------------------------------------------FIRST LANE EXTENSION-------------------------------------------------

            # Calculate lane extension probabilities for all batches
            lane_ext_probs = torch.mean(out_cls[0][:,:,0][sample].unsqueeze(0), dim=1)

            # Checking if lane exists if its probability is above threshold
            if lane_ext_probs > self.thr_ext:

                corr_idxs_b4 = out_cls[0][sample,:,:]
                corr_idxs = torch.argmax(corr_idxs_b4, dim=1)

                ext_corr_idxs_list.append(corr_idxs)

                # Create temp_token for valid batches
                temp_token = torch.zeros((dim_feat, image_height, 1 + 2 * off_grid)).to(self.device)

                for idx_h in range(image_height):
                    corr_idx = corr_idxs[idx_h] + off_grid
                    temp_token[:, idx_h, :] = row_feat_pad[sample, :, idx_h, corr_idx-off_grid:corr_idx+off_grid+1]

                temp_token = self.to_token(temp_token.reshape(1, -1)) + self.emb_0
                temp_token = temp_token.squeeze()

                # Append temp_token to ext_lane_tokens_list
                ext_lane_tokens_list.append(torch.unsqueeze(torch.unsqueeze(temp_token, 0),0))

            #-------------------------------------------------SECOND LANE EXTENSION-------------------------------------------------

            # Calculate lane extension probabilities for all batches
            lane_ext_probs = torch.mean(out_cls[0][:,:,0][sample].unsqueeze(0), dim=1)

            # Checking if lane exists if its probability is above threshold
            if lane_ext_probs > self.thr_ext:

                corr_idxs_b4 = out_cls[0][sample,:,:]
                corr_idxs = torch.argmax(corr_idxs_b4, dim=1)

                ext_corr_idxs_list.append(corr_idxs)

                # Create temp_token for valid batches
                temp_token = torch.zeros((dim_feat, image_height, 1 + 2 * off_grid)).to(self.device)

                for idx_h in range(image_height):
                    corr_idx = corr_idxs[idx_h] + off_grid
                    temp_token[:, idx_h, :] = row_feat_pad[sample, :, idx_h, corr_idx-off_grid:corr_idx+off_grid+1]

                temp_token = self.to_token(temp_token.reshape(1, -1)) + self.emb_0
                temp_token = temp_token.squeeze()

                # Append temp_token to ext_lane_tokens_list
                ext_lane_tokens_list.append(torch.unsqueeze(torch.unsqueeze(temp_token, 0),0))

            #-------------------------------------------------THIRD LANE EXTENSION-------------------------------------------------

            # Calculate lane extension probabilities for all batches
            lane_ext_probs = torch.mean(out_cls[0][:,:,0][sample].unsqueeze(0), dim=1)

            # Checking if lane exists if its probability is above threshold
            if lane_ext_probs > self.thr_ext:

                corr_idxs_b4 = out_cls[0][sample,:,:]
                corr_idxs = torch.argmax(corr_idxs_b4, dim=1)

                ext_corr_idxs_list.append(corr_idxs)

                # Create temp_token for valid batches
                temp_token = torch.zeros((dim_feat, image_height, 1 + 2 * off_grid)).to(self.device)

                for idx_h in range(image_height):
                    corr_idx = corr_idxs[idx_h] + off_grid
                    temp_token[:, idx_h, :] = row_feat_pad[sample, :, idx_h, corr_idx-off_grid:corr_idx+off_grid+1]

                temp_token = self.to_token(temp_token.reshape(1, -1)) + self.emb_0
                temp_token = temp_token.squeeze()

                # Append temp_token to ext_lane_tokens_list
                ext_lane_tokens_list.append(torch.unsqueeze(torch.unsqueeze(temp_token, 0),0))

            #-------------------------------------------------FOURTH LANE EXTENSION-------------------------------------------------

            # Calculate lane extension probabilities for all batches
            lane_ext_probs = torch.mean(out_cls[0][:,:,0][sample].unsqueeze(0), dim=1)

            # Checking if lane exists if its probability is above threshold
            if lane_ext_probs > self.thr_ext:

                corr_idxs_b4 = out_cls[0][sample,:,:]
                corr_idxs = torch.argmax(corr_idxs_b4, dim=1)

                ext_corr_idxs_list.append(corr_idxs)

                # Create temp_token for valid batches
                temp_token = torch.zeros((dim_feat, image_height, 1 + 2 * off_grid)).to(self.device)

                for idx_h in range(image_height):
                    corr_idx = corr_idxs[idx_h] + off_grid
                    temp_token[:, idx_h, :] = row_feat_pad[sample, :, idx_h, corr_idx-off_grid:corr_idx+off_grid+1]

                temp_token = self.to_token(temp_token.reshape(1, -1)) + self.emb_0
                temp_token = temp_token.squeeze()

                # Append temp_token to ext_lane_tokens_list
                ext_lane_tokens_list.append(torch.unsqueeze(torch.unsqueeze(temp_token, 0),0))

            #-------------------------------------------------FIFTH LANE EXTENSION-------------------------------------------------

            # Calculate lane extension probabilities for all batches
            lane_ext_probs = torch.mean(out_cls[0][:,:,0][sample].unsqueeze(0), dim=1)

            # Checking if lane exists if its probability is above threshold
            if lane_ext_probs > self.thr_ext:

                corr_idxs_b4 = out_cls[0][sample,:,:]
                corr_idxs = torch.argmax(corr_idxs_b4, dim=1)

                ext_corr_idxs_list.append(corr_idxs)

                # Create temp_token for valid batches
                temp_token = torch.zeros((dim_feat, image_height, 1 + 2 * off_grid)).to(self.device)

                for idx_h in range(image_height):
                    corr_idx = corr_idxs[idx_h] + off_grid
                    temp_token[:, idx_h, :] = row_feat_pad[sample, :, idx_h, corr_idx-off_grid:corr_idx+off_grid+1]

                temp_token = self.to_token(temp_token.reshape(1, -1)) + self.emb_0
                temp_token = temp_token.squeeze()

                # Append temp_token to ext_lane_tokens_list
                ext_lane_tokens_list.append(torch.unsqueeze(torch.unsqueeze(temp_token, 0),0))

            #-------------------------------------------------SIXTH LANE EXTENSION-------------------------------------------------

            # Calculate lane extension probabilities for all batches
            lane_ext_probs = torch.mean(out_cls[0][:,:,0][sample].unsqueeze(0), dim=1)

            # Checking if lane exists if its probability is above threshold
            if lane_ext_probs > self.thr_ext:

                corr_idxs_b4 = out_cls[0][sample,:,:]
                corr_idxs = torch.argmax(corr_idxs_b4, dim=1)

                ext_corr_idxs_list.append(corr_idxs)

                # Create temp_token for valid batches
                temp_token = torch.zeros((dim_feat, image_height, 1 + 2 * off_grid)).to(self.device)

                for idx_h in range(image_height):
                    corr_idx = corr_idxs[idx_h] + off_grid
                    temp_token[:, idx_h, :] = row_feat_pad[sample, :, idx_h, corr_idx-off_grid:corr_idx+off_grid+1]

                temp_token = self.to_token(temp_token.reshape(1, -1)) + self.emb_0
                temp_token = temp_token.squeeze()

                # Append temp_token to ext_lane_tokens_list
                ext_lane_tokens_list.append(torch.unsqueeze(torch.unsqueeze(temp_token, 0),0))

            #-------------------------------------------------LANE EXTENSION CORRELATOR-------------------------------------------------
            if len(ext_lane_tokens_list) > 0:
                tokens = self.tr_lane_correlator(torch.cat(ext_lane_tokens_list, dim=1))
                tokens = tokens.view(1, tokens.shape[1], 8, 144, 5)
                
                # return to original row_feat_pad
                for idx, corr_idxs in enumerate(ext_corr_idxs_list):
                    for idx_h in range(144):
                        corr_idx = corr_idxs[idx_h]+off_grid
                        row_feat_pad[sample, :, idx_h, corr_idx-off_grid:corr_idx + off_grid + 1] = tokens[0, idx,:, idx_h, :]

        # Return to original row_feat_pad
        row_feat = row_feat_pad[:, :, :, off_grid:144+off_grid]
        row_feat = row_feat.permute(0, 1, 3, 2)
        row_tensor = row_feat.reshape(row_feat.shape[0], row_feat.shape[1]*row_feat.shape[2], row_feat.shape[3])

        # Get lane extension probabilities for all batches (2nd stage processing)
        out_ext_2[0] = torch.nn.functional.softmax(self.ext2_0(row_tensor),dim=2)
        out_cls_2[0] = torch.nn.functional.softmax(self.cls2_0(row_tensor),dim=2)
        out_ext_2[1] = torch.nn.functional.softmax(self.ext2_0(row_tensor),dim=2)
        out_cls_2[1] = torch.nn.functional.softmax(self.cls2_0(row_tensor),dim=2)
        out_ext_2[2] = torch.nn.functional.softmax(self.ext2_0(row_tensor),dim=2)
        out_cls_2[2] = torch.nn.functional.softmax(self.cls2_0(row_tensor),dim=2)
        out_ext_2[3] = torch.nn.functional.softmax(self.ext2_0(row_tensor),dim=2)
        out_cls_2[3] = torch.nn.functional.softmax(self.cls2_0(row_tensor),dim=2)
        out_ext_2[4] = torch.nn.functional.softmax(self.ext2_0(row_tensor),dim=2)
        out_cls_2[4] = torch.nn.functional.softmax(self.cls2_0(row_tensor),dim=2)
        out_ext_2[5] = torch.nn.functional.softmax(self.ext2_0(row_tensor),dim=2)
        out_cls_2[5] = torch.nn.functional.softmax(self.cls2_0(row_tensor),dim=2)

        return out_ext, out_cls, out_cls_2, out_ext_2