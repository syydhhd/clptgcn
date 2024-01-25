import copy
import math
from transformers import BertTokenizer, BertPreTrainedModel
import torch

import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform, BertModel
from transformers.models.bart.modeling_bart import shift_tokens_right


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        hidden_state, prediction_scores = self.predictions(sequence_output)
        return prediction_scores, hidden_state


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states_trans = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states_trans)
        return hidden_states_trans, hidden_states


class CLPTGCN(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.bert = bert
        self.opt = opt
        self.layers = opt.num_layers
        self.mem_dim = opt.bert_dim // 2
        self.attention_heads = opt.attention_heads
        self.bert_dim = opt.bert_dim
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        self.lstm_drop = nn.Dropout(opt.input_dropout)
        self.layernorm = LayerNorm(opt.bert_dim)
        self.lm_head = BertOnlyMLMHead(config=opt.config)

        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert_dim
            self.W.append(nn.Linear(input_dim, self.bert_dim))

        self.dense_sent = nn.Linear(self.bert_dim, self.bert_dim)
        self.dense_aspect = nn.Linear(self.bert_dim, self.bert_dim)
        self.dense_mask = nn.Linear(self.bert_dim, self.bert_dim)

        self.sim = Similarity(temp=0.05)
        self.cls = nn.Linear(opt.bert_dim*3, opt.polarities_dim)

        # self.cls_aul = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        tids_mask, tids_pos, tids_neg_1, tids_neg_2, bert_segments_ids_aul, attention_mask_aul, \
        adj_dep, src_mask, aspect_mask, label_id, loc_mask = inputs

        output = self.bert(tids_mask, attention_mask=attention_mask_aul,
                           token_type_ids=bert_segments_ids_aul, return_dict=False)

        out_pos = self.bert(tids_pos, attention_mask=attention_mask_aul,
                           token_type_ids=bert_segments_ids_aul, return_dict=False)

        out_neg_1 = self.bert(tids_neg_1, attention_mask=attention_mask_aul,
                            token_type_ids=bert_segments_ids_aul, return_dict=False)

        out_neg_2 = self.bert(tids_neg_2, attention_mask=attention_mask_aul,
                              token_type_ids=bert_segments_ids_aul, return_dict=False)


        sequence_output_mask = output[0]
        sequence_output_pos = out_pos[0]
        scores, hidden_states = self.lm_head(sequence_output_mask)






        loc_mask_bert_dim = loc_mask.unsqueeze(-1).repeat(1, 1, self.opt.bert_dim)
        mask_state_ori = (sequence_output_mask * loc_mask_bert_dim).sum(dim=1)
        pos_state_ori = (sequence_output_pos * loc_mask_bert_dim).sum(dim=1)

        pooled_output = self.pooled_drop(output[1])
        pooled_output_pos = self.pooled_drop(out_pos[1])
        pooled_output_neg_1 = self.pooled_drop(out_neg_1[1])
        pooled_output_neg_2 = self.pooled_drop(out_neg_2[1])

        cos_sim = self.sim(pooled_output.unsqueeze(1), pooled_output_pos.unsqueeze(0))
        sample_mask = torch.eye(cos_sim.size(0)).cuda()
        cos_sim = torch.mm(cos_sim, sample_mask).sum(dim=1).unsqueeze(-1)
        cos_sim_n1 = self.sim(pooled_output.unsqueeze(1), pooled_output_neg_1.unsqueeze(0))

        cos_sim_n2 = self.sim(pooled_output.unsqueeze(1), pooled_output_neg_2.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, cos_sim_n1, cos_sim_n2], dim=-1)
        labels = torch.zeros(cos_sim.size(0)).long().cuda()
        loss_fct = nn.CrossEntropyLoss()

        loss_cl = loss_fct(cos_sim, labels)


        sequence_output_mask = self.bert_drop(sequence_output_mask)
        sequence_output_pos = self.bert_drop(sequence_output_pos)
        src_mask_cl = src_mask.unsqueeze(-1).repeat(1, 1, self.opt.bert_dim)
        h_s_mask = sequence_output_mask * src_mask_cl
        h_s_pos = sequence_output_pos * src_mask_cl



        h_s_mask_log_softmax = torch.log_softmax(h_s_mask, dim=-1)
        label_h_s_pos = torch.softmax(h_s_pos, dim=-1)
        kl_fct = torch.nn.KLDivLoss(reduction='batchmean')
        loss_kl = kl_fct(h_s_mask_log_softmax, label_h_s_pos)


        outputs_dep_mask = h_s_mask
        outputs_dep_pos = h_s_pos
        denom_dep = adj_dep.sum(2).unsqueeze(2) + 1
        for l in range(self.layers):
            # ************SynGCN_mask*************
            Ax_dep = adj_dep.bmm(outputs_dep_mask)
            AxW_dep = self.W[l](Ax_dep)
            AxW_dep = AxW_dep / denom_dep
            gAxW_dep = F.relu(AxW_dep)
            outputs_dep_mask = self.gcn_drop(gAxW_dep) if l < self.layers - 1 else gAxW_dep

            # ************SynGCN_pos*************
            Ax_dep_pos = adj_dep.bmm(outputs_dep_pos)
            AxW_dep_pos = self.W[l](Ax_dep_pos)
            AxW_dep_pos = AxW_dep_pos / denom_dep
            gAxW_dep_pos = F.relu(AxW_dep_pos)
            outputs_dep_pos = self.gcn_drop(gAxW_dep_pos) if l < self.layers - 1 else gAxW_dep_pos
        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.bert_dim)
        outputs_asp_dep_mask = (outputs_dep_mask * aspect_mask).sum(dim=1) / asp_wn
        outputs_asp_dep_pos = (outputs_dep_pos * aspect_mask).sum(dim=1) / asp_wn


        outputs_dep_mask_d = self.dense_sent(outputs_dep_mask)

        outputs_asp_dep = self.dense_aspect(outputs_asp_dep_mask)
        mask_state = self.dense_mask(mask_state_ori)
        temp_rep = (outputs_asp_dep + mask_state).unsqueeze(1)
        outputs_dep_t = torch.transpose(outputs_dep_mask_d, 1, 2)
        weight = torch.bmm(temp_rep, outputs_dep_t).squeeze(1)
        weight = weight.masked_fill(src_mask == 0, -1e9)
        target_weight = torch.softmax(weight, dim=-1).unsqueeze(1)
        target_representation = torch.bmm(target_weight, outputs_dep_mask_d).squeeze(1)

        final_rep = torch.cat((target_representation, pooled_output, mask_state_ori), dim=-1)

        outputs_dep_pos_d = self.dense_sent(outputs_dep_pos)

        outputs_asp_dep_pos = self.dense_aspect(outputs_asp_dep_pos)
        pos_state = self.dense_mask(pos_state_ori)
        temp_rep_pos = (outputs_asp_dep_pos + pos_state).unsqueeze(1)
        outputs_dep_t_pos = torch.transpose(outputs_dep_pos_d, 1, 2)
        weight_pos = torch.bmm(temp_rep_pos, outputs_dep_t_pos).squeeze(1)
        weight_pos = weight_pos.masked_fill(src_mask == 0, -1e9)
        target_weight_pos = torch.softmax(weight_pos, dim=-1).unsqueeze(1)
        target_representation_pos = torch.bmm(target_weight_pos, outputs_dep_pos_d).squeeze(1)

        final_rep_pos = torch.cat((target_representation_pos, pooled_output_pos, pos_state_ori), dim=-1)
        #



        logits = self.cls(final_rep)
        logits_pos = self.cls(final_rep_pos)
        labels_pos = torch.argmax(logits_pos, dim=-1)
        loss_dl = loss_fct(logits, labels_pos)

        return logits, logits_pos, scores, label_id, loss_kl, loss_cl, loss_dl, pooled_output_pos


class GCNAbsaModel(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn = GCNBert(bert, opt, opt.num_layers, )

    def forward(self, inputs):
        text_bert_indices_mask, text_bert_indices, text_bert_indices_aul, bert_segments_ids, \
        attention_mask, bert_segments_ids_aul, attention_mask_aul, adj_dep, src_mask, aspect_mask = inputs
        batch_size = text_bert_indices.size(0)
        num_sent = text_bert_indices.size(1)
        if text_bert_indices.ndim > 2:
            aspect_mask = aspect_mask.reshape((batch_size * num_sent, aspect_mask.shape[-1]))
            adj_dep = adj_dep[:, 0]
        if self.opt.do_mlm == 1:
            h1, h2, adj_ag, pooled_output, final_out, final_out_aul = self.gcn(inputs)
        else:
            h1, h2, adj_ag, pooled_output = self.gcn(inputs)
        if text_bert_indices.ndim > 2:
            adj_ag = adj_ag.reshape((batch_size, num_sent, adj_ag.shape[-1], adj_ag.shape[-1]))
            adj_ag = adj_ag[:, 0]

        # avg pooling asp feature

        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.bert_dim // 2)
        outputs1 = (h1 * aspect_mask).sum(dim=1) / asp_wn
        outputs2 = (h2 * aspect_mask).sum(dim=1) / asp_wn

        if self.opt.do_mlm == 1:
            return outputs1, outputs2, adj_ag, adj_dep, pooled_output, final_out, final_out_aul
        else:
            return outputs1, outputs2, adj_ag, adj_dep, pooled_output


class GCNBert(nn.Module):
    def __init__(self, bert, opt, num_layers):
        super(GCNBert, self).__init__()
        self.bert = bert
        self.opt = opt
        self.layers = num_layers
        self.mem_dim = opt.bert_dim // 2
        self.attention_heads = opt.attention_heads
        self.bert_dim = opt.bert_dim
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        self.layernorm = LayerNorm(opt.bert_dim)

        # self.pre_classifier = torch.nn.Linear(opt.bert_dim, opt.bert_dim * 2)
        # self.classifier = nn.Linear(opt.bert_dim * 2, opt.polarities_dim)
        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

        self.attn = MultiHeadAttention(opt.attention_heads, self.bert_dim)
        self.weight_list = nn.ModuleList()
        for j in range(self.layers):
            input_dim = self.bert_dim if j == 0 else self.mem_dim
            self.weight_list.append(nn.Linear(input_dim, self.mem_dim))

        self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))

    #        data_attr = ['text_bert_indices_mask', 'text_bert_indices', 'text_bert_indices_aul', 'bert_segments_ids',
    #                     'attention_mask','bert_segments_ids_aul', 'attention_mask_aul','adj_matrix', 'src_mask', 'aspect_mask']
    def forward(self, inputs):
        text_bert_indices_mask, text_bert_indices, text_bert_indices_aul, bert_segments_ids, \
        attention_mask, bert_segments_ids_aul, attention_mask_aul, adj_dep, src_mask, aspect_mask = inputs

        src_mask_out = src_mask.unsqueeze(-2)
        # sent_loc = torch.sum(attention_mask == 1, dim=1) - 1
        # sent_loc = sent_loc.unsqueeze(1).expand((attention_mask.shape[0], self.opt.bert_dim)).unsqueeze(1).long()
        output = self.bert(text_bert_indices, attention_mask=attention_mask,
                           token_type_ids=bert_segments_ids, return_dict=True)
        # output_mlm = self.bert(text_bert_indices_ori, attention_mask=attention_mask,
        #                                            token_type_ids=bert_segments_ids, return_dict=True)
        sequence_output = output.last_hidden_state
        pooled_output = output.pooler_output
        # sequence_output_mlm = output_mlm.last_hidden_state
        # pooled_output_mlm = output_mlm.pooler_output
        sequence_output_out = sequence_output
        # sequence_output_out_rnn, (_, _) = self.rnn(sequence_output_out)
        # sequence_output_out_rnn = sequence_output_out_rnn.gather(1, sent_loc)
        pooled_output_out = pooled_output

        sequence_output_out = self.layernorm(sequence_output_out)

        if self.opt.do_mlm == 1:
            output_mask = self.bert(text_bert_indices_mask, attention_mask=attention_mask_aul,
                                    token_type_ids=bert_segments_ids_aul, return_dict=True)
            sequence_output_mask = output_mask.last_hidden_state[:, 0]
            final_out = sequence_output_mask

            output_aul = self.bert(text_bert_indices_aul, attention_mask=attention_mask_aul,
                                   token_type_ids=bert_segments_ids_aul, return_dict=True)
            sequence_output_aul = output_aul.last_hidden_state[:, 0]
            final_out_aul = sequence_output_aul
        gcn_inputs = self.bert_drop(sequence_output_out)
        pooled_output_out = self.pooled_drop(pooled_output_out)

        denom_dep = adj_dep.sum(2).unsqueeze(2) + 1
        attn_tensor = self.attn(gcn_inputs, gcn_inputs, src_mask_out)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        adj_ag = None
        # * Average Multi-head Attention matrixes
        for i in range(self.attention_heads):
            if adj_ag is None:
                adj_ag = attn_adj_list[i]
            else:
                adj_ag = adj_ag + attn_adj_list[i]
        adj_ag = adj_ag / self.attention_heads

        for j in range(adj_ag.size(0)):
            adj_ag[j] -= torch.diag(torch.diag(adj_ag[j]))
            adj_ag[j] += torch.eye(adj_ag[j].size(0)).cuda()
        src_mask_out = src_mask_out.transpose(1, 2)
        adj_ag = src_mask_out * adj_ag

        denom_ag = adj_ag.sum(2).unsqueeze(2) + 1
        outputs_ag = gcn_inputs
        outputs_dep = gcn_inputs

        for l in range(self.layers):
            # ************SynGCN*************
            Ax_dep = adj_dep.bmm(outputs_dep)
            AxW_dep = self.W[l](Ax_dep)
            AxW_dep = AxW_dep / denom_dep
            gAxW_dep = F.relu(AxW_dep)

            # ************SemGCN*************
            Ax_ag = adj_ag.bmm(outputs_ag)
            AxW_ag = self.weight_list[l](Ax_ag)
            AxW_ag = AxW_ag / denom_ag
            gAxW_ag = F.relu(AxW_ag)

            # * mutual Biaffine module
            A1 = F.softmax(torch.bmm(torch.matmul(gAxW_dep, self.affine1), torch.transpose(gAxW_ag, 1, 2)), dim=-1)
            A2 = F.softmax(torch.bmm(torch.matmul(gAxW_ag, self.affine2), torch.transpose(gAxW_dep, 1, 2)), dim=-1)
            gAxW_dep, gAxW_ag = torch.bmm(A1, gAxW_ag), torch.bmm(A2, gAxW_dep)
            outputs_dep = self.gcn_drop(gAxW_dep) if l < self.layers - 1 else gAxW_dep
            outputs_ag = self.gcn_drop(gAxW_ag) if l < self.layers - 1 else gAxW_ag

        if self.opt.do_mlm == 1:
            return outputs_ag, outputs_dep, adj_ag, pooled_output_out, final_out, final_out_aul
        else:
            return outputs_ag, outputs_dep, adj_ag, pooled_output_out


# class GCNBert(nn.Module):
#     def __init__(self, bert, config, opt, num_layers):
#         super(GCNBert, self).__init__()
#         self.bert = bert
#         self.opt = opt
#         self.layers = num_layers
#         self.mem_dim = opt.bert_dim // 2
#         self.attention_heads = opt.attention_heads
#         self.bert_dim = opt.bert_dim
#         self.bert_drop = nn.Dropout(opt.bert_dropout)
#         self.pooled_drop = nn.Dropout(opt.bert_dropout)
#         self.gcn_drop = nn.Dropout(opt.gcn_dropout)
#         self.layernorm = LayerNorm(opt.bert_dim)
#         self.mlm_drop = nn.Dropout(opt.bert_mlm_drop)
#         self.config = config
#         self.sim = Similarity(temp=0.05)
#         self.lm_head = BertLMPredictionHead(config)
#         # gcn layer
#         self.W = nn.ModuleList()
#         self.do_mlm=opt.do_mlm
#         for layer in range(self.layers):
#             input_dim = self.bert_dim if layer == 0 else self.mem_dim
#             self.W.append(nn.Linear(input_dim, self.mem_dim))
#
#         self.attn = MultiHeadAttention(opt.attention_heads, self.bert_dim)
#         self.weight_list = nn.ModuleList()
#         for j in range(self.layers):
#             input_dim = self.bert_dim if j == 0 else self.mem_dim
#             self.weight_list.append(nn.Linear(input_dim, self.mem_dim))
#
#         self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
#         self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
#
#     def forward(self, inputs):
#         text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, adj_dep, src_mask, aspect_mask = inputs
#         # loss_fct = nn.CrossEntropyLoss()
#         # if text_bert_indices.ndim > 2:
#         #     batch_size = text_bert_indices.size(0)
#         #     num_sent = text_bert_indices.size(1)
#         #     text_bert_indices_mlm = text_bert_indices[:, -1]
#         #     text_bert_indices = text_bert_indices.reshape((batch_size*num_sent, text_bert_indices.shape[-1]))
#         #     bert_segments_ids = bert_segments_ids.reshape((batch_size * num_sent, bert_segments_ids.shape[-1]))
#         #     attention_mask = attention_mask.reshape((batch_size * num_sent, attention_mask.shape[-1]))
#         #
#         #     src_mask_out = src_mask[:, 0]
#         #     aspect_mask_out = aspect_mask[:, 0]
#         #     adj_dep_mlm = adj_dep[:, -1]
#         #     src_mask_mlm = src_mask[:, -1]
#         #     adj_dep = adj_dep[:, 0]
#         # else:
#         #     batch_size = text_bert_indices.size(0)
#         #     num_sent = 1
#         #     text_bert_indices = text_bert_indices
#         #     bert_segments_ids = bert_segments_ids
#         #     attention_mask = attention_mask
#         #     adj_dep = adj_dep
#         #     aspect_mask_out = aspect_mask
#         #     src_mask_out = src_mask
#         src_mask_out = src_mask_out.unsqueeze(-2)
#         output = self.bert(text_bert_indices, attention_mask=attention_mask,
#                                                    token_type_ids=bert_segments_ids, return_dict=True)
#         sequence_output = output.last_hidden_state
#         pooled_output = output.pooler_output
#
#         ############################ctl################################################
#         # if num_sent > 1:
#         #     sequence_output = sequence_output.reshape(
#         #         (batch_size, num_sent, sequence_output.shape[-2], sequence_output.shape[-1]))
#         #     pooled_output = pooled_output.reshape(
#         #         (batch_size, num_sent, pooled_output.shape[-1]))
#         #     sequence_output_out = sequence_output[:, 0]
#         #     pooled_output_out = pooled_output[:, 0]
#         #
#         #     # z1, z2, z3, z4 = sequence_output[:, 0, 0], sequence_output[:, 1, 0], sequence_output[:, 2, 0], sequence_output[:, 3, 0]
#         #     if self.do_mlm==1:
#         #         sequence_output_mlm = sequence_output[:, -1]  # batch_size x Maxlenth x dim
#         #         sequence_output_mlm = self.bert_drop(sequence_output_mlm)
#         #         mlm_labels = text_bert_indices_mlm
#         #         prediction_scores = self.lm_head(self.bert_drop(sequence_output_mlm))
#         #         loss_mlm = loss_fct(prediction_scores.reshape((prediction_scores.shape[0]*prediction_scores.shape[1], self.config.vocab_size)), mlm_labels.reshape((mlm_labels.shape[0]*mlm_labels.shape[1])))
#         #
#         #
#         #     # cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
#         #     # z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
#         #     # z1_z4_cos = self.sim(z1.unsqueeze(1), z4.unsqueeze(0))
#         #     #
#         #     # cos_sim = torch.cat([cos_sim, z1_z3_cos, z1_z4_cos], 1)
#         #     #
#         #     # # Hard negative
#         #     # # print("cos_sim.shape", cos_sim.shape)
#         #     # labels = torch.arange(cos_sim.size(0)).long().cuda()
#         #     # loss_fct = nn.CrossEntropyLoss()
#         #     #
#         #     # # Calculate loss with hard negatives
#         #     #
#         #     # # Note that weights are actually logits of weights
#         #     # weights = torch.tensor(
#         #     #     [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1) - z1_z4_cos.size(-1)) + [0.0] * i + [1] + [
#         #     #         0.0] * (
#         #     #              z1_z4_cos.size(-1) + z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
#         #     # ).cuda()
#         #     # cos_sim = cos_sim + weights
#         #
#         #
#         #     z1_pooled, z2_pooled, z3_pooled, z4_pooled = pooled_output[:, 0], pooled_output[:, 1], pooled_output[:, 2], pooled_output[:, 3]
#         #
#         #     cos_sim_pooled = self.sim(z1_pooled.unsqueeze(1), z2_pooled.unsqueeze(0))
#         #     z1_z3_cos_pooled = self.sim(z1_pooled.unsqueeze(1), z3_pooled.unsqueeze(0))
#         #     z1_z4_cos_pooled = self.sim(z1_pooled.unsqueeze(1), z4_pooled.unsqueeze(0))
#         #
#         #     cos_sim_pooled = torch.cat([cos_sim_pooled, z1_z3_cos_pooled, z1_z4_cos_pooled], 1)
#         #
#         #     # Hard negative
#         #     # print("cos_sim.shape", cos_sim.shape)
#         #     labels_pooled = torch.arange(cos_sim_pooled.size(0)).long().cuda()
#         #
#         #
#         #     # Calculate loss with hard negatives
#         #
#         #     # Note that weights are actually logits of weights
#         #     weights_pooled = torch.tensor(
#         #         [[0.0] * (cos_sim_pooled.size(-1) - z1_z3_cos_pooled.size(-1) - z1_z4_cos_pooled.size(-1)) + [0.0] * i + [1] + [
#         #             0.0] * (
#         #                  z1_z4_cos_pooled.size(-1) + z1_z3_cos_pooled.size(-1) - i - 1) for i in range(z1_z3_cos_pooled.size(-1))]
#         #     ).cuda()
#         #     cos_sim_pooled = cos_sim_pooled + weights_pooled
#         #
#         #     if self.do_mlm==1:
#         #         ctl_loss = loss_fct(cos_sim_pooled, labels_pooled) + loss_mlm*0.1
#         #     else:
#         #         ctl_loss = loss_fct(cos_sim_pooled, labels_pooled)
#         # else:
#         sequence_output_out = sequence_output
#         pooled_output_out = pooled_output
#
#         sequence_output_out = self.layernorm(sequence_output_out)
#         gcn_inputs = self.bert_drop(sequence_output_out)
#         pooled_output_out = self.pooled_drop(pooled_output_out)
#
#         denom_dep = adj_dep.sum(2).unsqueeze(2) + 1
#         attn_tensor = self.attn(gcn_inputs, gcn_inputs, src_mask_out)
#         attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
#         adj_ag = None
#         # * Average Multi-head Attention matrixes
#         for i in range(self.attention_heads):
#             if adj_ag is None:
#                 adj_ag = attn_adj_list[i]
#             else:
#                 adj_ag = attn_adj_list[i] + adj_ag
#         adj_ag = adj_ag / self.attention_heads
#
#         for j in range(adj_ag.size(0)):
#             adj_ag[j] -= torch.diag(torch.diag(adj_ag[j]))
#             adj_ag[j] += torch.eye(adj_ag[j].size(0)).cuda()
#         adj_ag = src_mask_out.transpose(1, 2) * adj_ag
#
#         denom_ag = adj_ag.sum(2).unsqueeze(2) + 1
#         outputs_ag = gcn_inputs
#         outputs_dep = gcn_inputs
#
#         for l in range(self.layers):
#             # ************SynGCN*************
#             Ax_dep = adj_dep.bmm(outputs_dep)
#             AxW_dep = self.W[l](Ax_dep)
#             AxW_dep = AxW_dep / denom_dep
#             gAxW_dep = F.relu(AxW_dep)
#
#             # ************SemGCN*************
#             Ax_ag = adj_ag.bmm(outputs_ag)
#             AxW_ag = self.weight_list[l](Ax_ag)
#             AxW_ag = AxW_ag / denom_ag
#             gAxW_ag = F.relu(AxW_ag)
#
#             # * mutual Biaffine module
#             A1 = F.softmax(torch.bmm(torch.matmul(gAxW_dep, self.affine1), torch.transpose(gAxW_ag, 1, 2)), dim=-1)
#             A2 = F.softmax(torch.bmm(torch.matmul(gAxW_ag, self.affine2), torch.transpose(gAxW_dep, 1, 2)), dim=-1)
#             gAxW_dep, gAxW_ag = torch.bmm(A1, gAxW_ag), torch.bmm(A2, gAxW_dep)
#             outputs_dep = self.gcn_drop(gAxW_dep) if l < self.layers - 1 else gAxW_dep
#             outputs_ag = self.gcn_drop(gAxW_ag) if l < self.layers - 1 else gAxW_ag
#         if num_sent > 2:
#             return outputs_ag, outputs_dep, adj_ag, pooled_output_out, ctl_loss
#         else:
#             return outputs_ag, outputs_dep, adj_ag, pooled_output_out
# class GCNBert(nn.Module):
#     def __init__(self, bert, opt, num_layers):
#         super(GCNBert, self).__init__()
#         self.bert = bert
#         self.opt = opt
#         self.moco = MoCo_ctl(opt, bert)
#         self.layers = num_layers
#         self.mem_dim = opt.bert_dim // 2
#         self.attention_heads = opt.attention_heads
#         self.bert_dim = opt.bert_dim
#         self.bert_drop = nn.Dropout(opt.bert_dropout)
#         self.pooled_drop = nn.Dropout(opt.bert_dropout)
#         self.gcn_drop = nn.Dropout(opt.gcn_dropout)
#         self.layernorm = LayerNorm(opt.bert_dim)
#
#         # gcn layer
#         self.W = nn.ModuleList()
#         for layer in range(self.layers):
#             input_dim = self.bert_dim if layer == 0 else self.mem_dim
#             self.W.append(nn.Linear(input_dim, self.mem_dim))
#         self.pool_out = False
#         self.attn = MultiHeadAttention(opt.attention_heads, self.bert_dim)
#         self.weight_list = nn.ModuleList()
#         for j in range(self.layers):
#             input_dim = self.bert_dim if j == 0 else self.mem_dim
#             self.weight_list.append(nn.Linear(input_dim, self.mem_dim))
#         self.sim = Similarity(temp=0.05)
#         self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
#         self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
#
#     def forward(self,inputs):
#         text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, adj_dep, src_mask, aspect_mask = inputs
#
#         if text_bert_indices.ndim > 2:
#             text_bert_indices_out = text_bert_indices[:, :3]
#             bert_segments_ids_out = bert_segments_ids[:, :3]
#             attention_mask_out = attention_mask[:, :3]
#             adj_dep_out = adj_dep[:, :3]
#             src_mask_out = src_mask[:, :3]
#             batch_size = text_bert_indices_out.size(0)
#             num_sent = text_bert_indices_out.size(1)
#             text_bert_indices_out = text_bert_indices_out.reshape((batch_size*3, text_bert_indices_out.shape[-1]))
#             bert_segments_ids_out = bert_segments_ids_out.reshape((batch_size*3, bert_segments_ids_out.shape[-1]))
#             attention_mask_out = attention_mask_out.reshape((batch_size*3, attention_mask_out.shape[-1]))
#             adj_dep_out = adj_dep_out.reshape((batch_size*3, adj_dep_out.shape[-1], adj_dep_out.shape[-1]))
#             src_mask_out = src_mask_out.reshape((batch_size*3, src_mask_out.shape[-1]))
#
#
#         else:
#             text_bert_indices_out = text_bert_indices
#             bert_segments_ids_out = bert_segments_ids
#             attention_mask_out = attention_mask
#             adj_dep_out = adj_dep
#             src_mask_out = src_mask
#         src_mask_out_1 = src_mask_out
#         src_mask_out = src_mask_out.unsqueeze(-2)
#
#         # out = self.moco(inputs, polarity=0, train=False)
#         # sequence_output, pooled_output = out.last_hidden_state, out.pooler_output
#
#
#         out = self.bert(text_bert_indices_out, attention_mask=attention_mask_out, token_type_ids=bert_segments_ids_out, return_dict=True)
#         sequence_output, pooled_output = out.last_hidden_state, out.pooler_output
#         sequence_output.requires_grad = True
#         pooled_output.requires_grad = True
#         sequence_output = self.layernorm(sequence_output)
#
#         gcn_inputs = self.bert_drop(sequence_output)
#         pooled_output = self.pooled_drop(pooled_output)
#
#         denom_dep = adj_dep_out.sum(2).unsqueeze(2) + 1
#         attn_tensor = self.attn(gcn_inputs, gcn_inputs, src_mask_out)
#         attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
#         adj_ag = None
#
#         # * Average Multi-head Attention matrixes
#         for i in range(self.attention_heads):
#             if adj_ag is None:
#                 adj_ag_1 = attn_adj_list[i]
#             else:
#                 adj_ag_1 = attn_adj_list[i] + adj_ag
#         adj_ag_1 = adj_ag_1 / self.attention_heads
#
#         for j in range(adj_ag_1.size(0)):
#             adj_ag_1[j] -= torch.diag(torch.diag(adj_ag_1[j]))
#             adj_ag_1[j] += torch.eye(adj_ag_1[j].size(0)).cuda()
#         adj_ag_1 = src_mask_out.transpose(1, 2) * adj_ag_1
#
#         denom_ag = adj_ag_1.sum(2).unsqueeze(2) + 1
#         outputs_ag = gcn_inputs
#         outputs_dep = gcn_inputs
#
#         for l in range(self.layers):
#             # ************SynGCN*************
#             Ax_dep = adj_dep_out.bmm(outputs_dep)
#             AxW_dep = self.W[l](Ax_dep)
#             AxW_dep = AxW_dep / denom_dep
#             gAxW_dep = F.relu(AxW_dep)
#
#             # ************SemGCN*************
#             Ax_ag = adj_ag_1.bmm(outputs_ag)
#             AxW_ag = self.weight_list[l](Ax_ag)
#             AxW_ag = AxW_ag / denom_ag
#             gAxW_ag = F.relu(AxW_ag)
#
#             # * mutual Biaffine module
#             A1 = F.softmax(torch.bmm(torch.matmul(gAxW_dep, self.affine1), torch.transpose(gAxW_ag, 1, 2)), dim=-1)
#             A2 = F.softmax(torch.bmm(torch.matmul(gAxW_ag, self.affine2), torch.transpose(gAxW_dep, 1, 2)), dim=-1)
#             gAxW_dep, gAxW_ag = torch.bmm(A1, gAxW_ag), torch.bmm(A2, gAxW_dep)
#             outputs_dep = self.gcn_drop(gAxW_dep) if l < self.layers - 1 else gAxW_dep
#             outputs_ag = self.gcn_drop(gAxW_ag) if l < self.layers - 1 else gAxW_ag
#
#
#         ####################ctl###################################
#         if text_bert_indices.ndim > 2:
#             # if num_sent == 4:
#             #     text_bert_indices_ctl = text_bert_indices.reshape((batch_size * num_sent, text_bert_indices.size(-1)))
#             #     bert_segments_ids_ctl = bert_segments_ids.reshape((batch_size * num_sent, text_bert_indices.size(-1)))
#             #     attention_mask_ctl = attention_mask.reshape((batch_size * num_sent, text_bert_indices.size(-1)))
#             # sequence_output_ctl, pooled_output_ctl = self.bert(text_bert_indices_ctl, attention_mask=attention_mask_ctl,
#             #                                                    token_type_ids=bert_segments_ids_ctl)
#             # if self.pool_out:
#             #     ctl_data = pooled_output_ctl
#             # else:
#             #     ctl_data = sequence_output_ctl[:, 0]
#             # ctl_data = ctl_data.reshape((batch_size, num_sent, ctl_data.size(-1)))
#             # if num_sent == 4:
#             #     z1, z2, z3, z4 = ctl_data[:, 0], ctl_data[:, 1], ctl_data[:, 2], ctl_data[:, 3]
#             # else:
#             #     z1, z2, z3, z4 = ctl_data[:, 0], ctl_data[:, 2], ctl_data[:, 3], ctl_data[:, 4]
#             #
#             # cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
#             # z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
#             # z1_z4_cos = self.sim(z1.unsqueeze(1), z4.unsqueeze(0))
#             #
#             # cos_sim = torch.cat([cos_sim, z1_z3_cos, z1_z4_cos], 1)
#             #
#             # # Hard negative
#             # # print("cos_sim.shape", cos_sim.shape)
#             # labels = torch.arange(cos_sim.size(0)).long().cuda()
#             # loss_fct = nn.CrossEntropyLoss()
#             #
#             # # Calculate loss with hard negatives
#             #
#             # # Note that weights are actually logits of weights
#             # weights = torch.tensor(
#             #     [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1) - z1_z4_cos.size(-1)) + [0.0] * i + [1] + [
#             #         0.0] * (
#             #              z1_z4_cos.size(-1) + z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
#             # ).cuda()
#             # cos_sim = cos_sim + weights
#             # ctl_loss = loss_fct(cos_sim, labels)
#             return outputs_ag, outputs_dep, adj_ag_1, pooled_output, adj_dep_out, src_mask_out_1
#         else:
#             return outputs_ag, outputs_dep, adj_ag_1, pooled_output, adj_dep_out, src_mask_out_1


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]

        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn