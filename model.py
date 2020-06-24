 ##TODO
import dynet_config
dynet_config.set(mem='3072', random_seed=1314159)
import dynet as dy
import math
import numpy as np
import time
from utils import bio2ot
from treelib import Node, Tree

class WDEmb:
    def __init__(self, pc, n_words, dim_w, pretrained_embeddings=None):
        self.pc = pc.add_subcollection()
        self.n_words = n_words
        self.dim_w = dim_w
        self.W = self.pc.add_lookup_parameters((self.n_words, self.dim_w))
        if pretrained_embeddings is not None:
            print("Use pretrained word embeddings")
            self.W.init_from_array(pretrained_embeddings)

    def parametrize(self):
        """
        note: lookup parameters do not need parametrization
        :return:
        """
        pass

    def __call__(self, xs):
        """

        :param xs: a list of ngrams (or words if win is set to 1)
        :return: embeddings looked from tables
        """
        embeddings = [dy.concatenate([self.W[w] for w in ngram]) for ngram in xs]
        return embeddings

class MultiWeightLayer:
    def __init__(self, pc, n_in, n_out, dropout_rate):
        self.n_in = n_in
        self.n_out = n_out
        self.dropout_rate = dropout_rate
        self.pc = pc.add_subcollection()
        
        self._W1 = self.pc.add_parameters((self.n_out, self.n_out), init=dy.UniformInitializer(0.2))
        self._W2 = self.pc.add_parameters((self.n_out, self.n_out), init=dy.UniformInitializer(0.2))
        self._W3 = self.pc.add_parameters((self.n_out, self.n_out), init=dy.UniformInitializer(0.2))

        self._bd = self.pc.add_parameters((self.n_out), init=dy.ConstInitializer(0.0))

    def parametrize(self):
        """
        note: lookup parameters do not need parametrization
        :return:
        """
        self.W1 = dy.parameter(self._W1)
        self.W2 = dy.parameter(self._W2)
        self.W3 = dy.parameter(self._W3)
        self.bd = dy.parameter(self._bd) 

    def __call__(self, H1,H2,H3,is_train=True):
        """

        :param xs: a list of ngrams (or words if win is set to 1)
        :return: embeddings looked from tables
        """
        seq_len = len(H1)
        
        if is_train:
            # in the training phase, perform dropout
            W1 = dy.dropout(self.W1, self.dropout_rate)
            W2 = dy.dropout(self.W2, self.dropout_rate)
            W3 = dy.dropout(self.W3, self.dropout_rate)
        else:
            W1= self.W1
            W2 = self.W2  
            W3 = self.W3 
            
      
        H = []
        for t in range(seq_len):
            ht_hat = dy.tanh(W1*H1[t]+W2*H2[t]+W3*H3[t]+self.bd)
            H.append(ht_hat) 
            
        return H  
   
class BiAttention:
    def __init__(self, pc, n_in, n_out, dropout_rate):
        self.n_in = n_in
        self.n_out = n_out
        self.dropout_rate = dropout_rate
        self.pc = pc.add_subcollection()
        
        self._v = self.pc.add_parameters((self.n_out,), init=dy.UniformInitializer(0.2))
        self._W1 = self.pc.add_parameters((self.n_out, self.n_out), init=dy.UniformInitializer(0.2))
        self._W2 = self.pc.add_parameters((self.n_out, self.n_out), init=dy.UniformInitializer(0.2))
        self._bd = self.pc.add_parameters((self.n_out), init=dy.ConstInitializer(0.0))

    def parametrize(self):
        """
        note: lookup parameters do not need parametrization
        :return:
        """
        self.v=self._v
        self.W1 = dy.parameter(self._W1)
        self.W2 = dy.parameter(self._W2)
        
        self.bd = dy.parameter(self._bd) 

    def __call__(self, H,is_train=True):
        """

        :param xs: a list of ngrams (or words if win is set to 1)
        :return: embeddings looked from tables
        """
        
        seq_len = len(H)
        if is_train:
            # in the training phase, perform dropout
            W1 = dy.dropout(self.W1, self.dropout_rate)
            W2 = dy.dropout(self.W2, self.dropout_rate)
        else:
            W1= self.W1
            W2 = self.W2  
        
        pool= dy.average(H)
           
        aspect_attentions = []
        Weights=[]
        for t in range(seq_len):
            ht = H[t]
            scores = dy.tanh(dy.transpose(ht)*W1*pool+self.bd)
#             print(scores.value())
            Weights.append(scores)
            ht_hat=dy.cmult(dy.softmax(scores),ht)
#             print(ht_hat.value())
            aspect_attentions.append(ht_hat)
        
        Weights_np=[]
        return aspect_attentions,Weights_np      

 
class DTreeBuilder:
    def __init__(self, pc, n_in, n_out, dropout_rate):
        self.n_in = n_in
        self.n_out = n_out
        self.dropout_rate = dropout_rate
        self.pc = pc.add_subcollection()
        
        self._WC = self.pc.add_parameters((self.n_out, self.n_in), init=dy.UniformInitializer(0.2))
        self._WP = self.pc.add_parameters((self.n_out, self.n_in), init=dy.UniformInitializer(0.2))
        self._WR = self.pc.add_parameters((self.n_out, self.n_in), init=dy.UniformInitializer(0.2))
        self._UP = self.pc.add_parameters((self.n_out, self.n_out), init=dy.UniformInitializer(0.2))
        self._UR = self.pc.add_parameters((self.n_out, self.n_out), init=dy.UniformInitializer(0.2))

        self._bc = self.pc.add_parameters((self.n_out), init=dy.ConstInitializer(0.0))
        self._bp = self.pc.add_parameters((self.n_out), init=dy.ConstInitializer(0.0))
        self._br = self.pc.add_parameters((self.n_out), init=dy.ConstInitializer(0.0))

#         self.E = pc_embed.add_lookup_parameters((len(word_vocab), n_in), init=word_embed)
#         self.w2i = word_vocab

    def parametrize(self):
        """
        note: lookup parameters do not need parametrization
        :return:
        """
        self.WC = dy.parameter(self._WC)
        self.WP = dy.parameter(self._WP)
        self.WR = dy.parameter(self._WR)
        self.UP = dy.parameter(self._UP)
        self.UR = dy.parameter(self._UR)

        self.bc = dy.parameter(self._bc)
        self.bp = dy.parameter(self._bp)
        self.br = dy.parameter(self._br)

    def __call__(self, inputs,words,dep,is_train=True):
        """
        :param xs: a list of ngrams (or words if win is set to 1)
        :return: embeddings looked from tables
        """
        list=eval(dep)
        tree = Tree()
        
        finish=False
        err_node=[]
        root_index=0
        while 1:
            if finish:
                break;
            if len(tree.all_nodes())==len(list):
                finish=True;
            for i in range(0,len(list)):
                arr=list[i]
                parentIdx=arr[1]
                nodeIdx=arr[2]
   
                if not tree.contains(nid=nodeIdx):
                    if i==0:
                        tree.create_node(words[nodeIdx-1],identifier=nodeIdx)
                        root_index=nodeIdx-1
                    else:
                        if tree.contains(nid=parentIdx):
                            tree.create_node(words[nodeIdx-1],identifier=nodeIdx,parent=parentIdx)
                                
        H=[]
        
        for idx in range(0,len(inputs)):
            h=self.expr_for_tree(xt=inputs[idx],tree=tree,node=tree.get_node(idx+1),is_train=is_train)
            H.append(h)
        
        return H,root_index
    
    def expr_for_tree(self,xt,tree,node,is_train):
        if is_train:
            # in the training phase, perform dropout
            W_dropout = dy.dropout(self.WP, self.dropout_rate)
            WR_dropout = dy.dropout(self.WR, self.dropout_rate)
            WC_dropout = dy.dropout(self.WC, self.dropout_rate)
        else:
            W_dropout = self.WP
            WR_dropout = self.WR
            WC_dropout = self.WC
            
            
        if node is None or node.is_leaf():
            Wx = W_dropout * xt
            h = dy.tanh(Wx + self.bc)
#             h = dy.tanh(dy.affine_transform([self.bc, self.WC, xt]))
            return h
        
        #需根据当前node的序号获取子节点数组        
        children=tree.children(node.identifier)
        children_sum=dy.zeros((self.n_out))
        for i in range(len(children)):
#             children_sum=children_sum+WC_dropout*self.expr_for_tree(xt=xt,tree=tree,node=children[i],is_train=is_train)
            hc=self.expr_for_tree(xt=xt,tree=tree,node=children[i],is_train=is_train)
            rt = dy.logistic(self.WR * xt +self.UR*hc+self.br)
            children_sum=children_sum+dy.cmult(rt, hc)
#             children_sum=children_sum+WC_dropout*self.expr_for_tree(xt=xt,tree=tree,node=children[i],is_train=is_train)
        
        Wx = W_dropout * xt
        h = dy.tanh(Wx + self.bc+self.UP*children_sum)
        return h     



class Linear:
    # fully connected layer
    def __init__(self, pc, n_in, n_out, use_bias=False):
        self.pc = pc.add_subcollection()
        self.n_in = n_in
        self.n_out = n_out
        self.use_bias = use_bias
        self._W = self.pc.add_parameters((self.n_out, self.n_in), init=dy.UniformInitializer(0.2))
        if self.use_bias:
            self._b = self.pc.add_parameters((self.n_out,), init=dy.ConstInitializer(0.0))

    def parametrize(self):
        """

        :return:
        """
        self.W = dy.parameter(self._W)
        if self.use_bias:
            self.b = dy.parameter(self._b)

    def __call__(self, x):
        """

        :param x: input feature vector
        :return:
        """
        Wx = self.W * x
        if self.use_bias:
            Wx = Wx + self.b
        return Wx


class MyModel:
    def __init__(self, params, vocab, label2tag, pretrained_embeddings=None):
        """

        :param params:
        :param vocab:
        :param label2tag:
        :param pretrained_embeddings:
        """
        self.dim_w = params.dim_w
        self.win = params.win
        self.vocab = vocab
        self.n_words = len(self.vocab)
        self.dim_asp = params.dim_asp
        self.dim_y_asp = params.n_asp_tags
        self.n_steps = params.n_steps
        self.asp_label2tag = label2tag
        self.opi_label2tag = {0: 'O', 1: 'T'}
        self.dropout_asp = params.dropout_asp
        self.dropout = params.dropout
        self.ds_name = params.ds_name
        self.model_name = params.model_name
        self.attention_type = params.attention_type

        self.pc = dy.ParameterCollection()
        self.Emb = WDEmb(pc=self.pc, n_words=self.n_words, dim_w=self.dim_w,
                         pretrained_embeddings=pretrained_embeddings)
        
        self.DEP_RecNN = DTreeBuilder(pc=self.pc, n_in=self.win * self.dim_w, n_out=self.dim_asp, dropout_rate=self.dropout_asp)
        
        self.ASP_RNN = dy.LSTMBuilder(1, self.win * self.dim_w, self.dim_asp, self.pc)

        self.BiAttention_F=BiAttention(pc=self.pc, n_in=self.dim_asp, n_out=self.dim_asp, dropout_rate=self.dropout_asp)
        self.BiAttention_B=BiAttention(pc=self.pc, n_in=self.dim_asp, n_out=self.dim_asp, dropout_rate=self.dropout_asp)
        self.BiAttention_T=BiAttention(pc=self.pc, n_in=self.dim_asp, n_out=self.dim_asp, dropout_rate=self.dropout_asp)

        self.MultiWeightLayer=MultiWeightLayer(pc=self.pc, n_in=self.dim_w, n_out=self.dim_asp, dropout_rate=self.dropout_asp)

        self.ASP_FC = Linear(pc=self.pc, n_in=self.dim_asp, n_out=self.dim_y_asp)
        
        self.layers = [self.ASP_FC,self.DEP_RecNN,self.BiAttention_F,self.BiAttention_B,self.BiAttention_T,self.MultiWeightLayer]

        if params.optimizer == 'sgd':
            self.optimizer = dy.SimpleSGDTrainer(self.pc, params.sgd_lr)
        elif params.optimizer == 'momentum':
            self.optimizer = dy.MomentumSGDTrainer(self.pc, 0.01, 0.9)
        elif params.optimizer == 'adam':
            self.optimizer = dy.AdamTrainer(self.pc, 0.001, 0.9, 0.9)
        elif params.optimizer == 'adagrad':
            self.optimizer = dy.AdagradTrainer(self.pc)
        elif params.optimizer == 'adadelta':
            self.optimizer = dy.AdadeltaTrainer(self.pc)
        else:
            raise Exception("Invalid optimizer!!")

    def parametrize(self):
        """

        :return:
        """
        for layer in self.layers:
            layer.parametrize()

    def __call__(self, dataset, is_train=True):
        """

        :param dataset: input dataset
        :param is_train: train flag
        :return:
        """
        #sentence count
        n_samples = len(dataset)
        total_loss = 0.0
        Y_pred_asp= []

        aspect_attention_outputs = []
        time_costs = []
        for i in range(n_samples):
            beg = time.time()
            dy.renew_cg()
            self.parametrize()

            self.words = dataset[i]['wids']
            self.y_asp = dataset[i]['labels']
            raw_words = dataset[i]['words']
            self.dep_record= dataset[i]['dep_record']

            input_embeddings = self.Emb(xs=self.words)
            
            f_asp = self.ASP_RNN.initial_state()
            b_asp = self.ASP_RNN.initial_state()
            # these operations are equivalent to partial dropout in LSTM
            if is_train:
                input_asp = [dy.dropout(x, self.dropout_asp) for x in input_embeddings]
            else:
                input_asp = input_embeddings
            H_asp_f = f_asp.transduce(input_asp)
            H_asp_b = b_asp.transduce(input_asp[::-1])[::-1]

            dp_asp=self.DEP_RecNN(inputs=input_asp,words=raw_words,dep=self.dep_record,is_train=is_train)
        
            asp_predictions = []
            seq_len = len(input_embeddings)
           
            att_f,attentions=self.BiAttention_F(H=H_asp_f)
            att_B,attentions=self.BiAttention_B(H=H_asp_b)
            att_T,attentions=self.BiAttention_T(H=dp_asp)

            H=self.MultiWeightLayer(att_f,att_B,att_T,is_train=is_train)
            
            losses = []
            for t in range(seq_len):
                htA = H[t]
                asp_feat = dy.concatenate([htA])

                if is_train:
                    # in the training phase, perform dropout
                    asp_feat = dy.dropout(asp_feat, self.dropout)
                p_y_x_asp = self.ASP_FC(x=asp_feat)
                asp_predictions.append(p_y_x_asp.npvalue())

                target_asp = self.y_asp[t]
                loss_asp = dy.pickneglogsoftmax(p_y_x_asp, target_asp)

                losses.append(loss_asp)
            end = time.time()
            time_cost = end - beg
            time_costs.append(time_cost)
            loss = dy.esum(losses)
            total_loss += loss.scalar_value()
            if is_train:
                loss.backward()
                self.optimizer.update()
            pred_asp_labels = np.argmax(np.array(asp_predictions), axis=1)
            pred_asp_tags = bio2ot(tag_sequence=[self.asp_label2tag[l] for l in pred_asp_labels])
            Y_pred_asp.append(pred_asp_tags)
            
        print("Y_pred_asp:",Y_pred_asp)
        return total_loss, Y_pred_asp 
    
