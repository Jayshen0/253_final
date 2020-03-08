from modules import *


class Model(nn.Module):
    def __init__(self, usernum, itemnum, args):
        self.usernum = usernum
        self.itemnum = itemnum
        self.args = args
        
        self.item_emb_table = nn.Embedding(self.itemnum+1, self.args.hidden_units)
        self.item_emb_table.weight.data[0,:] = torch.zeros([1,num_units])
        
        self.pos_emb_table = nn.Embedding(self.args.maxlen, self.args.hidden_units)
        self.dropout1 = torch.nn.Dropout(self.args.dropout_rate)
        
        
        in_feature = self.seq.shape[-1]
        self.layer = []
        for i in range(self.args.num_blocks):
            self.layer.append([])
            for j in range(3):
                self.layer[-1].append(nn.Linear(in_feature,self.args.num_units))
    
        self.ff_layer = []
        for i in range(self.args.num_blocks):
            self.ff_layer.append([])
            for j in range(2):
                self.ff_layers[-1].append(nn.Conv1d(self.args.hidden_units, self.args.hidden_units, 1))
            for j in range(2):
                self.ff_layers[-1].append(torch.nn.Dropout(self.args.dropout_rate))
        
        self.normlize = [nn.batchNorm3d(self.args.hidden_units) for i in range(2*self.args.num_blocks + 1)]
    
    def feedforward(self,inputs, 
                step,
                dropout_rate=0.2,
                is_training=True):
    '''Point-wise feed forward net.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    
    outputs = self.ff_layers[step][0](inputs)
    outputs = F.relu(outputs)
    
    if is_training:
        outputs = self.ff_layers[step][2](outputs)  
        
    outputs = self.ff_layers[step][1](inputs)
    
    if is_training:
        outputs = self.ff_layers[step][3](outputs)  
    
    outputs += inputs
    
    
    return outputs

    def forward(self, u, input_seq, pos, neg, is_training):
        
        mask = input_seq.ne(0).float()
        mask = mask.unsqueeze(-1)
        
        self.seq = self.item_emb_table(input_seq)
        self.seq = self.seq * (self.args.hidden_units ** 0.5) 
        
        pos = torch.Tensor(list(range(self.input_seq.shape[1])))
        pos = pos.repeat(self.input_seq.shape[0], 1)
        
        t = self.pos_emb_table(pos)
        
        self.seq += t
        # Dropout
        self.seq = self.dropout1(self.seq)
        self.seq *= mask
        #Build blocks
        
        for i in range(self.args.num_blocks):
            self.seq = multihead_attention(self.layer[i],
                                                   queries=self.normalize[2*i](self.seq),
                                                   keys=self.seq,
                                                   num_units=args.hidden_units,
                                                   num_heads=args.num_heads,
                                                   dropout_rate=args.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

            self.seq = self.feedforward(self.normalize[2*i+1](self.seq), i,
                                           dropout_rate=args.dropout_rate, is_training=self.is_training)
            self.seq *= mask
    

        self.seq = self.normalize[-1](self.seq)

        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])

        self.test_item = tf.placeholder(tf.int32, shape=(101))
        test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
        self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, 101])
        self.test_logits = self.test_logits[:, -1, :]

        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        self.loss = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
            tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)

        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, u, seq, item_idx):
        return sess.run(self.test_logits,
                        {self.u: u, self.input_seq: seq, self.test_item: item_idx, self.is_training: False})