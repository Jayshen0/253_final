# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:34:01 2020

@author: Ajie
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def positional_encoding(dim, sentence_length, dtype=tf.float32):

    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])

    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)

def normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    
    if len(inputs.shape) == 2:
        m = nn.batchNorm2d(input.shape[-1])
    elif len(inputs.shape) == 3:
        m = nn.batchNorm3d(input.shape[-1])
        
    return m(inputs)


def embedding(inputs, 
              vocab_size, 
              num_units, 
              zero_pad=True, 
              scale=True,
              l2_reg=0.0,
              scope="embedding", 
              with_t=False,
              reuse=None):
    '''Embeds a given tensor.
    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.
        
    '''
    
    
    

    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       #initializer=tf.contrib.layers.xavier_initializer(),
                                       regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        
        if scale:
            outputs = outputs * (num_units ** 0.5) 
    if with_t: return outputs,lookup_table
    else: return outputs


def multihead_attention(layers,
                        queries, 
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None,
                        with_qk=False):
    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    
    in_feature = queries.shape[-1]
    
    if num_units is None:
        num_units = in_feature
        
    Q = layers[0](queries)
    K = layers[1](keys)
    V = layers[2](keys)
    
    Q_ = torch.cat(torch.split(Q,num_heads,dim=2), 0)
    K_ = torch.cat(torch.split(K,num_heads,dim=2), 0)
    V_ = torch.cat(torch.split(V,num_heads,dim=2), 0)
    
    outputs = torch.matmul(Q_, torch.transpose(K_,1,2))
    
    outputs = outputs / (K_.shape[-1] ** 0.5)
    
    key_masks = torch.sign(torch.abs(torch.sum(keys,axis=-1)))
    key_masks = key_masks.repeat(num_heads,1)
    key_masks = torch.unsqueeze(key_masks, 1)
    key_masks = key_masks.repeat(1,queries.shape[1],1)
    
    paddings = torch.ones_like(outputs) * (-2**32 + 1)
    outputs = torch.where(key_masks.eq(0), paddings, outputs)
    
    if causality:
        diag_vals = torch.ones_like(outputs[0,:,:])
        tril = torch.tril(diag_vals, diagonal=0, out=None)
        tril = tril.unsqueeze(tril,0)
        masks = tril.repeat(tril,outputs.shape[0],1,1)
        
        paddings = torch.ones_like(masks)*(-2**32+1)
        outputs = torch.where(masks.eq(0),paddings, outputs)
    
    outputs = F.softmax(outputs)
    
    #query masking
    
    query_masks = torch.sign(torch.abs(torch.sum(queries,-1)))
    query_masks = query_masks.repeat(num_heads,1)
    query_masks = query_masks.unsqueeze(-1)
    query_masks = query_masks.repeat(1,1,keys.shape[1])
    outputs *= query_masks
    
    if is_training:
        outputs = torch.nn.Dropout(dropout_rate)(outputs)
    
    outputs = torch.matmul(outputs, V_)
    
    outputs = torch.cat(torch.split(outputs,num_heads,dim=0), 2)
    
    outputs += queries
    
    if with_qk: 
        return Q,K
    else:
        return outputs
    


    
    