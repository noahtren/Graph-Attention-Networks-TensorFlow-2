"""Implementation of Graph Attention Networks (GATs) By Petar Veličković,
Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio
https://arxiv.org/abs/1710.10903
"""


import tensorflow as tf


class GraphAttention(tf.keras.layers.Layer):
  def __init__(self, num_heads:int, hidden_size:int, max_neighbors:int):
    super(GraphAttention, self).__init__()
    self.hidden_size = hidden_size
    self.max_neighbors = max_neighbors
    self.num_heads = num_heads
    # trainable alignment matrix for general-style self-attention
    self.attn_ws = [tf.keras.layers.Dense(hidden_size) for _ in range(num_heads)]
    # output layer
    self.out_w = tf.keras.layers.Dense(hidden_size)


  def call(self, inputs, debug=False):
    """Calculate scores of all neighboring nodes using general Luong-style
    self-attention. If multiple attention heads are used, contexts are
    concatenated.

    # Inputs:
        value: the vector relating to the node to encode. Should be of size:
          [batch_size, 1, hidden_size]
        query: a tensor providing embeddings for the neighbors of the node
          which are to be attended to. Each query should be of size:
          [batch_size, max_neighbors, hidden_size]
        num_neighbors: the number of neighbors to attend to from the query
          tensor. Only the first `max_neighbors` nodes are attended to.
    """

    value = inputs['value']
    query = inputs['query']
    num_neighbors = inputs['num_neighbors']

    assert value.shape[1] == 1, f'second dim of value should be 1, but was {value.shape[1]}'
    assert query.shape[1] == self.max_neighbors, f'second dim of query should equal max_neighbors, but was {query.shape[1]}'
    assert num_neighbors < self.max_neighbors, f'num_neighbors input of {num_neighbors} cannot be greater than max neighbors'

    # aggregate features from all neighbors, including the node itself
    query = tf.concat([value, query], axis=1)

    # multi-head self-attention
    contexts = []
    for i in range(self.num_heads):
      query = self.attn_ws[i](query)
      e = tf.matmul(value, query, transpose_b=True)
      mask = tf.sequence_mask(num_neighbors + 1, maxlen=self.max_neighbors)[:, tf.newaxis]
      e = tf.nn.swish(e)
      # apply mask before softmaxing
      e = tf.where(mask, e, tf.ones_like(e) * -1e9)
      scores = tf.nn.softmax(e)
      # sum all query embeddings according to attention scores
      context = tf.matmul(scores, query)
      contexts.append(context)

    # concatenate contexts from each attention head
    contexts = tf.concat(contexts, axis=-1)

    # produce new features from full context
    x = self.out_w(contexts)
    x = tf.nn.relu(x)
    return x


if __name__ == "__main__":
  value = tf.random.normal([8, 1, 512])
  query = tf.random.normal([8, 10, 512])
  g_attn = GraphAttention(num_heads=4, hidden_size=512, max_neighbors=10)
  x = g_attn({
      'query': query,
      'value': value,
      'num_neighbors': 5
  })
  print(f'Input shape: {value.shape}')
  print(f'Output shape: {x.shape}')
