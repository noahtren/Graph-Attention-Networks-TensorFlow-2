"""Implementation of Graph Attention Networks (GAT)
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
    self.out_w = tf.keras.layers.Dense(hidden_size)


  def call(self, inputs, debug=False):
    """Calculate scores of all neighboring nodes using general Luong-style
    self-attention with additive aggregation.

    # Inputs:
        value: the vector relating to the node to encode. Should be of size:
          [batch_size, 1, hidden_size]
        queries: a list of tensors, each providing embeddings for the neighbors
          of the node which are attended to. Each query should be of size:
          [batch_size, max_neighbors, hidden_size]
    """

    query = inputs['query']
    value = inputs['value']

    assert value.shape[1] == 1, f'second dim of value should be 1, but was {value.shape[1]}'
    assert query.shape[1] == self.max_neighbors, f'second dim of query should equal max_neighbors, but was {query.shape[1]}'

    # aggregate features from all neighbors, including the node itself
    query = tf.concat([query, value], axis=1)

    contexts = []
    # multi-head  self-attention
    for i in range(self.num_heads):
      query = self.attn_ws[i](query)
      e = tf.matmul(value, query, transpose_b=True)
      scores = tf.nn.softmax(e)
      # sum all query embeddings according to attention scores
      context = tf.matmul(scores, query)
      contexts.append(context)

    contexts = tf.concat(contexts, axis=-1)

    # produce new features from context
    x = self.out_w(contexts)
    x = tf.nn.relu(x)
    return x


if __name__ == "__main__":
  query = tf.random.normal([8, 10, 512])
  value = tf.random.normal([8, 1, 512])
  g_attn = GraphAttention(num_heads=4, hidden_size=512, max_neighbors=10)
  x = g_attn({
      'query': query,
      'value': value
  })
  print(f'Input shape: {value.shape}')
  print(f'Output shape: {x.shape}')