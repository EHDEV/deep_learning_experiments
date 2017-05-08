How to structure your TensorFlow model

## Phase 1: assemble your graph

#### 1. Define placeholders for input and output
 - Input is the center word and output is the target (context) word. 
 - Instead of using one-hot vectors, we input the index of those words directly. 
      -> For example, if the center word is the 1001th word in the vocabulary, we input the number 1001.

 - Each sample input (and output) is a scalar, the placeholder for BATCH_SIZE sample inputs (and outputs) will have shape [BATCH_SIZE].

```python
center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
```

#### 2. Define the weights

- Each row corresponds to the representation vector of one word. 
- If one word is represented with a vector of size EMBED_SIZE, then the embedding matrix will have shape [VOCAB_SIZE, EMBED_SIZE]. 
- We initialize the embedding matrix to value from a random distribution (uniform, normal, etc)

```python
embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0))
```

#### 3. Define the inference model

- To get the representation of all the center words in the batch, we get the slice of all corresponding rows in the embedding matrix. - - TensorFlow provides a convenient method to do so called tf.nn.embedding_lookup().

```python
embed = tf.nn.embedding_lookup(embed_matrix, center_words)
# [0 1 0 0]   x   [12 4 13
                   33 4 66
                  * 6 5 55 *
                   15 54 53]
```

#### 4. Define loss function

- While NCE is cumbersome to implement in pure Python, TensorFlow already implemented it for us.
- For nce_loss, we need weights and biases for the hidden layer to calculate NCE loss.

```python
# nce_loss definition
  tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, num_true=1,
  sampled_values=None, remove_accidental_hits=False, partition_strategy='mod',
  name='nce_loss')

# Defining weights and biases
nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE], stddev=1.0 / EMBED_SIZE ** 0.5))
nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]))

# Defining loss

loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                        biases=nce_bias,
                        labels=target_words,
                        inputs=embed,
                        num_sampled=NUM_SAMPLED,
                        num_classes=VOCAB_SIZE))

```


#### 5. Define optimizer

```python
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
```


## Phase 2: execute the computation

- Create a session within the session
- Use feed_dict to feed inputs and outputs into the placeholders 
- Run the optimizer to minimize the loss
- And fetch the loss value to report back to us.

```python
with tf.Session() as sess:
 # 1. Initialize all model variables for the first time.
 sess.run(tf.global_variables_initializer())
 average_loss = 0.0
 for index in xrange(NUM_TRAIN_STEPS):
     # 2. Feed in the training data. Might involve randomizing the order of data samples.
     batch = batch_gen.next()
     # 3. Execute the inference model on the training data, so it calculates for each training input
example the output with the current model parameters.
     # 4. Compute the cost
     loss_batch, _ = sess.run([loss, optimizer], feed_dict={center_words: batch[0], target_words: batch[1]})
     # 5. Adjust the model parameters to minimize/maximize the cost depending on the model.
     average_loss += loss_batch
     if (index + 1) % 2000 == 0:
         print('Average loss at step {}: {:5.1f}'.format(index + 1, average_loss / (index + 1)))
```

### Group nodes/ops in TensorBoard

- How can we tell TensorBoard to know which nodes should be grouped together? For example, we would like to group all ops related to input/output together, and all ops related to NCE loss together. 
- Thankfully, TensorFlow lets us do that with name scope. You can just put all the ops that you want to group together under the block:

```python
# Example 
with tf.name_scope('data'):
center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='center_words')
target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1], name='target_words')

with tf.name_scope('embed'):
 embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0),name='embed_matrix')

with tf.name_scope('loss'):
 embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')
 nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE], stddev=1.0 / math.sqrt(EMBED_SIZE)), name='nce_weight')
 nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')
 loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
 biases=nce_bias, labels=target_words, inputs=embed, num_sampled=NUM_SAMPLED, num_classes=VOCAB_SIZE), name='loss')

 ```

### Visualize the computation graph (using TensorBoard)

- Solid lines in TensorBoard represent data flow edges. For example, the value of op tf.add(x + y) depends on the value of x and y. 
- Dotted arrows represent control dependence edges. For example, a variable can only be used after being initialized, as you see variable embed_matrix depends on the op init). 
- Control dependencies can also be declared using tf.Graph.control_dependencies(control_inputs).

```python
 # Step 1: define the placeholders for input and output
with tf.name_scope("data"):
 center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='center_words')
 target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1], name='target_words')
# Assemble this part of the graph on the CPU. You can change it to GPU if you have GPU
with tf.device('/cpu:0'):
 with tf.name_scope("embed"):
 # Step 2: define weights. In word2vec, it's actually the weights that we care about
 embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0), name='embed_matrix')
 # Step 3 + 4: define the inference + the loss function
 with tf.name_scope("loss"):
 # Step 3: define the inference
 embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')
 # Step 4: construct variables for NCE loss
 nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE], stddev=1.0 / math.sqrt(EMBED_SIZE)), name='nce_weight')
 nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')
 # define loss function to be NCE loss function
 loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, biases=nce_bias, labels=target_words, inputs=embed, num_sampled=NUM_SAMPLED, 
                        num_classes=VOCAB_SIZE), name='loss')
```

### Make your model reusable

Question: how do we make our model most easy to reuse?
Hint: take advantage of Python’s object-oriented-ness.
Answer: build our model as a class!
Our class should follow the interface. We combined step 3 and 4 because we want to put
embed under the name scope of “NCE loss”.


```python
class SkipGramModel:
 """ Build the graph for word2vec model """
 def __init__(self, params):
 pass
 def _create_placeholders(self):
 """ Step 1: define the placeholders for input and output """
 pass
 def _create_embedding(self):
 """ Step 2: define weights. In word2vec, it's actually the weights that we care
about """
 pass
 def _create_loss(self):
 """ Step 3 + 4: define the inference + the loss function """
 pass
 def _create_optimizer(self):
 """ Step 5: define optimizer """
 pass
```

 # Step 5: define optimizer
 
 ```
 optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
 ```

### Visualize Word Embeddings in 2D

Follow these steps 

```python
from tensorflow.contrib.tensorboard.plugins import projector
# obtain the embedding_matrix after you’ve trained it
final_embed_matrix = sess.run(model.embed_matrix)
# create a variable to hold your embeddings. It has to be a variable. Constants
# don’t work. You also can’t just use the embed_matrix we defined earlier for our model. Why
# is that so? I don’t know. I get the 500 most popular words.
embedding_var = tf.Variable(final_embed_matrix[:500], name='embedding')
sess.run(embedding_var.initializer)
config = projector.ProjectorConfig()
summary_writer = tf.summary.FileWriter(LOGDIR)
# add embeddings to config
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
# link the embeddings to their metadata file. In this case, the file that contains
# the 500 most popular words in our vocabulary
embedding.metadata_path = LOGDIR + '/vocab_500.tsv'
# save a configuration file that TensorBoard will read during startup
projector.visualize_embeddings(summary_writer, config)
# save our embedding
saver_embed = tf.train.Saver([embedding_var])
saver_embed.save(sess, LOGDIR + '/skip-gram.ckpt', 1)
```

+ Now we run our model again, then again run tensorboard. If you go to http://localhost:6006, clickon the Embeddings tab, you’ll see all the visualization.

+ You can visualize more than word embeddings, aka, you can visualize any embeddings.