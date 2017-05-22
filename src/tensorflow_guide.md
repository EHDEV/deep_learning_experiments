##### Some notes from Stanford's *CS 20SI: Tensorflow for Deep Learning Research* class


# How to structure your TensorFlow model

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
#                * 33 4 66 *
#                  6 5 55 
#                 15 54 53]
```

#### 4. Define loss function

- While NCE is cumbersome to implement in pure Python, TensorFlow already implemented it for us.
- For nce_loss, we need weights and biases for the hidden layer to calculate NCE loss.

```python
# nce_loss definition
tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, num_true=1, sampled_values=None, 
               remove_accidental_hits=False, partition_strategy='mod', name='nce_loss')

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
     # 3. Execute the inference model on the training data, 
     # so it calculates for each training input example the output with the current model parameters.
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
         loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, biases=nce_bias, 
                                         labels=target_words, inputs=embed, num_sampled=NUM_SAMPLED, 
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

# II.  How to manage your experiments in TensorFlow

- A good practice is to periodically save the model’s parameters after a certain number of steps so that we can restore/retrain our model from that step if need be. 
- The tf.train.Saver() classallows us to do so by saving the graph’s variables in binary files.
```python
tf.train.Saver.save(sess, save_path, global_step=None, latest_filename=None,
meta_graph_suffix='meta', write_meta_graph=True, write_state=True)
```
For example, if we want to save the variables of the graph after every 1000 training steps, we
do the following:

```python
# define model
# create a saver object
saver = tf.train.Saver()
# launch a session to compute the graph
with tf.Session() as sess:
 # actual training loop
    for step in range(training_steps):
        sess.run([optimizer])
        if (step + 1) % 1000==0:
            saver.save(sess, 'checkpoint_directory/model_name', global_step=model.global_step)
```
In TensorFlow lingo, the step at which you save your graph’s variables is called a checkpoint.
Since we will be creating many checkpoints, it’s helpful to append the number of training steps
our model has gone through in a variable called global_step. It’s a very common variable to see
in TensorFlow program. We first need to create it, initialize it to 0 and set it to be not trainable,
since we don’t want to TensorFlow to optimize it.

``` python
self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
```
We need to pass global_step as a parameter to the optimizer so it knows to increment
global_step by one with each training step:

```python
self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
```
To save the session’s variables in the folder ‘checkpoints’ with name model-name-global-step,
we use this:
```python
saver.save(sess, 'checkpoints/skip-gram', global_step=model.global_step)
```
So our training loop for word2vec now looks like this:

```python
self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss,
global_step=self.global_step)
saver = tf.train.Saver() # defaults to saving all variables
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    average_loss = 0.0
    writer = tf.summary.FileWriter('./improved_graph', sess.graph)
    for index in xrange(num_train_steps):
        batch = batch_gen.next()
        loss_batch, _ = sess.run([model.loss, model.optimizer],
                                  feed_dict={model.center_words: batch[0],
                                             model.target_words: batch[1]})
        average_loss += loss_batch
        if (index + 1) % 1000 == 0:
            saver.save(sess, 'checkpoints/skip-gram', global_step=model.global_step)
```
To restore the variables, we use tf.train.Saver.restore(sess, save_path). For example, you want
to restore the checkpoint at 10,000th step

```python
saver.restore(sess, 'checkpoints/skip-gram-10000')
```


```python
ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
if ckpt and ckpt.model_checkpoint_path:
 saver.restore(sess, ckpt.model_checkpoint_path)
```

If there is a checkpoint, restore it. If there isn’t, train from the start.
TensorFlow allows you to get checkpoint from a directory with
tf.train.get_checkpoint_state(‘directory-name’). The code for checking looks something like this:

The file checkpoint automatically updates the path to the latest checkpoint.

```python
ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
if ckpt and ckpt.model_checkpoint_path:
 saver.restore(sess, ckpt.model_checkpoint_path)
```

By default, saver.save() stores all variables of the graph, and this is recommended. However,
you can also choose what variables to store by passing them in as a list or a dict when we
create the saver object.

```python
v1 = tf.Variable(..., name='v1')
v2 = tf.Variable(..., name='v2')
# pass the variables as a dict:
saver = tf.train.Saver({'v1': v1, 'v2': v2})
# pass them as a list
saver = tf.train.Saver([v1, v2])
# passing a list is equivalent to passing a dict with the variable op names # as keys
saver = tf.train.Saver({v.op.name: v for v in [v1, v2]})
```
Note that savers only save variables, not the entire graph, so we still have to create the graph
ourselves, and then load in variables. The checkpoints specify the way to map from variable
names to tensors.
What people usually is not just save the parameters from the last iteration, but also save the
parameters that give the best result so far so that you can evaluate your model on the best
parameters so far.

___

TensorBoard provides us with a great set of tools to visualize our summary statistics during our training. Some popular statistics to visualize is loss, average loss, accuracy. You can visualize them as scalar plots, histograms, or even images. So we have a new namescope in our graph to hold all the summary ops

```python
def _create_summaries(self):
    with tf.name_scope("summaries"):
        tf.summary.scalar("loss", self.loss
        tf.summary.scalar("accuracy", self.accuracy)
        tf.summary.histogram("histogram loss", self.loss)
        # because you have several summaries, we should merge them all
        # into one op to make it easier to manage
        self.summary_op = tf.summary.merge_all()
loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op],
                                  feed_dict=feed_dict)

Now you’ve obtained the summary, you need to write the summary to file using the same
FileWriter object we created to visual our graph.

```python
writer.add_summary(summary, global_step=step)
```
Now, if you go run tensorboard and go to http://localhost:6006/, in the Scalars page, you will see
the plot of your scalar summaries. This is the summary of your loss in scalar plot.

If you save your summaries into different sub-folder in your graph folder, you can compare your progresses. For example, the first time we run our model with learning rate 1.0, we save it in ‘improved_graph/lr1.0’ and the second time we run our model, we save it in ‘improved_graph/lr0.5’, on the left corner of the Scalars page, we can toggle the plots of these two runs to compare them. This can be really helpful when you want to compare the progress made with different optimizers or different parameters

You can write a Python script to automate the naming of folders where you store the
graphs/plots of each experiment.
You can visualize the statistics as images using tf.summary.image.

```python
tf.summary.image(name, tensor, max_outputs=3, collections=None)
```

## Control Randomization

...


## Reading Data in TensorFlow

There are two main ways to load data into a TensorFlow graph: 
     + through feed_dict
     + through readers that allow us to read tensors directly from file.

##### feed_dict

Feed_dict will first send data from the storage system to the client, and then
from client to the worker process. This will cause the data to slow down, especially if the client is
on a different machine from the worker process. TensorFlow has readers that allow us to load
data directly into the worker process.

```python
tf.TextLineReader
Outputs the lines of a file delimited by newlines
E.g. text files, CSV files
tf.FixedLengthRecordReader
Outputs the entire file when all files have same fixed lengths
E.g. each MNIST file has 28 x 28 pixels, CIFAR-10 32 x 32 x 3
tf.WholeFileReader
Outputs the entire file content
tf.TFRecordReader
Reads samples from TensorFlow's own binary format (TFRecord)
tf.ReaderBase
Allows you to create your own readers
```
Data can be read in as individual data examples or in batches of examples.



