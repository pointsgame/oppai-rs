import tensorflow as tf
import numpy as np

field_width = 39
field_height = 32

trunk_layers = 19
fc_width = 256
policy_conv_width = 2
value_conv_width = 1

def oppai_activation():
  return tf.keras.layers.Activation(
    activation = tf.keras.activations.relu,
  )

def oppai_batchn():
  return tf.keras.layers.BatchNormalization(
    axis = -1,
    momentum = .95,
    epsilon = 1e-5,
    center = True,
    scale = True,
    fused = True,
  )

def oppai_conv2d():
  return tf.keras.layers.Conv2D(
    filters = 256,
    kernel_size = 3,
    padding='same',
    use_bias = True,
    data_format='channels_last',
    kernel_regularizer = tf.keras.regularizers.l2(0.01),
    bias_regularizer = tf.keras.regularizers.l2(0.01),
  )

inputs = tf.keras.Input(
  shape = (field_width, field_height, 2),
  dtype = 'float32',
  name = 'input_tensor',
)

initial_block = oppai_activation()(oppai_batchn()(oppai_conv2d()(inputs)))

# residual blocks
shared_output = initial_block
for _ in range(trunk_layers):
  conv_layer1 = oppai_batchn()(oppai_conv2d()(shared_output))
  initial_output = oppai_activation()(conv_layer1)
  conv_layer2 = oppai_batchn()(oppai_conv2d()(initial_output))
  shared_output = oppai_activation()(shared_output + conv_layer2)

policy_conv = tf.keras.layers.Conv2D(
  filters = policy_conv_width,
  kernel_size = 1,
  padding='same',
  use_bias = True,
  data_format='channels_last',
  kernel_regularizer = tf.keras.regularizers.l2(0.01),
  bias_regularizer = tf.keras.regularizers.l2(0.01),
)(shared_output)
policy_conv = oppai_activation()(tf.keras.layers.BatchNormalization(
  axis = -1,
  momentum = .95,
  epsilon = 1e-5,
  center = False,
  scale = False,
  fused = True,
)(policy_conv))

logits = tf.keras.layers.Dense(
  units = field_width * field_height,
  kernel_regularizer = tf.keras.regularizers.l2(0.01),
  bias_regularizer = tf.keras.regularizers.l2(0.01),
)(tf.keras.layers.Reshape(
  target_shape = (policy_conv_width * field_width * field_height,),
)(policy_conv))

policy_output = tf.keras.layers.Softmax(
  name = 'policy_output',
)(logits)

value_conv = tf.keras.layers.Conv2D(
  filters = value_conv_width,
  kernel_size = 1,
  padding='same',
  use_bias = True,
  data_format='channels_last',
  kernel_regularizer = tf.keras.regularizers.l2(0.01),
  bias_regularizer = tf.keras.regularizers.l2(0.01),
)(shared_output)
value_conv = oppai_activation()(tf.keras.layers.BatchNormalization(
  axis = -1,
  momentum = .95,
  epsilon = 1e-5,
  center = False,
  scale = False,
  fused = True,
)(value_conv))

value_fc_hidden = tf.keras.layers.Dense(
  units = fc_width,
  kernel_regularizer = tf.keras.regularizers.l2(0.01),
  bias_regularizer = tf.keras.regularizers.l2(0.01),
)(tf.keras.layers.Reshape(
  target_shape = (value_conv_width * field_width * field_height,),
)(value_conv))

value_output = tf.keras.layers.Activation(
  activation = tf.keras.activations.tanh,
  name = 'value_output',
)(tf.keras.layers.Dense(
  units = 1,
  kernel_regularizer = tf.keras.regularizers.l2(0.01),
  bias_regularizer = tf.keras.regularizers.l2(0.01),
)(value_fc_hidden))

model = tf.keras.Model(
  inputs = inputs,
  outputs = [policy_output, value_output],
)

model.compile(
  optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001, momentum = 0.9, decay = 0., nesterov = False),
  loss = {
    'policy_output': tf.keras.losses.CategoricalCrossentropy(),
    'value_output': tf.keras.losses.MSE
  },
)

model.summary()

model.save('model.tf')

tf.keras.utils.plot_model(
  model,
  to_file = 'model.png',
  show_shapes = True,
)
