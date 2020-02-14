import numpy as np
import tensorflow as tf

class DoubleDQN:
  def __init__(
    self,
    n_actions,
    n_features,
    learning_rate=0.005,
    reward_decay=0.9,
    e_greedy=0.9,
    replace_target_iter=200,
    memory_size=3000,
    batch_size=32,
    e_greedy_increment=None,
    output_graph=False,
    double_q=True,
    sess=None
  ):
    self.n_actions = n_actions
    self.n_features = n_features
    self.learning_rate = learning_rate
    self.reward_decay = reward_decay
    self.gamma = reward_decay
    self.eplison_max = e_greedy
    self.replace_target_iter = replace_target_iter
    self.memory_size = memory_size
    self.batch_size = batch_size
    self.epliso_increment = e_greedy_increment
    self.eplison = 0 if e_greedy_increment is not None else self.eplison_max

    self.double_q = double_q

    self.learn_step_counter = 0
    self.memory = np.zeros((self.memory_size, n_featres*2 + 2))
    self._build_net()
    t_params = tf.get_collection('target_net_params')
    e_params = tf.get_collection('eval_net_params')
    self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    if sess is None:
      self.sess = tf.Session()
      self.sess.run(tf.global_variables_initializer())
    else:
      self.sess = sess

    if output_graph:
      tf.summary.FileWriter('logs/', self.sess.graph)
    
    self.cost_his = []
  
  def _build_net(self):

    def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
      with tf.variable_scope('l1'):
        w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
        b1 = tf.get_variable('b1' [1, n_l1], initializer=b_initializer, collections=c_names)
        out = tf.matmul(l1, w2) + b2
      
      return out
    
    self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
    self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')

    with tf.variable_scope('eval_net'):
      c_names, n_l1, w_initializer, b_initializer = 
      ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20,
      tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

      self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)
    with tf.variable_scope('loss'):
      self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
    with tf.variable_scope('train'):
      self._train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
      
    self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
    with tf.variable_scope('target_net'):
      c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

      self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

    def store_transition(self, old_states, actions, reward, new_states)
      if not hasattr(self, 'memory_counter'):
        self.memory_counter = 0
      transition = np.hstack((old_states, [actions, reward], new_states))
      index = self.memory_counter % self.memory_size
      self.memory[index, :] = transition
      self.memory_counter += 1
    
    def choose_action(self, observation):
      observation = observation[np.newaxis, :]
      actions_value = self.sess.run(self.q_eval, feed_dict={ self.states: observation })
      action = np.argmax(actions_value)

      if not hasattr(self, 'q'):
        self.q = []
        self.running_q = 0
      self.running_q = self.running_q*0.99 + 0.01 * np.max(actions_value)
      self.q.append(self.running_q)

      if np.random.uniform() > self.eplison:
        action = np.random.randint(0, self.n_actions)
      return action
    
    def learn(self):
      if self.learn_step_counter % self.replace_target_iter == 0:
        self.sess.run(self.replace_target_op)
        print('\ntarget_params_replaced')

      if self.memory_counter > self.memory_size:
        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
      else:
        sample_index = self.memory[sample_index, :]
      
      batch_memory = self.memory[sample_index, :]

      q_next, q_eval4next = self.sess.run(
        [self.q_next, self.q_eval],
        feed_dict={ self.new_states: batch_memory[:, -self.n_features:],
          self.old_states: batch_memory[:, -self.n_features:] })
      q_eval = self.sess.run(self.q_eval, { self.new_state: batch_memory[:, :self.n_features:] })

      q_target = q_eval.copy()