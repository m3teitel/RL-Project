import sys
import numpy as np
from tqdm import trange, tqdm

import keras
from keras import backend as K

from mushroom_rl.core import Serializable
from mushroom_rl.utils.minibatches import minibatch_generator

class KerasApproximator(Serializable):
	"""
	class to interface a keras model to the mushroom Regressor interface.
	This class is needed to use a generic keras model and train it using a
	specified optimizer and objective function.
	
	"""
	
	def __init__(self, input_shape, output_shape, network, optimizer=None,
		loss=None, batch_size=0, n_fit_targets=1, quiet=True, **params):
		"""
		Constructor.
		
		Args:
			input_shape (tuple): shape of the input of the network;
			output_shape (tuple): shape of the output of the network;
			network (function/type): network class constructor;
			optimizer(dict): the optimizer used for every fit step;
			loss (tbd): ;
			batch_size (int, 0): the size of each minibatch. If 0, the whole
				dataset is fed to the optimizer at each epoch.;
			n_fit_targets (int, 1): the number of fit targets used by the fit
				method of the network;
			**params: dictionary of parameters needed to construct the network/
		
		"""
			
		
		self._batch_size = batch_size
		self._quiet = quiet
		self._n_fit_targets = n_fit_targets
		
		self.network = network(input_shape, output_shape, **params)
		#self.network.summary()
		
		if (optimizer is not None):
			self._optimizer = optimizer
		
		self._loss = loss
	
	def predict(self, *args, output_tensor=False, **kwargs):
		"""
		Predict.
		
		Args:
			*args: input;
			output_tensor (bool, False): whether to return the output as a
				tensor or not;
			**kwargs: other parameters used by the predict method or the
				regressor;
		
		Returns:
			The predictions of the model.
		
		"""
		
		keras_args = K.constant(np.array(*args))
		val = self.network(keras_args)
		#print(val, file=sys.stderr)
		#print(K.eval(val), file=sys.stderr)
		
		if output_tensor or isinstance(val, tuple):
			return val
		else:
			return K.eval(val)
	
	def fit(self, *args, n_epochs=None, weights=None, epsilon=None, patience=1.,
		validation_split=1., **kwargs):
		"""
		Fit the model.
		
		Args:
			*args: input, where the last ``n_fit_targets`` elements are
				considered as the target, while the others are considered as
				input;
			n_epochs (int, None): the number of training epochs;
			weights (np.ndarray, None): the weights of each sample in the
				computation of the loss;
			epsilon (float, None): the coefficient used for early stopping;
			patience (float, 1.): the number of epochs to wait until stop
				the learning if not improving;
			validation_split (float, 1.): the percentage of the dataset to use
				as training set;
			**kwargs: other parameters used by the fit method of the regressor.
		
		"""

		'''if self._reinitialize:
			self.network.weights_init()

		if self._dropout:
			self.network.train()'''

		if epsilon is not None:
			n_epochs = np.inf if n_epochs is None else n_epochs
			check_loss = True
		else:
			n_epochs = 1 if n_epochs is None else n_epochs
			check_loss = False
		
		if weights is not None:
			args += (weights,)
			use_weights = True
		else:
			use_weights = False
		
		if 0 < validation_split <= 1:
			train_len = np.ceil(len(args[0]) * validation_split).astype(np.int)
			train_args = [a[:train_len] for a in args]
			val_args = [a[train_len:] for a in args]
		else:
			raise ValueError
		
		patience_count = 0
		best_loss = np.inf
		epochs_count = 0
		if check_loss:
			with tqdm(total=n_epochs if n_epochs < n.inf else None,
					dynamic_ncols=True, disable=self._quiet,
					leave=False) as t_epochs:
				while patience_count < patience and epochs_count < n_epochs:
					mean_loss_current = self._fit_epoch(train_args, use_weights,
							kwargs)
					if len(val_args[0]):
						mean_val_loss_current = self._compute_batch_loss(
								val_args, use_weights, kwargs)
						
						loss = mean_val_loss_current.item()
					else:
						loss = mean_loss_current
					
					if not self._quiet:
						t_epochs.set_postfix(loss=loss)
						t_epochs.update(1)
					
					if best_loss - loss > epsilon:
						patience_count = 0
						best_loss = loss
					else:
						patience_count += 1
					
					epochs_count += 1
		
		else:
			with trange(n_epochs, disable=self._quiet) as t_epochs:
				for _ in t_epochs:
					mean_loss_current = self._fit_epoch(train_args, use_weights,
							kwargs)
					
					if not self._quiet:
						t_epochs.set_postfix(loss=mean_loss_current)
		
		'''if self._dropout:
			self.network.eval()'''
	
	def _fit_epoch(self, args, use_weights, kwargs):
		if self._batch_size > 0:
			batches = minibatch_generator(self._batch_size, *args)
		else:
			batches = [args]
		loss_current = []
		for batch in batches:
			loss_current.append(self._fit_batch(batch, use_weights, kwargs))
		
		return np.mean(loss_current)
	
	def _fit_batch(self, batch, use_weights, kwargs):
		loss = self._compute_batch_loss(batch, use_weights, kwargs)
		print('_fit_batch:179', loss, file=sys.stderr)
		raise NotImplementedError
	
	def _compute_batch_loss(self, batch, use_weights, kwargs):
		#if use_weights:
			#weights = K.eval(batch[-1])
			#batch = batch[:-1]
		#batch = batch[:-1]
		
		raw_args = [ K.constant(x) for x in batch ]
		x = raw_args[:-self._n_fit_targets]
		actions = x[1] if len(x) > 1 else None
		x = x[0]

		y_hat = self.network(x, **kwargs)
		if actions is not None:
			actions = K.eval(K.cast(K.squeeze(actions, 1), 'int32'))
			y_hat = K.constant([ r[actions[i]] for i, r in enumerate(K.eval(y_hat)) ])
		
		y = K.squeeze(K.constant(np.array(batch[-self._n_fit_targets:])), 0)
		#print(y_hat, file=sys.stderr)
		#print(y, file=sys.stderr)
		
		if not use_weights:
			#print(y.shape)
			#print(y_hat.shape)
			loss = self._loss(y_hat, y[0])
		else:
			raise NotImplementedError
		
		return loss
		
	def set_weights(self, weights):
		"""
		Setter.
		
		Args:
			weights (np.ndarray): the set of weights to set.
		
		"""
		self.network.set_weights(weights)
	
	def get_weights(self):
		return self.network.get_weights()
	
	@property
	def weights_size(self):
		print('weights_size:206', self.network.count_params(), file=sys.stderr)
		return self.network.count_params()
		
	
	def diff(self, *args, **kwargs):
		raise NotImplementedError
