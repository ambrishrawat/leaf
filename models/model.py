"""Interfaces for ClientModel and ServerModel."""

from abc import ABC, abstractmethod
import numpy as np
import os
import sys
import tensorflow as tf

from baseline_constants import ACCURACY_KEY

from utils.model_utils import batch_data
from utils.tf_utils import graph_size

import sys
sys.path.append('/Users/ambrish/github/adversarial-robustness-toolbox/')
from art.estimators.classification import TensorFlowClassifier
from art.defences.trainer import AdversarialTrainerMadryPGD, AdversarialTrainer
from art.attacks.evasion import ProjectedGradientDescent

class Model(ABC):

    def __init__(self, seed, lr, optimizer=None):
        self.lr = lr
        self.seed = seed
        self._optimizer = optimizer

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123 + self.seed)
            self.features, self.labels, self.train_op, self.eval_metric_ops, self.loss, self.logits = self.create_model()
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        self.size = graph_size(self.graph)

        self.classifier = TensorFlowClassifier(
            clip_values=(0,1),
            input_ph=self.features,
            output=self.logits,
            labels_ph=self.labels,
            train=self.train_op,
            loss=self.loss,
            learning=None,
            sess=self.sess,
            preprocessing_defences=[],
        )

        self.attack = ProjectedGradientDescent(
            self.classifier, eps=0.3, eps_step=0.01, max_iter=40, num_random_init=1,
        )

        # self.adv_trainer = AdversarialTrainerMadryPGD(
        #     self.classifier,
        #     nb_epochs=1,
        #     eps=0.1,
        #     eps_step=0.01,
        #     max_iter=10,
        #     batch_size=10,
        #     num_random_init=1,
        # )

        self.adv_trainer = AdversarialTrainer(self.classifier, self.attack, ratio=0.1)

        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops

        np.random.seed(self.seed)

    def set_params(self, model_params):
        with self.graph.as_default():
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, model_params):
                variable.load(value, self.sess)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    @property
    def optimizer(self):
        """Optimizer to be used by the model."""
        if self._optimizer is None:
            self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

        return self._optimizer

    @property
    def sess_(self):
        return self.sess


    @abstractmethod
    def create_model(self):
        """Creates the model for the task.

        Returns:
            A 4-tuple consisting of:
                features: A placeholder for the samples' features.
                labels: A placeholder for the samples' labels.
                train_op: A Tensorflow operation that, when run with the features and
                    the labels, trains the model.
                eval_metric_ops: A Tensorflow operation that, when run with features and labels,
                    returns the accuracy of the model.
        """
        return None, None, None, None, None, None

    def train(self, data, num_epochs=1, batch_size=10):
        """
        Trains the client model.

        Args:
            data: Dict of the form {'x': [list], 'y': [list]}.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            comp: Number of FLOPs computed while training given data
            update: List of np.ndarray weights, with each weight array
                corresponding to a variable in the resulting graph
        """
        for _ in range(num_epochs):
            self.run_epoch(data, batch_size)

        update = self.get_params()
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops
        return comp, update

    def run_epoch(self, data, batch_size):

        print(data['x'].shape)
        for batched_x, batched_y in batch_data(data, batch_size, seed=self.seed):
            
            input_data = self.process_x(batched_x)
            target_data = self.process_y(batched_y)
            self.adv_trainer.fit(input_data, target_data, batch_size=input_data.shape[0], nb_epochs=1)
            # with self.graph.as_default():
            #     self.sess.run(self.train_op,
            #         feed_dict={
            #             self.features: input_data,
            #             self.labels: target_data
            #         })

    def test(self, data):
        """
        Tests the current model on the given data.

        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Return:
            dict of metrics that will be recorded by the simulation.
        """
        x_vecs = self.process_x(data['x'])
        labels = self.process_y(data['y'])
        with self.graph.as_default():
            tot_acc, loss = self.sess.run(
                [self.eval_metric_ops, self.loss],
                feed_dict={self.features: x_vecs, self.labels: labels}
            )
        acc = float(tot_acc) / x_vecs.shape[0]
        return {ACCURACY_KEY: acc, 'loss': loss}

    def close(self):
        self.sess.close()

    @abstractmethod
    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        pass

    @abstractmethod
    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        pass


class ServerModel:
    def __init__(self, model):
        self.model = model

    @property
    def size(self):
        return self.model.size

    @property
    def cur_model(self):
        return self.model

    def send_to(self, clients):
        """Copies server model variables to each of the given clients

        Args:
            clients: list of Client objects
        """
        var_vals = {}
        with self.model.graph.as_default():
            all_vars = tf.trainable_variables()
            for v in all_vars:
                val = self.model.sess.run(v)
                var_vals[v.name] = val
        for c in clients:
            with c.model.graph.as_default():
                all_vars = tf.trainable_variables()
                for v in all_vars:
                    v.load(var_vals[v.name], c.model.sess)

    def save(self, path='checkpoints/model.ckpt'):
        return self.model.saver.save(self.model.sess, path)

    def close(self):
        self.model.close()
