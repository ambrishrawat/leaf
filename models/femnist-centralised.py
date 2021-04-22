import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

import sys
import time

sys.path.append('/dccstor/ambrish1/adversarial-robustness-toolbox/')
from art.estimators.classification import TensorFlowClassifier
from art.defences.trainer import AdversarialTrainerMadryPGD, AdversarialTrainer
from art.attacks.evasion import ProjectedGradientDescent

IMAGE_SIZE = 28
NB_CLASSES = 62


def model():
    """Model function for CNN."""
    features = tf.placeholder(
        tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE], name='features')
    labels = tf.placeholder(tf.int64, shape=[None], name='labels')
    input_layer = tf.reshape(features, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=2048, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense, units=NB_CLASSES)
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.004)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
    return features, labels, train_op, eval_metric_ops, loss, logits


x = np.load('../data/femnist/data/raw_data/data.npy')[:, :, :, 0]
x = x.reshape((x.shape[0], -1))
x = x / 255.0
x = x.astype('float32')
y = np.load('../data/femnist/data/raw_data/labels.npy')
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)
features, labels, train_op, eval_metric_ops, loss, logits = model()
sess = tf.Session()

# X_train = X_train[:200]
# y_train = y_train[:200]
# X_test = X_test[:200]
# y_test = y_test[:200]

classifier = TensorFlowClassifier(
    clip_values=(0, 1),
    input_ph=features,
    output=logits,
    labels_ph=labels,
    train=train_op,
    loss=loss,
    learning=None,
    sess=sess,
    preprocessing_defences=[],
)

attack = ProjectedGradientDescent(
    classifier, eps=0.3, eps_step=0.01, max_iter=40, num_random_init=1,
)


sess.run(tf.global_variables_initializer())
counter = 0
indices = np.arange(X_train.shape[0])
for e in range(3):
    epoch_s = time.time()
    np.random.shuffle(indices)
    for batch_index in range(int(np.ceil(X_train.shape[0] / float(128)))):
        batch_s = time.time()
        counter+=1
        begin, end = (
            batch_index * 128,
            min((batch_index + 1) * 128, X_train.shape[0]),
        )
        X_batch, y_batch = X_train[indices[begin:end]], y_train[indices[begin:end]]

        ratio = np.min(((int(counter / 1000) + 1) / 10.0, 1.0))

        adv_trainer = AdversarialTrainer(classifier, attack, ratio=ratio)
        adv_trainer.fit(X_batch, y_batch, batch_size=128, nb_epochs=1)
        print(counter, 'Batch time: ', time.time() - batch_s, 'ratio', ratio, flush=True)

    acc = np.mean(np.argmax(classifier.predict(X_test), axis=1) == y_test)
    print('accuracy: ', acc, 'Epoch time: ', time.time()-epoch_s, flush=True)

'''

for e in range(3):
    epoch_s = time.time()
    ratio = np.max(((int(e / 10) + 1) / 10.0, 1.0))
    print('running epoch %d ratio %f', e, ratio, flush=True)
    adv_trainer = AdversarialTrainer(classifier, attack, ratio=1.0)
    adv_trainer.fit(X_train, y_train, batch_size=128, nb_epochs=1)

    acc = np.mean(np.argmax(classifier.predict(X_test), axis=1) == y_test)
    print('accuracy:\t ', acc, '\tepoch time: \t', time.time()-epoch_s, flush=True)
'''
acc = np.mean(np.argmax(classifier.predict(X_test), axis=1) == y_test)

preds = attack.generate(X_test, y_test)
adv_acc = np.mean(np.argmax(classifier.predict(preds), axis=1) == y_test)
print(acc, adv_acc, flush=True)
