import tensorflow as tf
import datetime
from functions import readdata
from functions import calcAttributesOfEmail

inp = []
out = []
readdata(inp, out)

Target = tf.placeholder('float', shape=[None, 1], name="Target")

with tf.name_scope("Input_Layer") as scope:
    Input = tf.placeholder('float', shape=[None, 57], name="Input")
    inputBias = tf.Variable(initial_value=tf.random_normal(shape=[1], stddev=0.4, dtype='float', name="input_bias"))

with tf.name_scope("Hidden_Layer") as scope:
    weights = tf.Variable(initial_value=tf.random_normal(shape=[57, 3], stddev=0.4, dtype='float', name="hidden_weights"))
    hiddenBias = tf.Variable(initial_value=tf.random_normal(shape=[1], stddev=0.4, dtype='float', name="hidden_bias"))
    tf.summary.histogram(name="Weights_1", values=weights)
    hiddenLayer = tf.matmul(Input, weights) + inputBias
    hiddenLayer = tf.sigmoid(hiddenLayer, name="hidden_layer_activation")

with tf.name_scope("Output_Layer") as scope:
    outputWeigths = tf.Variable(initial_value=tf.random_normal(shape=[3, 1], stddev=0.4, dtype='float', name="output_weights"))
    tf.summary.histogram(name="Weights_2", values=outputWeigths)
    output = tf.matmul(hiddenLayer, outputWeigths) + hiddenBias
    output = tf.sigmoid(output, name="output_activation")

with tf.name_scope("Optimizer") as scope:
    cost = tf.squared_difference(Target, output)
    cost = tf.reduce_mean(cost)
    tf.summary.scalar("error_1", cost)
    tf.summary.scalar("error_2", cost)
    tf.summary.scalar("lr_1", cost)
    tf.summary.scalar("lr_2", cost)

    optimizer = tf.train.AdamOptimizer().minimize(cost)


saver = tf.train.Saver()
model_save_path = "./models/"
model_name = 'EmailClassifier'

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver.restore(sess, model_save_path + model_name)
    print("Model saved in path: " + model_save_path + model_name)

    inp = calcAttributesOfEmail('D:/learningPython/testsets/Mails/Spam/Spam1.txt')
    print("input is Spam1.txt")
    print(sess.run([output], feed_dict={Input: inp})[0][0])

    inp = calcAttributesOfEmail('D:/learningPython/testsets/Mails/Spam/Spam2.txt')
    print("input is Spam2.txt")
    print(sess.run([output], feed_dict={Input: inp})[0][0])

    i = 1
    while i <= 7:
        inp = calcAttributesOfEmail('D:/learningPython/testsets/Mails/Spam/Spam' + str(i) + '.txt')
        print("input is Spam" + str(i) + ".txt")
        print(sess.run([output], feed_dict={Input: inp})[0][0])
        i += 1

