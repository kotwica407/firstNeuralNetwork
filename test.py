import tensorflow as tf
import datetime

Target = tf.placeholder('float', shape=[None, 1], name="Target")

with tf.name_scope("Input_Layer") as scope:
    Input = tf.placeholder('float', shape=[None, 2], name="Input")
    inputBias = tf.Variable(initial_value=tf.random_normal(shape=[1], stddev=0.4, dtype='float', name="input_bias"))

with tf.name_scope("Hidden_Layer") as scope:
    weights = tf.Variable(initial_value=tf.random_normal(shape=[2, 3], stddev=0.4, dtype='float', name="hidden_weights"))
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

inp = [[1, 1], [1, 0], [0, 1], [0, 0]]
out = [[0], [1], [1], [0]]
epochs = 10000

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    mergedSummary = tf.summary.merge_all()
    fileName = "./summary_log/run" + datetime.datetime.now().strftime("%H-%M-%S")
    writer = tf.summary.FileWriter(fileName, sess.graph)
    for i in range(epochs):
        err, _, summaryOutput = sess.run([cost, optimizer, mergedSummary], feed_dict={Input: inp, Target: out})
        writer.add_summary(summaryOutput, i)

    while True:
        inp = [[0, 0]]
        inp[0][0] = input("type first input")
        inp[0][1] = input("type second input")
        print("input is " + str(inp))
        print(sess.run([output], feed_dict={Input: inp})[0][0])
