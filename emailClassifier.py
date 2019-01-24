import tensorflow as tf
import datetime


def readdata(inp, out):
    with open('D:/learningPython/trainsets/SpamBaseData/spambase.txt', 'r') as content:
        dane = []
        for linia in content:
            linia = linia.replace("\n", "")
            linia = linia.replace("\r", "")
            temp = tuple(linia.split(" "))
            danet = []
            for i in range(len(temp)):
                danet.append(float(temp[i]))
            dane.append(danet)
        for example in dane:
            inpt = []
            for i in range(len(example)):
                if i == (len(example)-1):
                    out.append([example[i]])
                else:
                    inpt.append(example[i])
            inp.append(inpt)


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

epochs = 10000
print(1)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    mergedSummary = tf.summary.merge_all()
    fileName = "./summary_log/runEmailClassifier" + datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    writer = tf.summary.FileWriter(fileName, sess.graph)
    print(2)
    for i in range(epochs):
        err, _, summaryOutput = sess.run([cost, optimizer, mergedSummary], feed_dict={Input: inp, Target: out})
        writer.add_summary(summaryOutput, i)
        print(i)


