import tensorflow as tf
import numpy as np
import os
import time


def createInputTarget(section):
    input = section[:-1]
    target = section[1:]
    return input, target


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(
        labels,
        logits,
        from_logits=True
    )


def buildModel(vocabSize, embeddingDimension, rnnUnits, batchSize):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocabSize, embeddingDimension,
                                  batch_input_shape=[batchSize, None]),
        tf.keras.layers.GRU(rnnUnits,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocabSize)
    ])
    return model


def generateText(model, start):
    numGenerate = 1000

    inputEval = [charToNum[s] for s in start]
    inputEval = tf.expand_dims(inputEval, 0)

    textGenerated = []

    temperature = 1.0

    model.reset_states()
    for i in range(numGenerate):
        predictions = model(inputEval)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predictedId = tf.random.categorical(
            predictions,
            num_samples=1
        )[-1, 0].numpy()

        inputEval = tf.expand_dims([predictedId], 0)

        textGenerated.append(numToChar[predictedId])

    return (start + ''.join(textGenerated))


sequenceLength = 100
batchSize = 64
bufferSize = 10000
epochs = 1000

# Load text in as a string and create a list of unique characters in the text
sampleText = open('dxd.txt', 'r').read()
uniqueChars = sorted(set(sampleText))

# Create two lookup tables for converting characters to integers and vice-versa
charToNum = {u: i for i, u in enumerate(uniqueChars)}
numToChar = np.array(uniqueChars)

# Maps each character in the sample text to an integer represenation
textAsNum = np.array([charToNum[c] for c in sampleText])

perEpoch = len(sampleText) // (sequenceLength + 1)

charDataset = tf.data.Dataset.from_tensor_slices(textAsNum)

sequences = charDataset.batch(sequenceLength + 1, drop_remainder=True)
dataset = sequences.map(createInputTarget)

dataset = dataset.shuffle(bufferSize).batch(batchSize, drop_remainder=True)

vocabSize = len(uniqueChars)
embeddingDimension = 256
rnnUnits = 1024

model = buildModel(
    vocabSize=len(uniqueChars),
    embeddingDimension=embeddingDimension,
    rnnUnits=rnnUnits,
    batchSize=batchSize)

model.compile(optimizer='adam', loss=loss)

checkpointDirectory = './trainingCheckpoints'
checkpointPrefix = os.path.join(checkpointDirectory, "ckpt_{epoch}")

checkpointCallback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpointPrefix, save_weights_only=True)

history = model.fit(dataset, epochs=epochs, callbacks=[checkpointCallback])

tf.train.latest_checkpoint(checkpointDirectory)
model = buildModel(vocabSize, embeddingDimension, rnnUnits, batchSize=1)
model.load_weights(tf.train.latest_checkpoint(checkpointDirectory))
model.build(tf.TensorShape([1, None]))

print(generateText(model, start="Breasts "))
