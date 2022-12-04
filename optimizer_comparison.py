import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = models.Sequential()
# model.add(layers.ZeroPadding2D(padding=(2, 2)))
model.add(layers.Conv2D(filters=64,
                        kernel_size=(5, 5),
                        strides=1, padding='same',
                        activation='relu',
                        input_shape=(28, 28, 1)
                        ))

model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

# model.add(layers.ZeroPadding2D(padding=(2, 2)))
model.add(layers.Conv2D(filters=64,
                        kernel_size=(5, 5),
                        strides=1, padding='same',
                        activation='relu',
                        # input_shape=(64, 14, 14, 3)))
                        ))

model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))


# model.add(layers.ZeroPadding2D(padding=(2, 2)))
model.add(layers.Conv2D(filters=64,
                        kernel_size=(5, 5),
                        strides=1, padding='same',
                        activation='relu',
                        # input_shape=(64, 7, 7, 3)
                        ))

model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adadelta',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

history1 = model.fit(train_images, train_labels, batch_size=64, epochs=10)
plt.plot(history1.history['accuracy'], label='Adadelta')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

history2 = model.fit(train_images, train_labels, batch_size=64, epochs=10)
plt.plot(history2.history['accuracy'], label='Adam')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

model.compile(optimizer='adagrad',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

history3 = model.fit(train_images, train_labels, batch_size=64, epochs=10)
plt.plot(history3.history['accuracy'], label='Adagrad')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

model.compile(optimizer='sgd',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

history4 = model.fit(train_images, train_labels, batch_size=64, epochs=10)
plt.plot(history4.history['accuracy'], label='SGD')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
