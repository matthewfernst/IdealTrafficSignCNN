from sklearn.model_selection import train_test_split
import tensorflow as tf
from utilities import *
import time
import matplotlib.pyplot as plt

classes = ["Stop Sign", "No U Turn", "Speed Limit", "Yield", "School Zone"]
def main():
        features, label,= load_data("data/5class.pickle")
        print(f"Number of features: {len(features)}")
        x_train, x_test, y_train, y_test = train_test_split(features,label,test_size=0.1,shuffle=True)
        x_train,x_test= tf.cast(x_train,tf.float32),tf.cast(x_test,tf.float32)
        train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train))
        train_data = train_data.batch(batch_size=3)
        model = tf.keras.models.load_model(f'myVgg16.h5')
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam()
        train_loss = tf.keras.metrics.Mean(name='train_loss')

        train_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(name='train_accuracy')

        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_object(y_true=labels, y_pred=predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

            train_loss(loss)
            train_accuracy(labels, predictions)

        for epoch in range(3):
            start=time.time()
            train_loss.reset_states()
            train_accuracy.reset_states()

            step = 0

            for images, labels in train_data:
                step += 1
                train_step(images, labels)
                if step % 2 == 0:
                    print('=> epoch: %i, loss: %.4f, train_accuracy: %.10f' % (epoch + 1,train_loss.result(), train_accuracy.result()))
            print(f"Time From epoch: {epoch} took {(time.time()-start)/60**2:.4f} hours")
        model.save(f'models/myVgg16Trained5classes.h5')


def main2():
    classes = ["Stop Sign", "No U Turn", "Speed Limit", "Yield", "School Zone"]
    #make_data("data/5classesMore.pickle")
    features, label, = load_data("data/5classesMore.pickle")
    # plt.figure(figsize=(8,8))
    # for i in range(25):
    #     plt.subplot(6,5,i+1)
    #     plt.imshow(features[i])
    #     plt.xlabel(classes[label[i]])
    #     plt.xticks([])
    # plt.show()
    # exit()
    print(f"Number of features: {len(features)}")
    x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.1, shuffle=True, random_state=1001)
    classes = ["Stop Sign", "No U Turn", "Speed Limit", "Yield", "School Zone"]
    model = tf.keras.models.load_model(f'myVgg16.h5')
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    history = model.fit(x_train,y_train,epochs=2)
    print(history)
    model.save(f'models/myVgg16Trained5classesMore20Epoch.h5')
    print(f"Evaluate: {model.evaluate(x_test, y_test)}")

if __name__ == '__main__':
    main2()
    # model = tf.keras.models.load_model('models/myVgg16Trained5classes.h5')
    # features, label, = load_data("data/5class.pickle")
    # x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.1, shuffle=True, random_state=42)
    # print(model.evaluate(x_test, y_test))
