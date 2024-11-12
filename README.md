# Making an auto encoder using Tensorflow

Ive made an auto encoder using Tensorflow. Training data consists of a small segment of the imagenet dataset available on Kaggle. Training done using GPU's on Kaggle.

The encoder section contains 4 encoder blocks. Their input and output shapes are shown below:
![__results___11_0](https://github.com/user-attachments/assets/1abc0f67-6fa6-45ae-8614-6d2a3f5562b2)

The decoder sections also contains 4 decoder blocks. Their input and output shares are shown below:
![__results___12_0](https://github.com/user-attachments/assets/ba9c5ff7-70e0-4ff2-84bc-83118a891189)

Every block is a sequential model that was instantiated made using the following code:
```
def encode_block(n, f):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(n, (f, f), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(n, (f, f), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D((2, 2), padding='same')
    ])

def decode_block(n, f):
    return tf.keras.models.Sequential([
        tf.keras.layers.UpSampling2D((2, 2)),
        tf.keras.layers.Conv2DTranspose(n, (f, f), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(n, (f, f), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
    ])
```

I used the Adam optimizer and the Mean Square Error loss function. Here is a sample output (Original image on the left, reconstructed image on the right):
![__results___20_1](https://github.com/user-attachments/assets/6077d4f7-9388-4f70-a354-a479dd5b8617)

## Improvements
Using a bigger model and training on a larger dataset would decrease the loss of the model and thus make it more accurate.

Here is the link of the actual file on Kaggle: https://www.kaggle.com/tanmaychoudhary/auto-encoder-using-tensorflow
