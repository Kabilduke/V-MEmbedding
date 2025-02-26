model = models.Sequential([
    layers.Embedding(input_dim = vocal_size, output_dim = 64),

    layers.Reshape((max_length, 8, 8, 1)), #Reshaping into Matrix Embeddings
    
    layers.Conv3D(32, kernel_size = (3, 3, 3), activation = 'relu'),
    layers.MaxPooling3D(pool_size = (2, 2, 2)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Conv3D(16, kernel_size = (2, 2, 2), activation = 'elu'),
    layers.MaxPooling3D(pool_size = (1, 1, 1)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Flatten(),
    layers.Dense(16, activation = 'elu'),
    layers.Dropout(0.1),
    
    layers.Dense(6, activation = 'softmax')
])

model.build(input_shape =(None, max_length))

model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

model.summary()
#-----------------------------------------
Total params: 677,062 (2.58 MB)
Trainable params: 676,966 (2.58 MB)
Non-trainable params: 96 (384.00 B)
#-----------------------------------------
Epoch 1/10
320/320 ━━━━━━━━━━━━━━━━━━━━ 21s 62ms/step - accuracy: 0.2866 - loss: 1.7470 - val_accuracy: 0.3070 - val_loss: 1.5807
Epoch 2/10
320/320 ━━━━━━━━━━━━━━━━━━━━ 20s 63ms/step - accuracy: 0.4481 - loss: 1.3999 - val_accuracy: 0.5059 - val_loss: 1.3619
Epoch 3/10
320/320 ━━━━━━━━━━━━━━━━━━━━ 20s 61ms/step - accuracy: 0.8172 - loss: 0.5406 - val_accuracy: 0.7801 - val_loss: 0.6895
Epoch 4/10
320/320 ━━━━━━━━━━━━━━━━━━━━ 19s 61ms/step - accuracy: 0.9320 - loss: 0.2169 - val_accuracy: 0.7859 - val_loss: 0.7194
Epoch 5/10
320/320 ━━━━━━━━━━━━━━━━━━━━ 21s 64ms/step - accuracy: 0.9639 - loss: 0.1062 - val_accuracy: 0.7910 - val_loss: 0.7263
Epoch 6/10
320/320 ━━━━━━━━━━━━━━━━━━━━ 20s 62ms/step - accuracy: 0.9758 - loss: 0.0809 - val_accuracy: 0.8359 - val_loss: 0.6926
Epoch 7/10
320/320 ━━━━━━━━━━━━━━━━━━━━ 20s 63ms/step - accuracy: 0.9793 - loss: 0.0599 - val_accuracy: 0.8328 - val_loss: 0.6868
Epoch 8/10
320/320 ━━━━━━━━━━━━━━━━━━━━ 19s 60ms/step - accuracy: 0.9823 - loss: 0.0532 - val_accuracy: 0.8492 - val_loss: 0.7714
Epoch 9/10
320/320 ━━━━━━━━━━━━━━━━━━━━ 19s 59ms/step - accuracy: 0.9812 - loss: 0.0600 - val_accuracy: 0.8441 - val_loss: 0.6282
Epoch 10/10
320/320 ━━━━━━━━━━━━━━━━━━━━ 19s 59ms/step - accuracy: 0.9871 - loss: 0.0395 - val_accuracy: 0.8590 - val_loss: 0.6915

