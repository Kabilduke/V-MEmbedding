Emodel = models.Sequential([
    layers.Embedding(input_dim = vocal_size, output_dim = 256), #With Vocal Size = 10000, 256-dimensions
    
    layers.Conv1D(64, kernel_size = 3, activation = 'relu'),
    layers.MaxPooling1D(pool_size = 4),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Conv1D(32, kernel_size = 3, activation = 'elu'),
    layers.MaxPooling1D(pool_size = 3),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Flatten(),
    layers.Dense(16, activation = 'elu'),
    layers.Dropout(0.2),
    
    layers.Dense(6, activation = 'softmax')
])

Emodel.build(input_shape =(None, max_length)) #Max_length = 66

Emodel.compile(
    optimizer='rmsprop', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

Emodel.summary()

Epoch 1/10
320/320 ━━━━━━━━━━━━━━━━━━━━ 4s 12ms/step - accuracy: 0.2842 - loss: 1.9353 - val_accuracy: 0.4676 - val_loss: 1.5695
Epoch 2/10
320/320 ━━━━━━━━━━━━━━━━━━━━ 4s 13ms/step - accuracy: 0.5723 - loss: 1.1715 - val_accuracy: 0.7547 - val_loss: 0.6991
Epoch 3/10
320/320 ━━━━━━━━━━━━━━━━━━━━ 4s 13ms/step - accuracy: 0.8157 - loss: 0.5562 - val_accuracy: 0.8734 - val_loss: 0.3821
Epoch 4/10
320/320 ━━━━━━━━━━━━━━━━━━━━ 4s 13ms/step - accuracy: 0.8987 - loss: 0.3219 - val_accuracy: 0.8852 - val_loss: 0.3220
Epoch 5/10
320/320 ━━━━━━━━━━━━━━━━━━━━ 4s 12ms/step - accuracy: 0.9302 - loss: 0.2259 - val_accuracy: 0.8949 - val_loss: 0.3135
Epoch 6/10
320/320 ━━━━━━━━━━━━━━━━━━━━ 4s 13ms/step - accuracy: 0.9438 - loss: 0.1827 - val_accuracy: 0.8934 - val_loss: 0.3309
Epoch 7/10
320/320 ━━━━━━━━━━━━━━━━━━━━ 4s 13ms/step - accuracy: 0.9487 - loss: 0.1584 - val_accuracy: 0.8953 - val_loss: 0.3320
Epoch 8/10
320/320 ━━━━━━━━━━━━━━━━━━━━ 4s 13ms/step - accuracy: 0.9588 - loss: 0.1375 - val_accuracy: 0.8922 - val_loss: 0.3306
Epoch 9/10
320/320 ━━━━━━━━━━━━━━━━━━━━ 4s 12ms/step - accuracy: 0.9632 - loss: 0.1182 - val_accuracy: 0.8848 - val_loss: 0.4043
Epoch 10/10
320/320 ━━━━━━━━━━━━━━━━━━━━ 4s 12ms/step - accuracy: 0.9693 - loss: 0.0971 - val_accuracy: 0.8926 - val_loss: 0.4021
