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
