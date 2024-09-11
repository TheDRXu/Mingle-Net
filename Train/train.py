import time
import gc
import tensorflow as tf
from Evaluation_Metrics import metrics as m
from Augmentation import augment

import time
import gc
import tensorflow as tf

def train_one_epoch(epoch, model, X_train, y_train, X_val, y_val, X_test, y_test, learning_rate, min_loss_for_saving, model_path, EPOCHS):
    print(f'Epoch {epoch+1}/{EPOCHS}: Training')

    # Clear Keras session to free memory before each epoch
    tf.keras.backend.clear_session()
    
    # Generate dynamically augmented images for the current epoch
    X_augmented, y_augmented = augment.augment_images(X_train, y_train)

    # Train the model for one epoch on the dynamically augmented data
    model.fit(x=X_augmented, y=y_augmented, epochs=1, batch_size=8, validation_data=(X_val, y_val), verbose=1)

    # Evaluate on validation set
    val_loss = calculate_loss(model, X_val, y_val)
    print(f"Validation Loss: {val_loss:.4f}")

    # Evaluate on test set
    test_loss = calculate_loss(model, X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")

    # Save the model if the validation loss improves
    if val_loss < min_loss_for_saving:
        print(f"Validation loss improved from {min_loss_for_saving:.4f} to {val_loss:.4f}. Saving model...")
        model.save(model_path)
        min_loss_for_saving = val_loss

    # Clean up memory
    del X_augmented, y_augmented
    tf.keras.backend.clear_session()  # Consider removing tf.compat.v1.reset_default_graph()
    gc.collect()

    return min_loss_for_saving

def calculate_loss(model, X, y_true):
    predictions = model.predict(X, verbose=0, batch_size=8)
    loss = m.dice_metric_loss(y_true, predictions).numpy()
    return loss

def train_model(model, EPOCHS, X_train, y_train, X_val, y_val, X_test, y_test, learning_rate, model_path):
    print(f"Starting training with learning rate: {learning_rate}")
    
    min_loss_for_saving = 0.3  # Initial minimum validation loss to beat
    start_time = time.time()

    for epoch in range(EPOCHS):
        min_loss_for_saving = train_one_epoch(
            epoch, model, X_train, y_train, X_val, y_val, X_test, y_test, learning_rate, min_loss_for_saving, model_path, EPOCHS)
    
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training complete. Elapsed time: {int(hours)}h {int(minutes)}m {int(seconds)}s")