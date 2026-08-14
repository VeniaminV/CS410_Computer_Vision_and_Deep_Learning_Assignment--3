"""
Veniamin Velikoretskikh
veniamin@pdx.edu
CS 410: Computer Vision & Deep Learning
Programming Assignment #3
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras import layers, models
from keras.applications import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input


DATA_DIR = "cats_dogs_dataset/dataset"
 
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# load InceptionResNetV2 with the weights from training on ImageNet
pre_model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(150, 150, 3))

pre_model.summary()  # will print every layer

# trainable_weights is every weight that isn't frozen yet 
trainable_params = sum(p.numpy().size for p in pre_model.trainable_weights)
print(f"\nTrainable parameters: {trainable_params:,}")


def visualize_first_layer_filters(model, save_path, num_filters=32):
    # grab the first conv layer in the model
    first_conv = None
    for layer in model.layers:
        if "Conv2D" in layer.__class__.__name__:
            first_conv = layer
            break

    # filters shape is (kernel_h, kernel_w, in_channels, out_channels)
    # for the first layer in_channels=3 (RGB) so each filter is basically
    # a tiny 3x3 image we can just plot directly
    filters = first_conv.get_weights()[0]
    print(f"First conv layer: '{first_conv.name}', filter shape: {filters.shape}")

    num_filters = min(num_filters, filters.shape[-1])
    cols = 8
    rows = int(np.ceil(num_filters / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    for i in range(rows * cols):
        ax = axes.flat[i]
        if i < num_filters:
            f = filters[:, :, :, i]
            # filter values aren't in [0,1] so normalize each one 
            f_norm = (f - f.min()) / (f.max() - f.min() + 1e-8)
            ax.imshow(f_norm)
        ax.axis("off")

    fig.suptitle(f"First-layer filters ({first_conv.name})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)


visualize_first_layer_filters(pre_model, "step1_first_layer_filters.png")



#step 2 -----------------------------------------------------------

# image_dataset_from_directory reads the images straight from the folder
# structure (dataset/training_set/cats, dataset/training_set/dogs, etc.)
# and figures out the labels from the folder names automatically
train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/training_set",
    label_mode="binary",   # gives labels as 0/1, matches our sigmoid output
    image_size=IMG_SIZE,   # resizes every image to 150x150 as it loads
    batch_size=BATCH_SIZE,
    shuffle=True,
)
 
test_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA_DIR}/test_set",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,  # keep test set in order so it lines up with labels later
)
 
class_names = train_ds.class_names
print("Classes:", class_names)  # alphabetical, so cats=0, dogs=1
 

train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))
 
# prefetch just lets the next batch load while the current one is being used
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
 
# check that everything loaded right
for images, labels in train_ds.take(1):
    print("batch shape:", images.shape)
    print("pixel value range:", float(images.numpy().min()), "to", float(images.numpy().max()))
    print("sample labels:", labels.numpy().flatten()[:8])



#step 3 -----------------------------------------------------------

# stack our own classifier on top of the pretrained base. pre_model
# outputs a feature map, not a single prediction, so we still need to
# flatten it and add dense layers to actually get a cat/dog prediction
model = models.Sequential()
model.add(pre_model)                              # the pretrained base
model.add(layers.Flatten())                        # flattens the feature map into a 1D vector
model.add(layers.Dense(256, activation="relu"))     # new hidden layer, starts randomly initialized
model.add(layers.Dense(1, activation="sigmoid"))    # single output, 0=cat 1=dog

model.summary()

# freeze the pretrained base so training only updates the new layers we
# just added, not the 54 million pretrained weights
pre_model.trainable = False

trainable_params = sum(p.numpy().size for p in model.trainable_weights)
non_trainable_params = sum(p.numpy().size for p in model.non_trainable_weights)
print(f"\nAfter freezing -- trainable: {trainable_params:,}  frozen: {non_trainable_params:,}")



#step 4(i) -----------------------------------------------------------

# need to compile before we can use the model, even just for predicting
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# run the model over the whole test set and collect predictions. the
# head is still randomly initialized at this point (we haven't trained
# it yet), so this is just a baseline -- expect accuracy close to 50%
y_true = []
y_pred = []
for images, labels in test_ds:
    probs = model.predict(images, verbose=0)
    preds = (probs > 0.5).astype(int).flatten()  # sigmoid output -> 0 or 1
    y_true.extend(labels.numpy().flatten().astype(int))
    y_pred.extend(preds)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

accuracy = (y_true == y_pred).mean()
print(f"\nTest accuracy (untrained head): {accuracy:.4f}")

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues")
plt.title(f"Untrained Head - Test Accuracy: {accuracy:.3f}")
plt.savefig("step4i_confusion_matrix.png", dpi=150)
print("Saved: step4i_confusion_matrix.png")
plt.close()



#step 4(ii) -----------------------------------------------------------

EPOCHS = 10  # start here, adjust based on how the per-epoch loss below looks

# actually train now, only the head updates since the base is frozen.
# validation_data=test_ds makes it report test loss/accuracy after every
# epoch too, which is what we need for "per-epoch test loss"
history = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)

print("\nPer-epoch test loss:")
for epoch, (loss, val_loss) in enumerate(zip(history.history["loss"], history.history["val_loss"]), 1):
    print(f"  epoch {epoch}: train_loss={loss:.4f}  test_loss={val_loss:.4f}")

# same evaluation as step 4(i), just re-run now that the head is trained
y_true = []
y_pred = []
for images, labels in test_ds:
    probs = model.predict(images, verbose=0)
    preds = (probs > 0.5).astype(int).flatten()
    y_true.extend(labels.numpy().flatten().astype(int))
    y_pred.extend(preds)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

accuracy = (y_true == y_pred).mean()
print(f"\nTest accuracy (trained head): {accuracy:.4f}")

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues")
plt.title(f"Trained Head - Test Accuracy: {accuracy:.3f}")
plt.savefig("step4ii_confusion_matrix.png", dpi=150)
print("Saved: step4ii_confusion_matrix.png")
plt.close()



#step 4(iii) -----------------------------------------------------------

# doing the same thing but with only PART of the pretrained network
# instead of the whole thing. cutting at "block35_5_ac" this is the
# end of the 5th (of 10 total) Inception-ResNet-A blocks, right before
# the network moves on to its next stage. picked this because it's a
# clean architectural boundary, not some random layer in the middle of
# a block
SUBNETWORK_CUTOFF = "block35_5_ac"
sub_model = models.Model(
    inputs=pre_model.input,
    outputs=pre_model.get_layer(SUBNETWORK_CUTOFF).output,
    name="subnetwork",
)

print(f"Sub-network cutoff layer: '{SUBNETWORK_CUTOFF}'")
print(f"Sub-network output shape: {sub_model.output_shape}")
print(f"Sub-network params: {sub_model.count_params():,}  (full network: {pre_model.count_params():,})")

# same transfer head as before, just stacked on the smaller sub_model instead
model2 = models.Sequential()
model2.add(sub_model)
model2.add(layers.Flatten())
model2.add(layers.Dense(256, activation="relu"))
model2.add(layers.Dense(1, activation="sigmoid"))

model2.summary()

sub_model.trainable = False

model2.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history2 = model2.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)

print("\nPer-epoch test loss (sub-network):")
for epoch, (loss, val_loss) in enumerate(zip(history2.history["loss"], history2.history["val_loss"]), 1):
    print(f"  epoch {epoch}: train_loss={loss:.4f}  test_loss={val_loss:.4f}")

y_true = []
y_pred = []
for images, labels in test_ds:
    probs = model2.predict(images, verbose=0)
    preds = (probs > 0.5).astype(int).flatten()
    y_true.extend(labels.numpy().flatten().astype(int))
    y_pred.extend(preds)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

accuracy2 = (y_true == y_pred).mean()
print(f"\nTest accuracy (sub-network, trained head): {accuracy2:.4f}")

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues")
plt.title(f"Sub-Network Transfer - Test Accuracy: {accuracy2:.3f}")
plt.savefig("step4iii_confusion_matrix.png", dpi=150)
print("Saved: step4iii_confusion_matrix.png")
plt.close()

print(f"\nFull network test accuracy (step 4ii): {accuracy:.4f}")
print(f"Sub-network test accuracy (step 4iii): {accuracy2:.4f}")