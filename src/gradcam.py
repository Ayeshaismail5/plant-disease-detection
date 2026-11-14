import tensorflow as tf
import numpy as np
import cv2

def gradcam(model, img_array, class_index=None, alpha=0.4):
    """
    Computes and overlays Grad-CAM heatmaps for Keras 3 models containing nested backbones.
    - img_array: A single raw RGB image array of shape (224, 224, 3) with values in range [0-255].
    """
    # Add batch dimension and convert to float32 tensor
    img_tensor = tf.expand_dims(tf.cast(img_array, tf.float32), axis=0)

    # Identify the nested base model by type
    base = [l for l in model.layers if isinstance(l, tf.keras.Model)][0]

    # Automatically detect the last convolutional layer of the base model safely
    last_conv_layer = None
    for layer in reversed(base.layers):
        try:
            # Check layer name/class or if output shape has 4 dimensions
            if hasattr(layer, 'output_shape') and layer.output_shape is not None:
                shape = layer.output_shape
                if len(shape) == 4:
                    last_conv_layer = layer
                    break
            # Fallback for layers like activation/normalization where shape may be hidden
            elif len(layer.output.shape) == 4:
                last_conv_layer = layer
                break
        except Exception:
            continue

    # Fallback to known layer names for common backbones if auto-detection failed
    if last_conv_layer is None:
        for layer_name in ['top_activation', 'out_relu', 'conv_pw_13_relu']:
            try:
                last_conv_layer = base.get_layer(layer_name)
                break
            except Exception:
                continue

    if last_conv_layer is None:
        raise ValueError("Could not find a 4D output convolutional layer in the base model.")

    # Build a sub-model from the base's input to [last_conv_layer.output, base.output]
    sub_model = tf.keras.Model(inputs=base.inputs, outputs=[last_conv_layer.output, base.output])

    # Track gradients from the custom final layers manually under a GradientTape
    with tf.GradientTape() as tape:
        # Forward pass through layers leading to the base model
        x = img_tensor
        for layer in model.layers:
            if layer == base:
                break
            # Skip InputLayer to avoid signature/argument errors
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            # Safely pass training=False only if the layer signature accepts it
            try:
                x = layer(x, training=False)
            except TypeError:
                x = layer(x)

        # Pass through base sub-model
        conv_outputs, base_features = sub_model(x, training=False)

        # Forward pass through the top classification head
        base_idx = model.layers.index(base)
        head_x = base_features
        for layer in model.layers[base_idx+1:]:
            try:
                head_x = layer(head_x, training=False)
            except TypeError:
                head_x = layer(head_x)

        # Extract target class score
        if class_index is None:
            class_index = tf.argmax(head_x[0])
        loss = head_x[:, class_index]

    # Calculate gradients of loss w.r.t the last convolutional feature map
    grads = tape.gradient(loss, conv_outputs)

    # Compute channel importance weights via global average pooling of gradients
    guided_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Compute weighted sum of the conv feature maps
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(guided_grads * conv_outputs, axis=-1)

    # Apply ReLU activation and normalize the heatmap
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val
    heatmap = heatmap.numpy()

    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply JET colormap
    color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

    # Superimpose the heatmap onto the original image
    overlay = np.float32(color_heatmap) * alpha + np.float32(img_array)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay, int(class_index)
