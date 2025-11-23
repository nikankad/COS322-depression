import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from tensorflow import keras


class NeuralNetworkVisualizer:
    def __init__(self, model, feature_names=None, class_labels=["No", "Yes"]):
        self.model = model
        self.feature_names = feature_names
        self.class_labels = class_labels

        # Filter visualization layers (ignore Dropout)
        self.visual_layers = [
            l for l in model.layers if not isinstance(l, keras.layers.Dropout)
        ]

        # Get input dimension
        try:
            input_size = model.input_shape[1]
        except Exception:
            input_size = model.layers[0].input_shape[1]

        # layer_sizes = input layer + units of each visualized layer
        self.layer_sizes = [input_size] + [l.units for l in self.visual_layers]

    def _get_activations(self, input_sample):
        """Forward pass collecting activations layer-by-layer."""
        current_value = np.array(input_sample).reshape(1, -1)
        activations = [current_value]  # input layer

        for layer in self.model.layers:
            current_value = layer(current_value)
            if layer in self.visual_layers:
                activations.append(current_value.numpy())

        return activations

    def create_animation(self, input_sample):
        activations = self._get_activations(input_sample)

        # ==========================================================
        # TRUE MODEL PREDICTION (FIXED!)
        # ==========================================================
        final_raw = self.model(np.array(input_sample).reshape(1, -1)).numpy().ravel()

        # Decide if model is sigmoid or softmax
        if final_raw.shape[0] == 1:
            # Binary sigmoid
            display_val = float(final_raw[0])
            predicted_idx = int(np.round(display_val))
        else:
            # Softmax
            exps = np.exp(final_raw - np.max(final_raw))
            probs = exps / exps.sum()
            display_val = float(probs[1])     # Probability of class 1 (Yes/Depression)
            predicted_idx = int(np.argmax(probs))

        predicted_class = (
            self.class_labels[predicted_idx]
            if predicted_idx < len(self.class_labels)
            else str(predicted_idx)
        )

        # ==========================================================
        # PLOTTING BASE
        # ==========================================================
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis("off")

        # Neuron positions
        x_positions, y_positions = [], []
        max_neurons = max(self.layer_sizes)

        for i, size in enumerate(self.layer_sizes):
            x_positions.append([i] * size)
            y_positions.append(np.linspace(0, max_neurons, size + 2)[1:-1])

        # ==========================================================
        # STATIC RED CONNECTION LINES
        # ==========================================================
        for i in range(len(self.layer_sizes) - 1):
            n_from, n_to = len(y_positions[i]), len(y_positions[i + 1])
            dense = n_from * n_to

            if dense < 2000:
                alpha = 0.15
                lw = 0.7
            else:
                alpha = 0.05
                lw = 0.4

            for y1 in y_positions[i]:
                for y2 in y_positions[i + 1]:
                    ax.plot([i, i + 1], [y1, y2], "r-", alpha=alpha, lw=lw, zorder=1)

        # ==========================================================
        # INPUT LABELS
        # ==========================================================
        if self.feature_names:
            for idx, y in enumerate(y_positions[0]):
                if idx < len(self.feature_names):
                    name = self.feature_names[idx]
                    val = float(input_sample[idx])
                    ax.text(-0.1, y, f"{name}\n({val:.2f})",
                            ha="right", va="center",
                            fontsize=9, color="#333")

        # ==========================================================
        # OUTPUT LABEL (HIDDEN INITIALLY)
        # ==========================================================
        out_x = len(self.layer_sizes) - 1
        out_y = y_positions[-1][0]
        result_text = f"Output: {display_val:.3f}\nPred: {predicted_class}"

        output_text_obj = ax.text(
            out_x + 0.1, out_y, result_text,
            ha="left", va="center",
            fontsize=12, fontweight="bold",
            alpha=0.0,  # hidden until last frame
            zorder=20
        )

        # ==========================================================
        # DRAW NEURONS
        # ==========================================================
        scatter_points = []
        for i in range(len(self.layer_sizes)):
            initial_colors = np.full(self.layer_sizes[i], 0.5)
            scat = ax.scatter(
                x_positions[i], y_positions[i],
                s=300,
                c=initial_colors,
                cmap="plasma",
                vmin=0.0, vmax=1.0,
                edgecolors="black",
                zorder=10
            )
            scatter_points.append(scat)

        # ==========================================================
        # COLORBAR (BOTTOM)
        # ==========================================================
        sm = plt.cm.ScalarMappable(cmap="plasma",
                                   norm=plt.Normalize(0.0, 1.0))
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax,
                            orientation="horizontal",
                            fraction=0.05, pad=0.08)
        cbar.set_label("Relative activation (low → high)", fontsize=10)

        ax.set_title("Neural Network Processing Animation", fontsize=16)

        # ==========================================================
        # ANIMATION UPDATE FUNCTION
        # ==========================================================
        def update(frame):
            if frame >= len(activations):
                return []

            artists = []

            # Min–max normalization across all layers up to current frame
            all_vals = np.concatenate([
                activations[k].flatten().astype(float)
                for k in range(frame + 1)
            ])

            vmin = np.percentile(all_vals, 5)
            vmax = np.percentile(all_vals, 95)
            if vmax <= vmin:
                vmax = vmin + 1e-6

            # Update layer color values
            for layer_idx in range(frame + 1):
                vals = activations[layer_idx].flatten().astype(float)
                vals = np.clip(vals, vmin, vmax)
                norm_vals = (vals - vmin) / (vmax - vmin)

                n = min(len(norm_vals), self.layer_sizes[layer_idx])
                scatter_points[layer_idx].set_array(norm_vals[:n])
                artists.append(scatter_points[layer_idx])

            # Reveal output only at last frame
            if frame == len(activations) - 1:
                output_text_obj.set_alpha(1.0)
            else:
                output_text_obj.set_alpha(0.0)

            return artists

        # ==========================================================
        # RUN ANIMATION
        # ==========================================================
        anim = FuncAnimation(
            fig, update,
            frames=len(activations),
            interval=700,
            blit=False
        )

        plt.close(fig)
        return anim
