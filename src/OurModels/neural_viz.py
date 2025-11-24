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

    def _get_all_sample_results(self, X):
        """
        Forward pass for ALL samples, collecting activations and final predictions.
        
        Returns:
            all_activations: list of lists of np.arrays ( [ [sample1_activations], [sample2_activations], ... ] )
            all_predictions: list of dicts ( [{'display_val': ..., 'predicted_class': ...}, ...] )
        """
        X = np.array(X)
        all_activations = []
        all_predictions = []

        for input_sample in X:
            current_value = input_sample.reshape(1, -1)
            sample_activations = [current_value]  # input layer

            for layer in self.model.layers:
                current_value = layer(current_value)
                if layer in self.visual_layers:
                    sample_activations.append(current_value.numpy())

            # --- Calculate Model Prediction ---
            final_raw = self.model(input_sample.reshape(1, -1)).numpy().ravel()
            
            if final_raw.shape[0] == 1:
                # Binary sigmoid
                display_val = float(final_raw[0])
                predicted_idx = int(np.round(display_val))
            else:
                # Softmax
                exps = np.exp(final_raw - np.max(final_raw))
                probs = exps / exps.sum()
                # Probability of class 1 (Assuming two classes)
                display_val = float(probs[1]) 
                predicted_idx = int(np.argmax(probs))

            predicted_class = (
                self.class_labels[predicted_idx]
                if predicted_idx < len(self.class_labels)
                else str(predicted_idx)
            )
            
            all_activations.append(sample_activations)
            all_predictions.append({
                'display_val': display_val,
                'predicted_class': predicted_class,
                'input_sample': input_sample
            })

        return all_activations, all_predictions

    def create_animation(self, X):
        """
        Generates an animation visualizing the forward pass for multiple input samples in X.
        """
        all_activations, all_predictions = self._get_all_sample_results(X)
        num_samples = len(X)
        num_layers = len(self.layer_sizes)

        # Total frames: (layers per sample) * num_samples
        total_frames = num_samples * num_layers

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
        for i in range(num_layers - 1):
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
        # DYNAMIC LABELS
        # ==========================================================
        
        # Input Labels
        input_label_objs = []
        if self.feature_names:
            for idx, y in enumerate(y_positions[0]):
                if idx < len(self.feature_names):
                    obj = ax.text(-0.1, y, "",
                                  ha="right", va="center",
                                  fontsize=9, color="#333")
                    input_label_objs.append(obj)

        # Output Label
        out_x = num_layers - 1
        out_y = y_positions[-1][0]
        output_text_obj = ax.text(
            out_x + 0.1, out_y, "",
            ha="left", va="center",
            fontsize=12, fontweight="bold",
            alpha=0.0,
            zorder=20
        )
        
        # Sample Indicator
        sample_indicator = ax.text(
            0.5, 1.05, f"Sample 1 of {num_samples}",
            ha="center", va="bottom",
            fontsize=14, fontweight="bold",
            transform=ax.transAxes
        )


        # ==========================================================
        # DRAW NEURONS
        # ==========================================================
        scatter_points = []
        for i in range(num_layers):
            initial_colors = np.full(self.layer_sizes[i], 0.5)
            scat = ax.scatter(
                x_positions[i], y_positions[i],
                s=300,
                c=initial_colors,
                cmap="viridis",
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

        ax.set_title("Neural Network Multi-Sample Processing Animation", fontsize=16)

        # ==========================================================
        # ANIMATION UPDATE FUNCTION
        # ==========================================================
        def update(frame):
            # Determine which sample and which layer we are on
            sample_idx = frame // num_layers
            layer_idx = frame % num_layers
            
            # Use activations and prediction results for the current sample
            current_activations = all_activations[sample_idx]
            current_result = all_predictions[sample_idx]
            current_input = current_result['input_sample']

            artists = []
            
            # --- Update Sample Indicator ---
            sample_indicator.set_text(f"Sample {sample_idx + 1} of {num_samples}")
            artists.append(sample_indicator)

            # --- Update Input Labels in Layer 0 ---
            if layer_idx == 0 and self.feature_names:
                for idx, name in enumerate(self.feature_names):
                    if idx < len(current_input):
                        val = float(current_input[idx])
                        input_label_objs[idx].set_text(f"{name}\n({val:.2f})")
                        
            # --- Reset and Update Neuron Colors ---
            # Reset colors of later layers when moving to a new sample
            if layer_idx == 0 and frame > 0:
                for scat in scatter_points:
                    scat.set_array(np.full(len(scat.get_offsets()), 0.5))
                output_text_obj.set_alpha(0.0)
            
            # Min–max normalization across all layers processed *so far* for the current sample
            all_vals = np.concatenate([
                current_activations[k].flatten().astype(float)
                for k in range(layer_idx + 1)
            ])

            vmin = np.percentile(all_vals, 5)
            vmax = np.percentile(all_vals, 95)
            if vmax <= vmin:
                vmax = vmin + 1e-6

            # Update layer color values
            for l_idx in range(layer_idx + 1):
                vals = current_activations[l_idx].flatten().astype(float)
                vals = np.clip(vals, vmin, vmax)
                norm_vals = (vals - vmin) / (vmax - vmin)

                n = min(len(norm_vals), self.layer_sizes[l_idx])
                scatter_points[l_idx].set_array(norm_vals[:n])
                artists.append(scatter_points[l_idx])
            
            # --- Update Output Text ---
            if layer_idx == num_layers - 1:
                result_text = (
                    f"Output: {current_result['display_val']:.3f}\n"
                    f"Pred: {current_result['predicted_class']}"
                )
                output_text_obj.set_text(result_text)
                output_text_obj.set_alpha(1.0)
            else:
                output_text_obj.set_alpha(0.0)

            return artists

        # ==========================================================
        # RUN ANIMATION
        # ==========================================================
        anim = FuncAnimation(
            fig, update,
            frames=total_frames,
            interval=700,
            blit=False
        )

        plt.close(fig)
        return anim