import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class VITBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, residual=True, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.residual = residual

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim)
        ])

    def call(self, inputs):
        # Self-attention block
        x_norm = self.norm1(inputs)
        attn_output = self.attn(x_norm, x_norm)
        if self.residual:
            x = inputs + attn_output
        else:
            x = attn_output

        # MLP block
        x_norm2 = self.norm2(x)
        mlp_output = self.mlp(x_norm2)
        if self.residual:
            x = x + mlp_output
        else:
            x = mlp_output
        return x
