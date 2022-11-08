class CustomModel(tf.keras.models.Model):

    def CustomSteps(self, steps):
        self.steps = steps
        self.step_counter = 0

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        #self.step_counter = self.step_counter + 1
        self.loss['categorical'].increase_counter()# = tf.cast(self.step_counter/self.steps, tf.float32)
        #tf.print (tf.cast(self.step_counter/self.steps, tf.float32))
        return super().train_step(data)
        

        #with tf.GradientTape() as tape:
        #    y_pred = self(x, training=True)  # Forward pass
        #    # Compute the loss value
        #    # (the loss function is configured in `compile()`)
        #    print (self.loss)
        #    loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
#
        ## Compute gradients
        #trainable_vars = self.trainable_variables
        #gradients = tape.gradient(loss, trainable_vars)
        ## Update weights
        #self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        ## Update metrics (includes the metric that tracks the loss)
        #self.compiled_metrics.update_state(y, y_pred)
        ## Return a dict mapping metric names to current value
        #return {m.name: m.result() for m in self.metrics}