import tensorflow as tf
import enum

class MappingType(enum.Enum):
    Identity = 0
    Linear = 1
    Affine = 2


class ODESolver(enum.Enum):
    SemiImplicit = 0
    Explicit = 1
    RungeKutta = 2
    

class LTCCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, units, ode_steps=6, **kwargs):
        self.units = units
        super(LTCCell, self).__init__(**kwargs)
        # Number of ODE solver steps in one RNN step
        self.ode_solver_unfolds = ode_steps
        self.solver = ODESolver.SemiImplicit

        self.input_mapping = MappingType.Affine

        self.erev_init_factor = 1

        self.w_init_max = 1.0
        self.w_init_min = 0.01
        self.cm_init_min = 0.5
        self.cm_init_max = 0.5
        self.gleak_init_min = 1
        self.gleak_init_max = 1

        self.w_min_value = 1e-5
        self.w_max_value = 1e3
        self.gleak_min_value = 1e-5
        self.gleak_max_value = 1e3
        self.cm_t_min_value = 1e-6
        self.cm_t_max_value = 1e3

        self.fix_cm = None
        self.fix_gleak = None
        self.fix_vleak = None


    @property
    def state_size(self):
        
        return (self.units)
    

    def build(self, input_shape):
        
        self.input_size = input_shape[-1]
        
        shape = (self.input_size,)
        if self.input_mapping in (MappingType.Affine, MappingType.Linear):
            self.input_w = tf.Variable(
                name="input_w", 
                shape=shape, 
                trainable=True, 
                initial_value=tf.ones(shape)
            )
            
        shape = (self.input_size,)
        if self.input_mapping == MappingType.Affine:
            self.input_b = tf.Variable(
                name="input_b",
                shape=shape,
                trainable=True,
                initial_value=tf.zeros(shape)
            )
        
        shape = (self.input_size, self.units)
        self.sensory_mu = tf.Variable(
            name="sensory_mu",
            shape=shape,
            trainable=True,
            initial_value=tf.random.uniform(shape, minval=0.3, maxval=0.8)
        )
        
        shape = (self.input_size, self.units)
        self.sensory_sigma = tf.Variable(
            name="sensory_sigma",
            shape=shape,
            trainable=True,
            initial_value=tf.random.uniform(shape, minval=3.0, maxval=8.0)
        )
        
        shape = (self.input_size, self.units)
        self.sensory_W = tf.Variable(
            name="sensory_W",
            shape=shape,
            trainable=True,
            initial_value=tf.random.uniform(
                shape, 
                minval=self.w_init_min, 
                maxval=self.w_init_max
            )
        )
        
        shape = (self.input_size, self.units)
        sensory_erev_init = 2 * tf.random.uniform(
            shape, 
            minval=0, 
            maxval=2, 
            dtype=tf.dtypes.int32
        ) - 1
        sensory_erev_init = tf.cast(sensory_erev_init, tf.dtypes.float32)
        self.sensory_erev = tf.Variable(
            name="sensory_erev",
            shape=shape,
            trainable=True,
            initial_value=sensory_erev_init*self.erev_init_factor
        )

        shape = (self.units, self.units)
        self.mu = tf.Variable(
            name="mu",
            shape=shape,
            trainable=True,
            initial_value=tf.random.uniform(shape, minval=0.3, maxval=0.8)
        )
        
        shape = (self.units, self.units)
        self.sigma = tf.Variable(
            name="sigma",
            shape=shape,
            trainable=True,
            initial_value=tf.random.uniform(shape, minval=3.0, maxval=8.0)
        )
        
        shape = (self.units, self.units)
        self.W = tf.Variable(
            name="W",
            shape=shape,
            trainable=True,
            initial_value=tf.random.uniform(
                shape,
                minval=self.w_init_min,
                maxval=self.w_init_max
            )
        )

        shape = (self.units, self.units)
        erev_init = 2 * tf.random.uniform(
            shape, 
            minval=0, 
            maxval=2, 
            dtype=tf.dtypes.int32
        ) - 1
        erev_init = tf.cast(erev_init, tf.dtypes.float32)
        self.erev = tf.Variable(
            name="erev",
            shape=shape,
            trainable=True,
            initial_value=erev_init*self.erev_init_factor
        )

        shape = (self.units,)
        if self.fix_vleak is None:
            self.vleak = tf.Variable(
                name="vleak",
                shape=shape,
                trainable=True,
                initial_value=tf.random.uniform(shape, minval=-0.2, maxval=0.2)
            )
        else:
            self.vleak = tf.Variable(
                name="vleak",
                shape=shape,
                trainable=False,
                initial_value=self.fix_vleak*tf.ones(shape)
            )
            
        shape = (self.units,)
        if self.fix_gleak is None:
            if self.gleak_init_max > self.gleak_init_min:
                initial_value = tf.random.uniform(
                    shape,
                    minval=self.gleak_init_min, 
                    maxval=self.gleak_init_max
                )
            else:
                initial_value = self.gleak_init_min * tf.ones(shape)
            self.gleak = tf.Variable(
                name="gleak",
                shape=shape,
                trainable=True,
                initial_value=initial_value
            )
        else:
            self.gleak = tf.Variable(
                name="gleak",
                shape=shape,
                trainable=False,
                initial_value=self.fix_gleak*tf.ones(shape)
            )

        shape = (self.units,)
        if self.fix_cm is None:
            if self.cm_init_max > self.cm_init_min:
                initial_value = tf.random.uniform(
                    shape,
                    minval=self.cm_init_min, 
                    maxval=self.cm_init_max
                )
            else:
                initial_value = self.cm_init_min*tf.ones(shape)
            self.cm_t = tf.Variable(
                name="cm_t",
                shape=shape,
                trainable=True,
                initial_value=initial_value
            )
        else:
            self.cm_t = tf.Variable(
                name="cm_t",
                shape=shape,
                trainable=False,
                initializer=self.fix_cm*tf.ones(shape),
            )
            

        self.built = True


    def call(self, inputs, states):
        
        state = states[0]
        
        inputs = self.map_inputs(inputs)
        
        if self.solver == ODESolver.Explicit:
            next_state = self.ode_step_explicit(
                inputs, state, ode_solver_unfolds=self.ode_solver_unfolds
            )

        elif self.solver == ODESolver.SemiImplicit:
            next_state = self.ode_step(inputs, state)

        elif self.solver == ODESolver.RungeKutta:
            next_state = self.ode_step_runge_kutta(inputs, state)

        else:
            raise ValueError(f"Unknown ODE solver '{str(self.solver)}'")

        outputs = next_state

        return (outputs, next_state)


    def get_config(self):
        config = super(LTCCell, self).get_config()
        config.update({
            "units": self.units, 
            "ode_steps": self.ode_solver_unfolds
        })

        return (config)
    
    @classmethod
    def from_config(cls, config):
        return (cls(**config))
        


    def get_param_constrain_op(self):

        cm_clipping_op = self.cm_t.assign(
            tf.clip_by_value(self.cm_t, self.cm_t_min_value, self.cm_t_max_value)
        )
        gleak_clipping_op = self.gleak.assign(
            tf.clip_by_value(self.gleak, self.gleak_min_value, self.gleak_max_value)
        )
        w_clipping_op = self.W.assign(
            tf.clip_by_value(self.W, self.w_min_value, self.w_max_value)
        )
        sensory_w_clipping_op = self.sensory_W.assign(
            tf.clip_by_value(self.sensory_W, self.w_min_value, self.w_max_value)
        )

        return (cm_clipping_op, gleak_clipping_op, w_clipping_op, sensory_w_clipping_op)


    def map_inputs(self, inputs):
        
        if self.input_mapping in (MappingType.Affine, MappingType.Linear):
            inputs = inputs * self.input_w
        if self.input_mapping == MappingType.Affine:
            inputs = inputs + self.input_b
            
        return (inputs)
    
    # Hybrid euler method
    def ode_step(self, inputs, state):
        v_pre = state

        sensory_w_activation = self.sensory_W * self.sigmoid(
            inputs, 
            self.sensory_mu, 
            self.sensory_sigma
        )
        
        sensory_rev_activation = sensory_w_activation * self.sensory_erev

        w_numerator_sensory = tf.reduce_sum(sensory_rev_activation, axis=1)
        w_denominator_sensory = tf.reduce_sum(sensory_w_activation, axis=1)

        for t in range(self.ode_solver_unfolds):
            w_activation = self.W * self.sigmoid(v_pre, self.mu, self.sigma)

            rev_activation = w_activation * self.erev

            w_numerator = tf.reduce_sum(rev_activation, axis=1) + w_numerator_sensory
            w_denominator = tf.reduce_sum(w_activation, axis=1) + w_denominator_sensory

            numerator = self.cm_t * v_pre + self.gleak * self.vleak + w_numerator
            denominator = self.cm_t + self.gleak + w_denominator

            v_pre = numerator / denominator

        return (v_pre)
    

    def sigmoid(self, v_pre, mu, sigma):
        v_pre = tf.reshape(v_pre, (-1, v_pre.shape[-1], 1))
        
        return (tf.nn.sigmoid(sigma * (v_pre - mu)))


class LTCLayer(tf.keras.layers.RNN):
    def __init__(self, units, *args, ode_steps=6, **kwargs):
        if "time_major" not in kwargs:
            print("Defaulting to time_major=True for LTCLayer.")
            kwargs["time_major"] = True
        elif kwargs["time_major"] is False:
            raise (NotImplementedError(
                "time_major=False has not been implemented for LTCLayer."
            ))
        self.units = units
        self.ode_steps = ode_steps
        
        super(LTCLayer, self).__init__(
            LTCCell(units, ode_steps=ode_steps),
            *args,
            **kwargs
        )


    @classmethod
    def from_config(cls, config):
        del (config["cell"])
        return (cls(**config))


    def get_config(self):
        config = super(LTCLayer, self).get_config()
        config.update({
            "units": self.units,
            "ode_steps": self.ode_steps
        })
        return (config)