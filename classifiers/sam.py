import tensorflow as tf

class SAMOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, base_optimizer, rho=0.05, eps=1e-12, **kwargs):
        super(SAMOptimizer, self).__init__(**kwargs)
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.eps = eps

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "e_w")

    def _resource_apply_dense(self, grad, var):
        e_w = self.get_slot(var, "e_w")
        grad_norm = tf.linalg.global_norm([grad])
        ew_multiplier = self.rho / (grad_norm + self.eps)
        e_w.assign(grad * ew_multiplier)
        self.base_optimizer._resource_apply_dense(e_w, var)

    def _resource_apply_sparse(self, grad, var, indices):
        print("salam")

    def get_config(self):
        config = super().get_config()
        config.update({
            'base_optimizer': tf.keras.optimizers.serialize(self.base_optimizer),
            'rho': self.rho,
            'eps': self.eps
        })
        return config
