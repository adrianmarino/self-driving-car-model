class SimplePIController:

    @staticmethod
    def create_from(config):
        return SimplePIController(
            k_p=config['pi_controller']['k_p'],
            k_i=config['pi_controller']['k_i'],
            set_point=config['pi_controller']['set_point']
        )

    def __init__(self, k_p=0.1, k_i=0.002, set_point=0.):
        self.k_p = float(k_p)
        self.k_i = float(k_i)
        self.set_point = float(set_point)
        self.error = 0.
        self.integral = 0.

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.k_p * self.error + self.k_i * self.integral
