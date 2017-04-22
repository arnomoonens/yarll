class FunctionApproximator(object):
    """Map states and actions using a function."""
    def __init__(self, n_actions):
        super(FunctionApproximator, self).__init__()
        self.n_actions = n_actions
        self.thetas = []
        self.features_shape = (0)

    def get_summed_thetas(state, action):
        return 0

    def set_thetas(self, addition):
        self.thetas += addition
