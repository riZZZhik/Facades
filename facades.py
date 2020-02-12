class Facades:  # TODO: Predict  # TODO: Comments
    def __init__(self, dim=(720, 1080), n_channels=3):
        self.dim = dim
        self.n_channels = n_channels
        self.shape = dim + (n_channels,)

        self.model = None
        self.train_generator, self.valid_generator = None, None
        self.batch_size = None

        # TODO: Train init bool
        self.training_init_check = False


if __name__ == "__main__":
    facades = Facades((512, 1024))
