from options.base_options import BaseOptions


class TestT2MOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
        self.parser.add_argument('--top_k', type=int, default=100)

        self.parser.add_argument('--do_denoise', action="store_true")

        self.parser.add_argument('--sample', action="store_true")
        self.parser.add_argument('--repeat_times', type=int, default=5)
        self.parser.add_argument('--beam_size', type=int, default=2)
        self.parser.add_argument('--split_file', type=str, default='test.txt')

        self.parser.add_argument('--text_file', type=str, default="./input.txt", help='Frequency of plot points')
        self.parser.add_argument('--aist_split_file', type=str, default=None, help='Frequency of plot points')
        self.parser.add_argument('--random_text_list', type=str, default=None, help='Frequency of plot points')
        self.parser.add_argument('--mtp_file', type=str, default=None, help='Frequency of plot points')

        self.parser.add_argument('--which_epoch', type=str, default="finest", help='Frequency of plot points')
        self.parser.add_argument('--which_vqvae', type=str, default="finest", help='Frequency of plot points')
        self.parser.add_argument('--result_path', type=str, default="./eval_results/", help='Frequency of plot points')
        self.parser.add_argument('--num_results', type=int, default=40, help='Batch size of pose discriminator')

        self.parser.add_argument('--ext', type=str, default='default', help='Batch size of pose discriminator')
        self.parser.add_argument('--eval_mode', type=str, default='vis', help='Batch size of pose discriminator')

        self.is_train = False
