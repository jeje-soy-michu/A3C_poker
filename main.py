import argparse

from master import Master

parser = argparse.ArgumentParser(description='Run A3C algorithm on the game '
                                             'Poker.')

parser.add_argument('--train', dest='train', action='store_true',
                    help='Train our model.')
parser.add_argument('--new_model', dest='new_model', action='store_true',
                    help='Train our model.')
parser.add_argument('--lr', default=3e-3,
                    help='Learning rate for the shared optimizer.')
parser.add_argument('--max-eps', default=1000, type=int,
                    help='Global maximum number of episodes to run.')
parser.add_argument('--debug', default=1, type=int,
                    help='Debug level.')
parser.add_argument('--gamma', default=0.99,
                    help='Discount factor of rewards.')
parser.add_argument('--save-dir', default='./models/', type=str,
                    help='Directory in which you desire to save the model.')

def main():
    args = parser.parse_args()
    master = Master(args)
    if args.train:
        master.train()
    else:
        master.play()

if __name__ == '__main__':
    main()
