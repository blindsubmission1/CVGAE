import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--eval', action='store_true', help='only evaluation, no training.')

parser.add_argument('--name', default='tmp', help='model name determines the log path.')
parser.add_argument('--ds', default='mit', help='mit|ut|cgqa|fashion200k')
parser.add_argument('--close_world', action='store_true', help='open world in default')
parser.add_argument('--train_only', action='store_true')
parser.add_argument('--cpu_eval', action='store_true')
parser.add_argument('--ir_model', action='store_true')

parser.add_argument('--resnet', default='18', help='18|50|101|152')
parser.add_argument('--resnet_trainable', action='store_true')
parser.add_argument('--resnet_lr', type=float, default=5e-6)

parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--wd', type=float, default=0, help='weight decay')
parser.add_argument('--epoch', type=int, default=2000)
parser.add_argument('--bs', type=int, default=128, help='batch size')
parser.add_argument('--eval_every', type=int, default=1)

args = parser.parse_args()