import argparse
from message.monitor import FileMonitor


parser = argparse.ArgumentParser(usage='"run_minimal.py -h" to show help.')
parser.add_argument('-p', '--path', dest='path', type=str, required=True, help='Monitor path')
parser.add_argument('-t', '--interval', dest='interval', type=int, default=600, help='Monitor run interval in seconds, default 600')
parser.add_argument('-at', '--at_mobiles', dest='at_mobiles', type=str, default="", help='Monitor dingtalk message at, default empty')
args = parser.parse_args()

f = FileMonitor(args.path, "minimal/param", ".npz", interval=args.interval, at_mobiles=args.at_mobiles.split(','))
f.run()
