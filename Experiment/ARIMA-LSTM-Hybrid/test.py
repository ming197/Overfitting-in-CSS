
import argparse
import ast

parser = argparse.ArgumentParser(description='Parameters Configuration!')
parser.add_argument('--reg', '-r', help='正则化', type=ast.literal_eval, default=False)
parser.add_argument('--dropout', '-d', help='Dropout', type=ast.literal_eval, default=False)
args = parser.parse_args()
Reg = args.reg
Dropout = args.dropout


print('Reg: ', Reg)

print('Dropout: ', Dropout)