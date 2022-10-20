from master.parameters.parameters import hyper_para
from master.src.utils import *


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if hyper_para.mode == 'train':
        train()
    elif hyper_para.mode == 'test':
        test()


if __name__ == "__main__":

    if hyper_para.mode == 'train':
        train()
    elif hyper_para.mode == 'test':
        test()