import random
from dataloader import Dataset
from model import *
import warnings
import argparse
from warnings import simplefilter
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description='HiGSP')
    parser.add_argument('--dataset', default='ml100k', type=str)
    parser.add_argument('--alpha1', default=0.08, type=float)
    parser.add_argument('--alpha2', default=0.73, type=float)
    parser.add_argument('--order1', default=2, type=int)
    parser.add_argument('--order2', default=12, type=int)
    parser.add_argument('--pri_factor', default=80, type=int)
    parser.add_argument('--n_clusters', default=25, type=int)
    args = parser.parse_args()

    random.seed(2020)
    np.random.seed(2020)

    dataloader = Dataset(args.dataset)
    init_adj_mat = dataloader.get_init_mats()
    user_interacted_items_dict, valid_user_items_dict, test_user_items_dict = dataloader.get_train_test_data()

    lm = HiGSP(init_adj_mat,
               alpha1=args.alpha1, alpha2=args.alpha2, order1=args.order1, order2=args.order2, n_clusters=args.n_clusters, pri_factor=args.pri_factor,
               user_interacted_items_dict=user_interacted_items_dict)
    print('Train model...')
    ratings = lm.train()

    print('Evaluate model...')
    valid_f1s, valid_mrrs, valid_ndcgs = lm.eval_test(valid_user_items_dict, ratings)  # for hyper-parameter tuning
    test_f1s, test_mrrs, test_ndcgs = lm.eval_test(test_user_items_dict, ratings)
    test_metric_info = '\tF1@5:{:.4f}\tMRR@5:{:.4f}\tNDCG@5:{:.4f}\n\tF1@10:{:.4f}\tMRR@10:{:.4f}\tNDCG@10:{:.4f}'\
        .format(test_f1s[0], test_mrrs[0], test_ndcgs[0], test_f1s[1], test_mrrs[1], test_ndcgs[1])
    print('[Test]\n{}'.format(test_metric_info))


if __name__ == '__main__':
    main()
