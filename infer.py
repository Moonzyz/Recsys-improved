import argparse
import torch
import numpy as np
import random
from tqdm import tqdm
from models.a_llmrec_model import *
from pre_train.sasrec.utils import data_partition

def evaluate(model, dataset, args, k=10):
    [train, valid, test, usernum, itemnum] = dataset

    ndcg = 0.0
    hit_rate = 0.0
    valid_user = 0

    users = random.sample(range(1, usernum + 1), 10000) if usernum > 10000 else range(1, usernum + 1)

    for u in tqdm(users):
        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        # Lấy chuỗi lịch sử người dùng
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]  
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        # Negative sampling
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]  # Sản phẩm positive
        for _ in range(19):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        with torch.no_grad():
            u_tensor = torch.LongTensor([u]).to(args.device)
            seq_tensor = torch.LongTensor([seq]).to(args.device)
            item_idx_tensor = torch.LongTensor(item_idx).to(args.device)

            logits = model.predict(u_tensor, seq_tensor, item_idx_tensor)

        # Tính rank của sản phẩm positive
        rank = logits.cpu().numpy().argsort().argsort()[0, 0]

        valid_user += 1

        # Tính NDCG@k
        if rank < k:
            ndcg += 1 / np.log2(rank + 2)

        # Tính Hit Rate@k
        if rank < k:
            hit_rate += 1

    ndcg /= valid_user
    hit_rate /= valid_user

    return ndcg, hit_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # GPU train options
    parser.add_argument("--multi_gpu", action='store_true')
    parser.add_argument('--gpu_num', type=int, default=0)

    # Model setting
    parser.add_argument("--llm", type=str, default='opt', help='flan_t5, opt, vicuna')
    parser.add_argument("--recsys", type=str, default='sasrec')

    # Dataset setting
    parser.add_argument("--rec_pre_trained_data", type=str, default='Movies_and_TV')

    # Train phase setting
    parser.add_argument("--pretrain_stage1", action='store_true')
    parser.add_argument("--inference", action='store_true')

    # Hyperparameters
    parser.add_argument('--batch_size1', default=32, type=int)
    parser.add_argument('--batch_size_infer', default=2, type=int)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument("--stage1_lr", type=float, default=0.0001)
    
    args = parser.parse_args()

    # Thiết lập thiết bị
    args.device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')

    print("Loading model...")
    model = A_llmrec_model(args).to(args.device)

    model_path = "/home4/khanhnd/hieupt/test2/A-LLMRec/models/saved_models"
    model.load_model(args, phase1_epoch=1)  

    print("Loading dataset...")
    dataset_path = f'./data/amazon/{args.rec_pre_trained_data}.txt'
    dataset = data_partition(args.rec_pre_trained_data, path=dataset_path)
    [train, valid, test, usernum, itemnum] = dataset

    print("Evaluating model...")
    ndcg, hit_rate = evaluate(model, [train, valid, test, usernum, itemnum], args, k=10)
    print(f"NDCG@10: {ndcg:.4f}, Hit Rate@10: {hit_rate:.4f}")