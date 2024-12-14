import torch.utils.data as Data
import Dataset
import argparse
from Runner import make_model,Runner
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch
import random
from torch.utils.data.distributed import DistributedSampler
import warnings

def setup(rank, world_size, port=12365):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, args, port):
    setup(rank, world_size, port)

    model, config = make_model(args)
    runner = Runner(model, args, config, rank)

    train_data = config["datasets"][args.sensor]["train"]
    test_reduce_data = config["datasets"][args.sensor]["test_reduce"]
    test_full_data = config["datasets"][args.sensor]["test_full"]
    image_scale = config["datasets"][args.sensor]["image_scale"]
    if rank == 0:
        print('Loading dataset ...\n')

    dataset_reduce_test = Dataset.Dataset(test_reduce_data, img_scale = image_scale, train = False)
    dataset_full_test = Dataset.Dataset(test_full_data, img_scale = image_scale, train = False)


    reduce_test_sampler = DistributedSampler(dataset_reduce_test, num_replicas=world_size, rank=rank, shuffle=False)
    full_test_sampler = DistributedSampler(dataset_full_test, num_replicas=world_size, rank=rank, shuffle=False)
    # full_test_sampler = DistributedSampler(dataset_reduce_test, num_replicas=world_size, rank=rank, shuffle=False)



    test_reduce_loader = Data.DataLoader(dataset=dataset_reduce_test, num_workers=4, batch_size=1,
                                         sampler=reduce_test_sampler)
    test_full_loader = Data.DataLoader(dataset=dataset_full_test, num_workers=4, batch_size=1,
                                       sampler=full_test_sampler)
    # test_full_loader = Data.DataLoader(dataset=dataset_reduce_test, num_workers=4, batch_size=1,
    #                                    sampler=full_test_sampler)

    if args.mode == "train":
        dataset_train = Dataset.Dataset(train_data, img_scale=image_scale, train=True)
        train_sampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank,
                                           shuffle=config["train"]["use_shuffle"])
        train_loader = Data.DataLoader(dataset=dataset_train, num_workers=4, batch_size=config["train"]["batch_size"],
                                       sampler=train_sampler)
        if rank == 0:
            print("# of training samples: %d\n" % int(len(dataset_train)))

        runner.train(args,train_loader,test_reduce_loader,test_full_loader)

    else:
        if args.path == None and rank == 0:
            raise ValueError("The folder where the test weights are located needs to be provided!")
        if len((args.cuda).split(',')) > 1 and rank == 0:
            warnings.warn("Using more than one graphics card for testing may cause deviations in the average indicators.")

        runner.test_reduce(test_reduce_loader,args.recode)
        runner.test_full(test_full_loader,args.sensor, args.recode)
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cuda", type=str, default="2,7", help="gpu to train")
    parser.add_argument("-r", "--recode", help="choose whether to recode", action="store_true")
    parser.add_argument("-p", "--path", type=str, default=None, help="pre weight path")
    parser.add_argument("-s", "--sensor", type=str, default="GF2", help="choose dataset sensor")
    parser.add_argument("-m", "--mode", type=str, default="train", choices=["train","test"], help="choose to train or test")

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 按PCI总线ID排序，确保GPU顺序与它们在物理硬件中的位置一致。
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    WORLD_SIZE = torch.cuda.device_count()
    port = random.randint(10000, 20000)

    mp.spawn(main, args=(WORLD_SIZE, args, port), nprocs=WORLD_SIZE, join=True)




