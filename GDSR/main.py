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



    scale = config["scale"]
    nyu_test_data = config["datasets"]["test"]["NYUv2"]
    Middlebury_test_data = config["datasets"]["test"]["Middlebury"]
    Lu_test_data = config["datasets"]["test"]["Lu"]
    RGBDD_data = config["datasets"]["test"]["RGBDD"]





    if rank == 0:
        print('Loading dataset ...\n')

    dataset_nyu_test = Dataset.Dataset_NYU(None,nyu_test_data,False,scale,train=False)
    dataset_middle_test = Dataset.Dataset_Middle_LU(Middlebury_test_data, scale)
    dataset_Lu_test = Dataset.Dataset_Middle_LU(Lu_test_data, scale)
    dataset_RGBDD_syn_test = Dataset.Dataset_RGBDD(RGBDD_data, scale,True)


    nyu_test_sampler = DistributedSampler(dataset_nyu_test, num_replicas=world_size, rank=rank, shuffle=False)
    middle_test_sampler = DistributedSampler(dataset_middle_test, num_replicas=world_size, rank=rank, shuffle=False)
    Lu_test_sampler = DistributedSampler(dataset_Lu_test, num_replicas=world_size, rank=rank, shuffle=False)
    RGBDD_syn_test_sampler = DistributedSampler(dataset_RGBDD_syn_test, num_replicas=world_size, rank=rank, shuffle=False)



    test_nyu_loader = Data.DataLoader(dataset=dataset_nyu_test, num_workers=4, batch_size=1, sampler=nyu_test_sampler)
    test_middle_loader = Data.DataLoader(dataset=dataset_middle_test, num_workers=4, batch_size=1, sampler=middle_test_sampler)
    test_Lu_loader = Data.DataLoader(dataset=dataset_Lu_test, num_workers=4, batch_size=1, sampler=Lu_test_sampler)
    test_RGBDD_syn_loader = Data.DataLoader(dataset=dataset_RGBDD_syn_test, num_workers=4, batch_size=1, sampler=RGBDD_syn_test_sampler)



    if args.mode == "train":

        train_data = config["datasets"]["train"]["path"]
        patch_size = config["datasets"]["train"]["patch_size"]
        aug_mode = config["datasets"]["train"]["aug_mode"]

        dataset_train = Dataset.Dataset_NYU(patch_size,train_data,aug_mode,scale,train=True)


        train_sampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank,
                                           shuffle=config["datasets"]["train"]["use_shuffle"])

        train_loader = Data.DataLoader(dataset=dataset_train, num_workers=4, batch_size=config["datasets"]["train"]["batch_size"],
                                       sampler=train_sampler)

        test_loader_dict = {"NYU": test_nyu_loader,
                            "MiddleBury": test_middle_loader,
                            "Lu": test_Lu_loader,
                            "RGBDD": test_RGBDD_syn_loader}
        if rank == 0:
            print("# of training samples: %d\n" % int(len(dataset_train)))

        runner.train(args,train_loader,test_loader_dict)

    else:

        dataset_RGBDD_real_test = Dataset.Dataset_RGBDD(RGBDD_data, scale, False)
        RGBDD_real_test_sampler = DistributedSampler(dataset_RGBDD_real_test, num_replicas=world_size, rank=rank,
                                                     shuffle=False)
        test_RGBDD_real_loader = Data.DataLoader(dataset=dataset_RGBDD_real_test, num_workers=4, batch_size=1,
                                                 sampler=RGBDD_real_test_sampler)

        if args.path == None and rank == 0:
            raise ValueError("The folder where the test weights are located needs to be provided!")
        if len((args.cuda).split(',')) > 1 and rank == 0:
            warnings.warn("Using more than one graphics card for testing may cause deviations in the average indicators.")
        runner.test_NYU(test_nyu_loader, args.recode)
        runner.test_Middle_Lu(test_middle_loader,"MiddleBury", args.recode)
        runner.test_Middle_Lu(test_Lu_loader, "Lu", args.recode)
        runner.test_RGBDD(test_RGBDD_syn_loader,True, args.recode)
        runner.test_RGBDD(test_RGBDD_real_loader, False, args.recode)

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cuda", type=str, default="2,7", help="gpu to train")
    parser.add_argument("-r", "--recode", help="choose whether to recode", action="store_true")
    parser.add_argument("-p", "--path", type=str, default=None, help="pre weight path")
    parser.add_argument("-m", "--mode", type=str, default="train", choices=["train","test"], help="choose to train or test")

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 按PCI总线ID排序，确保GPU顺序与它们在物理硬件中的位置一致。
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    WORLD_SIZE = torch.cuda.device_count()
    port = random.randint(10000, 20000)

    mp.spawn(main, args=(WORLD_SIZE, args, port), nprocs=WORLD_SIZE, join=True)




