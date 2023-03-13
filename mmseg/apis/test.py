import os.path as osp
import os
import pickle
import shutil
import tempfile
import timeit
import numpy as np
import torch
import torch.distributed as dist

import mmcv
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info


def single_gpu_test(model, data_loader, show=False, out_dir=None, test_fps=False):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped
        into the directory to save output results.
        test_fps (str, optional): If True, to test model inference fps.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    total_time = 0.0
    for i, total_data in enumerate(data_loader):
        with torch.no_grad():
            start_time = timeit.default_timer()

            data = total_data[0]
            seg_logit = model(return_loss=False, **data)
            for i in range(1,len(total_data)):
                data = total_data[i]
                cur_seg_logit = model(return_loss=False, **data)
                seg_logit += cur_seg_logit
            seg_logit /= len(total_data)
            seg_pred = seg_logit.argmax(dim=1)
            seg_pred = seg_pred.cpu().numpy()
            result = list(seg_pred)

        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)
        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta, seg_pred in zip(imgs, img_metas, result):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))
                if out_dir:
                    ori_filename = img_meta['ori_filename']
                    if ori_filename[-7:] == '5_5.png':
                        ori_filename = ori_filename[0:-8]+'.png'
                        img_meta['ori_filename'] = ori_filename
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    seg_pred,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file)
                
                if os.path.exists('numpy_res') == False:
                    os.makedirs('numpy_res')
                numpy_name = ori_filename[:-4]  
                info = numpy_name.split('/')
                if seg_pred.shape[0] == 480:
                    if os.path.exists('numpy_res/syn') == False:
                        os.makedirs('numpy_res/syn')
                    numpy_name = 'numpy_res/syn/'+info[-1]
                    os.mkdir(numpy_name)
                    np.save(numpy_name+'/label.npy',(seg_pred+1).astype(np.uint8))
                else:
                    if os.path.exists('numpy_res/real') == False:
                        os.makedirs('numpy_res/real')
                    numpy_name = 'numpy_res/real/'+info[-1]
                    os.mkdir(numpy_name)
                    np.save(numpy_name+'/label.npy',(seg_pred+1).astype(np.uint8))
                
        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, total_data in enumerate(data_loader):
        with torch.no_grad():
            data = total_data[0]
            seg_logit = model(return_loss=False, **data)
            for i in range(1,len(total_data)):
                data = total_data[i]
                cur_seg_logit = model(return_loss=False, **data)
                seg_logit += cur_seg_logit
            seg_logit /= len(total_data)
            seg_pred = seg_logit.argmax(dim=1)
            seg_pred = seg_pred.cpu().numpy()
            result = list(seg_pred)
        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results with CPU."""
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results with GPU."""
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
