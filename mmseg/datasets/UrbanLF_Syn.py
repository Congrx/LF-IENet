import os.path as osp
import tempfile
import random
import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image
from torch.utils.data import Dataset
from .builder import DATASETS
from .custom_lf import CustomLFDataset
import os

@DATASETS.register_module()
class UrbanLFSynDataset(CustomLFDataset):
    """
        UrbanLF dataset.
    """

    CLASSES = ('bike','building','fence','others','person','pole','road', 'sidewalk', 
               'traffic sign', 'vegetation', 'vehicle', 'bridge','rider','sky')

    PALETTE = [[168, 198, 168], [0, 0, 198], [198, 154, 202], [0, 0, 0],
               [198, 198, 100], [0, 100, 198], [198, 42, 52], [192, 52, 154],
               [168, 0, 198], [0, 198, 0], [90, 186, 198], [161, 107, 108],
               [26, 200, 156], [202, 179, 158]]


    def __init__(self, **kwargs):
        super(UrbanLFSynDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.npy',
            sai_suffix='.png',
            **kwargs)
        
        def indexToName(index):
            if index % 9 == 0:
                return str(index//9)+'_9'
            else:
                return str(index//9+1)+'_'+str(index%9)
        
        self.img_infos = []
        
        self.datalist = os.listdir(self.img_dir) 
        for filename in self.datalist:
            img_info = dict(filename=filename + '/5_5.png') 
            img_info['ann'] = dict(seg_map=filename + '/5_5_label.npy') 
            sais = []
            sais_index = []
            if self.sai_number - 1 == 4:
                index_list = [[0,10,20,30],[4,13,22,31],[8,16,24,32],[36,37,38,39],[44,43,42,41],[72,64,56,48],[80,70,60,50],[76,67,58,49]]
                random_list = random.sample(index_list,1)[0]
            if self.sai_number - 1 == 3:
                index_list = [[10,20,30],[13,22,31],[16,24,32],[37,38,39],[43,42,41],[64,56,48],[70,60,50],[67,58,49]]
                random_list = random.sample(index_list,1)[0]

            for i in random_list:
                png_name = indexToName(i+1)
                sais.append(filename+'/'+png_name+'.png')  
                sais_index.append([int(png_name[0]),int(png_name[-1])])    
            img_info['sai_sequence'] = dict(sais=sais)
            img_info['sai_sequence_index'] = dict(sais_index=sais_index)
            img_info['dis'] = dict(dis_map=filename + '/5_5_disparity_OAVC.npy') 
            self.img_infos.append(img_info) 
        print("dataset length is :",len(self.img_infos))

    def results2img(self, results, imgfile_prefix, to_label_id):
        """Write the segmentation results to images.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]
            if to_label_id:
                result = self._convert_to_label_id(result)
            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            output = Image.fromarray(result.astype(np.uint8)).convert('P')
            import cityscapesscripts.helpers.labels as CSLabels
            palette = np.zeros((len(CSLabels.id2label), 3), dtype=np.uint8)
            for label_id, label in CSLabels.id2label.items():
                palette[label_id] = label.color

            output.putpalette(palette)
            output.save(png_filename)
            result_files.append(png_filename)
            prog_bar.update()

        return result_files

    def format_results(self, results, imgfile_prefix=None, to_label_id=True):
        """Format the results into dir (standard format for Cityscapes
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        if imgfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            imgfile_prefix = tmp_dir.name
        else:
            tmp_dir = None
        result_files = self.results2img(results, imgfile_prefix, to_label_id)

        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 imgfile_prefix=None):
        """Evaluation in Cityscapes/default protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file,
                for cityscapes evaluation only. It includes the file path and
                the prefix of filename, e.g., "a/b/prefix".
                If results are evaluated with cityscapes protocol, it would be
                the prefix of output png files. The output files would be
                png images under folder "a/b/prefix/xxx.png", where "xxx" is
                the image name of cityscapes. If not specified, a temp file
                will be created for evaluation.
                Default: None.

        Returns:
            dict[str, float]: Cityscapes/default metrics.
        """

        eval_results = dict()
        metrics = metric.copy() if isinstance(metric, list) else [metric]
        if len(metrics) > 0:
            eval_results.update(
                super(UrbanLFSynDataset,
                      self).evaluate(results, metrics, logger))

        return eval_results

    
