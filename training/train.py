import argparse

import util


if __name__ == "__main__":
    
    """
    annotations is in yolo format, this is: 
            class, xc, yc, w, h
    data-directory
    ----- train
    --------- imgs
    ------------ filename0001.jpg
    ------------ ....
    --------- anns
    ------------ filename0001.txt
    ------------ ....
    ----- val
    --------- imgs
    ------------ filename0001.jpg
    ------------ ....
    --------- anns
    ------------ filename0001.txt
    ------------ ....
    """

# default - cpu training for comparison
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--class-list', default='./class.names')
#     parser.add_argument('--data-dir', default='./data')
#     parser.add_argument('--output-dir', default='./output')
#     parser.add_argument('--device', default='cpu')
#     parser.add_argument('--learning-rate', default=0.0005)
#     parser.add_argument('--batch-size', default=2)
#     parser.add_argument('--iterations', default=100)
#     parser.add_argument('--checkpoint-period', default=20)
#     parser.add_argument('--model', default='COCO-Detection/retinanet_R_101_FPN_3x.yaml')

# tuning 2 - increased learning rate
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--class-list', default='./class.names')
#     parser.add_argument('--data-dir', default='./data')
#     parser.add_argument('--output-dir', default='./output_high_lr')
#     parser.add_argument('--device', default='cuda')
#     parser.add_argument('--learning-rate', default=0.001)  # Higher learning rate
#     parser.add_argument('--batch-size', default=4)         # Moderate batch size
#     parser.add_argument('--iterations', default=150)       # More iterations
#     parser.add_argument('--checkpoint-period', default=30) # Moderate checkpoint frequency
#     parser.add_argument('--model', default='COCO-Detection/retinanet_R_101_FPN_3x.yaml')

# tuning 3 - smaller batch size and lower learning rate
    parser = argparse.ArgumentParser()
    parser.add_argument('--class-list', default='./class.names')
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--output-dir', default='./output_small_lr_small_batch')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--learning-rate', default=0.0001) # Lower learning rate
    parser.add_argument('--batch-size', default=1)         # Small batch size
    parser.add_argument('--iterations', default=150)       # More iterations to compensate small batch size
    parser.add_argument('--checkpoint-period', default=10) # Frequent checkpoints
    parser.add_argument('--model', default='COCO-Detection/retinanet_R_101_FPN_3x.yaml')

# tuning 4 - increased batch size 
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--class-list', default='./class.names')
#     parser.add_argument('--data-dir', default='./data')
#     parser.add_argument('--output-dir', default='./output_gpu_large_batch')
#     parser.add_argument('--device', default='cuda')        # Enable GPU
#     parser.add_argument('--learning-rate', default=0.0002) # Lower LR for larger batch size
#     parser.add_argument('--batch-size', default=8)         # Larger batch size for GPU
#     parser.add_argument('--iterations', default=200)       # More iterations for better convergence
#     parser.add_argument('--checkpoint-period', default=50) # Save checkpoint less frequently
#     parser.add_argument('--model', default='COCO-Detection/retinanet_R_101_FPN_3x.yaml')
# ini gabisa karena gpu ran out of memory (batch size terlalu besar)

    args = parser.parse_args()

    util.train(args.output_dir,
               args.data_dir,
               args.class_list,
               device=args.device,
               learning_rate=float(args.learning_rate),
               batch_size=int(args.batch_size),
               iterations=int(args.iterations),
               checkpoint_period=int(args.checkpoint_period),
               model=args.model)
