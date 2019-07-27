
Dense models 'vgg-dense' and 'alex-dense'. These models do not downsample (stride=2). Instead, they use stride=1 and accumulate the factor as dilation in subsequent layers. Try `python test_network.py --net alex-dense --spatial`.
