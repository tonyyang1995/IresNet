def get_dataset(opt):
	dataset = None
	name = opt.dataset_mode

	if name == '2D':
		from data.twoDdataset import twoDDataset
		dataset = twoDDataset(opt)
	elif name == 'test2D':
		from data.twoDdataset import testDataset
		dataset = testDataset(opt)
	elif name == '3D':
		from data.DCMDataset import DCMDataset
		dataset = DCMDataset(opt)
	elif name == 'test3D':
		from data.DCMDataset import DCMtestDataset
		dataset = DCMtestDataset(opt)
	elif name == '3D_cube':
		from data.CubeDataset import CubeDataset
		dataset = CubeDataset(opt)
	elif name == 'dcm':
		from data.originDataset import OriginDataset
		dataset = OriginDataset(opt)
	elif name == 'abide':
		from data.AbideDataset import AbideDataset
		dataset = AbideDataset(opt)
	elif name == 'abidetest':
		from data.AbideDataset import AbideDatasetTest
		dataset = AbideDatasetTest(opt)
	elif name == 'Ours':
		from data.AbideDataset import OursDataset
		dataset = OursDataset(opt)
	else:
		raise NotImplementedError('the dataset [%s] is not implemented' % name)

	return dataset