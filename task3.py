from some_model_zoo import ResnetGenerator, UnetGenerator, NLayerDiscriminator, FeatureResNet34
from collections import OrderedDict
from torch.autograd import Variable
from torch import nn
from some_utils import ImagePool
import torch, os, itertools, functools
from skimage.color import rgb2lab, deltaE_ciede2000
from some_utils import tensor2img

import numpy as np


class Net:
    def __init__(self, config, logger):
        self._config = config
        self._logger = logger
        self._gpu_ids = config['gpu_ids']
        self._Tensor = torch.cuda.FloatTensor if self._gpu_ids else torch.Tensor
        self._save_dir = os.path.join(config['checkpoints_dir'], str(config['name']))
        self.name = self.__class__.__name__

        self.initialize()

    def initialize(self):
        nb = self._config['batchSize']
        size = self._config['fineSize']
        self.input_A = self._Tensor(nb, self._config['input_nc'], size, size)
        self.input_B = self._Tensor(nb, self._config['output_nc'], size, size)

        # load/define networks
        self.netG_A = define_G(self._config['input_nc'], self._config['output_nc'], self._config['ngf'],
                                        self._config['which_model_netG'], norm=self._config['norm'],
                                        use_dropout=not self._config['no_dropout'],
                                        padding_type=self._config['padding_type'], gpu_ids=self._gpu_ids)
        self.netG_B = define_G(self._config['output_nc'], self._config['input_nc'], self._config['ngf'],
                                        self._config['which_model_netG'], norm=self._config['norm'],
                                        use_dropout=not self._config['no_dropout'],
                                        padding_type=self._config['padding_type'], gpu_ids=self._gpu_ids)

        if self._config['is_train']:
            use_sigmoid = False
            self.netD_A = define_D((self._config['output_nc'], self._config['fineSize'],
                                             self._config['fineSize']), self._config['ndf'],
                                            self._config['which_model_netD'],
                                            self._config['n_layers_D'], self._config['norm'], use_sigmoid,
                                            self._config['gpu_ids'])
            self.netD_B = define_D((self._config['input_nc'], self._config['fineSize'],
                                             self._config['fineSize']), self._config['ndf'],
                                            self._config['which_model_netD'],
                                            self._config['n_layers_D'], self._config['norm'], use_sigmoid,
                                            self._config['gpu_ids'])
            self.netFeat = define_feature_network(self._config['which_model_feat'], self._config['gpu_ids'])

        if not self._config['is_train'] or self._config['continue_train']:
            which_epoch = self._config['which_epoch']
            self._logger.info(f'epoch to load: {which_epoch}')
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)

        if self._config['is_train']:
            self.old_lr = float(self._config['lr'])
            self.fake_A_pool = ImagePool(self._config['pool_size'])
            self.fake_B_pool = ImagePool(self._config['pool_size'])

            # define loss functions
            target_label_real = 0.9 if self._config['smoothed_label'] == 'fixed' else 1.0
            target_label_fake = 0.1 if self._config['smoothed_label'] == 'fixed' else 0.0
            self.criterionGAN = GANLoss(target_real_label=target_label_real,
                                target_fake_label=target_label_fake, tensor=self._Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            idt_options = {'l1': torch.nn.L1Loss(), 'lab_mse': lambda x, y: lab_identity_loss(x, y, 'mse'),
                           'lab_mae': lambda x, y: lab_identity_loss(x, y, 'mae')}
            self.criterionIdt = idt_options[self._config['identity_loss']]
            self.criterionFeat = get_mse({})
            self.criterionColor = color_loss
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=self._config['lr'], betas=(self._config['beta1'], 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=self._config['lr'],
                                                  betas=(self._config['beta1'], 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=self._config['lr'],
                                                  betas=(self._config['beta1'], 0.999))

        if self._config['verbosity'] >= 1:
            self._logger.info('---------- Networks initialized -------------')

    def set_input(self, input):
        AtoB = self._config['which_direction'] == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        AtoB_test_mode = AtoB and not self._config['is_train']
        if not AtoB_test_mode:
            input_B = input['B' if AtoB else 'A']
            self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        with torch.no_grad():  # Replaced Variable(self.input_A, volatile=True) with it to avoid warning.
            self.real_A = Variable(self.input_A)
        self.fake_B = self.netG_A.forward(self.real_A)
        self.rec_A = self.netG_B.forward(self.fake_B)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):

        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)

        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5 * float(self._config['lambda_D'])

        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = float(self._config['identity'])
        lambda_A = float(self._config['lambda_A'])
        lambda_B = float(self._config['lambda_B'])

        lambda_feat_color = float(self._config['lambda_feat_color'])

        lambda_feat_AfB = float(self._config['lambda_feat_AfB'])
        lambda_feat_BfA = float(self._config['lambda_feat_BfA'])

        lambda_feat_fArecB = float(self._config['lambda_feat_fArecB'])
        lambda_feat_fBrecA = float(self._config['lambda_feat_fBrecA'])

        lambda_feat_ArecA = float(self._config['lambda_feat_ArecA'])
        lambda_feat_BrecB = float(self._config['lambda_feat_BrecB'])

        # Identity loss
        if lambda_idt > 0:

            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A.forward(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt

            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B.forward(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        self.fake_B = self.netG_A.forward(self.real_A)
        pred_fake = self.netD_A.forward(self.fake_B)
        self.loss_G_A = self.criterionGAN(pred_fake, True)

        self.fake_A = self.netG_B.forward(self.real_B)
        pred_fake = self.netD_B.forward(self.fake_A)
        self.loss_G_B = self.criterionGAN(pred_fake, True)

        # Forward cycle loss
        self.rec_A = self.netG_B.forward(self.fake_B)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A

        # Backward cycle loss
        self.rec_B = self.netG_A.forward(self.fake_A)
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        self.feat_loss_AfB = self.criterionFeat(self.netFeat(self.real_A), self.netFeat(self.fake_B)) * lambda_feat_AfB
        self.feat_loss_BfA = self.criterionFeat(self.netFeat(self.real_B), self.netFeat(self.fake_A)) * lambda_feat_BfA

        self.feat_loss_fArecB = self.criterionFeat(self.netFeat(self.fake_A),
                                                   self.netFeat(self.rec_B)) * lambda_feat_fArecB
        self.feat_loss_fBrecA = self.criterionFeat(self.netFeat(self.fake_B),
                                                   self.netFeat(self.rec_A)) * lambda_feat_fBrecA

        self.feat_loss_ArecA = self.criterionFeat(self.netFeat(self.real_A),
                                                  self.netFeat(self.rec_A)) * lambda_feat_ArecA
        self.feat_loss_BrecB = self.criterionFeat(self.netFeat(self.real_B),
                                                  self.netFeat(self.rec_B)) * lambda_feat_BrecB

        self.loss_color = self.criterionColor(self.real_A, self.fake_B, n=4) * lambda_feat_color
        self.feat_loss = self.feat_loss_AfB + self.feat_loss_BfA + self.feat_loss_fArecB + self.feat_loss_fBrecA + \
                         self.feat_loss_ArecA + self.feat_loss_BrecB + self.loss_color

        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + \
                      self.loss_idt_A + self.loss_idt_B + self.feat_loss
        self.loss_G.backward()

    def optimize_parameters(self):

        # forward
        self.forward()

        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()

        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_errors(self):
        D_A = self.loss_D_A.data[0]
        G_A = self.loss_G_A.data[0]
        Cyc_A = self.loss_cycle_A.data[0]
        D_B = self.loss_D_B.data[0]
        G_B = self.loss_G_B.data[0]
        Cyc_B = self.loss_cycle_B.data[0]
        if self._config['identity'] > 0.0:
            idt_A = self.loss_idt_A.data[0]
            idt_B = self.loss_idt_B.data[0]
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B)])
        else:
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B)])

    def get_current_visuals(self):

        def helper(tensors, names):
            return OrderedDict([(name, tensor2img(tensor.data)) for name, tensor in zip(names, tensors)])

        tensors, names = [self.real_A, self.fake_B, self.rec_A], ['real_A', 'fake_B', 'rec_A']
        AtoB_test_mode = self._config['which_direction'] == 'AtoB' and not self._config['is_train']
        if not AtoB_test_mode:
            tensors += [self.real_B, self.fake_A, self.rec_B]
            names += ['real_B', 'fake_A', 'rec_B']
            if self._config['identity'] > 0.0:
                tensors.insert(3, self.idt_A)
                names.insert(3, 'idt_A')
                tensors.append(self.idt_B)
                names.append('idt_B')

        return helper(tensors, names)

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self._gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self._gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self._gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self._gpu_ids)

    def update_learning_rate(self):
        lrd = self._config['lr'] / self._config['niter_decay']
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_B.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        self._logger.info(f'update learning rate: {self.old_lr} -> {lr}')
        self.old_lr = lr

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device=gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self._save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


def define_G(input_nc, output_nc, ngf, which_model_netG,
             norm='batch', use_dropout=False, padding_type='reflect', gpu_ids=[]):
    netG = None
    use_gpu = len(str(gpu_ids)) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    resnet_G = lambda n: ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                         padding_type=padding_type, n_blocks=n, gpu_ids=gpu_ids)
    unet_G = lambda n: UnetGenerator(input_nc, output_nc, n, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                     gpu_ids=gpu_ids)
    if which_model_netG == 'resnet_12blocks':
        netG = resnet_G(12)
    elif which_model_netG == 'resnet_9blocks':
        netG = resnet_G(9)
    elif which_model_netG == 'resnet_6blocks':
        netG = resnet_G(6)
    elif which_model_netG == 'unet_128':
        netG = unet_G(7)
    elif which_model_netG == 'unet_256':
        netG = unet_G(8)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(str(gpu_ids)) > 0:
        netG.cuda(device='cuda')
    netG.apply(weights_init)
    return netG


def define_D(input_shape, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_shape[0], ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_shape[0], ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(device=gpu_ids[0])
    netD.apply(weights_init)
    return netD


def define_feature_network(which_model_netFeat, gpu_ids=[]):
    netFeat = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model_netFeat == 'resnet34':
        netFeat = FeatureResNet34(gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Feature model name [%s] is not recognized' %
                                  which_model_netFeat)
    if use_gpu:
        netFeat.cuda(device=gpu_ids[0])

    return netFeat


def lab_identity_loss(tensor1, tensor2, reduce_type='mse'):
    def transform2lab(tensor):
        chw_img = (tensor.data[0].cpu().numpy() + 1.) / 2.
        return rgb2lab(np.transpose(chw_img, (1, 2, 0)))

    img_lab1 = transform2lab(tensor1)
    img_lab2 = transform2lab(tensor2)
    reduce_options = {'mse': np.linalg.norm, 'mae': lambda x: np.abs(x).mean()}
    lab_norm = reduce_options[reduce_type](deltaE_ciede2000(img_lab1, img_lab2))
    return torch.tensor(lab_norm).float().to('cuda')


def color_loss(predict: torch.Tensor, target: torch.Tensor, n: int) -> torch.Tensor:
    # TODO: please implement color loss that meets the following requirements:
    #     1. The loss is computed on pair of image tensors (predict, target) of shapes (n_batches, height, width, n_channels)
    #     2. The loss is computed only on regions of interest (ROIs) that contain object of interest: person or avatar
    #     3. The loss computes RMSE between colors within ROIs of passed tensors.
    #        Let color be the average value of region of (n x n) pixels.
    #
    #     You are allowed to use you favourite search engine to check APIs and docs of used libraries 
    pass


def get_mse(config):
    def mse_loss(input_tn, target_tn):
        return ((input_tn - target_tn) ** 2).mean()

    return mse_loss


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
