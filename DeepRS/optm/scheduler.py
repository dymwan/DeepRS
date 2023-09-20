
import math


__all__ = ["get_lr_scheduler"]




class LR_Scheduler_Batch(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:  :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, cfg, iters_per_epoch=0, para_pretrained=True):
        
        self.mode = cfg.SOLVER.LR.LR_SCHEDULER
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = cfg.SOLVER.LR.BASE_LR
        self.target_lr = cfg.SOLVER.WARMUP.TARGET_LR
        self.poly_power = cfg.SOLVER.LR.POLY.POWER
        self.lr_step = cfg.SOLVER.LR.STEP.LR_STEP
        self.lr_decay = cfg.SOLVER.LR.STEP.LR_DECAY
        self.iters_per_epoch = iters_per_epoch
        num_epochs = cfg.TRAIN.MAX_EPOCH
        self.num_iter = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup = cfg.SOLVER.WARMUP.WARMUP
        warmup_epochs = cfg.SOLVER.WARMUP.WARMUP_EPOCH
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.para_pretrained = para_pretrained

        if self.warmup:
            print('Using warmup_{} LR Scheduler! Warmup {} epochs!'.format(self.mode, warmup_epochs))
        else:
            print('Using {} LR Scheduler!'.format(self.mode))

    def __call__(self, optimizer, i, epoch):
        if self.mode != 'auto':
            cur_iter = epoch * self.iters_per_epoch + i
            # warm up lr schedule
            if self.warmup and self.warmup_iters > 0 and cur_iter < self.warmup_iters:
                lr = self.target_lr * 1.0 * cur_iter / self.warmup_iters
            else:
                # return normal lr schedule
                if self.mode == 'cos':
                    lr = 0.5 * self.lr * (1 + math.cos(1.0 * cur_iter * math.pi / self.num_iter))
                elif self.mode == 'poly':
                    lr = self.lr * pow((1 - 1.0 * cur_iter / self.num_iter), self.poly_power)
                elif self.mode == 'step':
                    lr = self.lr * (self.lr_decay ** (epoch // self.lr_step))
                else:
                    raise NotImplemented

            if epoch > self.epoch:
                self.epoch = epoch
            assert lr >= 0
            self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        for i in range(0, len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr
        if self.para_pretrained == True:
            optimizer.param_groups[0]['lr'] = lr * self.adjust_lr
            
    def _adjust_learning_rate(self, optimizer, lr, reduce_lr):
        for i in range(0, len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr * reduce_lr
        # TODO
        if self.para_pretrained == True:
            optimizer.param_groups[0]['lr'] = lr * self.adjust_lr * reduce_lr

class LR_Scheduler_Epoch(object):

    def __init__(self, cfg, iters_per_epoch=0, para_pretrained=False):

        self.mode = cfg.SOLVER.LR.LR_SCHEDULER
        self.base_lr = cfg.SOLVER.LR.BASE_LR
        #TODO 添加到配置文件
        self.min_lr = self.base_lr / 100
        self.adjust_lr = cfg.SOLVER.LR.ADJUST_LR
        ## cycle
        self.c_lr = cfg.SOLVER.LR.CYCLE_LR
        self.c_lr_step = cfg.SOLVER.LR.CYCLE_LR_STEP
        ## scheduler
        self.poly_power = cfg.SOLVER.LR.POLY.POWER
        self.lr_step = cfg.SOLVER.LR.STEP.LR_STEP
        self.lr_decay = cfg.SOLVER.LR.STEP.LR_DECAY
        self.num_epochs = cfg.TRAIN.END_EPOCH
        self.epoch = -1
        ## warmup
        self.warmup = cfg.SOLVER.WARMUP.WARMUP
        self.warmup_power = cfg.SOLVER.WARMUP.POWER
        self.warmup_epochs = cfg.SOLVER.WARMUP.WARMUP_EPOCH
        self.warmup_flag = False
        self.para_pretrained = para_pretrained
        ## swa
        self.swa = False
        self.swa_start = 30
        self.swa_lr = 0.05
        self.swa_c_epochs = 10
        if self.warmup:
            print('Using warmup_{} LR Scheduler! Warmup {} epochs!'.format(self.mode, self.warmup_epochs))
        else:
            print('Using {} LR Scheduler!'.format(self.mode))

    def __call__(self, optimizer, epoch, reduce_lr=1):

        ## warm up lr schedule
        if self.warmup and self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            # TODO
            # self.warmup_flag = True
            lr = self.base_lr * self.warmup_power ** (self.warmup_epochs - epoch)
        ## return normal lr schedule
        elif self.mode == 'fix':
            lr = self.base_lr
        elif self.mode == 'cos':
            lr = self.cos_lr(epoch)
        elif self.mode == 'poly':
            lr = self.poly_lr(epoch)
        elif self.mode == 'step':
            lr = self.step_lr(epoch)
        else:
            raise RuntimeError('unknown schedule mode : {}'.format(self.mode))
        ## 更新epoch
        if epoch > self.epoch:
            self.epoch = epoch
        assert lr >= 0
        ## SWA
        if self.swa and self.epoch >= self.swa_start:
            lr_ratio = self.swa_lr / self.base_lr
            reduce_lr = reduce_lr * lr_ratio
        ## 调整optimizer的学习率
        self._adjust_learning_rate(optimizer, lr, reduce_lr)

    def cos_lr(self, epoch):
        if not self.c_lr:
            cur_iter = epoch
            return 0.5 * self.base_lr * (1 + math.cos(cur_iter / self.num_epochs * math.pi)) + self.min_lr
        else:
            cur_iter = epoch % self.c_lr_step
            cycle_iter = epoch // self.c_lr_step
            max_cycle = self.num_epochs // self.c_lr_step + 1
            c_base_lr = 0.5 * self.base_lr * (1 + math.cos(cycle_iter / max_cycle * math.pi))
            return 0.5 * c_base_lr * (1 + math.cos(2 * (cur_iter / self.c_lr_step - 0.5) * math.pi)) + self.min_lr

    def poly_lr(self, epoch):
        if not self.c_lr:
            cur_iter = epoch
            return self.base_lr * pow((1 - 1.0 * cur_iter / self.num_epochs), self.poly_power) + self.min_lr
        else:
            cur_iter = epoch % self.c_lr_step
            cycle_iter = epoch // self.c_lr_step
            max_cycle = self.num_epochs // self.c_lr_step + 1
            c_base_lr = self.base_lr * pow((1 - 1.0 * cycle_iter / max_cycle), self.poly_power)
            return c_base_lr * pow((1 - 1.0 * cur_iter / self.c_lr_step), self.poly_power) + self.min_lr

    def step_lr(self, epoch):
        if not self.c_lr:
            cur_iter = epoch
            return self.base_lr * (self.lr_decay ** (epoch // self.lr_step)) + self.min_lr
        else:
            cur_iter = epoch % self.c_lr_step
            cycle_iter = epoch // self.c_lr_step
            max_cycle = self.num_epochs // self.c_lr_step + 1
            c_base_lr = self.base_lr * (self.lr_decay ** (cycle_iter // max_cycle))
            return c_base_lr * (self.lr_decay ** (cur_iter // self.c_lr_step)) + self.min_lr

    def _adjust_learning_rate(self, optimizer, lr, reduce_lr):
        for i in range(0, len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr * reduce_lr
        # TODO
        if self.para_pretrained == True:
            optimizer.param_groups[0]['lr'] = lr * self.adjust_lr * reduce_lr



lr_scheduler = {
    'LR_Scheduler_Batch': LR_Scheduler_Batch,
    'LR_Scheduler_Epoch': LR_Scheduler_Epoch,
}

def get_lr_scheduler(name, cfg, iters_per_epoch, para_pretrained):
    return lr_scheduler[name](cfg, iters_per_epoch, para_pretrained)


