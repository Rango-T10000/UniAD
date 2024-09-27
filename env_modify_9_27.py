#这个函数位于home2 > wzc › anaconda3 › envs › uniad2 › lib › python3.8 › site-packages › mmcv › parallel ›
#data_parallel.py › *; MMDataParallel >train_step

#是最lib中代码的修改，如果需要forward_train_IMU完整运行，需要做如下相应修改
    def train_step(self, *inputs, **kwargs):
        if not self.device_ids:
            # We add the following line thus the module could gather and
            # convert datas containers as those in GPU inference
            inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            return self.module.train_step(*inputs[0], **kwargs[0])

        assert len(self.device_ids) == 1, \
            ('MMDataParallel only supports single GPU training, if you need to'
             ' train with multiple GPUs, please use MMDistributedDataParallel'
             ' instead.')

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    'module must have its parameters and buffers '
                    f'on device {self.src_device_obj} (device_ids[0]) but '
                    f'found one of them on device: {t.device}')
        #--------scatter 方法的关键作用是将数据从 DataContainer 中解包并转换为 Tensor，然后分发到指定的设备上-------
        #--------经过这句，数据从DataContainer解包为tensor-------
        # inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids) #反正也不启动dist，所以不过这句话
        
        # 把数据中 DataContainer objects 转换为 tensors，顺便放到GPU上
        processed_inputs = {}
        for key, value in inputs[0].items():
            if isinstance(value, DataContainer) and key != 'img_metas':
                # 先判断 value.data 是否是 list
                if isinstance(value.data, list):
                    processed_inputs[key] = [v.to(self.device_ids[0]) for v in value.data]
                else:
                    processed_inputs[key] = value.data.to(self.device_ids[0])
            elif isinstance(value, DataContainer) and key == 'img_metas':
                processed_inputs[key] = value.data
            else:
                # 判断 value 是否是 list
                if isinstance(value, list):
                    processed_inputs[key] = [v.to(self.device_ids[0]) for v in value]
                else:
                    processed_inputs[key] = value.to(self.device_ids[0])

        inputs = (processed_inputs, inputs[1])
        kwargs = (kwargs, ) #为了防止下一句报错
        inputs = (inputs, )
        return self.module.train_step(*inputs[0], **kwargs[0])