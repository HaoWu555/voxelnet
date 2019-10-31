tf.config.set_soft_device_placement(True)
class RPN3D(object):
    def __init__(self,cls="Car",single_batch_size=1,learning_rate=0.001,max_gradient_norm=5.0,
                 alpha=1.5,beta=1,training=False,avail_gpus=['0']):
        self.cls = cls
        self.single_batch_size = single_batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(1, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)
        self.alpha = alpha
        self.max_gradient_norm = max_gradient_norm
        self.beta = beta
        self.avail_gpus = avail_gpus

        boundaries = [80, 120]
        values = [ self.learning_rate, self.learning_rate * 0.1, self.learning_rate * 0.01 ]
        self.lr = tf.compat.v1.train.piecewise_constant(self.epoch, boundaries, values)

        #build graph
        self.is_train=training

        self.vox_feature = []
        self.vox_number = []
        self.vox_coordinate = []
        self.targets = []
        self.pos_equal_one = []
        self.pos_equal_one_sum = []
        self.pos_equal_one_for_reg = []
        self.neg_equal_one = []
        self.neg_equal_one_sum = []

        self.delta_output = []
        self.prob_output = []
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.gradient_norm = []
        self.tower_grads = []
        self.rpn = MiddleAndRPN()

    def __call__(self,a):
        print(a)

        for batch in iterate_data(train_dir,batch_size=1,multi_gpu_sum=len(self.avail_gpus)):
            #batches = iterate_data(train_dir)
            #batch = next(batches)
            tag = batch[0]
            labels = batch[1]
            voxel_feature = batch[2]
            vox_number = batch[3] # original is int64
            voxel_coordinate =batch[4] # original is int64
            rgb = batch[5]
            raw_lidar = batch[6]
            # get from labels

            for idx, dev in enumerate(self.avail_gpus):
                print("{}:{}".format("idx",idx))
                print("{}:{}".format("dev",dev))
                if idx==2:
                    return None

                with tf.device("/gpu:{}".format(dev)):
                    pos_equal_one, neg_equal_one, targets = cal_rpn_target(labels[idx * self.single_batch_size:
                                           (idx + 1) * self.single_batch_size], [cfg.FEATURE_HEIGHT,cfg.FEATURE_WIDTH], anchors)
                    pos_equal_one_for_reg = np.concatenate([np.tile(pos_equal_one[..., [0]], 7), np.tile(pos_equal_one[..., [1]], 7)], axis=-1)
                    pos_equal_one_sum = np.clip(np.sum(pos_equal_one, axis=(1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)
                    neg_equal_one_sum = np.clip(np.sum(neg_equal_one, axis=(1, 2, 3)).reshape(-1, 1, 1, 1), a_min=1, a_max=None)

                    info={"pos_equal_one":pos_equal_one,"neg_equal_one":neg_equal_one,"targets":tf.cast(targets,dtype=tf.float32),
                          "pos_equal_one_for_reg":pos_equal_one_for_reg,"pos_equal_one_sum":pos_equal_one_sum,
                          "neg_equal_one_sum":neg_equal_one_sum}

                    # model running
                    with tf.GradientTape() as tape:
                        # feature is initial address for saving class FeatureNet
                        feature=FeatureNet(voxel_feature[idx],voxel_coordinate[idx],training=True,batch_size=1)
                        rpn=self.rpn(inputs=feature.outputs,info=info,alpha=self.alpha, beta=self.beta, training=self.is_train)

                    print("{}:{}".format("loss",rpn.loss))

                    #print(gradients)
                    # input
                    self.vox_feature.append(voxel_feature)
                    self.vox_number.append(vox_number)
                    self.vox_coordinate.append(feature.coordinate)
                    self.targets.append(targets)
                    self.pos_equal_one.append(pos_equal_one)
                    self.pos_equal_one_sum.append(pos_equal_one_sum)
                    self.pos_equal_one_for_reg.append(pos_equal_one_for_reg)
                    self.neg_equal_one.append(neg_equal_one)
                    self.neg_equal_one_sum.append(neg_equal_one_sum)

                    # output
                    feature_output = feature.outputs
                    delta_output = rpn.delta_output
                    prob_output = rpn.prob_output

                    # loss and grad
                    self.loss = rpn.loss
                    self.reg_loss = rpn.reg_loss
                    self.cls_loss = rpn.cls_loss
                    self.cls_pos_loss = rpn.cls_pos_loss_rec
                    self.cls_neg_loss = rpn.cls_neg_loss_rec

                    # list add A + B
                    #self.params = rpn.variables+feature.variables
                    self.trainable_params = rpn.trainable_variables+feature.trainable_variables
                    #print("Variables is equal to trainable:{}".format(self.params==self.trainable_params))
                    #for i in self.params:
                    #    print("{}:{}".format("variables",i.name))
                    #for i in self.trainable_params:
                    #    print("{}:{}".format("trainable_variables",i.name))
                    # calculate gradients and update it
                    gradients = tape.gradient(rpn.loss,self.trainable_params)
                    # clip the gradients
                    clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                    #self.opt.apply_gradients(zip(clipped_gradients,self.trainable_params))
                    self.delta_output.append(delta_output)
                    self.prob_output.append(prob_output)
                    self.tower_grads.append(clipped_gradients)
                    self.gradient_norm.append(gradient_norm)
                    self.rpn_output_shape= rpn.output_shapes

            # loss and optimizer
            # self.xxxloss is only the loss for the lowest tower
            with tf.device('/gpu:{}'.format(self.avail_gpus[0])):
                self.grads = average_gradients(self.tower_grads)
                self.update = [self.opt.apply_gradients(zip(self.grads, self.trainable_params))]
                self.gradient_norm = tf.group(*self.gradient_norm)


