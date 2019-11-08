# %load model/rpn_1.py
import tensorflow as tf
import numpy as np

from config import cfg


small_addon_for_BCE = 1e-6

class MiddleAndRPN(tf.keras.Model):
    def __init__(self):
        # scale = [batchsize, 10, 400/200, 352/240, 128] should be the output of feature learning network
        super(MiddleAndRPN,self).__init__()
        self.conv1 = ConvMD(3,128,64,3,(2,1,1),(1,1,1),name="conv1")
        self.conv2 = ConvMD(3,64,64,3,(1,1,1),(0,1,1),name="conv2")
        self.conv3 = ConvMD(3,64,64,3,(2,1,1),(1,1,1),name="conv3")
        self.conv4 = ConvMD(2,128,128,3,(2,2),(1,1),name = "conv4")
        self.conv5 = ConvMD(2,128,128,3,(1,1),(1,1),name = "conv5")
        self.conv6 = ConvMD(2,128,128,3,(1,1),(1,1),name = "conv6")
        self.conv7 = ConvMD(2,128,128,3,(1,1),(1,1),name = "conv7")

        self.deconv1 = Deconv2D(128,256,3,(1,1),(0,0),name="deconv1")

        self.conv8 = ConvMD(2,128,128,3,(2,2),(1,1),name="conv8")
        self.conv9 = ConvMD(2,128,128,3,(1,1),(1,1),name="conv9")
        self.conv10 = ConvMD(2,128,128,3,(1,1),(1,1),name="conv10")
        self.conv11 = ConvMD(2,128,128,3,(1,1),(1,1),name="conv11")
        self.conv12 = ConvMD(2,128,128,3,(1,1),(1,1),name="conv12")
        self.conv13= ConvMD(2,128,128,3,(1,1),(1,1),name="conv13")

        self.deconv2 = Deconv2D(128,256,2,(2,2),(0,0),name="deconv2")

        self.conv14 = ConvMD(2,128,256,3,(2,2),(1,1),name="conv14")
        self.conv15 = ConvMD(2,256,256,3,(1,1),(1,1),name="conv15")
        self.conv16 = ConvMD(2,256,256,3,(1,1),(1,1),name="conv16")
        self.conv17 = ConvMD(2,256,256,3,(1,1),(1,1),name="conv17")
        self.conv18 = ConvMD(2,256,256,3,(1,1),(1,1),name="conv18")
        self.conv19 = ConvMD(2,256,256,3,(1,1),(1,1),name="conv19")
        self.deconv3 = Deconv2D(256,256,4,(4,4),(0,0),name="deconv3")

        self.conv2d_pro = ConvMD(2,768,2,1,(1,1),(0,0),name="score_pro")
        self.conv2d_reg = ConvMD(2,768,14,1,(1,1),(0,0),name = "reg_pro")

    def __call__(self,inputs,pos_equal_one,neg_equal_one,targets,pos_equal_one_for_reg,pos_equal_one_sum,neg_equal_one_sum,alpha=1.5,beta=1,sigma=3,training=True,name=''):
        # scale = [batchsize, 10, 400/200, 352/240, 128] should be the output of feature learning network
        self.inputs = inputs
        self.training=training
        self.targets= targets
        self.pos_equal_one = pos_equal_one
        self.pos_equal_one_sum = pos_equal_one_sum
        self.pos_equal_one_for_reg = pos_equal_one_for_reg
        self.neg_equal_one = neg_equal_one
        self.neg_equal_one_sum = neg_equal_one_sum

        temp_conv=self.conv1(self.inputs)
        temp_conv=self.conv2(temp_conv)
        temp_conv=self.conv3(temp_conv)
        temp_conv = tf.transpose(a=temp_conv,perm=[0,2,3,4,1])
        temp_conv=tf.reshape(temp_conv,[-1,cfg.INPUT_HEIGHT,cfg.INPUT_WIDTH,128])

        #rpn
        #block1:
        temp_conv = self.conv4(temp_conv,training=self.training)
        temp_conv = self.conv5(temp_conv,training=self.training)
        temp_conv = self.conv6(temp_conv,training=self.training)
        temp_conv = self.conv7(temp_conv,training=self.training)
        deconv1 = self.deconv1(temp_conv,training=self.training)

        #block2:
        temp_conv= self.conv8(temp_conv,training=self.training)
        temp_conv= self.conv9(temp_conv,training=self.training)
        temp_conv= self.conv10(temp_conv,training=self.training)
        temp_conv= self.conv11(temp_conv,training=self.training)
        temp_conv= self.conv12(temp_conv,training=self.training)
        temp_conv= self.conv13(temp_conv,training=self.training)
        deconv2 = self.deconv2(temp_conv,training=self.training)

        #block3:
        temp_conv=self.conv14(temp_conv,training=self.training)
        temp_conv=self.conv15(temp_conv,training=self.training)
        temp_conv=self.conv16(temp_conv,training=self.training)
        temp_conv=self.conv17(temp_conv,training=self.training)
        temp_conv=self.conv18(temp_conv,training=self.training)
        temp_conv=self.conv19(temp_conv,training=self.training)
        deconv3 = self.deconv3(temp_conv,training=self.training)

        temp_conv=tf.concat([deconv3,deconv2,deconv1],-1)
        p_map=self.conv2d_pro(temp_conv,training=self.training,activation=False,bn=False)
        r_map=self.conv2d_reg(temp_conv,training=self.training,activation=False,bn=False)
        # softmax output for positive anchor and negative anchor, scale = [None, 200/100, 176/120, 1(2)]

        self.p_pos = tf.sigmoid(p_map)
        #self.p_pos = tf.nn.softmax(p_map, dim=3)
        self.output_shapes = [cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH]

        self.cls_pos_loss = (-self.pos_equal_one * tf.math.log(self.p_pos + small_addon_for_BCE)) / self.pos_equal_one_sum
        self.cls_neg_loss = (-self.neg_equal_one * tf.math.log(1 - self.p_pos + small_addon_for_BCE)) / self.neg_equal_one_sum

        self.cls_loss = tf.reduce_sum( input_tensor=alpha * self.cls_pos_loss + beta * self.cls_neg_loss )
        self.cls_pos_loss_rec = tf.reduce_sum( input_tensor=self.cls_pos_loss )
        self.cls_neg_loss_rec = tf.reduce_sum( input_tensor=self.cls_neg_loss )

        self.reg_loss = smooth_l1(r_map * self.pos_equal_one_for_reg, self.targets *
                                  self.pos_equal_one_for_reg, sigma) / self.pos_equal_one_sum
        self.reg_loss = tf.reduce_sum(input_tensor=self.reg_loss)

        self.loss = tf.reduce_sum(input_tensor=self.cls_loss + self.reg_loss)

        self.delta_output = r_map
        self.prob_output = self.p_pos


#@tf.function
def smooth_l1(deltas, targets, sigma=3.0):
    sigma2 = sigma * sigma
    diffs = tf.subtract(deltas, targets)
    smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

    smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
    smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + \
        tf.multiply(smooth_l1_option2, 1 - smooth_l1_signs)
    smooth_l1 = smooth_l1_add
    return smooth_l1

# bulid layers
class ConvMD(tf.keras.Model):
    def __init__(self,M,Cin,Cout,k,s,p,name=""):
        super(ConvMD,self).__init__()
        temp_p=np.array(p)
        temp_p=np.lib.pad(temp_p,(1,1),'constant',constant_values=(0,0))
        if M==3:
            # padding only valid for depth,width and height, not bathch_size and in_channel
            self.paddings = (np.array(temp_p)).repeat(2).reshape(5,2)
            #pad = tf.pad(tensor=inputs,paddings=paddings,mode="CONSTANT",constant_values=0)
            self.conv = tf.keras.layers.Conv3D(Cout,kernel_size=k,strides=s,padding="valid",name=name)
        if M==2:
            self.paddings=(np.array(temp_p)).repeat(2).reshape(4,2)
            #pad=tf.pad(tensor=inputs,paddings=paddings,mode="CONSTANT",constant_values=0)
            self.conv=tf.keras.layers.Conv2D(Cout,kernel_size=k,strides=s,padding="valid",name=name)
        self.batchnorm = tf.keras.layers.BatchNormalization(name=name+'/batch_norm')
        self.relu = tf.keras.layers.ReLU(name=name)

    def __call__(self,inputs,training=True,activation=True,bn=True):
        pad = tf.pad(tensor=inputs,paddings=self.paddings,mode="CONSTANT",constant_values=0)
        temp_conv = self.conv(pad)
        if bn:
            temp_conv =  self.batchnorm(temp_conv,training=training)
        if activation:
            temp_conv = self.relu(temp_conv)
        return temp_conv

class Deconv2D(tf.keras.Model):
    def __init__(self,Cin,Cout,k,s,p,name=""):
        super(Deconv2D,self).__init__()
        temp_p = np.array(p)
        temp_p = np.lib.pad(temp_p,(1,1),mode='constant',constant_values=(0,0))
        self.padding = np.array(temp_p).repeat(2).reshape(4,2)
        self.conv2dtranspose = tf.keras.layers.Conv2DTranspose(Cout,kernel_size=k,data_format=
                                "channels_last",strides=s,padding="SAME",name=name)
        self.batchnorm=tf.keras.layers.BatchNormalization(name=name+'/batch_norm')
        self.relu = tf.keras.layers.ReLU(name=name)

    def __call__(self,inputs,training=True, activation=True,bn=True):
        pad = tf.pad(tensor=inputs,paddings=self.padding,mode="CONSTANT")
        temp_conv = self.conv2dtranspose(pad)
        if bn:
            temp_conv = self.batchnorm(temp_conv,training=training)
        if activation:
            temp_conv = self.relu(temp_conv)
        return temp_conv

#if(__name__ == "__main__"):
#    m = MiddleAndRPN(tf.compat.v1.placeholder(
#        tf.float32, [None, 10, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128]))
