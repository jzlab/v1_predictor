from .parser import gen_parser
from keras.layers import Conv2D,Dense,Flatten,Input,MaxPooling2D

parser = gen_parser()
FLAGS,unparsed = parser.parse_known_args()



#1 all-to-all hidden layer
dense_list = [FLAGS.hidden1] # [300]
keep_prob = FLAGS.dropout # 0.55
# Attempt to make port wk's class definitions to keras to make easier to work with
class ConvNetDrop(object):
    def __init__(self,images,
                num_filter_list,
                filter_size_list,
                pool_stride_list,
                pool_k_list,
                dense_list,
                keep_prob):

        self.num_filter_list = num_filter_list
        self.filter_size_list = filter_size_list
        self.pool_stride_list = pool_stride_list
        self.pool_k_list = pool_k_list
        self.dense_list = dense_list
        self.input = images

        self.build(self.input)

    def build(self,x):
        net = x
        generator = zip(self.num_filter_list,self.filter_size_list,self.pool_stride_list,self.pool_k_list)

        for filters,k_sz,pstride,pk in generator:
                net = Conv2D(filters,k_sz,kernel_initializer='truncated_normal')(net)
                net = MaxPooling2D(pool_size=(pk,pk),strides=(pstride,pstride))(net)

        net = Flatten()(net)

        for units in self.dense_list:
                net = Dense(units)(net)

        self.output = net