#ifndef DDRNET_COMMON_H_
#define DDRNET_COMMON_H_

#include <iostream>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "dirent.h"
#include "NvInfer.h"
#include <chrono>



#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

using namespace nvinfer1;

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--) {
        Weights wt{ DataType::kFLOAT, nullptr, 0 };
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x) {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
//        std::cout<<""<<name<<std::endl;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}


ILayer* basicBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int stride, bool downsample, bool no_relu, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* conv1 = network->addConvolution(input, outch, DimsHW{ 3, 3 }, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{ stride, stride });
    conv1->setPadding(DimsHW{ 1, 1 });

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolution(*relu1->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setPadding(DimsHW{ 1, 1 });

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);


    if(downsample){
        IConvolutionLayer* convdown = network->addConvolution(input, outch, DimsHW{ 1, 1 }, weightMap[lname + "downsample.0.weight"], emptywts);
        assert(convdown);
        convdown->setStride(DimsHW{ stride, stride});
        convdown->setPadding(DimsHW{ 0, 0 });
        IScaleLayer* bndown = addBatchNorm2d(network, weightMap, *convdown->getOutput(0), lname + "downsample.1", 1e-5);


        IElementWiseLayer* ew1 = network->addElementWise(*bn2->getOutput(0), *bndown->getOutput(0), ElementWiseOperation::kSUM);
        if(no_relu){
            return ew1;
        }else{
            IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
            assert(relu3);
            return relu3;
        }
    }
    IElementWiseLayer* ew2 = network->addElementWise(input, *bn2->getOutput(0), ElementWiseOperation::kSUM);
    if(no_relu){
        return ew2;
    }else{
        IActivationLayer* relu3 = network->addActivation(*ew2->getOutput(0), ActivationType::kRELU);
        assert(relu1);
        return relu3;
    }
}

ILayer* Bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int planes, int stride, bool downsample, bool no_relu, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int expansion = 2;

    IConvolutionLayer* conv1 = network->addConvolution(input, planes, DimsHW{ 1, 1 }, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{ 1, 1 });
//    conv1->setPadding(DimsHW{ 0, 0 });

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);


    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolution(*relu1->getOutput(0), planes, DimsHW{ 3, 3 }, weightMap[lname + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStride(DimsHW{ stride, stride });
    conv2->setPadding(DimsHW{ 1, 1 });

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);


    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IConvolutionLayer* conv3 = network->addConvolution(*relu2->getOutput(0), planes*expansion, DimsHW{ 1, 1 }, weightMap[lname + "conv3.weight"], emptywts);
    assert(conv3);
    conv3->setStride(DimsHW{ 1, 1 });
//    conv3->setPadding(DimsHW{ 0, 0 });

    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "bn3", 1e-5);

    if(downsample){
        IConvolutionLayer* convdown = network->addConvolution(input, planes*expansion, DimsHW{ 1, 1 }, weightMap[lname + "downsample.0.weight"], emptywts);
        assert(convdown);
        convdown->setStride(DimsHW{ stride, stride });
//        conv1->setPadding(DimsHW{ 0, 0 });

        IScaleLayer* bndown = addBatchNorm2d(network, weightMap, *convdown->getOutput(0), lname + "downsample.1", 1e-5);


        IElementWiseLayer* ew1 = network->addElementWise(*bn3->getOutput(0), *bndown->getOutput(0), ElementWiseOperation::kSUM);
        if(no_relu){
            return ew1;
        }else{
            IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
            assert(relu1);
            return relu3;
        }
    }
    IElementWiseLayer* ew2 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    if(no_relu){
        return ew2;
    }else{
        IActivationLayer* relu3 = network->addActivation(*ew2->getOutput(0), ActivationType::kRELU);
        assert(relu1);
        return relu3;
    }
}

IScaleLayer* ConvBnNoStride(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* conv1 = network->addConvolution(input, outch, DimsHW{1, 1 }, weightMap[lname + "0.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{ 1, 1 });
    conv1->setPadding(DimsHW{ 0, 0 });
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-5);

    return bn1;
}


IActivationLayer* Conv1(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, std::string lname) {
    //self.conv1 =  nn.Sequential(
    //nn.Conv2d(3,planes,kernel_size=3, stride=2, padding=1),
    //BatchNorm2d(planes, momentum=bn_mom),
    //nn.ReLU(inplace=True),
    //nn.Conv2d(planes,planes,kernel_size=3, stride=2, padding=1),
    //BatchNorm2d(planes, momentum=bn_mom),
    //nn.ReLU(inplace=True),
    //)

    IConvolutionLayer* conv1 = network->addConvolution(input, outch, DimsHW{ 3, 3 }, weightMap[lname + "0.weight"], weightMap[lname + "0.bias"]);
    assert(conv1);
    conv1->setStride(DimsHW{ 2, 2 });
    conv1->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    IConvolutionLayer* conv2 = network->addConvolution(*relu1->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + "3.weight"], weightMap[lname + "3.bias"]);
    assert(conv2);
    conv2->setStride(DimsHW{ 2, 2 });
    conv2->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "4", 1e-5);
    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    return relu2;
}



IScaleLayer* ConvBn(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int kernelsize, int stride, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* conv1 = network->addConvolution(input, outch, DimsHW{ kernelsize, kernelsize }, weightMap[lname + "0.weight"], emptywts);
    assert(conv1);
    int padding = (int) (stride/2);
    conv1->setStride(DimsHW{ stride, stride });
    conv1->setPadding(DimsHW{ padding, padding });
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-5);

    return bn1;
}


ILayer* compression3(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int highres_planes, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* conv1 = network->addConvolution(input, highres_planes , DimsHW{ 1, 1 }, weightMap[lname + "0.weight"], emptywts);
    assert(conv1);
    conv1->setPadding(DimsHW{ 0, 0 });
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-5);
    return bn1;
}

ILayer* PagFM(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  ITensor& input2, int mid_channels, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int inch = input.getDimensions().d[1];
    ILayer* y_q = ConvBnNoStride(network, weightMap, input2, mid_channels, lname + "f_y.");

    ILayer* x_k = ConvBnNoStride(network, weightMap, input, mid_channels, lname + "f_x.");
    auto y_q_up = network->addResize(*y_q->getOutput(0));
    y_q_up->setResizeMode(ResizeMode::kLINEAR);
    y_q_up->setOutputDimensions(x_k->getOutput(0)->getDimensions());
    assert(y_q_up);


    //sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
    IElementWiseLayer* sim_map = network->addElementWise(*x_k->getOutput(0), *y_q_up->getOutput(0), ElementWiseOperation::kPROD);

    float *deval2 = reinterpret_cast<float*>(malloc(sizeof(float) * mid_channels * 1 * 1));
    for (int i = 0; i < mid_channels * 1 * 1; i++) {
        deval2[i] = 1.0;
    }
    Weights deconvwts2{ DataType::kFLOAT, deval2, mid_channels * 1 * 1 };
    IConvolutionLayer* sim_map_unsqueeze = network->addConvolution(*sim_map->getOutput(0), 1 , DimsHW{ 1, 1 }, deconvwts2, emptywts);

    assert(sim_map_unsqueeze);
    sim_map_unsqueeze->setPadding(DimsHW{ 0, 0 });
    IActivationLayer* thresh = network->addActivation(*sim_map_unsqueeze->getOutput(0), ActivationType::kSIGMOID);
    assert(thresh);


    float *deval3 = reinterpret_cast<float*>(malloc(sizeof(float) * inch * 1 * 1));
    for (int i = 0; i < inch * 1 * 1; i++) {
        deval3[i] = 1.0;
    }
    Weights deconvwts3{ DataType::kFLOAT, deval3, inch * 1 * 1 };
    IConvolutionLayer* thresh_broadcast = network->addConvolution(*thresh->getOutput(0), inch , DimsHW{ 1, 1 }, deconvwts3, emptywts);

    //y = F.interpolate(y, size=[input_size[2], input_size[3]],    mode='nearest')
    auto y_up = network->addResize(input2);
    assert(y_up);
    y_up->setResizeMode(ResizeMode::kLINEAR);
    y_up->setOutputDimensions(input.getDimensions());
    //x = (1-sim_map)*x + sim_map*y

    IElementWiseLayer* sim_map_x = network->addElementWise(*thresh->getOutput(0), input, ElementWiseOperation::kPROD);

    // thresh:( 1; 1; 64; 128)   input2 :(1, 64; 32; 64)
    IElementWiseLayer* sim_map_y = network->addElementWise(*thresh->getOutput(0), *y_up->getOutput(0), ElementWiseOperation::kPROD);

    IElementWiseLayer* sim_map_x2 = network->addElementWise(input, *sim_map_x->getOutput(0), ElementWiseOperation::kSUB);

    IElementWiseLayer* out = network->addElementWise(*sim_map_x2->getOutput(0), *sim_map_y->getOutput(0), ElementWiseOperation::kSUM);

    return out;
}



ILayer* diffblock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* conv1 = network->addConvolution(input, outch , DimsHW{ 3, 3 }, weightMap[lname + "0.weight"], emptywts);
    assert(conv1);
    conv1->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-5);
    return bn1;


}


ILayer* compression(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int highres_planes, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    IConvolutionLayer* conv1 = network->addConvolution(input, highres_planes , DimsHW{ 1, 1 }, weightMap[lname + "0.weight"], emptywts);
    assert(conv1);
    conv1->setPadding(DimsHW{ 0, 0 });
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-5);
    return bn1;
}


ILayer* PoolingSingle(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int pooling_kernel_h, int pooling_kernel_w, int stride, int padding, int outch, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IPoolingLayer* avgpooling = network->addPooling(input, PoolingType::kAVERAGE, DimsHW{ pooling_kernel_h, pooling_kernel_w });
    assert(avgpooling);
    avgpooling->setStride(DimsHW{ stride, stride });
    avgpooling->setPadding(DimsHW{ padding, padding });
    avgpooling->setAverageCountExcludesPadding(false);
    return avgpooling;
}



ILayer* PoolingBnReluConv(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int pooling_kernel_h, int pooling_kernel_w, int stride, int padding, int outch, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IPoolingLayer* avgpooling = network->addPooling(input, PoolingType::kAVERAGE, DimsHW{ pooling_kernel_h, pooling_kernel_w });
    assert(avgpooling);
    avgpooling->setStrideNd(DimsHW{ stride, stride });
    avgpooling->setPaddingNd(DimsHW{ padding, padding });
    avgpooling->setAverageCountExcludesPadding(false);

    IScaleLayer* scalebn = addBatchNorm2d(network, weightMap, *avgpooling->getOutput(0), lname + "1", 1e-5);


    IActivationLayer* scalerelu = network->addActivation(*scalebn->getOutput(0), ActivationType::kRELU);


    IConvolutionLayer* scaleconv = network->addConvolution(*scalerelu->getOutput(0), outch , DimsHW{ 1, 1 }, weightMap[lname + "3.weight"], emptywts);
    assert(scaleconv);
//    scaleconv->setPadding(DimsHW{ 0, 0 });

    return scaleconv;
}


ILayer* BnReluConv(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int kernel, int stride, int groups, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };


    IScaleLayer* scalebn = addBatchNorm2d(network, weightMap, input, lname + "0", 1e-5);


    IActivationLayer* scalerelu = network->addActivation(*scalebn->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* scaleconv = network->addConvolution(*scalerelu->getOutput(0), outch , DimsHW{ kernel, kernel }, weightMap[lname + "2.weight"], emptywts);
    assert(scaleconv);
    scaleconv->setStride(DimsHW{ stride, stride });
    int padsize = (int) (kernel/2);
    scaleconv->setPadding(DimsHW{ padsize, padsize });
    scaleconv->setNbGroups(groups);
    return scaleconv;
}


ILayer* Light_Bag(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input_p, ITensor& input_i, ITensor& input_d, int outch, std::string lname) {
    IActivationLayer* edge_att = network->addActivation(input_d, ActivationType::kSIGMOID);
    //p_add = self.conv_p((1-edge_att)*i + p)
    //(1-edge_att)*i
    IElementWiseLayer* edge_att_i = network->addElementWise(*edge_att->getOutput(0), input_i, ElementWiseOperation::kPROD);
    IElementWiseLayer* edge_att_2 = network->addElementWise(input_i, *edge_att_i->getOutput(0), ElementWiseOperation::kSUB);
    IElementWiseLayer* edge_att_p = network->addElementWise(*edge_att_2->getOutput(0), input_p, ElementWiseOperation::kSUM);
    //p_add = self.conv_p((1-edge_att)*i + p)
    IScaleLayer* p_add = ConvBn(network, weightMap, *edge_att_p->getOutput(0), outch, 1, 1, lname + "conv_p.");

    //i + edge_att*p
    IElementWiseLayer* edge_att_p2 = network->addElementWise(*edge_att->getOutput(0), input_p, ElementWiseOperation::kPROD);
    IElementWiseLayer* i_add_input = network->addElementWise(input_i, *edge_att_p2->getOutput(0), ElementWiseOperation::kSUM);
    //i_add = self.conv_i(i + edge_att*p)
    ILayer* i_add = ConvBn(network, weightMap, *i_add_input->getOutput(0), outch, 1, 1, lname + "conv_i.");
    IElementWiseLayer* out = network->addElementWise(*p_add->getOutput(0), *i_add->getOutput(0), ElementWiseOperation::kSUM);

    return out;
}


ILayer* down3(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int planes, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* conv1 = network->addConvolution(input, planes , DimsHW{ 3, 3 }, weightMap[lname + "0.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{ 2, 2 });
    conv1->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-5);
    return bn1;
}

ILayer* down4(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int planes, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* conv1 = network->addConvolution(input, planes , DimsHW{ 3, 3 }, weightMap[lname + "0.weight"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{ 2, 2 });
    conv1->setPadding(DimsHW{ 1, 1 });
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);

    IConvolutionLayer* conv2 = network->addConvolution(*relu1->getOutput(0), planes*2 , DimsHW{ 3, 3 }, weightMap[lname + "3.weight"], emptywts);
    assert(conv2);
    conv2->setStride(DimsHW{ 2, 2 });
    conv2->setPadding(DimsHW{ 1, 1 });

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "4", 1e-5);
    return bn2;
}


ILayer* PAPPM(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int branch_planes, int outplanes, std::string lname) {
    //x_ = self.scale0(x)
    int input_w = input.getDimensions().d[3];
    int input_h = input.getDimensions().d[2];  // nchw
    ILayer* scale0 = BnReluConv(network, weightMap, input, branch_planes, 1, 1, 1,  lname + "scale0.");
    //scale_list.append(F.interpolate(self.scale1(x), size=[height, width],    mode='bilinear', align_corners=algc)+x_)
    ILayer* scale1 = PoolingBnReluConv(network, weightMap, input, 5, 5, 2,  2,  branch_planes, lname + "scale1.");

    auto scale_list1 = network->addResize(*scale1->getOutput(0));
    assert(scale_list1);
    scale_list1->setResizeMode(ResizeMode::kLINEAR);
    scale_list1->setOutputDimensions(Dims4{ 1, branch_planes, input_h, input_w});

    IElementWiseLayer* scale_list1add = network->addElementWise(*scale_list1->getOutput(0), *scale0->getOutput(0), ElementWiseOperation::kSUM);


    //scale_list.append(F.interpolate(self.scale2(x), size=[height, width],    mode='bilinear', align_corners=algc)+x_)
    ILayer* scale2 = PoolingBnReluConv(network, weightMap, input, 9, 9, 4, 4, branch_planes, lname + "scale2.");
    auto scale_list2 = network->addResize(*scale2->getOutput(0));
    assert(scale_list2);
    scale_list2->setResizeMode(ResizeMode::kLINEAR);
    scale_list2->setOutputDimensions(Dims4{ 1, branch_planes, input_h, input_w});
    IElementWiseLayer* scale_list2add = network->addElementWise(*scale_list2->getOutput(0), *scale0->getOutput(0), ElementWiseOperation::kSUM);

    // scale_list.append(F.interpolate(self.scale3(x), size=[height, width],    mode='bilinear', align_corners=algc)+x_)
    ILayer* scale3 = PoolingBnReluConv(network, weightMap, input, 17, 17, 8, 8,  branch_planes, lname + "scale3.");
    auto scale_list3 = network->addResize(*scale3->getOutput(0));
    assert(scale_list3);
    scale_list3->setResizeMode(ResizeMode::kLINEAR);
    scale_list3->setOutputDimensions(Dims4{ 1, branch_planes, input_h, input_w});
    IElementWiseLayer* scale_list3add = network->addElementWise(*scale_list3->getOutput(0), *scale0->getOutput(0), ElementWiseOperation::kSUM);


    //scale_list.append(F.interpolate(self.scale4(x), size=[height, width],    mode='bilinear', align_corners=algc)+x_)
    int globle_kernel_h = input.getDimensions().d[2];
    int globle_kernel_w = input.getDimensions().d[3];
    ILayer* scale4 = PoolingBnReluConv(network, weightMap, input, globle_kernel_h, globle_kernel_w, 1, 0, branch_planes, lname + "scale4.");

    auto scale_list4 = network->addResize(*scale4->getOutput(0));
    assert(scale_list4);
    scale_list4->setResizeMode(ResizeMode::kLINEAR);
    scale_list4->setOutputDimensions(Dims4{ 1, branch_planes, input_h, input_w});

    IElementWiseLayer* scale_list4add = network->addElementWise(*scale_list4->getOutput(0), *scale0->getOutput(0), ElementWiseOperation::kSUM);
    //  torch.cat(scale_list, 1)
        // out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
    ITensor* inputTensors[] = {scale_list1add->getOutput(0),  scale_list2add->getOutput(0) ,  scale_list3add->getOutput(0), scale_list4add->getOutput(0)};
    IConcatenationLayer* neck_cat = network->addConcatenation(inputTensors, 4);
    //scale_out = self.scale_process(torch.cat(scale_list, 1))
    ILayer* scale_process = BnReluConv(network, weightMap, *neck_cat->getOutput(0), branch_planes*4, 3, 1, 4,  lname + "scale_process.");

    //torch.cat([x_,scale_out], 1)
    ITensor* inputTensors2[] = {scale0->getOutput(0),  scale_process->getOutput(0)};
    IConcatenationLayer* neck_cat2 = network->addConcatenation(inputTensors2, 2);

    ILayer* compression = BnReluConv(network, weightMap, *neck_cat2->getOutput(0), outplanes, 1, 1, 1, lname + "compression.");

    ILayer* shortcut = BnReluConv(network, weightMap, input, outplanes, 1, 1, 1,  lname + "shortcut.");

    IElementWiseLayer* out = network->addElementWise(*compression->getOutput(0), *shortcut->getOutput(0), ElementWiseOperation::kSUM);

    return out;
}




ILayer* segmenthead(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int interplanes, int outplanes, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, input, lname + "bn1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* conv1 = network->addConvolution(*relu1->getOutput(0), interplanes , DimsHW{ 3, 3 }, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);
    conv1->setPadding(DimsHW{ 1, 1 });

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn2", 1e-5);


    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    IConvolutionLayer* conv2 = network->addConvolution(*relu2->getOutput(0), outplanes , DimsHW{ 1, 1 }, weightMap[lname + "conv2.weight"], weightMap[lname + "conv2.bias"]);
    assert(conv2);
    conv2->setPadding(DimsHW{ 0, 0 });

    return conv2;
}

int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }
    closedir(p_dir);
    return 0;
}

static const int map_[19][3] = { {128, 64,128},
                                {244, 35,232},
                                { 70, 70, 70},
                                {102,102,156},
                                {190,153,153},
                                {153,153,153},
                                {250,170, 30},
                                {220,220,  0},
                                {107,142, 35},
                                {152,251,152},
                                { 70,130,180},
                                {220, 20, 60},
                                {255,  0,  0},
                                {  0,  0,142},
                                {  0,  0, 70},
                                {  0, 60,100},
                                {  0, 80,100},
                                {  0,  0,230},
                                {119, 11, 32}};


static const int maparea_[2][3] = { {0,0,0} ,
                                {0,255,0}};


cv::Mat read2mat(float * prob, cv::Mat out)
{
    for (int i = 0; i < 128; ++i)
    {
        cv::Vec<float, 19> *p1 = out.ptr<cv::Vec<float, 19>>(i);
        for (int j = 0; j < 128; ++j)
        {
            for (int c = 0; c < 19; ++c)
            {
                p1[j][c] = prob[c * 128 * 128 + i * 128 + j];
            }
        }
    }
    return out;
}


cv::Mat map2cityscape(cv::Mat real_out,cv::Mat real_out_)
{
    for (int i = 0; i < 128; ++i)
    {
        uchar *p1 = real_out.ptr<uchar>(i);
        cv::Vec3b *p2 = real_out_.ptr<cv::Vec3b>(i);
        for (int j = 0; j < 128; ++j)
        {
            int index = p1[j];

            p2[j][0] = map_[index][2];
            p2[j][1] = map_[index][1];
            p2[j][2] = map_[index][0];

        }
    }
    return real_out_;
}

void map2mask(cv::Mat& mat,cv::Mat& out_cls)
{
    for (int i = 0; i < 128; ++i)
    {
        cv::Vec<float, 19> *p1 = mat.ptr<cv::Vec<float, 19>>(i);
        uchar *p2 = out_cls.ptr<uchar>(i);
        for (int j = 0; j < 128; ++j)
        {
            int index = 0;
            float swap;
            for (int c = 0; c < 19; ++c)
            {
                if (p1[j][0] < p1[j][c])
                {
                    swap = p1[j][0];
                    p1[j][0] = p1[j][c];
                    p1[j][c] = swap;
                    index = c;
                }
            }
            p2[j] = index;
        }
    }
}



void map2area(cv::Mat real_out,cv::Mat &real_out_)
{
    for (int i = 0; i < 64; ++i)
    {
        uchar *p1 = real_out.ptr<uchar>(i);
        cv::Vec3b *p2 = real_out_.ptr<cv::Vec3b>(i);
        for (int j = 0; j < 64; ++j)
        {
            int index = p1[j];

            p2[j][0] = maparea_[index][2];
            p2[j][1] = maparea_[index][1];
            p2[j][2] = maparea_[index][0];

        }
    }
}


#endif

