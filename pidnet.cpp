#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include <math.h>
#include "segresize.h"

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id

static const int INPUT_W = 1024;
static const int INPUT_H = 1024;
static const int OUT_MAP_W = 128;
static const int OUT_MAP_H = 128;
static const int OUT_MAP_C = 19;



static const std::string wts_path = "/path-to-wts/PIDNet.wts";


const char* INPUT_BLOB_NAME = "input_0";
const char* OUTPUT_BLOB_NAME = "output_0";
static Logger gLogger;

#define CHECK(status)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cerr << "Cuda failure: " << ret << std::endl;                                                         \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << cudaGetErrorString(error_code) << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

inline void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}


// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);
    // Create input tensor of shape {1, 3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims4{ 1, 3, INPUT_H, INPUT_W });
    assert(data);


//    IActivationLayer* linear = network->addActivation(*data, ActivationType::kLEAKY_RELU);
//    linear->setAlpha(1);



    std::map<std::string, Weights> weightMap = loadWeights(wts_path);
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    //pytorch: self.conv1
    ILayer* conv1 = Conv1(network, weightMap, *data, 32, "conv1.");
    // layer1
    ILayer* layer1_0 = basicBlock(network, weightMap, *conv1->getOutput(0), 32, 1, false, false, "layer1.0.");
    ILayer* layer1_1 = basicBlock(network, weightMap, *layer1_0->getOutput(0), 32, 1, false, true, "layer1.1.");
    IActivationLayer* layer1_relu = network->addActivation(*layer1_1->getOutput(0), ActivationType::kRELU);
    assert(layer1_relu);

    // layer2
    ILayer* layer2_0 = basicBlock(network, weightMap, *layer1_relu->getOutput(0), 64, 2, true, false, "layer2.0.");
    ILayer* layer2_1 = basicBlock(network, weightMap, *layer2_0->getOutput(0), 64, 1, false, true, "layer2.1."); // 1/8
    IActivationLayer* layer2_relu = network->addActivation(*layer2_1->getOutput(0), ActivationType::kRELU);
    assert(layer2_relu);

    // layer3
    ILayer* layer3_0 = basicBlock(network, weightMap, *layer2_relu->getOutput(0), 128, 2, true, false, "layer3.0.");
    ILayer* layer3_1 = basicBlock(network, weightMap, *layer3_0->getOutput(0), 128, 1, false, false, "layer3.1."); // 1/16
    ILayer* layer3_2 = basicBlock(network, weightMap, *layer3_1->getOutput(0), 128, 1, false, true, "layer3.2."); // 1/16
    IActivationLayer* layer3_relu = network->addActivation(*layer3_2->getOutput(0), ActivationType::kRELU);
    assert(layer3_relu);   // layer[2]

    // x_ = self.layer3_(self.relu(layers[1]))
    // layer3_
    ILayer* layer3_10 = basicBlock(network, weightMap, *layer2_relu->getOutput(0), 64, 1, false, false, "layer3_.0.");
    ILayer* layer3_11 = basicBlock(network, weightMap, *layer3_10->getOutput(0), 64, 1, false, true, "layer3_.1."); // x_ = self.layer3_(self.relu(layers[1]))
    //x_ = self.pag3(x_, self.compression3(x))	#1/16
    //compression3
    ILayer* compression3_input = compression3(network, weightMap, *layer3_relu->getOutput(0), 64, "compression3.");

    //x_ = self.pag3(x_, self.compression3(x))	#1/16  : x:1/16, x_:1/8
    ILayer* x_pag3 = PagFM(network, weightMap, *layer3_11->getOutput(0), *compression3_input->getOutput(0), 32, "pag3." );

    //x_d = self.layer3_d(x)				#1/8
    ILayer* layer3_d0 = basicBlock(network, weightMap, *layer2_relu->getOutput(0), 32, 1, true, true, "layer3_d.");

    //x_d = x_d + F.interpolate(self.diff3(x),size=[height_output, width_output], mode='nearest')	#1/8
    ILayer* diff3out = diffblock(network, weightMap, *layer3_relu->getOutput(0), 32, "diff3.");

    auto updiff3out = network->addResize(*diff3out->getOutput(0));
    assert(updiff3out);
    updiff3out->setResizeMode(ResizeMode::kLINEAR);
    updiff3out->setOutputDimensions(layer3_d0->getOutput(0)->getDimensions());
    IElementWiseLayer* x_d = network->addElementWise(*layer3_d0->getOutput(0), *updiff3out->getOutput(0), ElementWiseOperation::kSUM);

    // layer4
//   x = self.relu(self.layer4(x))			#1/32
    ILayer* layer4_0 = basicBlock(network, weightMap, *layer3_relu->getOutput(0), 256, 2, true, false, "layer4.0.");
    //  x = self.layer4(self.relu(x))
    ILayer* layer4_1 = basicBlock(network, weightMap, *layer4_0->getOutput(0), 256, 1, false, false, "layer4.1."); // 1/32
    ILayer* layer4_2 = basicBlock(network, weightMap, *layer4_1->getOutput(0), 256, 1, false, true, "layer4.2."); // 1/32

    IActivationLayer* layer4_relu = network->addActivation(*layer4_2->getOutput(0), ActivationType::kRELU);
    assert(layer4_relu);

    // layer4_
    //x_ = self.layer4_(self.relu(x_))		#1/8
    IActivationLayer* layer4_1_input = network->addActivation(*x_pag3->getOutput(0), ActivationType::kRELU);
    ILayer* layer4_10 = basicBlock(network, weightMap, *layer4_1_input->getOutput(0), 64, 1, false, false, "layer4_.0.");
    //  x_ = self.layer4_(self.relu(x_))
    ILayer* layer4_11 = basicBlock(network, weightMap, *layer4_10->getOutput(0), 64, 1, false, true, "layer4_.1."); // 1/8

    //layer4_d
    // x_d = self.layer4_d(self.relu(x_d))		#1/8
    IActivationLayer* layer4_d_input = network->addActivation(*x_d->getOutput(0), ActivationType::kRELU);
    ILayer* layer4_d1 = Bottleneck(network, weightMap, *layer4_d_input->getOutput(0), 32, 1, true, true, "layer4_d.0."); // 1/8

    //x_ = self.pag4(x_, self.compression4(x))	#1/8
    //compression4
    ILayer* compression4_input = compression(network, weightMap, *layer4_relu->getOutput(0), 64, "compression4.");
    ILayer* x_pag4 = PagFM(network, weightMap, *layer4_11->getOutput(0), *compression4_input->getOutput(0), 32, "pag4." );


    // x_d = x_d + F.interpolate(    self.diff4(x),    size=[height_output, width_output],    mode='nearest')	#1/8
    ILayer* diff4out = diffblock(network, weightMap, *layer4_relu->getOutput(0), 64, "diff4.");
    auto updiff4out = network->addResize(*diff4out->getOutput(0));
    assert(updiff4out);
    updiff4out->setResizeMode(ResizeMode::kLINEAR);
    updiff4out->setOutputDimensions(layer4_d1->getOutput(0)->getDimensions());
    IElementWiseLayer* x_d_4 = network->addElementWise(*layer4_d1->getOutput(0), *updiff4out->getOutput(0), ElementWiseOperation::kSUM);

    //x_ = self.layer5_(self.relu(x_))		#1/8
    IActivationLayer* layer5_0_input = network->addActivation(*x_pag4->getOutput(0), ActivationType::kRELU);
    ILayer* layer5_0_out = Bottleneck(network, weightMap, *layer5_0_input->getOutput(0), 64, 1, true, true, "layer5_.0.");
    // x_d = self.layer5_d(self.relu(x_d))
    IActivationLayer* layer5_d_input = network->addActivation(*x_d_4->getOutput(0), ActivationType::kRELU);
    ILayer* layer5_d0 = Bottleneck(network, weightMap, *layer5_d_input->getOutput(0), 64, 1, true, true, "layer5_d.0."); // 1/8
    // x = F.interpolate(    self.spp(self.layer5(x)),    size=[height_output, width_output],    mode='nearest')	#1/64 => 1/8
    //self.layer5(x))
    ILayer* layer5_0 = Bottleneck(network, weightMap, *layer4_relu->getOutput(0), 256, 2, true, true, "layer5.0.");
    ILayer* layer5_1 = Bottleneck(network, weightMap, *layer5_0->getOutput(0), 256, 1, false, true, "layer5.1.");
    //self.spp(self.layer5(x))

    ILayer* scale1 = PoolingSingle(network, weightMap, *layer5_1->getOutput(0), 5, 5, 2,  2,  96, "spp.scale1.");


    ILayer* ssp = PAPPM(network, weightMap, *layer5_1->getOutput(0), 96, 128, "spp.");

    //x = F.interpolate(    self.spp(self.layer5(x)),    size=[height_output, width_output],    mode='nearest')	#1/64 => 1/8
    auto ssp_up = network->addResize(*ssp->getOutput(0));
    assert(ssp_up);
    ssp_up->setResizeMode(ResizeMode::kLINEAR);
    ssp_up->setOutputDimensions(Dims4{ 1, 128, OUT_MAP_H, OUT_MAP_W });

    //x_dfm = self.dfm(x_, x, x_d)
    ILayer* x_dfm = Light_Bag(network, weightMap, *layer5_0_out->getOutput(0), *ssp_up->getOutput(0), *layer5_d0->getOutput(0), 128, "dfm." );

    //x_ = self.final_layer(x_dfm)
    ILayer* seg_out= segmenthead(network, weightMap, *x_dfm->getOutput(0), 128, OUT_MAP_C, "final_layer.");

    // y = F.interpolate(y, size=(H, W))
    seg_out->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*seg_out->getOutput(0));
    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB

#if defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "../calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#elif defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap) {
        free((void*)(mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    //ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, void **buffers, float* output) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
//    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
//    context.setBindingDimensions(inputIndex, Dims4(1, 3, input_h, input_w));

    // Create GPU buffers on device
//    CHECK(cudaMalloc(&buffers[inputIndex], 3 * INPUT_H * INPUT_W * sizeof(float)));
//    CHECK(cudaMalloc(&buffers[outputIndex],   2* OUT_MAP_H* OUT_MAP_W * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
//    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
//    context.enqueueV2(buffers, stream, nullptr);
    context.enqueue(1, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex],   OUT_MAP_C * OUT_MAP_H* OUT_MAP_W * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{ nullptr };
    size_t size{ 0 };

    if (argc == 2 && std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{ nullptr };
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p("PIDNetS.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }
    else if (argc == 3 && std::string(argv[1]) == "-d") {
        std::ifstream file("PIDNetS.ztmodel", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    }
    else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./debnet -s  // serialize model to plan file" << std::endl;
        std::cerr << "./debnet -d ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // prepare input data ---------------------------
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    std::vector<std::string> file_names;
    if (read_files_in_dir(argv[2], file_names) < 0) {
        std::cout << argv[2] <<std::endl;
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }

//    std::vector<float> mean_value{ 0.406, 0.456, 0.485 };  // BGR
//    std::vector<float> std_value{ 0.225, 0.224, 0.229 };
    int fcount = 0;
    for (auto f : file_names) {
        fcount++;
        std::cout<<"read image:"<<f<<std::endl;
        cv::Mat pr_img = cv::imread(std::string(argv[2]) + "/" + f);
        auto start1 = std::chrono::system_clock::now();

//        cv::resize(pr_img,pr_img,cv::Size(INPUT_W,INPUT_H));
        if (pr_img.empty()) continue;

        void* buffers[2];
        const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
        const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
        assert(inputIndex == 0);
        assert(outputIndex == 1);
        CUDA_CHECK(cudaMalloc(&buffers[inputIndex], 3 * INPUT_H * INPUT_W * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buffers[outputIndex],   OUT_MAP_C * OUT_MAP_H* OUT_MAP_W* sizeof(float)));
//        cudaStream_t stream;
//        CUDA_CHECK(cudaStreamCreate(&stream));
        nvinfer1::Dims inputDim = engine->getBindingDimensions(0);
        bool keepRation = 0 ,keepCenter= 0;

        void *mCudaImg =  safeCudaMalloc(3000*2000*3* sizeof(uchar)); // max input image shape
        CUDA_CHECK(cudaMemcpy(mCudaImg,pr_img.data,pr_img.step[0]*pr_img.rows,cudaMemcpyHostToDevice));
        segresizeAndNorm(mCudaImg,(float*)buffers[0], 0, pr_img.cols,pr_img.rows,inputDim.d[2],inputDim.d[3], keepRation,keepCenter,0);
        CUDA_CHECK(cudaMemset(buffers[1],0, sizeof(int)));


        auto start2 = std::chrono::system_clock::now();


        std::cout <<"pre-process time : "<< std::chrono::duration_cast<std::chrono::milliseconds>(start2 - start1).count() << "ms" << std::endl;

//        std::cout<<data[0]<<" ; "<<data[1]<<" ; "<<data[2]<<" ;...;  "<<data[1024*1024]<<" ; "<<data[2*1024*1024]<<std::endl;

        float* prob = new float[ OUT_MAP_C * OUT_MAP_H* OUT_MAP_W];
        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, buffers, prob);

        auto end = std::chrono::system_clock::now();

        std::cout << "forward time:" << fcount << " time :"<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;


        cv::Mat out;
        out.create(OUT_MAP_H, OUT_MAP_W, CV_32FC(19));
        out = read2mat(prob, out);

        cv::Mat roadout;
        roadout.create(OUT_MAP_H, OUT_MAP_W, CV_8UC1);
        map2mask(out, roadout);

        cv::Mat maskroad;
        maskroad.create(OUT_MAP_H, OUT_MAP_W, CV_8UC3);
        map2cityscape(roadout, maskroad);



        cv::resize(maskroad,maskroad,cv::Size(pr_img.cols,pr_img.rows));
        cv::Mat result;
        cv::addWeighted(pr_img,0.6,maskroad,0.4,1,result);

        cv::imwrite("result_" + f, result);
        //cv::waitKey(0);

        delete prob;
//        delete data;
    }
    return 0;
}
