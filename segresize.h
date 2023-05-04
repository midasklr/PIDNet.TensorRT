//
// Created by yoloCao on 20-5-4.
//

#ifndef TENSORRT_YOLOV4_RESIZE_H
#define TENSORRT_YOLOV4_RESIZE_H
typedef unsigned char uchar;
int segresizeAndNorm(void * p,float *d, int index,int w,int h,int in_w,int in_h, bool keepration ,bool keepcenter,cudaStream_t stream);
#endif //TENSORRT_YOLOV4_RESIZE_H
