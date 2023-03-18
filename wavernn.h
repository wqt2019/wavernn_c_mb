
#define EIGEN_USE_MKL_ALL

#ifndef WAVERNN_H
#define WAVERNN_H

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>


using namespace Eigen;

const int SPARSE_GROUP_SIZE = 4; //When pruning we use groups of 4 to reduce index
const int MULAW_QUANTIZE_CHANNELS = 512;  //same as hparams.mulaw_quantize_channels
const uint8_t ROW_END_MARKER = 255;
const float mu = MULAW_QUANTIZE_CHANNELS - 1;

typedef Matrix<float, Dynamic, Dynamic, RowMajor> Matrixf;
typedef Matrix<float, 1, Dynamic> Vectorf;
typedef Matrix<uint8_t, 1, Dynamic> Vectori8;

Matrixf relu( const Matrixf& x);
Matrixf relu1( const Matrixf& x);
Matrixf concat_row( const Matrixf& x1, const Matrixf& x2 );


class BaseLayer {
public:
    BaseLayer() = default;
    virtual BaseLayer* loadNext( FILE* fd ) {assert(0); return nullptr;};
    virtual Matrixf apply( const Matrixf& x){assert(0); return Matrixf();};
    virtual Vectorf apply( const Vectorf& x){assert(0); return Vectorf();};
    virtual Vectorf apply( const Vectorf& x, const Vectorf& h){assert(0); return Vectorf();};

    virtual Matrixf apply( const Matrixf& x, const Matrixf& h){assert(0); return Matrixf();};

    virtual std::vector<int> shape(void) const { return std::vector<int>(); }

};

//TODO: This should be turned into a proper class factory pattern
class TorchLayer : public BaseLayer {
    struct  Header{
        //int size; //size of data blob, not including this header
        enum class LayerType : int { Conv1d=1, Conv2d=2, BatchNorm1d=3, Linear=4, GRU=5, Stretch2d=6 } layerType;
        char name[64]; //layer name for debugging
    };

    BaseLayer* impl;

public:
    BaseLayer* loadNext( FILE* fd );

    template< typename T> T operator()( const T& x){ return impl->apply( x ); }
    template< typename T, typename T2> T operator()( const T& x, const T2& h ){ return impl->apply( x, h );}
    virtual std::vector<int> shape( void ) const override { return impl->shape(); }

    virtual Matrixf apply( const Matrixf& x) override { return impl->apply(x); };
    virtual Vectorf apply( const Vectorf& x) override { return impl->apply(x); };
    virtual Vectorf apply( const Vectorf& x, const Vectorf& h) override { return impl->apply(x); };

    //
    virtual Matrixf apply( const Matrixf& x, const Matrixf& h) override { return impl->apply(x); };


    virtual ~TorchLayer(){
        delete impl;
        impl=nullptr;
    }
};

class Conv1dLayer : public TorchLayer{
    struct  Header{
        int elSize;  //size of each entry in bytes: 4 for float, 2 for fp16.
        int useBias;
        int inChannels;
        int outChannels;
        int kernelSize;
    };

    std::vector<Matrixf> weight;
    Vectorf bias;

    bool hasBias;
    int inChannels;
    int outChannels;
    int nKernel;
public:
    Conv1dLayer() = default;
    //call TorchLayer loadNext, not derived loadNext
    Conv1dLayer* loadNext( FILE* fd );
    Matrixf apply( const Matrixf& x ) override;
    virtual std::vector<int> shape( void ) const override { return std::vector<int>({inChannels, outChannels, nKernel}); }
};

class Conv2dLayer : public TorchLayer{
    struct  Header{
        int elSize;  //size of each entry in bytes: 4 for float, 2 for fp16.
        int nKernel;  //kernel size. special case of conv2d used in WaveRNN
    };

    Vectorf weight;
    int nKernel;

public:
    Conv2dLayer() = default;
    //call TorchLayer loadNext, not derived loadNext
    Conv2dLayer* loadNext( FILE* fd );
    Matrixf apply( const Matrixf& x ) override;
    virtual std::vector<int> shape(void) const override { return std::vector<int>({nKernel}); }
};

class BatchNorm1dLayer : public TorchLayer{
    struct  Header{
        int elSize;  //size of each entry in bytes: 4 for float, 2 for fp16.
        int inChannels;
        float eps;
    };

    Vectorf weight;
    Vectorf bias;
    Vectorf running_mean;
    Vectorf running_var;
    float eps;
    int nChannels;

public:
    //call TorchLayer loadNext, not derived loadNext
    BatchNorm1dLayer* loadNext( FILE* fd );

    Matrixf apply(const Matrixf &x ) override;
    virtual std::vector<int> shape(void) const override { return std::vector<int>({nChannels}); }
};


class LinearLayer : public TorchLayer{
    struct  Header{
        int elSize;  //size of each entry in bytes: 4 for float, 2 for fp16.
        int nRows;
        int nCols;
    };

    Matrixf mat;
    Vectorf bias;
    int nRows;
    int nCols;


public:
    LinearLayer() = default;
    //call TorchLayer loadNext, not derived loadNext
    LinearLayer* loadNext( FILE* fd );
    Vectorf apply( const Vectorf& x ) override;
    Matrixf apply(const Matrixf &x) override;
    virtual std::vector<int> shape(void) const override { return std::vector<int>({nRows, nCols}); }
};


class GRULayer : public TorchLayer{
    struct  Header{
        int elSize;  //size of each entry in bytes: 4 for float, 2 for fp16.
        int nHidden;
        int nInput;
    };

    Matrixf W_ir,W_iz,W_in;
    Matrixf W_hr,W_hz,W_hn;

//    Matrixf wi,wh;

    Vectorf b_ir,b_iz,b_in;
    Vectorf b_hr,b_hz,b_hn;

    Vectorf b_ihr,b_ihz;

    int nRows;
    int nCols;

public:
    GRULayer() = default;
    //call TorchLayer loadNext, not derived loadNext
    GRULayer* loadNext( FILE* fd );
    Vectorf apply( const Vectorf& x, const Vectorf& hx ) override;
    Matrixf apply( const Matrixf& x, const Matrixf& hx ) override;
    virtual std::vector<int> shape(void) const override { return std::vector<int>({nRows, nCols}); }


};

class Stretch2dLayer : public TorchLayer{
    struct  Header{
        int x_scale;
        int y_scale;
    };

    int x_scale;
    int y_scale;

public:
    Stretch2dLayer() = default;
    //call TorchLayer loadNext, not derived loadNext
    Stretch2dLayer* loadNext( FILE* fd );
    Matrixf apply(const Matrixf &x ) override;
    virtual std::vector<int> shape(void) const override { return std::vector<int>({0}); }
};


#endif // WAVERNN_H
