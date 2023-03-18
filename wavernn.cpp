/*
Copyright 2019 Eugene Ingerman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/


#include <stdio.h>
#include <iostream>
#include <cmath>
#include "wavernn.h"
#include "omp.h"
#include <pthread.h>
#include <xmmintrin.h>


Matrixf relu( const Matrixf& x)
{
    return x.array().max(0.f);
    //return x.unaryExpr([](float x){return std::max(0.f, x);});
}


Matrixf relu1( const Matrixf& x)
{
    return x.cwiseMax(0.f);
}

inline Vectorf sigmoid( const Vectorf& v )
{
    //TODO: optimize this
    //maybe use one of these approximations: https://stackoverflow.com/questions/10732027/fast-sigmoid-algorithm
    Vectorf y = 1.f / ( 1.f + Eigen::exp( - v.array()));

    return y;
}



inline Matrixf sigmoid1( const Matrixf& v )
{
    int b_size = v.rows();
    Matrixf sigmoid_result(b_size,v.cols()) ;

    //
    for(int i=0;i<b_size;++i) {
        Vectorf y = 1.f / ( 1.f + Eigen::exp( - v.row(i).array()));
        sigmoid_result.block(i,0,1,v.cols()) = y;
    }

    return sigmoid_result;
}

inline Vectorf tanh( const Vectorf& v )
{
    //TODO: optimize this
    Vectorf y = Eigen::tanh( v.array() );
    return y;
}

inline Matrixf tanh1( const Matrixf& v )
{
    int b_size = v.rows();
    Matrixf tanh_result(b_size,v.cols());

    for(int i=0;i<b_size;++i) {
        Vectorf y = Eigen::tanh(v.row(i).array());
        tanh_result.block(i,0,1,v.cols()) = y;
    }

    return tanh_result;
}




BaseLayer *TorchLayer::loadNext(FILE *fd)
{
    TorchLayer::Header header;
    fread(&header, sizeof(TorchLayer::Header), 1, fd);
    std::cerr << "Loading:" << header.name << std::endl;

    switch( header.layerType ){

    case TorchLayer::Header::LayerType::Linear:
    {
        impl = new LinearLayer();
        impl->loadNext(fd);
        return impl;
    }
    break;

    case TorchLayer::Header::LayerType::GRU:
    {
        impl = new GRULayer();
        impl->loadNext(fd);
        return impl;
    }
    break;

    case TorchLayer::Header::LayerType::Conv1d:
    {
        impl = new Conv1dLayer();
        impl->loadNext(fd);
        return impl;
    }
    case TorchLayer::Header::LayerType::Conv2d:{
        impl = new Conv2dLayer();
        impl->loadNext(fd);
        return impl;
    }
    case TorchLayer::Header::LayerType::BatchNorm1d:
    {
        impl = new BatchNorm1dLayer();
        impl->loadNext(fd);
        return impl;
    }
    case TorchLayer::Header::LayerType::Stretch2d:
    {
        impl = new Stretch2dLayer();
        impl->loadNext(fd);
        return impl;
    }

    default:
        std::cout << "nullptr:"  <<std::endl;
        return nullptr;
    }
}


LinearLayer* LinearLayer::loadNext(FILE *fd)
{
    LinearLayer::Header header;
    fread( &header, sizeof(LinearLayer::Header), 1, fd);
    assert(header.elSize==4 or header.elSize==2 or header.elSize==1);

    nRows = header.nRows;
    nCols = header.nCols;

    mat.resize(nRows,nCols);
    fread(mat.data(), header.elSize, nRows*nCols, fd);

    bias.resize(header.nRows);
    fread(bias.data(), header.elSize, header.nRows, fd);
    return this;
}

Matrixf LinearLayer::apply(const Matrixf &x)
{

    return (mat*x.transpose()).transpose().rowwise()+bias;
//    return (x*mat.transpose()).rowwise()+bias;
}

Vectorf LinearLayer::apply(const Vectorf &x)
{

    return (mat*x.transpose()).transpose().rowwise()+bias;
//    return (x*mat.transpose()).rowwise()+bias;
}



GRULayer* GRULayer::loadNext(FILE *fd)
{

    GRULayer::Header header;
    fread( &header, sizeof(GRULayer::Header), 1, fd);
    assert(header.elSize==4 or header.elSize==2 or header.elSize==1);

    nRows = header.nHidden;
    nCols = header.nInput;

    W_ir.resize(nRows,nCols);
    W_iz.resize(nRows,nCols);
    W_in.resize(nRows,nCols);

    W_hr.resize(nRows,nRows);
    W_hz.resize(nRows,nRows);
    W_hn.resize(nRows,nRows);

    b_ir.resize(header.nHidden);
    b_iz.resize(header.nHidden);
    b_in.resize(header.nHidden);

    b_hr.resize(header.nHidden);
    b_hz.resize(header.nHidden);
    b_hn.resize(header.nHidden);


    fread(W_ir.data(), header.elSize, nRows*nCols, fd);
    fread(W_iz.data(), header.elSize, nRows*nCols, fd);
    fread(W_in.data(), header.elSize, nRows*nCols, fd);

    fread(W_hr.data(), header.elSize, nRows*nRows, fd);
    fread(W_hz.data(), header.elSize, nRows*nRows, fd);
    fread(W_hn.data(), header.elSize, nRows*nRows, fd);


    fread(b_ir.data(), header.elSize, header.nHidden, fd);
    fread(b_iz.data(), header.elSize, header.nHidden, fd);
    fread(b_in.data(), header.elSize, header.nHidden, fd);

    fread(b_hr.data(), header.elSize, header.nHidden, fd);
    fread(b_hz.data(), header.elSize, header.nHidden, fd);
    fread(b_hn.data(), header.elSize, header.nHidden, fd);

//    wi = concat_row(concat_row(W_ir,W_iz),W_in); //合并w
//    wh = concat_row(concat_row(W_hr,W_hz),W_hn);

    b_ihr = b_ir + b_hr;
    b_ihz = b_iz + b_hz;

    return this;
}

Matrixf concat_row( const Matrixf& x1, const Matrixf& x2 )
{
    assert(x1.cols() == x2.cols());
    int cols = x1.cols();
    Matrixf temp = Matrixf::Zero(x1.rows()+x2.rows(),cols);
    temp.block(0,0,x1.rows(),cols) = x1;
    temp.block(x1.rows(),0,x2.rows(),cols) = x2;
    return temp;
}



Vectorf GRULayer::apply(const Vectorf &x, const Vectorf &hx)
{
    Vectorf r, z, n, hout;

    r = sigmoid((W_ir*x.transpose() + W_hr*hx.transpose()).transpose().rowwise() + b_ihr);
    z = sigmoid((W_iz*x.transpose() + W_hz*hx.transpose()).transpose().rowwise() + b_ihz);
    n = tanh(((W_in*x.transpose()).transpose().rowwise() + b_in) +
            (((W_hn*hx.transpose()).transpose().rowwise() + b_hn)).cwiseProduct(r));

    hout = (1.f - z.array()).matrix().cwiseProduct(n) + z.cwiseProduct(hx);

    return hout;
}


//不合并 w * x
Matrixf GRULayer::apply(const Matrixf &x, const Matrixf &hx)
{
    Matrixf r, z, n, hout;

    r.noalias() = sigmoid1((W_ir*x.transpose() + W_hr*hx.transpose()).transpose().rowwise() + b_ihr);
    z.noalias() = sigmoid1((W_iz*x.transpose() + W_hz*hx.transpose()).transpose().rowwise() + b_ihz);
    n.noalias() = tanh1(((W_in*x.transpose()).transpose().rowwise() + b_in) +
            (((W_hn*hx.transpose()).transpose().rowwise() + b_hn)).cwiseProduct(r));

    hout.noalias() = (1.f - z.array()).matrix().cwiseProduct(n) + z.cwiseProduct(hx);

    return hout;
}


//合并 w * x
//Matrixf GRULayer::apply(const Matrixf &x, const Matrixf &hx)
//{
//
//    Matrixf r, z, n, hout;
//    Matrixf w_ix,w_hx,w_ihx;
//
//    w_ix = wi * x.transpose();  //合并 w * x
//    w_hx = wh * hx.transpose();
//    w_ihx = w_ix + w_hx;
//
//    int w_i_rows = w_ix.rows()/3;
//    int w_i_cols = w_ix.cols();
//
//    r.noalias() = sigmoid1(w_ihx.block(0,0,w_i_rows,w_i_cols).transpose().rowwise() + b_ihr);
//    z.noalias() = sigmoid1(w_ihx.block(w_i_rows,0,w_i_rows,w_i_cols).transpose().rowwise() + b_ihz);
//    n.noalias() = tanh1((w_ix.block(w_i_rows*2,0,w_i_rows,w_i_cols).transpose().rowwise() + b_in) +
//            ((w_hx.block(w_i_rows*2,0,w_i_rows,w_i_cols).transpose().rowwise() + b_hn)).cwiseProduct(r));
//
//    hout.noalias() = (1.f - z.array()).matrix().cwiseProduct(n) + z.cwiseProduct(hx);
//
//    return hout;
//}



Conv1dLayer *Conv1dLayer::loadNext(FILE *fd)
{
    Conv1dLayer::Header header;
    fread( &header, sizeof(Conv1dLayer::Header), 1, fd);

    assert(header.elSize==4 or header.elSize==2  or header.elSize==1);

    hasBias = header.useBias;
    inChannels = header.inChannels;
    outChannels = header.outChannels;
    nKernel = header.kernelSize;


    if( nKernel==1 ){
        //if kernel is 1x then convolution is just matrix multiplication. Load weight into the first element
        //and handle separately
        weight.resize(1);
        weight[0].resize(inChannels, outChannels);
        fread(weight[0].data(), header.elSize, inChannels*outChannels*nKernel, fd);
    } else {
        weight.resize(outChannels);
        for(int i=0; i<outChannels; ++i){
            weight[i].resize(inChannels, nKernel);
            fread(weight[i].data(), header.elSize, inChannels*nKernel, fd);
        }
    }

    if( hasBias ){
        bias.resize(outChannels);
        fread(bias.data(), header.elSize, outChannels, fd);
    }


    return this;
}

Matrixf Conv1dLayer::apply(const Matrixf &x)
{
    int convDim = x.cols()-nKernel+1;
    Matrixf y(outChannels, convDim);

    if( nKernel == 1 ){
        //fast path for x1 convolution kernel
        y = weight[0] * x;
    } else {
        omp_set_num_threads(2);
        #pragma omp parallel for
        for(int outIdx = 0; outIdx<outChannels; ++outIdx){
            for(int kernIdx = 0; kernIdx < convDim; ++kernIdx ){
                y(outIdx, kernIdx) = ( x.block(0, kernIdx, inChannels, nKernel).cwiseProduct( weight[outIdx] ) ).sum();
            }
        }
    }

    if( hasBias ){
        //add bias to every column
        y.colwise() += bias.transpose();
    }

    return y;
}

Conv2dLayer *Conv2dLayer::loadNext(FILE *fd)
{
    Conv2dLayer::Header header;
    fread( &header, sizeof(Conv2dLayer::Header), 1, fd);
    assert(header.elSize==4 or header.elSize==2  or header.elSize==1);

    nKernel = header.nKernel;

    weight.resize(nKernel);
    fread(weight.data(), header.elSize, nKernel, fd);
    return this;
}

Matrixf Conv2dLayer::apply(const Matrixf &x)
{

    Matrixf y(x.rows(), x.cols());

    int nKernel = weight.size();
    int npad = (nKernel-1)/2;

    //TODO: possibly optimize
    omp_set_num_threads(2);
    #pragma omp parallel for
    for(int i=0; i<x.rows(); ++i){
        Vectorf temp = Vectorf::Zero(x.cols()+2*npad);
        temp.block(0, npad, 1, x.cols()) = x.block(i, 0, 1, x.cols());

        for(int j=0; j<x.cols(); ++j){
            y(i,j) = ( temp.block(0, j, 1, nKernel).array() * weight.array() ).sum();
        }
    }

    return y;
}

BatchNorm1dLayer *BatchNorm1dLayer::loadNext(FILE *fd)
{
    BatchNorm1dLayer::Header header;
    fread( &header, sizeof( BatchNorm1dLayer::Header), 1, fd);
    assert(header.elSize==4 or header.elSize==2  or header.elSize==1);

    eps = header.eps;
    nChannels = header.inChannels;

    weight.resize( header.inChannels );
    bias.resize( header.inChannels );
    running_mean.resize( header.inChannels );
    running_var.resize( header.inChannels );

    fread(weight.data(), header.elSize, header.inChannels, fd);
    fread(bias.data(), header.elSize, header.inChannels, fd);
    fread(running_mean.data(), header.elSize, header.inChannels, fd);
    fread(running_var.data(), header.elSize, header.inChannels, fd);

    return this;
}

Matrixf BatchNorm1dLayer::apply(const Matrixf &x)
{
    Matrixf y(x.rows(), x.cols());

    //y = ((x1[:,0]-running_mean)/(np.sqrt(running_var+eps)))*weight+bias

    Vectorf invstd = Eigen::rsqrt(running_var.array() + eps);
    Matrixf r1 = (x.colwise() - running_mean.transpose());
    y = ((r1.array().colwise()*invstd.transpose().array()).colwise()*weight.transpose().array()).colwise() + bias.transpose().array();
    return y;
}

Stretch2dLayer *Stretch2dLayer::loadNext(FILE *fd)
{
    Stretch2dLayer::Header header;
    fread( &header, sizeof(Stretch2dLayer::Header), 1, fd);

    x_scale = header.x_scale;
    y_scale = header.y_scale;
}

Matrixf Stretch2dLayer::apply(const Matrixf &x)
{
    Matrixf y(x.rows()*y_scale, x.cols()*x_scale);

    assert(y_scale==1); //TODO: implement for 2d scaling

//    int scaled_col = 0;
//    for(int i=0; i<x.cols(); ++i){
//        for(int j=0; j<x_scale; ++j){
//            y.col(scaled_col++) = x.col(i);
//        }
//    }

    omp_set_num_threads(2);
    #pragma omp parallel for
    for (int i = 0; i < x.cols(); ++i) {
        y.middleCols(x_scale * i, x_scale) = x.col(i).rowwise().replicate(x_scale);
    }

    return y;
}
