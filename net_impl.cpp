/* Copyright 2019 Eugene Ingerman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <string.h>
#include <stdio.h>
#include <vector>
#include <random>
#include <numeric>
#include <cmath>
#include <iostream>

#include "wavernn.h"
#include "net_impl.h"
#include <time.h>
#include <chrono>
#include "omp.h"
#include <pthread.h>
#include <fstream>

using namespace std;


__m256 exp256_ps(__m256 x) {

    __m256   exp_hi        = _mm256_set1_ps(88.3762626647949f);
    __m256   exp_lo        = _mm256_set1_ps(-88.3762626647949f);

    __m256   cephes_LOG2EF = _mm256_set1_ps(1.44269504088896341);
    __m256   cephes_exp_C1 = _mm256_set1_ps(0.693359375);
    __m256   cephes_exp_C2 = _mm256_set1_ps(-2.12194440e-4);

    __m256   cephes_exp_p0 = _mm256_set1_ps(1.9875691500E-4);
    __m256   cephes_exp_p1 = _mm256_set1_ps(1.3981999507E-3);
    __m256   cephes_exp_p2 = _mm256_set1_ps(8.3334519073E-3);
    __m256   cephes_exp_p3 = _mm256_set1_ps(4.1665795894E-2);
    __m256   cephes_exp_p4 = _mm256_set1_ps(1.6666665459E-1);
    __m256   cephes_exp_p5 = _mm256_set1_ps(5.0000001201E-1);
    __m256   tmp           = _mm256_setzero_ps(), fx;
    __m256i  imm0;
    __m256   one           = _mm256_set1_ps(1.0f);

    x     = _mm256_min_ps(x, exp_hi);
    x     = _mm256_max_ps(x, exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    fx    = _mm256_mul_ps(x, cephes_LOG2EF);
    fx    = _mm256_add_ps(fx, _mm256_set1_ps(0.5f));
    tmp   = _mm256_floor_ps(fx);
    __m256  mask  = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
    mask  = _mm256_and_ps(mask, one);
    fx    = _mm256_sub_ps(tmp, mask);
    tmp   = _mm256_mul_ps(fx, cephes_exp_C1);
    __m256  z     = _mm256_mul_ps(fx, cephes_exp_C2);
    x     = _mm256_sub_ps(x, tmp);
    x     = _mm256_sub_ps(x, z);
    z     = _mm256_mul_ps(x,x);

    __m256  y     = cephes_exp_p0;
    y     = _mm256_mul_ps(y, x);
    y     = _mm256_add_ps(y, cephes_exp_p1);
    y     = _mm256_mul_ps(y, x);
    y     = _mm256_add_ps(y, cephes_exp_p2);
    y     = _mm256_mul_ps(y, x);
    y     = _mm256_add_ps(y, cephes_exp_p3);
    y     = _mm256_mul_ps(y, x);
    y     = _mm256_add_ps(y, cephes_exp_p4);
    y     = _mm256_mul_ps(y, x);
    y     = _mm256_add_ps(y, cephes_exp_p5);
    y     = _mm256_mul_ps(y, z);
    y     = _mm256_add_ps(y, x);
    y     = _mm256_add_ps(y, one);

    /* build 2^n */
    imm0  = _mm256_cvttps_epi32(fx);
    imm0  = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
    imm0  = _mm256_slli_epi32(imm0, 23);
    __m256  pow2n = _mm256_castsi256_ps(imm0);
    y     = _mm256_mul_ps(y, pow2n);
    return y;
}


Vectorf exp_fast_avx(const Vectorf &v)
{
    int v_cols = v.cols();
    Vectorf y(v_cols);

    int j = 0,limit = v_cols - 8 + 1;
    for(j = 0;j<limit; j += 8)
    {
//        __m256 x = _mm256_setr_ps(v(j), v(j+1), v(j+2) ,v(j+3),v(j+4), v(j+5), v(j+6), v(j+7));
        __m256 x = _mm256_loadu_ps(&v[j]);
        __m256 y_256 = exp256_ps(x);
        _mm256_store_ps(&y(j),y_256);
    }

    float y1;
    for (; j < v_cols; j++)
    {
        y1 = std::exp(v(j));
        y(j) = y1;
    }

    return y;
}

Vectorf softmax( const Vectorf& x )
{
    float maxVal = x.maxCoeff();
    Vectorf y = x.array()-maxVal;
//    y = Eigen::exp(y.array());
    y = exp_fast_avx(y.array());

    float sum = y.sum();
    return y.array() / sum;
}

inline Vectorf exp_fast(const Vectorf &x1)
{
    Vectorf x = 1.f + x1.array()/(1024.f) ;
    for(int i=0;i<10;++i)
    {
        x.noalias() = x.cwiseProduct(x);
    }
    return x;
}


//inline float test(float x) {
//    x = 1.0 + x / 1024;
//    x *= x; x *= x; x *= x; x *= x;
//    x *= x; x *= x; x *= x; x *= x;
//    x *= x; x *= x;
//    return x;
//}


Matrixf softmax1( const Matrixf& x )
{
    int b_size = x.rows();
    Vectorf tmp;
    Matrixf softmax_result(b_size,x.cols());
    for(int i=0;i<b_size;++i) {
        float maxVal = x.row(i).maxCoeff();
        Vectorf y = x.row(i).array() - maxVal;
//        y = Eigen::exp(y.array());
//        y = exp_fast(y);
        y.noalias() = exp_fast_avx(y);
        float sum = y.sum();
        tmp = y.array() / sum;
        softmax_result.block(i,0,1,x.cols()) = tmp;
    }

    return softmax_result;
}


void ResBlock::loadNext(FILE *fd)
{
    resblock.resize( RES_BLOCKS*4 );
    for(int i=0; i<RES_BLOCKS*4; ++i){
        resblock[i].loadNext(fd);
    }
}

Matrixf ResBlock::apply(const Matrixf &x)
{
    Matrixf y = x;

    for(int i=0; i<RES_BLOCKS; ++i){
        Matrixf residual = y;

        y = resblock[4*i](y);    //conv1
        y = resblock[4*i+1](y);  //batch_norm1
        y = relu(y);
        y = resblock[4*i+2](y);  //conv2
        y = resblock[4*i+3](y);  //batch_norm2

        y += residual;
    }
    return y;
}

void Resnet::loadNext(FILE *fd)
{
    conv_in.loadNext(fd);
    batch_norm.loadNext(fd);
    resblock.loadNext(fd); //load the full stack
    conv_out.loadNext(fd);
    stretch2d.loadNext(fd);
}

Matrixf Resnet::apply(const Matrixf &x)
{
    Matrixf y = x;
    y=conv_in(y);
    y=batch_norm(y);
    y=relu(y);
    y=resblock.apply(y);
    y=conv_out(y);
    y=stretch2d(y);
    return y;
}

void UpsampleNetwork::loadNext(FILE *fd)
{
    up_layers.resize( UPSAMPLE_LAYERS*2 );
    for(int i=0; i<up_layers.size(); ++i){
        up_layers[i].loadNext(fd);
    }
}

Matrixf UpsampleNetwork::apply(const Matrixf &x)
{
    Matrixf y = x;
    for(int i=0; i<up_layers.size(); ++i){
        y = up_layers[i].apply( y );
    }
    return y;
}


void Model::loadNext(FILE *fd)
{
    fread( &header, sizeof( Model::Header ), 1, fd);

    resnet.loadNext(fd);
    upsample.loadNext(fd);

    I.loadNext(fd);

    rnn1.loadNext(fd);
    rnn2.loadNext(fd);

    fc1.loadNext(fd);
    fc2.loadNext(fd);
//    fc3.loadNext(fd);

    O1.loadNext(fd);
    O2.loadNext(fd);
    O3.loadNext(fd);
    O4.loadNext(fd);

    std::cout << "load model done:" << std::endl;
}


Matrixf pad( const Matrixf& x, int nPad )
{
    Matrixf y = Matrixf::Zero(x.rows(), x.cols()+2*nPad);
    y.block(0, nPad, x.rows(), x.cols()) = x;
    return y;
}

Vectorf vstack( const Vectorf& x1, const Vectorf& x2 )
{
    Vectorf temp(x1.size()+x2.size());
    temp << x1, x2;
    return temp;
}

Vectorf vstack( const Vectorf& x1, const Vectorf& x2, const Vectorf& x3 )
{
    return vstack( vstack( x1, x2), x3 );
}

inline float sampleCategorical( const VectorXf& probabilities )
{
    //Sampling using this algorithm https://en.wikipedia.org/wiki/Categorical_distribution#Sampling
    static std::ranlux24 rnd;
    std::vector<float> cdf(probabilities.size());
    float uniform_random = static_cast<float>(rnd()) / rnd.max();

    std::partial_sum(probabilities.data(), probabilities.data()+probabilities.size(), cdf.begin());
    auto it = std::find_if(cdf.cbegin(), cdf.cend(), [uniform_random](float x){ return (x >= uniform_random);});
    int pos = std::distance(cdf.cbegin(), it);
    return pos;
}

inline float invMulawQuantize( float x_mu )
{
//    const float mu = MULAW_QUANTIZE_CHANNELS - 1;
    float x = (x_mu / mu) * 2.f - 1.f;
    x = std::copysign(1.f, x) * (std::exp(std::fabs(x) * std::log1p(mu) ) - 1.f) / mu;
    return x;
}

template <typename T>
inline std::vector<T> linspace(T a, T b, size_t N) {
    T h = (b - a) / static_cast<T>(N-1);
    std::vector<T> xs(N);
    typename std::vector<T>::iterator x;
    T val;
    for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
        *x = val;
    return xs;
}

//def decode_mu_law(y, mu, from_labels=True):
//    # TODO: get rid of log2 - makes no sense
//    if from_labels:
//        y = label_2_float(y, math.log2(mu))
//    mu = mu - 1
//    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
//    return x

inline float decode_mu_law( float x_mu )
{
//    const float mu = MULAW_QUANTIZE_CHANNELS - 1;
    float x = std::copysign(1.f, x_mu) * (std::pow(1.f + mu ,std::fabs(x_mu)) - 1.f) / mu;

    return x;
}


Matrixf pad_after( const Matrixf& x, int nPad )
{
    Matrixf y = Matrixf::Zero(x.rows(), x.cols()+nPad);
    y.block(0, 0, x.rows(), x.cols()) = x;
    return y;
}


std::vector<Matrixf> fold_with_overlap(Matrixf mels_,int target, int overlap)
{

    int seq = target + 2 * overlap;
    int n_mel = mels_.rows();

    std::vector<Matrixf> folded;
    int total_len = mels_.cols();
    int num_folds = (total_len - overlap) / (target + overlap);
    int extended_len = num_folds * (overlap + target) + overlap;
    int remaining = total_len - extended_len;
    int padding=0;

    if(remaining != 0)
    {
        num_folds += 1;
        padding = target + 2 * overlap - remaining;
        mels_ = pad_after(mels_, padding);
    }
    Matrixf mels_trans = mels_.transpose();

    // 80*59200 ---> 14*4400*80 ---> 4400*14*80
    int start = 0,end = 0;
    Matrixf tmp(num_folds,n_mel);
    for(int j = 0;j<seq;++j)
    {
        for(int i=0;i<num_folds;++i)
        {
            start = i * (target + overlap);
            end = start + target + 2 * overlap;
            tmp.block(i,0,1,n_mel) = (mels_trans.block(start,0,(end-start),n_mel)).row(j);
        }
        folded.push_back(tmp);
    }

    return folded;
}


Vectorf xfade_and_unfold(Matrixf y,int target, int overlap)
{
    int num_folds= y.rows();
    int length = y.cols();
    target = length - 2 * overlap;
    int total_len = num_folds * (target + overlap) + overlap;
    int silence_len = overlap / 2;
    int fade_len = overlap - silence_len;

    Vectorf unfolded = Vectorf::Zero(total_len);
    Vectorf silence = Vectorf::Zero(silence_len);
    Vectorf linear = Vectorf::Ones(silence_len);

    Vectorf t = Vectorf::LinSpaced(fade_len,-1.0,1.0);
    Vectorf fade_in = Eigen::sqrt(0.5*(1.0 + t.array()));
    Vectorf fade_out = Eigen::sqrt(0.5*(1.0 - t.array()));

    fade_in = vstack( silence, fade_in);
    fade_out = vstack( linear, fade_out);

    int start = 0,end = 0;
    for(int i=0;i<num_folds;++i)
    {
        y.row(i).head(overlap) = y.row(i).head(overlap).array() * fade_in.array();
        y.row(i).tail(overlap) = y.row(i).tail(overlap).array() * fade_out.array();

        start = i * (target + overlap);
        end = start + target + 2 * overlap;
        unfolded.block(0,start,1,end-start) += y.row(i);
    }

    return unfolded;
}


Matrixf concat_col( const Matrixf& x1, const Matrixf& x2 )
{
    int bsize = x1.rows();
    Matrixf temp(bsize,x1.cols()+x2.cols());
    temp.block(0,0,bsize,x1.cols()) = x1;
    temp.block(0,x1.cols(),bsize,x2.cols()) = x2;
    return temp;
}


std::vector<Matrixf> matrixf_trans(std::vector<Matrixf> input_mat,int seq_size,int batch_size,int sub_bands)
{
    // 4400*3*4  ---> 4*3*4400
    // seq*b*sub ---> sub*b*seq

    std::vector<Matrixf> trans;
    Matrixf tmp(seq_size,batch_size);
    int i,j;

    for(i=0;i<sub_bands;i++)
    {
        for(j=0;j<seq_size;j++)
        {
            tmp.block(j,0,1,batch_size) = (input_mat[j].transpose()).row(i);
        }
        trans.push_back(tmp.transpose());
    }

    return trans;
}



////mb + no_batch
//Vectorf Model::apply(const Matrixf &mels_in)
//{
//    //
//    std::chrono::steady_clock::time_point  upsample1,upsample2,resnet1,resnet2,total1,total2;
//    std::chrono::duration<double> upsample_span,resnet_span,total_span;
//    double total_span_count;
//    total1 = std::chrono::steady_clock::now();
//
//    //
//    Matrixf mel_padded = pad(mels_in, header.nPad);
//
//    upsample1 = std::chrono::steady_clock::now();
//    Matrixf mels = upsample.apply(mel_padded);
//    upsample2 = std::chrono::steady_clock::now();
//    upsample_span = std::chrono::duration_cast<std::chrono::duration<double>>(upsample2 - upsample1);
//    std::cout << "upsample time: " << upsample_span.count() << " seconds."<<std::endl;
//
//    int indent = header.nPad * header.total_scale;
//
//    mels = mels.block(0,indent, mels.rows(), mels.cols()-2*indent ).eval(); //remove padding added in the previous step
//
//    resnet1 = std::chrono::steady_clock::now();
//    Matrixf aux = resnet.apply(mel_padded);
//    resnet2 = std::chrono::steady_clock::now();
//    resnet_span = std::chrono::duration_cast<std::chrono::duration<double>>(resnet2 - resnet1);
//    std::cout << "resnet time: " << resnet_span.count() << " seconds."<<std::endl;
//
//    assert(mels.cols() == aux.cols());
//    int seq_len = mels.cols();
//
//    Matrixf a1 = aux.block(0,          0, aux_dims, aux.cols());
//    Matrixf a2 = aux.block(aux_dims*1, 0, aux_dims, aux.cols());
//    Matrixf a3 = aux.block(aux_dims*2, 0, aux_dims, aux.cols());
//    Matrixf a4 = aux.block(aux_dims*3, 0, aux_dims, aux.cols());
//
////    Vectorf wav_out(seq_len);     //output vector
//    Matrixf wav_out = Matrixf::Zero(4,seq_len); //output vector
//    Vectorf x = Vectorf::Zero(4); //current sound amplitude
//
//    Vectorf h1 = Vectorf::Zero(rnn_dims);
//    Vectorf h2 = Vectorf::Zero(rnn_dims);
//
//    std::cout << "seq_len:" <<seq_len << std::endl;
//
//    //
//    std::chrono::steady_clock::time_point  i1,i2,rnn11,rnn12,rnn21,rnn22,fc11,fc12,fc21,fc22,mb1,mb2;
//    std::chrono::steady_clock::time_point  soft1,soft2,sam1,sam2,deco1,deco2,post1,post2;
//    std::chrono::duration<double> i_span,rnn1_span,rnn2_span,fc1_span,fc2_span,mb_span,soft_span,sam_span,deco_span,post_span;
//    double i_span_count,rnn1_span_count,rnn2_span_count,fc1_span_count,fc2_span_count,mb_span_count,soft_span_count,sam_span_count,deco_span_count,post_span_count;
//    //
//
//
//    for(int i=0; i<seq_len; ++i)
//    {
//        Vectorf y = vstack( x, mels.col(i), a1.col(i) );
//
//        i1 = std::chrono::steady_clock::now();
//        y = I( y );
//        i2 = std::chrono::steady_clock::now();
//        i_span = std::chrono::duration_cast<std::chrono::duration<double>>(i2 - i1);
//        i_span_count +=i_span.count();
//
//
//        rnn11 = std::chrono::steady_clock::now();
//        h1 = rnn1( y, h1 );
//        rnn12 = std::chrono::steady_clock::now();
//        rnn1_span = std::chrono::duration_cast<std::chrono::duration<double>>(rnn12 - rnn11);
//        rnn1_span_count +=rnn1_span.count();
//
//        y += h1;
//        rnn21 = std::chrono::steady_clock::now();
//        h2 = rnn2(vstack( y, a2.col(i)), h2 );
//        rnn22 = std::chrono::steady_clock::now();
//        rnn2_span = std::chrono::duration_cast<std::chrono::duration<double>>(rnn22 - rnn21);
//        rnn2_span_count +=rnn2_span.count();
//
//        y += h2;
//        y = vstack( y, a3.col(i) );
//        fc11 = std::chrono::steady_clock::now();
//        y = relu( fc1( y ) );
//        fc12 = std::chrono::steady_clock::now();
//        fc1_span = std::chrono::duration_cast<std::chrono::duration<double>>(fc12 - fc11);
//        fc1_span_count +=fc1_span.count();
//
//        y = vstack( y, a4.col(i) );
//        fc21 = std::chrono::steady_clock::now();
//        y = relu( fc2( y ) );
//        fc22 = std::chrono::steady_clock::now();
//        fc2_span = std::chrono::duration_cast<std::chrono::duration<double>>(fc22 - fc21);
//        fc2_span_count +=fc2_span.count();
//
//        mb1 = std::chrono::steady_clock::now();
//        Vectorf y_mb1 = O1( y );
//        Vectorf y_mb2 = O2( y );
//        Vectorf y_mb3 = O3( y );
//        Vectorf y_mb4 = O4( y );
//        mb2 = std::chrono::steady_clock::now();
//        mb_span = std::chrono::duration_cast<std::chrono::duration<double>>(mb2 - mb1);
//        mb_span_count +=mb_span.count();
//
//
//        soft1 = std::chrono::steady_clock::now();
//        Vectorf posterior1 = softmax( y_mb1 );
//        Vectorf posterior2 = softmax( y_mb2 );
//        Vectorf posterior3 = softmax( y_mb3 );
//        Vectorf posterior4 = softmax( y_mb4 );
//
//
//        soft2 = std::chrono::steady_clock::now();
//        soft_span = std::chrono::duration_cast<std::chrono::duration<double>>(soft2 - soft1);
//        soft_span_count +=soft_span.count();
//
//        sam1 = std::chrono::steady_clock::now();
//        float newAmplitude1 = sampleCategorical( posterior1 );
//        float newAmplitude2 = sampleCategorical( posterior2 );
//        float newAmplitude3 = sampleCategorical( posterior3 );
//        float newAmplitude4 = sampleCategorical( posterior4 );
//        newAmplitude1 = (2.*newAmplitude1) / posterior_size - 1.; //for bits output
//        newAmplitude2 = (2.*newAmplitude2) / posterior_size - 1.; //for bits output
//        newAmplitude3 = (2.*newAmplitude3) / posterior_size - 1.; //for bits output
//        newAmplitude4 = (2.*newAmplitude4) / posterior_size - 1.; //for bits output
//        sam2 = std::chrono::steady_clock::now();
//        sam_span = std::chrono::duration_cast<std::chrono::duration<double>>(sam2 - sam1);
//        sam_span_count +=sam_span.count();
//
//
//        deco1 = std::chrono::steady_clock::now();
//        float newAmplitude_de1 = decode_mu_law( newAmplitude1 );
//        float newAmplitude_de2 = decode_mu_law( newAmplitude2 );
//        float newAmplitude_de3 = decode_mu_law( newAmplitude3 );
//        float newAmplitude_de4 = decode_mu_law( newAmplitude4 );
//        deco2 = std::chrono::steady_clock::now();
//        deco_span = std::chrono::duration_cast<std::chrono::duration<double>>(deco2 - deco1);
//        deco_span_count +=deco_span.count();
//
//        x(0) = newAmplitude1;
//        x(1) = newAmplitude2;
//        x(2) = newAmplitude3;
//        x(3) = newAmplitude4;
//
//        wav_out(0,i) = newAmplitude_de1;
//        wav_out(1,i) = newAmplitude_de2;
//        wav_out(2,i) = newAmplitude_de3;
//        wav_out(3,i) = newAmplitude_de4;

//    }
//
//
//
//    //mb
//    post1 = std::chrono::steady_clock::now();
//
//    int conv_trans_ow = subbands * (seq_len - 1) -2*conv_trans_pad + conv_trans_ksize;
//    int conv_trans_pad_ow = conv_trans_ow + taps * 2;
//    float *conv_transpose1d=new float[subbands*conv_trans_pad_ow];
//    memset(conv_transpose1d, 0, sizeof(float)*subbands*conv_trans_pad_ow);
//    int conv1d_ow = int((conv_trans_ow + taps * 2 - conv1d_ksize + 2*conv1d_pad)/conv1d_stride)+1;
//    Vectorf wav_mb_out = Vectorf::Zero(conv1d_ow);
//
////    x1 = F.conv_transpose1d(x, self.updown_filter * self.subbands, stride=self.subbands)
////    x2 = self.pad_fn(x1)
//    int i=0,j=0;
//    for(i=0;i<seq_len;i++)
//    {
//        conv_transpose1d[0 * conv_trans_pad_ow + i * 4 + taps] = wav_out(0,i) * 4.f;
//        conv_transpose1d[1 * conv_trans_pad_ow + i * 4 + taps] = wav_out(1,i) * 4.f;
//        conv_transpose1d[2 * conv_trans_pad_ow + i * 4 + taps] = wav_out(2,i) * 4.f;
//        conv_transpose1d[3 * conv_trans_pad_ow + i * 4 + taps] = wav_out(3,i) * 4.f;
//    }
//
////    x3 = F.conv1d(x2, self.synthesis_filter)
//    for(i=0;i<conv1d_ow;i++)
//    {
//        float out0=0.f,out1=0.f,out2=0.f,out3=0.f;
//        for(j=0;j<conv1d_ksize;j++)
//        {
//            out0 += conv_transpose1d[0* conv_trans_pad_ow +i+j]*synthesis_filter[0*63+j];
//            out1 += conv_transpose1d[1* conv_trans_pad_ow +i+j]*synthesis_filter[1*63+j];
//            out2 += conv_transpose1d[2* conv_trans_pad_ow +i+j]*synthesis_filter[2*63+j];
//            out3 += conv_transpose1d[3* conv_trans_pad_ow +i+j]*synthesis_filter[3*63+j];
//        }
//        wav_mb_out(i) = out0+out1+out2+out3;
//    }
//
//    post2 = std::chrono::steady_clock::now();
//    post_span = std::chrono::duration_cast<std::chrono::duration<double>>(post2 - post1);
//    post_span_count =post_span.count();
//
//    total2 = std::chrono::steady_clock::now();
//    total_span = std::chrono::duration_cast<std::chrono::duration<double>>(total2 - total1);
//    total_span_count = total_span.count();
//
//
//    std::cout << "i time:" << i_span_count<<std::endl;
//    std::cout << "rnn1 time:" << rnn1_span_count<<std::endl;
//    std::cout << "rnn2 time:" << rnn2_span_count<<std::endl;
//    std::cout << "fc1 time:" << fc1_span_count<<std::endl;
//    std::cout << "fc2 time:" << fc2_span_count<<std::endl;
//    std::cout << "mb time:" << mb_span_count<<std::endl;
//    std::cout << "sofamax time:" << soft_span_count<<std::endl;
//    std::cout << "sampleCategorical time:" << sam_span_count<<std::endl;
//    std::cout << "decode_mu_law time:" << deco_span_count<<std::endl;
//    std::cout << "post time:" << post_span_count<<std::endl;
//    std::cout << "total_t time:" << total_span_count<<std::endl;
//
//    return wav_mb_out;
//}





//mb+batch
Vectorf Model::apply(const Matrixf &mels_in)
{

    //
    std::chrono::steady_clock::time_point  pad1,pad2,upsample1,upsample2,resnet1,resnet2,total1,total2;
    std::chrono::duration<double> pad_span,upsample_span,resnet_span,total_span;
    double pad_span_count=0,upsample_span_count=0,resnet_span_count=0,total_span_count=0 ;

    //
    std::chrono::steady_clock::time_point  post1,post2,fold1,fold2,batch_generate1,batch_generate2,unfold1,unfold2;
    std::chrono::duration<double> post_span,fold_span,batch_generate_span,unfold_span;
    double post_span_count=0,fold_span_count=0,batch_generate_span_count=0,unfold_span_count=0;
    //

    //
    std::chrono::steady_clock::time_point  i1,i2,rnn11,rnn12,rnn21,rnn22,fc11,fc12,fc21,fc22,mb1,mb2;
    std::chrono::steady_clock::time_point  soft1,soft2,sam1,sam2,deco1,deco2;
    std::chrono::duration<double> i_span,rnn1_span,rnn2_span,fc1_span,fc2_span,mb_span,soft_span,sam_span,deco_span;
    double i_span_count=0,rnn1_span_count=0,rnn2_span_count=0,fc1_span_count=0,fc2_span_count=0,
            mb_span_count=0,soft_span_count=0,sam_span_count=0,deco_span_count=0;
    //

    total1 = std::chrono::steady_clock::now();

    std::vector<Matrixf> mel_fold,aux_fold;
    std::vector<Matrixf> wav_out_vec;
    mel_fold.clear();
    aux_fold.clear();
    wav_out_vec.clear();


    Matrixf mel_padded = pad(mels_in, header.nPad);

    upsample1 = std::chrono::steady_clock::now();
    Matrixf mels = upsample.apply(mel_padded);

    upsample2 = std::chrono::steady_clock::now();
    upsample_span = std::chrono::duration_cast<std::chrono::duration<double>>(upsample2 - upsample1);
    upsample_span_count = upsample_span.count();
    std::cout << "upsample time: " << upsample_span_count << " seconds."<<std::endl;

    int indent = header.nPad * header.total_scale;

    mels = mels.block(0,indent, mels.rows(), mels.cols()-2*indent).eval(); //remove padding added in the previous step

    resnet1 = std::chrono::steady_clock::now();
    Matrixf aux = resnet.apply(mel_padded);

    resnet2 = std::chrono::steady_clock::now();
    resnet_span = std::chrono::duration_cast<std::chrono::duration<double>>(resnet2 - resnet1);
    resnet_span_count = resnet_span.count();
    std::cout << "resnet time: " << resnet_span_count << " seconds."<<std::endl;


    //batched
    fold1 = std::chrono::steady_clock::now();
    mel_fold = fold_with_overlap(mels, target, overlap);
    aux_fold = fold_with_overlap(aux, target, overlap);

    fold2 = std::chrono::steady_clock::now();
    fold_span = std::chrono::duration_cast<std::chrono::duration<double>>(fold2 - fold1);
    fold_span_count = fold_span.count();
    std::cout << "fold time: " << fold_span_count << " seconds."<<std::endl;
    //

    assert(mels.cols() == aux.cols());

    int seq_len = mel_fold.size();
    int b_size = mel_fold[0].rows();

    std::cout << "b_size:" <<b_size << std::endl;
    std::cout << "seq_len:" <<seq_len << std::endl;

    Matrixf wav_out = Matrixf::Zero(b_size,4);     //output vector
    Matrixf x = Matrixf::Zero(b_size,4);
    Matrixf m_t,a1_t,a2_t,a3_t,a4_t;
    Matrixf y;

    Matrixf h1 = Matrixf::Zero(b_size,rnn_dims);
    Matrixf h2 = Matrixf::Zero(b_size,rnn_dims);

    for (int i = 0; i < seq_len; ++i)
    {
        m_t = mel_fold[i];
        a1_t = aux_fold[i].block(0,aux_dims*0,b_size,aux_dims);
        a2_t = aux_fold[i].block(0,aux_dims*1,b_size,aux_dims);
        a3_t = aux_fold[i].block(0,aux_dims*2,b_size,aux_dims);
        a4_t = aux_fold[i].block(0,aux_dims*3,b_size,aux_dims);

        y = concat_col(x, concat_col(m_t, a1_t));

        i1 = std::chrono::steady_clock::now();
        y = I( y );
        i2 = std::chrono::steady_clock::now();
        i_span = std::chrono::duration_cast<std::chrono::duration<double>>(i2 - i1);
        i_span_count +=i_span.count();


        rnn11 = std::chrono::steady_clock::now();
        h1 = rnn1( y, h1 );
        rnn12 = std::chrono::steady_clock::now();
        rnn1_span = std::chrono::duration_cast<std::chrono::duration<double>>(rnn12 - rnn11);
        rnn1_span_count +=rnn1_span.count();


        y += h1;
        rnn21 = std::chrono::steady_clock::now();
        h2 = rnn2(concat_col(y, a2_t), h2);
        rnn22 = std::chrono::steady_clock::now();
        rnn2_span = std::chrono::duration_cast<std::chrono::duration<double>>(rnn22 - rnn21);
        rnn2_span_count +=rnn2_span.count();

        y += h2;
        y = concat_col(y, a3_t);

        fc11 = std::chrono::steady_clock::now();
        y = relu1( fc1( y ) );
        fc12 = std::chrono::steady_clock::now();
        fc1_span = std::chrono::duration_cast<std::chrono::duration<double>>(fc12 - fc11);
        fc1_span_count +=fc1_span.count();

        y = concat_col(y, a4_t);
        fc21 = std::chrono::steady_clock::now();
        y = relu1( fc2( y ) );
        fc22 = std::chrono::steady_clock::now();
        fc2_span = std::chrono::duration_cast<std::chrono::duration<double>>(fc22 - fc21);
        fc2_span_count +=fc2_span.count();

        mb1 = std::chrono::steady_clock::now();
        Matrixf y_mb1 = O1( y );
        Matrixf y_mb2 = O2( y );
        Matrixf y_mb3 = O3( y );
        Matrixf y_mb4 = O4( y );
        mb2 = std::chrono::steady_clock::now();
        mb_span = std::chrono::duration_cast<std::chrono::duration<double>>(mb2 - mb1);
        mb_span_count +=mb_span.count();

        soft1 = std::chrono::steady_clock::now();
        Matrixf posterior1 = softmax1( y_mb1 );
        Matrixf posterior2 = softmax1( y_mb2 );
        Matrixf posterior3 = softmax1( y_mb3 );
        Matrixf posterior4 = softmax1( y_mb4 );

        soft2 = std::chrono::steady_clock::now();
        soft_span = std::chrono::duration_cast<std::chrono::duration<double>>(soft2 - soft1);
        soft_span_count +=soft_span.count();

        for(int j=0;j<b_size;++j)
        {
            sam1 = std::chrono::steady_clock::now();
            float newAmplitude1 = sampleCategorical( posterior1.row(j) );
            float newAmplitude2 = sampleCategorical( posterior2.row(j) );
            float newAmplitude3 = sampleCategorical( posterior3.row(j) );
            float newAmplitude4 = sampleCategorical( posterior4.row(j) );
            newAmplitude1 = (2.*newAmplitude1) / posterior_size - 1.; //for bits output
            newAmplitude2 = (2.*newAmplitude2) / posterior_size - 1.; //for bits output
            newAmplitude3 = (2.*newAmplitude3) / posterior_size - 1.; //for bits output
            newAmplitude4 = (2.*newAmplitude4) / posterior_size - 1.; //for bits output

            sam2 = std::chrono::steady_clock::now();
            sam_span = std::chrono::duration_cast<std::chrono::duration<double>>(sam2 - sam1);
            sam_span_count +=sam_span.count();

            deco1 = std::chrono::steady_clock::now();
            float newAmplitude_de1 = decode_mu_law( newAmplitude1 );
            float newAmplitude_de2 = decode_mu_law( newAmplitude2 );
            float newAmplitude_de3 = decode_mu_law( newAmplitude3 );
            float newAmplitude_de4 = decode_mu_law( newAmplitude4 );

            deco2 = std::chrono::steady_clock::now();
            deco_span = std::chrono::duration_cast<std::chrono::duration<double>>(deco2 - deco1);
            deco_span_count +=deco_span.count();

            x(j,0) = newAmplitude1;
            x(j,1) = newAmplitude2;
            x(j,2) = newAmplitude3;
            x(j,3) = newAmplitude4;

            wav_out(j, 0) = newAmplitude_de1;
            wav_out(j, 1) = newAmplitude_de2;
            wav_out(j, 2) = newAmplitude_de3;
            wav_out(j, 3) = newAmplitude_de4;
        }
        wav_out_vec.push_back(wav_out);
    }


    post1 = std::chrono::steady_clock::now();

    std::vector<Matrixf> wav_out_trans_vec = matrixf_trans(wav_out_vec,seq_len,b_size,subbands);

    int wave_len = (mels_in.cols() - 1) * header.total_scale;
    int cut_len = 20 * header.total_scale;
    std::vector<double> fade_out = linspace(0.0, 1.0, cut_len);
    Matrixf wav_out_xfade = Matrixf::Zero(subbands,wave_len);

    for(int k=0;k<subbands;k++)
    {
        Vectorf trans = xfade_and_unfold(wav_out_trans_vec[k], target, overlap);
        for(int i = 0;i<wave_len;++i){
            if(i<(wave_len-cut_len)){
                wav_out_xfade(k,i) = trans(i);
            }
            else{
                wav_out_xfade(k,i) = trans(i) * fade_out[wave_len-i-1];
            }
        }
    }

    //mb
    int conv_trans_ow = subbands * (wave_len - 1) -2*conv_trans_pad + conv_trans_ksize;
    int conv_trans_pad_ow = conv_trans_ow + taps * 2;
    float *conv_transpose1d=new float[subbands*conv_trans_pad_ow];
    memset(conv_transpose1d, 0, sizeof(float)*subbands*conv_trans_pad_ow);
    int conv1d_ow = int((conv_trans_ow + taps * 2 - conv1d_ksize + 2*conv1d_pad)/conv1d_stride)+1;
    Vectorf wav_mb_out = Vectorf::Zero(conv1d_ow);

//    x1 = F.conv_transpose1d(x, self.updown_filter * self.subbands, stride=self.subbands)
//    x2 = self.pad_fn(x1)
    for(int i=0;i<wave_len;i++)
    {
        conv_transpose1d[0 * conv_trans_pad_ow + i * 4 + taps] = wav_out_xfade(0,i) * 4.f;
        conv_transpose1d[1 * conv_trans_pad_ow + i * 4 + taps] = wav_out_xfade(1,i) * 4.f;
        conv_transpose1d[2 * conv_trans_pad_ow + i * 4 + taps] = wav_out_xfade(2,i) * 4.f;
        conv_transpose1d[3 * conv_trans_pad_ow + i * 4 + taps] = wav_out_xfade(3,i) * 4.f;
    }

//    x3 = F.conv1d(x2, self.synthesis_filter)
    for(int i=0;i<conv1d_ow;i++)
    {
        float out0=0.f,out1=0.f,out2=0.f,out3=0.f;
        for(int j=0;j<conv1d_ksize;j++)
        {
            out0 += conv_transpose1d[0* conv_trans_pad_ow +i+j]*synthesis_filter[0*63+j];
            out1 += conv_transpose1d[1* conv_trans_pad_ow +i+j]*synthesis_filter[1*63+j];
            out2 += conv_transpose1d[2* conv_trans_pad_ow +i+j]*synthesis_filter[2*63+j];
            out3 += conv_transpose1d[3* conv_trans_pad_ow +i+j]*synthesis_filter[3*63+j];
        }
        wav_mb_out(i) = out0+out1+out2+out3;
    }

    delete []conv_transpose1d;


    post2 = std::chrono::steady_clock::now();
    post_span = std::chrono::duration_cast<std::chrono::duration<double>>(post2 - post1);
    post_span_count =post_span.count();

    total2 = std::chrono::steady_clock::now();
    total_span = std::chrono::duration_cast<std::chrono::duration<double>>(total2 - total1);
    total_span_count = total_span.count();


    std::cout << "i time:" << i_span_count<<std::endl;
    std::cout << "rnn1 time:" << rnn1_span_count<<std::endl;
    std::cout << "rnn2 time:" << rnn2_span_count<<std::endl;
    std::cout << "fc1 time:" << fc1_span_count<<std::endl;
    std::cout << "fc2 time:" << fc2_span_count<<std::endl;
    std::cout << "mb time:" << mb_span_count<<std::endl;
    std::cout << "sofamax time:" << soft_span_count<<std::endl;
    std::cout << "sampleCategorical time:" << sam_span_count<<std::endl;
    std::cout << "decode_mu_law time:" << deco_span_count<<std::endl;
    std::cout << "post time:" << post_span_count<<std::endl;
    std::cout << "total_t time:" << total_span_count<<std::endl;

    std::cout << "inference done:" << std::endl;
    return wav_mb_out;
}

