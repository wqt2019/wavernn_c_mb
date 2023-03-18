/*
Copyright 2019 Eugene Ingerman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/


#include <stdio.h>
#include <string>
#include <iostream>
#include "cxxopts.hpp"

#include <fstream>

#include "vocoder.h"
#include "net_impl.h"
#include "wavernn.h"
#include <time.h>
#include <chrono>

using namespace std;


Matrixf loadMel( FILE *fd )
{

    struct Header{
        int nRows, nCols;
    } header;
    fread( &header, sizeof( Header ), 1, fd);

    Matrixf mel( header.nRows, header.nCols );
    fread(mel.data(), sizeof(float), header.nRows*header.nCols, fd);

    return mel;
}


Matrixf ReadArrayXXf(const char *filename) {
    int nrows = 80;
    ifstream fin(filename);
    std::vector<float> points;
    float item = 0.0;
    if (fin.is_open()) {
        while (fin >> item) {
            points.push_back(item);
        }
        fin.close();
    }

    int ncols = points.size() / nrows;
    Matrixf X = Matrixf::Zero(nrows, ncols);
    for (int row = 0; row < nrows; row++) {
        for (int col = 0; col < ncols; col++) {
            X(row, col) = points[ncols * row + col];
        }
    }

    return X;
}

template <typename Word>
std::ostream& WriteWord(std::ostream& outs, Word value, unsigned size = sizeof(Word) )
{
    for (; size; --size, value >>= 8)
        outs.put( static_cast <char> (value & 0xFF) );
    return outs;
}

void Write(const std::string outpath, int sr, int channel,  const std::vector<char>& wav) {
    std::ofstream f(outpath, std::ios::binary);
    f << "RIFF----WAVEfmt ";
    WriteWord(f,          16, 4);  // no extension data
    WriteWord(f,           1, 2);  // PCM - integer samples
    WriteWord(f,     channel, 2);  // two channels (stereo file)
    WriteWord(f,          sr, 4);  // samples per second (Hz)
    WriteWord(f,  channel * sr * 2, 4);  // (Sample Rate * BitsPerSample * Channels) / 8
    WriteWord(f, channel * 2, 2);  // data block size (size of two integer samples, one for each channel, in bytes)
    WriteWord(f,          16, 2);  // number of bits per sample (use a multiple of 8)

    size_t data_chunk_pos = f.tellp();
    f << "data----";  // (chunk size to be filled in later)

    for (auto &&i : wav) f.put(i);

    size_t file_length = f.tellp();
    // Fix the data chunk header to contain the data size
    f.seekp(data_chunk_pos + 4);
    WriteWord(f, file_length - data_chunk_pos + 8);

    // Fix the file header to contain the proper RIFF chunk size, which is (file size - 8) bytes
    f.seekp(0 + 4);
    WriteWord(f, file_length - 8, 4);
    f.close();
}

int main(int argc, char* argv[])
{
    mkl_set_num_threads(2);

    string weights_file = "/home/wqt/software/clion-2020.1.2/work/wavernn_c_mb/16000/biaobei/model_mb_nostd.bin";
    char *mel_file = "/home/wqt/software/clion-2020.1.2/work/wavernn_c_mb/16000/txt/000001.txt";

    Matrixf mel = ReadArrayXXf(mel_file);

    FILE *fd = fopen(weights_file.c_str(), "rb");
    assert(fd);

    Model model;

    model.loadNext(fd);

    for(int i =0;i<1;++i) {
        std::chrono::steady_clock::time_point t1, t2;
        std::chrono::duration<double> pad_span;
        t1 = std::chrono::steady_clock::now();

        Vectorf wav = model.apply(mel);

        t2 = std::chrono::steady_clock::now();
        pad_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        std::cout << "time: " << pad_span.count() << " seconds." << std::endl;

        double points = wav.size() / pad_span.count();
        cout << "points/s:" << points << endl;

        std::cout << "done" << std::endl;
        std::cout << "-------------------------" << std::endl;
    }


    fclose(fd);
//    fclose(fdMel);
    return 0;
}
