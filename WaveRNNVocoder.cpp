
#define EIGEN_USE_MKL_ALL

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>
#include <Eigen/Core>

#include <stdio.h>
#include "net_impl.h"

namespace py = pybind11;

typedef Matrixf MatrixPy;

typedef MatrixPy::Scalar Scalar;
constexpr bool rowMajor = MatrixPy::Flags & Eigen::RowMajorBit;

class Vocoder {
    Model model;
    bool isLoaded;
public:
    Vocoder() { isLoaded = false; }
    void loadWeights( const std::string& fileName ){
        mkl_set_num_threads(2);

        FILE* fd = fopen(fileName.c_str(), "rb");
        if( not fd ){
            throw std::runtime_error("Cannot open file.");
        }
        model.loadNext(fd);
        isLoaded = true;

    }

    Vectorf melToWav( Eigen::Ref<const MatrixPy> mels ){


        if( not isLoaded ){
            throw std::runtime_error("Model hasn't been loaded. Call loadWeights first.");
        }
        return model.apply(mels);
    }

};

PYBIND11_MODULE(WaveRNNVocoder, m){
    m.doc() = "WaveRNN Vocoder";

    py::class_<MatrixPy>( m, "Matrix", py::buffer_protocol() )
            .def("__init__", [](MatrixPy &m, py::buffer b) {
        typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> Strides;

        /* Request a buffer descriptor from Python */
        py::buffer_info info = b.request();

        /* Some sanity checks ... */
        if (info.format != py::format_descriptor<Scalar>::format())
            throw std::runtime_error("Incompatible format: expected a float32 array!");

        if (info.ndim != 2)
            throw std::runtime_error("Incompatible buffer dimension!");

        auto strides = Strides(
                    info.strides[rowMajor ? 0 : 1] / (py::ssize_t)sizeof(Scalar),
                info.strides[rowMajor ? 1 : 0] / (py::ssize_t)sizeof(Scalar));

        auto map = Eigen::Map<MatrixPy, 0, Strides>(
                    static_cast<Scalar *>(info.ptr), info.shape[0], info.shape[1], strides);

        new (&m) MatrixPy(map);
    });

    py::class_<Vocoder>( m, "Vocoder")
            .def(py::init())
            .def("loadWeights", &Vocoder::loadWeights )
            .def("melToWav", &Vocoder::melToWav )
            ;
}
