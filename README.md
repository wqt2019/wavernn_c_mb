# wavernn_multi-band c++(eigen3+mkl) inference  
  
在c++下使用eigen3+mkl矩阵库实现wavernn_multi-band(cpu)的推理服务．先将模型权重保存为bin文件(非稀疏模型)，参考convert_model.py.  
  
### 编译  
  
解压eigen3，安装mkl库  
  
修改CMakeLists.txt中的eigen3 mkl的路径，修改wavernn.h net_impl.h中的参数，与训练时的参数相匹配  
  
mkdir cmake-build-debug  
  
cd cmake-build-debug  
  
cmake -D PYTHON_EXECUTABLE=/home/wqt/work/anaconda3/envs/python37/bin/python ../  
  
make -j4  
  
### 运行  
  
./cmake-build-debug/vocoder 或者 python test_wavernnvocoder.py  
  