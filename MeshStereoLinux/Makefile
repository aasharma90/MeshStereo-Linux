include Makefile.config
all: MeshStereo


MeshStereo: *.cpp *.h  Makefile
		$(CPP) $(CPPFLAGS) -o MeshStereo *.cpp $(FEATURES_UTILITY_LIB) -I$(OPENCV_INC_DIR) -I$(EIGEN_INC_DIR) -I$(PNG_INC_DIR) -I$(SLIC_INC_DIR) 
